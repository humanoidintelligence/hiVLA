"""openpi backend: a π0 / π0.5 / π0-FAST checkpoint served via ``Policy.infer``.

openpi's ``create_trained_policy`` already owns normalization, tokenization and
the robot-specific input/output transforms (``LiberoInputs``, ``AlohaInputs``,
...), so hiVLA only maps canonical keys onto the dict those transforms read
and does what openpi's eval clients do before sending (``examples/libero/main.py``):

* ``images[robot_cam]`` -> ``obs[openpi_key]`` per ``native_kwargs.cameras``,
  resized with ``openpi_client.image_tools.resize_with_pad`` to ``image_size``
  (224) and converted to uint8.
* ``joint_pos`` -> ``obs[state_key]`` (``observation/state``). Drivers put the
  model's state vector there (LIBERO: eef pos 3 | axis-angle 3 | gripper qpos 2).
* ``instruction`` -> ``prompt`` (``native_kwargs.task`` is openpi's default prompt).

``Policy.infer`` returns unnormalized ``actions (action_horizon, action_dim)``.
``native_kwargs.action_layout`` names how they leave the server: a list of
``[field, name, size]`` slices, ``field`` in ``ee_targets`` / ``gripper`` /
``joint_targets``. LIBERO is ``[[ee_targets, eef, 6], [gripper, gripper, 1]]``
with ``control_mode: cartesian_delta``; without a layout the whole vector is
``joint_targets`` (ALOHA, DROID joint configs).
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hivla.families.base import FamilyAdapter, rename_transforms
from hivla.runtimes.base import ModelBackend
from hivla.core.spec import ModelSpec

_FIELDS = ("joint_targets", "ee_targets", "gripper")


def parse_action_layout(layout: Any) -> List[Tuple[str, str, int]]:
    """Validate ``[[field, name, size], ...]`` into tuples."""
    out: List[Tuple[str, str, int]] = []
    for entry in layout or []:
        field, name, size = entry
        if field not in _FIELDS:
            raise ValueError(f"action_layout field must be one of {_FIELDS}, got {field!r}")
        out.append((str(field), str(name), int(size)))
    if sum(1 for f, _, _ in out if f == "joint_targets") > 1:
        raise ValueError("action_layout may have at most one joint_targets slice")
    return out


def split_actions(actions: np.ndarray, layout: List[Tuple[str, str, int]]) -> Dict[str, Any]:
    """Slice ``(T, D)`` actions into a chunk dict per ``layout`` (all of D if empty)."""
    actions = np.asarray(actions, dtype=np.float32)
    if not layout:
        return {"joint_targets": actions}
    width = sum(size for _, _, size in layout)
    if actions.shape[-1] != width:
        raise ValueError(f"actions have {actions.shape[-1]} dims, action_layout covers {width}")
    chunk: Dict[str, Any] = {}
    start = 0
    for field, name, size in layout:
        part = actions[:, start:start + size]
        start += size
        if field == "joint_targets":
            chunk["joint_targets"] = part
        else:
            # gripper commands are (T,) per name; ee_targets stay (T, size).
            chunk.setdefault(field, {})[name] = part[:, 0] if field == "gripper" else part
    return chunk


class OpenpiBackend(ModelBackend):
    """Run an openpi checkpoint directory (``params/`` + ``assets/``).

    Args:
        spec: The :class:`~hivla.core.spec.ModelSpec`. ``checkpoint`` is a local dir
            or a ``gs://openpi-assets/...`` URL (resolved through openpi's
            ``OPENPI_DATA_HOME`` cache); ``action_horizon`` and ``action_dim``
            apply. ``spec.extra["native_kwargs"]`` may set:

            * ``config``: openpi train-config name (required, e.g. ``pi05_libero``).
            * ``cameras``: ``{robot_camera: openpi_key}``.
            * ``state_key``: where ``joint_pos`` goes (default ``observation/state``).
            * ``image_size``: ``[h, w]`` for ``resize_with_pad`` (default 224x224).
            * ``action_layout``: see the module docstring.
            * ``task``: default prompt.
            * ``seed``: JAX sampling seed (openpi defaults to 0).
            * ``sample_kwargs``: passed to ``sample_actions`` (e.g. ``num_steps``).
    """

    def __init__(self, spec: ModelSpec) -> None:
        self._spec = spec
        self._kwargs: Dict[str, Any] = dict(
            (spec.model_extra or {}).get("native_kwargs", {}) or {}
        )
        if not self._kwargs.get("config"):
            raise ValueError("family 'openpi' requires native_kwargs.config (e.g. pi05_libero)")
        self._cameras: Dict[str, str] = dict(self._kwargs.get("cameras") or {})
        if not self._cameras:
            raise ValueError("family 'openpi' requires native_kwargs.cameras {robot: openpi key}")
        self._state_key: str = self._kwargs.get("state_key", "observation/state")
        h, w = (int(s) for s in self._kwargs.get("image_size", (224, 224)))
        self._image_size = (h, w)
        self._layout = parse_action_layout(self._kwargs.get("action_layout"))
        self._policy: Optional[Any] = None
        self._image_tools: Any = None
        self._lock = threading.Lock()

    def load(self) -> None:
        try:
            from openpi.policies import policy_config
            from openpi.training import config as openpi_config
            from openpi_client import image_tools
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "OpenpiBackend needs the 'openpi' package; serve from the openpi env "
                "(see https://github.com/Physical-Intelligence/openpi)."
            ) from exc
        if not self._spec.checkpoint:
            raise ValueError("family 'openpi' requires model.checkpoint (dir or gs:// URL)")

        train_config = openpi_config.get_config(self._kwargs["config"])
        horizon = int(train_config.model.action_horizon)
        if self._spec.action_horizon > horizon:
            raise ValueError(
                f"action_horizon {self._spec.action_horizon} > checkpoint horizon {horizon}"
            )
        policy = policy_config.create_trained_policy(
            train_config,
            self._spec.checkpoint,
            default_prompt=self._kwargs.get("task"),
            sample_kwargs=self._kwargs.get("sample_kwargs"),
        )
        if self._kwargs.get("seed") is not None:
            import jax

            # create_trained_policy has no rng argument; Policy keeps it in _rng.
            policy._rng = jax.random.key(int(self._kwargs["seed"]))
        self._image_tools = image_tools
        self._policy = policy

    def _obs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        images = inputs.get("images") or {}
        missing = [n for n in self._cameras if n not in images]
        if missing:
            raise ValueError(f"images are missing cameras {missing} (got {sorted(images)})")
        h, w = self._image_size
        obs: Dict[str, Any] = {
            key: self._image_tools.convert_to_uint8(
                self._image_tools.resize_with_pad(np.asarray(images[cam]), h, w)
            )
            for cam, key in self._cameras.items()
        }
        if "joint_pos" not in inputs:
            raise ValueError("observation needs joint_pos (the model's state vector)")
        obs[self._state_key] = np.asarray(inputs["joint_pos"], dtype=np.float32).reshape(-1)
        text = inputs.get("instruction") or inputs.get("task")
        if text:
            obs["prompt"] = str(text)
        return obs

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self._policy is None:
            raise RuntimeError("OpenpiBackend.load() must be called before predict().")
        obs = self._obs(inputs)
        # Policy.infer splits self._rng on every call; the server runs requests
        # from several connections in a thread pool.
        with self._lock:
            out = self._policy.infer(obs)
        actions = np.asarray(out["actions"])[: self._spec.action_horizon]
        return split_actions(actions, self._layout)

    # predict_batch: the base class's sequential fallback. Policy.infer is
    # unbatched (it adds the batch axis itself and JITs for batch size 1).

    def unload(self) -> None:
        self._policy = None


class OpenpiFamily(FamilyAdapter):
    """Serve a Physical Intelligence π0 / π0.5 / π0-FAST checkpoint (JAX or PyTorch).

    openpi's ``create_trained_policy`` builds the normalization and robot I/O
    transforms from a named train config; hiVLA maps canonical keys onto that
    config's input dict and slices the actions into ``ee_targets`` / ``gripper``
    / ``joint_targets`` (see :class:`~hivla.families.openpi.OpenpiBackend`).
    """

    name = "openpi"

    def build_backend(self, spec: ModelSpec) -> ModelBackend:
        return OpenpiBackend(spec)

    def build_transforms(self, spec: ModelSpec):
        return rename_transforms(spec)
