"""GR00T N1.6 / N1.7 backend: the checkpoint's own ``Gr00tPolicy``.

GR00T ships the whole observation -> action pipeline (Eagle VLM processor,
per-embodiment state/action normalization, relative->absolute action decoding)
in ``gr00t.policy.gr00t_policy.Gr00tPolicy``. hiVLA keeps that chain and only
translates between hiRobot's flat observation and GR00T's grouped, batched form:

* ``joint_pos`` is split into the embodiment's state groups, in the order of the
  checkpoint's modality config (gr1: ``left_arm, right_arm, left_hand,
  right_hand, waist``). A ``state`` dict of named groups is accepted instead.
* ``images`` (``name -> HxWx3 uint8``) map to the model's video keys.
* ``instruction`` (or ``task``) fills the single language key.

The returned action groups are concatenated in action-key order into one
``joint_targets`` array of shape ``(action_horizon, dof)``. GR00T already
unnormalizes, so values are in physical units (radians for gr1).
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hivla.families.base import FamilyAdapter, rename_transforms
from hivla.runtimes.base import ModelBackend
from hivla.core.spec import ModelSpec


class Gr00tBackend(ModelBackend):
    """Run a GR00T N1.6/N1.7 checkpoint through its ``Gr00tPolicy``.

    Args:
        spec: The :class:`~hivla.core.spec.ModelSpec`. ``checkpoint`` is the model
            directory; ``device`` and ``action_horizon`` apply.
            ``spec.extra["native_kwargs"]`` must set ``embodiment_tag`` (e.g.
            ``gr1``) and may set ``cameras`` (robot camera name -> GR00T video
            key; optional when the model has one video key and the robot sends
            one image), ``task`` (default instruction), ``strict`` (GR00T's
            input/output validation, default true) and ``compile`` (a
            ``torch.compile`` mode for the DiT action head, e.g.
            ``max-autotune``; off by default).
    """

    def __init__(self, spec: ModelSpec) -> None:
        self._spec = spec
        self._kwargs: Dict[str, Any] = dict(
            (spec.model_extra or {}).get("native_kwargs", {}) or {}
        )
        self._policy: Optional[Any] = None
        self._video_keys: List[str] = []
        self._state_layout: List[Tuple[str, int]] = []
        self._action_keys: List[str] = []
        self._language_key = ""
        # Serialize inference: at max_batch_size=1 the engine calls predict() from
        # one threadpool thread per in-flight request, so concurrent clients (e.g.
        # several envs in one eval worker) would otherwise run get_action on the
        # shared Gr00tPolicy at once. Even with GR00T_SEED pinned that interleaves
        # the CUDA matmul launch order -> bf16 reduction drifts -> actions diverge.
        # Matches OpenpiBackend / RlinfBackend. (Does not fix env-side MuJoCo RNG.)
        self._lock = threading.Lock()

    @property
    def state_layout(self) -> List[Tuple[str, int]]:
        """``(group, dim)`` pairs that ``joint_pos`` is split into, in order."""
        return list(self._state_layout)

    def load(self) -> None:
        try:
            from gr00t.data.embodiment_tags import EmbodimentTag
            from gr00t.policy.gr00t_policy import Gr00tPolicy
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "Gr00tBackend needs the 'gr00t' package (Isaac-GR00T N1.6/N1.7); "
                "serve from its env (see https://github.com/NVIDIA/Isaac-GR00T)."
            ) from exc
        if not self._spec.checkpoint:
            raise ValueError("family 'gr00t' requires model.checkpoint (model directory)")
        tag = self._kwargs.get("embodiment_tag")
        if not tag:
            raise ValueError("family 'gr00t' requires native_kwargs.embodiment_tag (e.g. gr1)")

        policy = Gr00tPolicy(
            EmbodimentTag(tag),
            self._spec.checkpoint,
            device=self._spec.device,
            strict=bool(self._kwargs.get("strict", True)),
        )
        configs = policy.get_modality_config()
        horizon = len(configs["action"].delta_indices)
        if self._spec.action_horizon > horizon:
            raise ValueError(
                f"action_horizon {self._spec.action_horizon} > GR00T {tag!r} horizon {horizon}"
            )
        for modality in ("video", "state", "language"):
            if len(configs[modality].delta_indices) != 1:
                # A history window needs per-client state hiVLA does not keep.
                raise NotImplementedError(
                    f"GR00T {tag!r} uses {len(configs[modality].delta_indices)} "
                    f"{modality} steps; only single-step observations are supported"
                )
        state_params = policy.processor.state_action_processor.norm_params[tag]["state"]
        self._state_layout = [
            (key, int(np.asarray(state_params[key]["dim"]).item()))
            for key in configs["state"].modality_keys
        ]
        self._video_keys = list(configs["video"].modality_keys)
        self._action_keys = list(configs["action"].modality_keys)
        self._language_key = configs["language"].modality_keys[0]

        mode = self._kwargs.get("compile")
        if mode:
            # Same swap as Isaac-GR00T's deployment script: only the DiT action
            # head is compiled; the Eagle backbone stays eager.
            import torch

            dit = policy.model.action_head.model
            dit.forward = torch.compile(dit.forward, mode=str(mode))
        self._policy = policy

    def _to_gr00t(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """One hiVLA observation -> GR00T's unbatched ``(T=1, ...)`` modality dict."""
        state = inputs.get("state")
        if isinstance(state, dict):
            groups = {k: np.asarray(v, dtype=np.float32).reshape(1, -1) for k, v in state.items()}
            missing = [k for k, _ in self._state_layout if k not in groups]
            if missing:
                raise ValueError(f"state is missing GR00T groups {missing}")
        else:
            if "joint_pos" not in inputs:
                raise ValueError("observation needs joint_pos (or a state dict of groups)")
            flat = np.asarray(inputs["joint_pos"], dtype=np.float32).reshape(-1)
            total = sum(dim for _, dim in self._state_layout)
            if flat.shape[0] != total:
                layout = ", ".join(f"{k}:{d}" for k, d in self._state_layout)
                raise ValueError(
                    f"joint_pos has {flat.shape[0]} dims, GR00T expects {total} ({layout})"
                )
            groups, start = {}, 0
            for key, dim in self._state_layout:
                groups[key] = flat[start:start + dim].reshape(1, dim)
                start += dim

        images = inputs.get("images") or {}
        cameras: Dict[str, str] = dict(self._kwargs.get("cameras") or {})
        if not cameras and len(self._video_keys) == 1 and len(images) == 1:
            cameras = {next(iter(images)): self._video_keys[0]}
        video = {}
        for name, img in images.items():
            key = cameras.get(name, name)
            if key in self._video_keys:
                video[key] = np.asarray(img, dtype=np.uint8)[None]  # (T=1, H, W, 3)
        missing = [k for k in self._video_keys if k not in video]
        if missing:
            raise ValueError(f"images are missing GR00T video keys {missing} (got {sorted(images)})")

        text = inputs.get("instruction") or inputs.get("task") or self._kwargs.get("task")
        if not isinstance(text, str) or not text:
            raise ValueError("observation needs an instruction (or set native_kwargs.task)")
        return {"video": video, "state": groups, "language": {self._language_key: [text]}}

    def warmup(self) -> None:
        """Run one zeroed forward so the first real request skips lazy init.

        Builds a dummy observation from the loaded modality layout: a zeroed
        ``joint_pos`` of the right total width, a 256x256 black frame per video
        key (the processor resizes, so the exact size only needs to be valid),
        and a placeholder instruction. Using each video key as its own image name
        makes ``_to_gr00t``'s camera map resolve to identity, so no camera config
        is needed. Compiling backends pay their trace here instead of mid-run.
        """
        if self._policy is None:
            raise RuntimeError("Gr00tBackend.load() must be called before warmup().")
        total = sum(dim for _, dim in self._state_layout)
        inputs = {
            "joint_pos": np.zeros(total, dtype=np.float32),
            "images": {key: np.zeros((256, 256, 3), dtype=np.uint8) for key in self._video_keys},
            "instruction": "warmup",
        }
        self.predict(inputs)

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self.predict_batch([inputs])[0]

    def predict_batch(self, batch: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """Stack the batch into one ``Gr00tPolicy.get_action`` call."""
        if self._policy is None:
            raise RuntimeError("Gr00tBackend.load() must be called before predict().")
        samples = [self._to_gr00t(inputs) for inputs in batch]
        obs: Dict[str, Any] = {
            "video": {k: np.stack([s["video"][k] for s in samples]) for k in self._video_keys},
            "state": {k: np.stack([s["state"][k] for s in samples]) for k, _ in self._state_layout},
            "language": {
                self._language_key: [s["language"][self._language_key] for s in samples]
            },
        }
        try:
            with self._lock:
                action, _ = self._policy.get_action(obs)
        except AssertionError as exc:
            # GR00T's strict checks assert; surface them as a client error.
            raise ValueError(f"GR00T rejected the observation: {exc}") from exc
        joint = np.concatenate([np.asarray(action[k]) for k in self._action_keys], axis=-1)
        horizon = self._spec.action_horizon
        return [{"joint_targets": joint[i, :horizon].astype(np.float32)} for i in range(len(batch))]

    def unload(self) -> None:
        self._policy = None


class Gr00tFamily(FamilyAdapter):
    """Delegate to an Isaac GR00T N1.6/N1.7 checkpoint's ``Gr00tPolicy``.

    GR00T normalizes, runs the VLM + DiT and decodes relative actions itself, so
    hiVLA only splits ``joint_pos`` into GR00T's state groups and joins the
    action groups back into ``joint_targets`` (see
    :class:`~hivla.families.gr00t.Gr00tBackend`).
    """

    name = "gr00t"

    def build_backend(self, spec: ModelSpec) -> ModelBackend:
        return Gr00tBackend(spec)

    def build_transforms(self, spec: ModelSpec):
        return rename_transforms(spec)
