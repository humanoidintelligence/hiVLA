"""RLinf backend: an RLinf π0 / π0.5 checkpoint via ``rlinf.models.embodiment.openpi``.

RLinf reimplements openpi π0/π0.5 in PyTorch (numerically aligned with the JAX
reference) and publishes its SFT and RL-finetuned checkpoints in that form:
``model.safetensors`` plus ``<asset_id>/norm_stats.json`` (e.g.
``RLinf/RLinf-Pi05-LIBERO-SFT``). ``get_model(cfg)`` with ``openpi.task: eval``
builds ``Pi0Eval`` and installs openpi's input/output transforms for
``config_name``, so normalization, the 224 resize and tokenization happen inside
the model, as in RLinf's own eval.

``Pi0Eval.predict_action_batch(env_obs)`` reads RLinf's env-worker dict (batched):

* ``images[robot_cam]`` -> ``env_obs[key]`` per ``native_kwargs.cameras``
  (LIBERO: ``agentview: main_images``, ``wrist: wrist_images``), uint8 as captured.
* ``joint_pos`` -> ``states`` (LIBERO: eef pos 3 | axis-angle 3 | gripper qpos 2).
* ``instruction`` -> ``task_descriptions``.

An observation that already carries ``main_images`` (openpi-wire form) is passed
through as is. Actions come back unnormalized as
``(action_horizon, action_dim)`` and leave per ``native_kwargs.action_layout``
(see :mod:`hivla.families.openpi`). ``predict_batch`` stacks requests
into one forward pass.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

import numpy as np

from hivla.families.base import FamilyAdapter, rename_transforms
from hivla.runtimes.base import ModelBackend
from hivla.families.openpi import parse_action_layout, split_actions
from hivla.core.spec import ModelSpec

# Pi0.5 shape and eval runtime, from RLinf's examples/embodiment/config/model/pi0_5.yaml.
_OPENPI_DEFAULTS: Dict[str, Any] = {
    "task": "eval",
    "model_action_dim": 32,
    "paligemma_variant": "gemma_2b",
    "action_expert_variant": "gemma_300m",
    "max_token_len": 200,
    "torch_compile": True,
}


class RlinfBackend(ModelBackend):
    """Run an RLinf openpi checkpoint (``model.safetensors`` + norm stats).

    Args:
        spec: The :class:`~hivla.core.spec.ModelSpec`. ``checkpoint`` is the local
            checkpoint dir; ``action_horizon`` is the executed chunk RLinf returns
            (``num_action_chunks``). ``spec.extra["native_kwargs"]`` may set:

            * ``config``: openpi train-config name (required, e.g. ``pi05_libero``).
            * ``cameras``: ``{robot_camera: env_obs key}``.
            * ``action_layout``: ``[[field, name, size], ...]``.
            * ``pi05``: π0.5 (default) or π0.
            * ``num_steps``: flow-matching denoise steps (RLinf's LIBERO evals: 3-5).
            * ``precision``: parameter dtype (``bf16`` ...; default: checkpoint's).
            * ``seed``: seed of the CUDA generator for the initial noise.
            * ``openpi``: extra ``cfg.openpi`` fields (e.g. ``torch_compile: false``).
    """

    def __init__(self, spec: ModelSpec) -> None:
        self._spec = spec
        self._kwargs: Dict[str, Any] = dict(
            (spec.model_extra or {}).get("native_kwargs", {}) or {}
        )
        if not self._kwargs.get("config"):
            raise ValueError("family 'rlinf' requires native_kwargs.config (e.g. pi05_libero)")
        self._cameras: Dict[str, str] = dict(self._kwargs.get("cameras") or {})
        self._layout = parse_action_layout(self._kwargs.get("action_layout"))
        self._model: Optional[Any] = None
        self._rng: Optional[Any] = None
        self._lock = threading.Lock()

    def build_cfg(self) -> Dict[str, Any]:
        """The ``cfg`` handed to ``get_model`` (an ``actor.model`` section)."""
        kw = self._kwargs
        horizon = int(self._spec.action_horizon)
        num_steps = int(kw.get("num_steps", 5))
        openpi = dict(_OPENPI_DEFAULTS)
        openpi.update(
            config_name=kw["config"],
            action_chunk=horizon,
            action_env_dim=int(self._spec.action_dim),
            num_steps=num_steps,
        )
        openpi.update(kw.get("openpi") or {})
        return {
            "model_type": "openpi",
            "model_path": self._spec.checkpoint,
            "precision": kw.get("precision"),
            "pi05": bool(kw.get("pi05", True)),
            "num_action_chunks": horizon,
            "action_dim": int(self._spec.action_dim),
            "num_steps": num_steps,
            "openpi": openpi,
        }

    def load(self) -> None:
        try:
            import torch
            from omegaconf import OmegaConf
            from rlinf.models.embodiment.openpi import get_model
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "RlinfBackend needs the 'rlinf' package; serve from the rlinf env "
                "(see https://github.com/RLinf/RLinf)."
            ) from exc
        if not self._spec.checkpoint:
            raise ValueError("family 'rlinf' requires model.checkpoint (local dir)")
        model = get_model(OmegaConf.create(self.build_cfg()))
        device = self._spec.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self._model = model.to(device).eval()
        if self._kwargs.get("seed") is not None:
            self._rng = torch.Generator(device=device).manual_seed(int(self._kwargs["seed"]))

    def _env_obs(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stack canonical observations into RLinf's batched env-obs dict."""
        if all("main_images" in obs for obs in batch):
            return {
                key: [obs[key] for obs in batch] if key == "task_descriptions"
                else np.stack([np.asarray(obs[key]) for obs in batch])
                for key in batch[0]
            }
        env_obs: Dict[str, Any] = {}
        for cam, key in self._cameras.items():
            frames = []
            for obs in batch:
                images = obs.get("images") or {}
                if cam not in images:
                    raise ValueError(f"images are missing camera {cam!r} (got {sorted(images)})")
                frames.append(np.asarray(images[cam], dtype=np.uint8))
            env_obs[key] = np.stack(frames)
        if any("joint_pos" not in obs for obs in batch):
            raise ValueError("observation needs joint_pos (the model's state vector)")
        env_obs["states"] = np.stack(
            [np.asarray(obs["joint_pos"], dtype=np.float32).reshape(-1) for obs in batch]
        )
        env_obs["task_descriptions"] = [
            str(obs.get("instruction") or obs.get("task") or "") for obs in batch
        ]
        return env_obs

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self.predict_batch([inputs])[0]

    def predict_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run the batch through one ``predict_action_batch`` call."""
        if self._model is None:
            raise RuntimeError("RlinfBackend.load() must be called before predict().")
        env_obs = self._env_obs(batch)
        # The generator advances on every call; requests arrive from a thread pool.
        with self._lock:
            actions, _ = self._model.predict_action_batch(env_obs, mode="eval", rng=self._rng)
        if hasattr(actions, "detach"):
            actions = actions.detach().float().cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)
        horizon = self._spec.action_horizon
        return [split_actions(a[:horizon], self._layout) for a in actions]

    def unload(self) -> None:
        self._model = None
        self._rng = None


class RlinfFamily(FamilyAdapter):
    """Delegate to RLinf's openpi PyTorch reimplementation.

    RLinf loads π0/π0.5 via ``rlinf.models.embodiment.openpi.get_model``. The
    model does its own preprocessing/normalization, so hiVLA only renames keys;
    obs should arrive
    in openpi-wire form (``main_images``/``states``/``task_descriptions``).
    """

    name = "rlinf"

    def build_backend(self, spec: ModelSpec) -> ModelBackend:
        return RlinfBackend(spec)

    def build_transforms(self, spec: ModelSpec):
        return rename_transforms(spec)
