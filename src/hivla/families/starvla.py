"""StarVLA backend: a ``baseframework`` checkpoint (QwenGR00T, QwenOFT, ...).

StarVLA's ``predict_action`` returns *normalized* actions and leaves state
encoding, image resizing and action unnormalization to the benchmark client
(``examples/<benchmark>/eval_files/model2*_interface.py``). hiVLA moves that
client-side work here so hiRobot can send plain observations, and reproduces
StarVLA's RoboCasa GR1 client (``model2robocasa_interface.py``) exactly:

* ``joint_pos`` is split into ``state_groups`` (gr1: ``7, 7, 6, 6, 3``) and, with
  ``state_encoding: sincos``, each group becomes ``[sin(g), cos(g)]`` (29 -> 58).
* ``images`` are taken in ``policy.cameras`` order and resized with OpenCV
  ``INTER_AREA`` to ``image_size`` (224x224).
* ``instruction`` (or ``task``) becomes ``lang``.
* ``normalized_actions`` are clipped to [-1, 1] and mapped back with the
  checkpoint's ``dataset_statistics.json`` min/max (masked dims pass through),
  returned as ``joint_targets`` of shape ``(action_horizon, action_dim)``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from hivla.families.base import FamilyAdapter, rename_transforms
from hivla.runtimes.base import ModelBackend
from hivla.core.spec import ModelSpec


def sincos_encode(groups: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate ``[sin(g), cos(g)]`` for each group, in order (StarVLA GR1)."""
    parts = [np.concatenate([np.sin(g), np.cos(g)], axis=-1) for g in groups]
    return np.concatenate(parts, axis=-1)


def minmax_unnormalize(normalized: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    """StarVLA's action unnormalization: clip to [-1, 1], then min/max where masked."""
    low = np.asarray(stats["min"], dtype=np.float64)
    high = np.asarray(stats["max"], dtype=np.float64)
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    clipped = np.clip(normalized, -1.0, 1.0)
    return np.where(mask, (clipped + 1.0) / 2.0 * (high - low) + low, clipped)


class StarvlaBackend(ModelBackend):
    """Run a StarVLA checkpoint (``.../checkpoints/steps_N_pytorch_model.pt``).

    Args:
        spec: The :class:`~hivla.core.spec.ModelSpec`. ``checkpoint`` is the ``.pt``
            file (``config.yaml`` and ``dataset_statistics.json`` sit two levels
            up, as StarVLA's ``read_mode_config`` expects); ``device``,
            ``action_horizon`` and ``cameras`` apply. ``spec.extra["native_kwargs"]``
            may set:

            * ``base_vlm``: override ``framework.qwenvl.base_vlm`` (checkpoints
              store a path relative to the training cwd).
            * ``unnorm_key``: dataset-statistics key (optional if there is one).
            * ``state_groups``: group sizes ``joint_pos`` is split into.
            * ``state_encoding``: ``sincos`` (StarVLA GR1) or ``none``.
            * ``image_size``: ``[w, h]`` resize target (default ``[224, 224]``).
            * ``bf16``: cast weights to bfloat16 (default true, as the eval server).
            * ``compile``: ``torch.compile`` mode for the DiT action head
              (e.g. ``max-autotune``); off by default.
            * ``task``: default instruction.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self._spec = spec
        self._kwargs: Dict[str, Any] = dict(
            (spec.model_extra or {}).get("native_kwargs", {}) or {}
        )
        self._model: Optional[Any] = None
        self._action_stats: Dict[str, Any] = {}
        self._state_groups: List[int] = [int(d) for d in self._kwargs.get("state_groups", [])]
        encoding = self._kwargs.get("state_encoding", "none")
        if encoding not in ("sincos", "none"):
            raise ValueError(f"state_encoding must be 'sincos' or 'none', got {encoding!r}")
        self._encoding = encoding
        self._image_size = tuple(int(s) for s in self._kwargs.get("image_size", (224, 224)))
        cams = spec.cameras
        self._cameras: List[str] = list(cams.keys() if isinstance(cams, dict) else cams or [])
        self._cv2: Any = None

    def load(self) -> None:
        try:
            import cv2
            import torch
            from starVLA.model.framework import build_framework
            from starVLA.model.framework.share_tools import dict_to_namespace, read_mode_config
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "StarvlaBackend needs the 'starVLA' package (and opencv); serve from "
                "a StarVLA env (see https://github.com/starVLA/starVLA)."
            ) from exc
        if not self._spec.checkpoint:
            raise ValueError("family 'starvla' requires model.checkpoint (the .pt file)")

        # Mirrors baseframework.from_pretrained, plus the base_vlm override.
        cfg, norm_stats = read_mode_config(self._spec.checkpoint)
        if self._kwargs.get("base_vlm"):
            cfg["framework"]["qwenvl"]["base_vlm"] = self._kwargs["base_vlm"]
        config = dict_to_namespace(cfg)
        config.trainer.pretrained_checkpoint = None
        model = build_framework(cfg=config)
        state_dict = torch.load(self._spec.checkpoint, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        del state_dict
        if self._kwargs.get("bf16", True):
            model = model.to(torch.bfloat16)
        model = model.to(self._spec.device).eval()

        mode = self._kwargs.get("compile")
        if mode:
            head = model.action_model.model
            head.forward = torch.compile(head.forward, mode=str(mode))

        key = self._kwargs.get("unnorm_key")
        if key is None:
            if len(norm_stats) != 1:
                raise ValueError(
                    f"checkpoint has several dataset statistics {sorted(norm_stats)}; "
                    "set native_kwargs.unnorm_key"
                )
            key = next(iter(norm_stats))
        if key not in norm_stats:
            raise ValueError(f"unnorm_key {key!r} not in dataset statistics {sorted(norm_stats)}")
        self._action_stats = norm_stats[key]["action"]
        dim = len(self._action_stats["min"])
        if dim != self._spec.action_dim:
            raise ValueError(
                f"checkpoint action dim {dim} != policy.action_dim {self._spec.action_dim}"
            )
        horizon = int(config.framework.action_model.action_horizon)
        if self._spec.action_horizon > horizon:
            raise ValueError(
                f"action_horizon {self._spec.action_horizon} > checkpoint horizon {horizon}"
            )
        self._cv2 = cv2
        self._model = model

    def _state(self, inputs: Dict[str, Any]) -> Optional[np.ndarray]:
        """``joint_pos`` -> the model's ``(1, state_dim)`` state, or None if unused."""
        if not self._state_groups:
            return None
        if "joint_pos" not in inputs:
            raise ValueError("observation needs joint_pos")
        flat = np.asarray(inputs["joint_pos"], dtype=np.float64).reshape(-1)
        total = sum(self._state_groups)
        if flat.shape[0] != total:
            raise ValueError(
                f"joint_pos has {flat.shape[0]} dims, expected {total} {self._state_groups}"
            )
        groups = np.split(flat, np.cumsum(self._state_groups)[:-1])
        state = sincos_encode(groups) if self._encoding == "sincos" else flat
        return state.reshape(1, -1)

    def _example(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        images = inputs.get("images") or {}
        names = self._cameras or sorted(images)
        missing = [n for n in names if n not in images]
        if missing or not names:
            raise ValueError(
                f"images are missing cameras {missing or names} (got {sorted(images)})"
            )
        views = [
            self._cv2.resize(np.asarray(images[n], dtype=np.uint8), self._image_size,
                             interpolation=self._cv2.INTER_AREA)
            for n in names
        ]
        text = inputs.get("instruction") or inputs.get("task") or self._kwargs.get("task")
        if not isinstance(text, str) or not text:
            raise ValueError("observation needs an instruction (or set native_kwargs.task)")
        example = {"image": views, "lang": text}
        state = self._state(inputs)
        if state is not None:
            example["state"] = state
        return example

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self.predict_batch([inputs])[0]

    def predict_batch(self, batch: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """Run the batch through one ``predict_action`` call."""
        if self._model is None:
            raise RuntimeError("StarvlaBackend.load() must be called before predict().")
        import torch

        examples = [self._example(inputs) for inputs in batch]
        with torch.inference_mode():
            out = self._model.predict_action(examples=examples)
        actions = minmax_unnormalize(np.asarray(out["normalized_actions"], dtype=np.float64),
                                     self._action_stats)
        horizon = self._spec.action_horizon
        return [{"joint_targets": a[:horizon].astype(np.float32)} for a in actions]

    def unload(self) -> None:
        self._model = None


class StarvlaFamily(FamilyAdapter):
    """Serve a StarVLA ``baseframework`` checkpoint (QwenGR00T, QwenOFT, ...).

    StarVLA leaves state encoding, image resizing and action unnormalization to
    its benchmark clients; hiVLA does them server-side (see
    :class:`~hivla.families.starvla.StarvlaBackend`) so the robot
    sends plain ``joint_pos`` / ``images`` / ``instruction``.
    """

    name = "starvla"

    def build_backend(self, spec: ModelSpec) -> ModelBackend:
        return StarvlaBackend(spec)

    def build_transforms(self, spec: ModelSpec):
        return rename_transforms(spec)
