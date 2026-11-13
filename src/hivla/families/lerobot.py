"""LeRobot (v0.6.2) backend: a checkpoint's policy plus its processor pipelines.

LeRobot >= 0.4 moved normalization out of the policy into serialized
``PolicyProcessorPipeline`` pairs saved next to the weights
(``policy_preprocessor.json`` / ``policy_postprocessor.json``). Serving a
checkpoint therefore means reproducing the chain lerobot's own
``SyncInferenceEngine`` runs::

    prepare_observation_for_inference -> preprocessor -> policy -> postprocessor

hiVLA returns a whole action chunk per request, so the policy is entered via
``predict_action_chunk`` (stateless, ``(B, chunk, dof)``) rather than
``select_action`` (which pops one action from an internal queue).

Only checkpoints in the processor format are supported. Legacy checkpoints (the
original ``lerobot/act_*`` / ``lerobot/diffusion_*`` hub repos, which carry
normalization buffers in ``model.safetensors``) must be converted once with
lerobot's ``src/lerobot/processor/migrate_policy_normalization.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from hivla.families.base import FamilyAdapter, rename_transforms
from hivla.runtimes.base import ModelBackend
from hivla.core.spec import ModelSpec


class LerobotBackend(ModelBackend):
    """Run a LeRobot ``PreTrainedPolicy`` with its pre/post-processor pipelines.

    Args:
        spec: The :class:`~hivla.core.spec.ModelSpec`. ``checkpoint`` is a local
            directory or hub repo id; ``device`` and ``action_horizon`` apply.
            ``spec.extra["native_kwargs"]`` may set ``task`` (default language
            instruction), ``robot_type`` and ``revision`` (hub revision).
    """

    def __init__(self, spec: ModelSpec) -> None:
        self._spec = spec
        self._kwargs: Dict[str, Any] = dict(
            (spec.model_extra or {}).get("native_kwargs", {}) or {}
        )
        self._policy: Optional[Any] = None
        self._preprocessor: Optional[Any] = None
        self._postprocessor: Optional[Any] = None
        self._prepare: Optional[Any] = None
        self._device: Optional[Any] = None

    def load(self) -> None:
        try:
            import torch
            from lerobot.configs import PreTrainedConfig
            from lerobot.policies import get_policy_class, prepare_observation_for_inference
            from lerobot.processor import load_pretrained_policy_processors
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "LerobotBackend needs the 'lerobot' package (v0.6.2); install it "
                "into the serving env (see https://github.com/huggingface/lerobot)."
            ) from exc
        if not self._spec.checkpoint:
            raise ValueError("family 'lerobot' requires model.checkpoint (dir or hub repo id)")

        path = self._spec.checkpoint
        revision = self._kwargs.get("revision")
        cfg = PreTrainedConfig.from_pretrained(path, revision=revision)
        if getattr(cfg, "n_obs_steps", 1) > 1:
            # predict_action_chunk on a fresh request has no observation history;
            # serving such policies needs per-client state hiVLA does not keep.
            raise NotImplementedError(
                f"lerobot policy {cfg.type!r} uses n_obs_steps={cfg.n_obs_steps}; "
                "only single-observation policies (n_obs_steps=1, e.g. ACT) are supported"
            )
        cfg.device = self._spec.device
        policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg, revision=revision)
        policy.eval()

        # Legacy checkpoints raise lerobot's ProcessorMigrationError here, whose
        # message already carries the migration command.
        preprocessor, postprocessor = load_pretrained_policy_processors(
            path,
            revision=revision,
            preprocessor_overrides={"device_processor": {"device": self._spec.device}},
        )

        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._prepare = prepare_observation_for_inference
        self._device = torch.device(self._spec.device)

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self._policy is None:
            raise RuntimeError("LerobotBackend.load() must be called before predict().")
        import torch

        obs: Dict[str, Any] = {}
        task = self._kwargs.get("task")
        for key, val in inputs.items():
            if key == "task" and isinstance(val, str):
                task = val
                continue
            arr = np.asarray(val)
            if arr.dtype == np.float64:
                arr = arr.astype(np.float32)  # lerobot weights are fp32
            obs[key] = arr

        with torch.inference_mode():
            batch = self._prepare(obs, self._device, task, self._kwargs.get("robot_type"))
            batch = self._preprocessor(batch)
            chunk = self._policy.predict_action_chunk(batch)
            chunk = self._postprocessor(chunk)

        arr = chunk.detach().cpu().numpy() if hasattr(chunk, "detach") else np.asarray(chunk)
        if arr.ndim == 3:
            arr = arr[0]  # drop the batch dim -> (chunk, dof)
        horizon = self._spec.action_horizon
        if arr.shape[0] < horizon:
            raise ValueError(
                f"lerobot policy predicts {arr.shape[0]} steps < action_horizon {horizon}"
            )
        return {"joint_targets": arr[:horizon].astype(np.float32)}

    def unload(self) -> None:
        self._policy = self._preprocessor = self._postprocessor = None


class LerobotFamily(FamilyAdapter):
    """Delegate to a LeRobot v0.6.2 checkpoint and its processor pipelines.

    The checkpoint's own ``policy_preprocessor`` / ``policy_postprocessor``
    normalize and unnormalize, so hiVLA only renames keys; obs should arrive
    with lerobot feature names (``observation.state``, ``observation.images.<cam>``
    as ``(H, W, 3)`` uint8, optional ``task`` string).
    """

    name = "lerobot"

    def build_backend(self, spec: ModelSpec) -> ModelBackend:
        return LerobotBackend(spec)

    def build_transforms(self, spec: ModelSpec):
        return rename_transforms(spec)
