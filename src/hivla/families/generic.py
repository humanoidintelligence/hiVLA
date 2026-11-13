"""The ``generic`` family: hiVLA loads a raw torch / ONNX / TensorRT checkpoint
through a :mod:`hivla.runtimes` runtime and owns normalization itself."""

from __future__ import annotations

from typing import List

from hivla.core.spec import ModelSpec
from hivla.core.transforms import DataTransformFn, Normalize, RenameKeys, Unnormalize
from hivla.families.base import FamilyAdapter
from hivla.runtimes.base import ModelBackend


class GenericFamily(FamilyAdapter):
    """hiVLA loads the raw checkpoint and owns normalization itself."""

    name = "generic"

    def build_backend(self, spec: ModelSpec) -> ModelBackend:
        from hivla.runtimes import build_backend

        return build_backend(
            spec.backend,
            {
                "checkpoint": spec.checkpoint,
                "action_dim": spec.action_dim,
                "action_horizon": spec.action_horizon,
                "device": spec.device,
                "precision": spec.precision,
            },
        )

    def build_transforms(self, spec: ModelSpec):
        inputs: List[DataTransformFn] = []
        outputs: List[DataTransformFn] = []
        if spec.rename_inputs:
            inputs.append(RenameKeys(spec.rename_inputs))
        stats = spec.load_norm_stats()
        if stats:
            inputs.append(Normalize(stats))
            outputs.append(Unnormalize(stats))
        if spec.rename_outputs:
            outputs.append(RenameKeys(spec.rename_outputs))
        return inputs, outputs
