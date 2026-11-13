"""The :class:`FamilyAdapter` contract every model family implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, List, Tuple

from hivla.core.spec import ModelSpec
from hivla.core.transforms import DataTransformFn, RenameKeys
from hivla.runtimes.base import ModelBackend


class FamilyAdapter(ABC):
    """Builds the backend and I/O transforms for one model family from a spec."""

    name: ClassVar[str] = "generic"

    @abstractmethod
    def build_backend(self, spec: ModelSpec) -> ModelBackend:
        """Return a loaded-ready backend for ``spec``."""

    @abstractmethod
    def build_transforms(
        self, spec: ModelSpec
    ) -> Tuple[List[DataTransformFn], List[DataTransformFn]]:
        """Return ``(input_transforms, output_transforms)`` wrapping the backend."""


def rename_transforms(spec: ModelSpec) -> Tuple[List[DataTransformFn], List[DataTransformFn]]:
    """Only ``rename_inputs`` / ``rename_outputs``: for families whose model
    normalizes and decodes by itself."""
    inputs: List[DataTransformFn] = [RenameKeys(spec.rename_inputs)] if spec.rename_inputs else []
    outputs: List[DataTransformFn] = (
        [RenameKeys(spec.rename_outputs)] if spec.rename_outputs else [])
    return inputs, outputs
