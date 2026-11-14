"""hiVLA — a unified serving layer for deploying diverse VLA models.

hiVLA loads VLA model checkpoints, runs them behind a chosen backend
(PyTorch / ONNX / TensorRT / remote), and serves inference over the network
with serving-time optimizations (batching, caching, precision). It is the
counterpart to hiRobot: hiRobot unifies the *robot* interface, and hiVLA's
wire format follows hiRobot's ``Observation`` / ``ActionChunk`` fields without
importing it. A caller (hiAgent) moves data between the two through
:mod:`hivla.client`.

A servable model is described by a :class:`hivla.core.spec.ModelSpec`, assembled
into an :class:`hivla.serving.engine.InferenceEngine` by :func:`build_engine`.

Names are imported on first use, so ``import hivla.protocol`` or
``import hivla.client`` does not load the backends, the engine or pydantic.
"""

from importlib import import_module
from typing import Any

__version__ = "0.0.3"

#: Public name -> module that defines it.
_EXPORTS = {
    "ModelBackend": "hivla.runtimes",
    "EchoBackend": "hivla.runtimes",
    "build_backend": "hivla.runtimes",
    "infer_backend": "hivla.runtimes",
    "InferenceEngine": "hivla.serving.engine",
    "AsyncInferenceEngine": "hivla.serving.engine",
    "ObservationCache": "hivla.serving.engine",
    "VlaPolicy": "hivla.core.policy",
    "ModelSpec": "hivla.core.spec",
    "FamilyAdapter": "hivla.families",
    "get_adapter": "hivla.families",
    "available_families": "hivla.families",
    "build_engine": "hivla.core.build",
    "build_engine_from_spec": "hivla.core.build",
    "load_config": "hivla.core.config",
}

__all__ = [*_EXPORTS, "__version__"]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'hivla' has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value  # later lookups skip __getattr__
    return value


def __dir__() -> list:
    return sorted(__all__)
