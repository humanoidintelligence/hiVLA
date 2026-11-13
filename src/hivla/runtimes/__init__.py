"""Model backends: format-specific loaders and runners for VLA models.

A backend maps a dict of (already-preprocessed) input tensors to a dict of
decoded action-chunk arrays. ``build_backend`` constructs one by name; the
``"auto"`` name infers the format from the checkpoint (openpi-style: file
presence decides JAX vs PyTorch). Heavy dependencies (torch / onnxruntime /
tensorrt / httpx) are imported lazily only when the matching backend is built.
"""

from __future__ import annotations

import os

from hivla.runtimes.base import ModelBackend
from hivla.runtimes.echo import EchoBackend

__all__ = ["ModelBackend", "EchoBackend", "build_backend", "infer_backend"]

_BUILDERS = {}


def _register(name):
    def deco(fn):
        _BUILDERS[name] = fn
        return fn

    return deco


@_register("echo")
def _build_echo(cfg):
    return EchoBackend(
        action_dim=cfg.get("action_dim", 7),
        action_horizon=cfg.get("action_horizon", 16),
    )


@_register("torch")
def _build_torch(cfg):
    from hivla.runtimes.torch import TorchBackend

    return TorchBackend(
        checkpoint=cfg["checkpoint"],
        device=cfg.get("device", "cuda"),
        precision=cfg.get("precision", "fp32"),
        module_factory=cfg.get("module_factory"),
    )


@_register("onnx")
def _build_onnx(cfg):
    from hivla.runtimes.onnx import OnnxBackend

    return OnnxBackend(checkpoint=cfg["checkpoint"], device=cfg.get("device", "cpu"))


@_register("tensorrt")
def _build_tensorrt(cfg):
    from hivla.runtimes.tensorrt import TensorrtBackend

    return TensorrtBackend(checkpoint=cfg["checkpoint"], device=cfg.get("device", "cuda"))


@_register("remote")
def _build_remote(cfg):
    from hivla.runtimes.remote import RemoteBackend

    return RemoteBackend(
        endpoint=cfg["endpoint"],
        timeout=cfg.get("timeout", 30.0),
        api_key=cfg.get("api_key"),
        wire=cfg.get("wire", "json"),
    )


def infer_backend(checkpoint) -> str:
    """Infer the backend name from a checkpoint path (openpi-style).

    ``*.engine``/``*.plan`` -> tensorrt, ``*.onnx`` -> onnx, anything else that
    looks like weights (``.safetensors``/``.pt``/``.pth``/``.bin`` or a HF dir) ->
    torch. No checkpoint -> echo (dependency-free placeholder).
    """
    if not checkpoint:
        return "echo"
    if os.path.isdir(checkpoint):
        names = os.listdir(checkpoint)
        for n in names:
            if n.endswith((".engine", ".plan")):
                return "tensorrt"
            if n.endswith(".onnx"):
                return "onnx"
        for n in names:
            if n.endswith((".safetensors", ".bin", ".pt", ".pth")):
                return "torch"
        return "torch"
    base = os.path.basename(str(checkpoint)).lower()
    if base.endswith((".engine", ".plan")):
        return "tensorrt"
    if base.endswith(".onnx"):
        return "onnx"
    return "torch"


def build_backend(name: str, cfg: dict) -> ModelBackend:
    """Construct a backend by name (or ``"auto"``) from a config dict."""
    if name == "auto":
        name = infer_backend(cfg.get("checkpoint"))
    if name not in _BUILDERS:
        raise ValueError(f"unknown backend {name!r}; available: {sorted(_BUILDERS)}")
    return _BUILDERS[name](cfg)
