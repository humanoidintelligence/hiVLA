"""PyTorch model backend.

Loads a TorchScript / ``torch.load``-able checkpoint and runs it as a pure
tensor-in / tensor-out forward pass. Normalization, padding, tokenization and
action decoding are **not** here — they live in :class:`~hivla.core.transforms`
and are driven by :class:`~hivla.core.policy.VlaPolicy`. ``torch`` is imported
lazily so the core package (and the echo-backend tests) need no torch.

This is a skeleton: :meth:`predict` maps a dict of arrays to tensors, calls the
module, and maps the result back. Adapt the forward signature and output key(s)
to your checkpoint when a real model lands.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional

import numpy as np

from hivla.runtimes.base import ModelBackend


class TorchBackend(ModelBackend):
    """Run a PyTorch VLA checkpoint behind the backend contract.

    Two loading modes:

    * Whole-module: ``torch.load`` a serialized ``nn.Module`` / ``ScriptModule``
      (the default).
    * Weights-only: provide ``module_factory`` (a callable or ``"dotted.path"``)
      that builds the architecture; weights are then loaded from ``checkpoint``
      via ``torch.load`` (``.pt``/``.pth``) or ``safetensors`` (``.safetensors``).

    Args:
        checkpoint: Path to the checkpoint to load.
        device: Torch device string (``"cuda"`` / ``"cpu"``).
        precision: ``"fp32"`` / ``"fp16"`` / ``"bf16"``.
        module_factory: Optional architecture builder (callable or dotted path).
    """

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        precision: str = "fp32",
        module_factory: Optional[Any] = None,
    ) -> None:
        self._checkpoint = checkpoint
        self._device = device
        self._precision = precision
        self._module_factory = module_factory
        self._torch = None
        self._module: Optional[Any] = None

    def load(self) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "TorchBackend needs torch; install the 'torch' extra: "
                "pip install 'hivla[torch]'."
            ) from exc
        self._torch = torch
        if self._module_factory is not None:
            module = self._build_module()
            module.load_state_dict(self._load_state_dict())
        else:
            module = torch.load(self._checkpoint, map_location=self._device)
        if hasattr(module, "eval"):
            module.eval()
        self._module = module

    def _build_module(self) -> Any:
        factory = self._module_factory
        if isinstance(factory, str):
            mod_path, _, attr = factory.rpartition(".")
            factory = getattr(importlib.import_module(mod_path), attr)
        return factory()

    def _load_state_dict(self) -> Dict[str, Any]:
        torch = self._torch
        if str(self._checkpoint).endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "Loading safetensors needs 'safetensors'; install the 'torch' extra."
                ) from exc
            return load_file(self._checkpoint)
        return torch.load(self._checkpoint, map_location=self._device)

    def _dtype(self):
        return {
            "fp32": self._torch.float32,
            "fp16": self._torch.float16,
            "bf16": self._torch.bfloat16,
        }.get(self._precision, self._torch.float32)

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self._module is None:
            raise RuntimeError("TorchBackend.load() must be called before predict().")
        torch = self._torch
        tensors = {}
        for key, val in inputs.items():
            if isinstance(val, np.ndarray) and val.dtype.kind in "fiub":
                t = torch.from_numpy(np.ascontiguousarray(val)).to(self._device)
                if val.dtype.kind == "f":
                    t = t.to(self._dtype())
                tensors[key] = t.unsqueeze(0)  # add batch dim
            else:
                tensors[key] = val

        with torch.no_grad():
            out = self._module(tensors)

        # Normalize the output into a dict of de-batched numpy arrays.
        if not isinstance(out, dict):
            out = {"joint_targets": out}
        result: Dict[str, Any] = {}
        for key, val in out.items():
            if hasattr(val, "detach"):
                result[key] = val.detach().float().cpu().numpy()[0]
            else:
                result[key] = val
        return result
