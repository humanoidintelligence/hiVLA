"""TensorRT backend.

Runs a serialized TensorRT engine (``*.engine`` / ``*.plan``) as a pure
tensor-in / tensor-out forward pass. ``tensorrt`` and ``torch`` are imported
lazily; install the ``tensorrt`` extra (and have a CUDA GPU) to use it.

This is a static-shape engine runner: input tensors are matched to the engine's
input bindings by name (falling back to positional order), outputs are returned
as de-batched numpy arrays. It is not exercised in CI — it needs a GPU and a
compiled engine — but follows the standard TRT runtime API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from hivla.runtimes.base import ModelBackend


class TensorrtBackend(ModelBackend):
    """Run a TensorRT engine behind the backend contract.

    Args:
        checkpoint: Path to the serialized engine (``*.engine``/``*.plan``).
        device: CUDA device index (default 0).
    """

    def __init__(self, checkpoint: str, device: str = "cuda") -> None:
        self._checkpoint = checkpoint
        self._device_index = int(device.split(":")[-1]) if device.startswith("cuda") else 0
        self._trt: Optional[Any] = None
        self._torch: Optional[Any] = None
        self._engine: Optional[Any] = None
        self._context: Optional[Any] = None

    def load(self) -> None:
        try:
            import tensorrt as trt
            import torch
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "TensorrtBackend needs tensorrt + torch; install the 'tensorrt' extra."
            ) from exc
        self._trt, self._torch = trt, torch
        with open(self._checkpoint, "rb") as f:
            engine_bytes = f.read()
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self._engine = runtime.deserialize_cuda_engine(engine_bytes)
        self._context = self._engine.create_execution_context()

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self._engine is None:
            raise RuntimeError("TensorrtBackend.load() must be called before predict().")
        trt, torch = self._trt, self._torch
        with torch.cuda.device(self._device_index):
            bindings: Dict[str, Any] = {}
            input_arrays = list(inputs.values())

            for i in range(self._engine.num_io_tensors):
                name = self._engine.get_tensor_name(i)
                if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    arr = self._pick_input(name, inputs, input_arrays, i)
                    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).cuda()
                else:
                    shape = tuple(self._context.get_tensor_shape(name))
                    t = torch.empty(shape, dtype=torch.float32, device="cuda")
                bindings[name] = t
                self._context.set_tensor_address(name, t.data_ptr())

            self._context.execute_v3()

            result: Dict[str, np.ndarray] = {}
            for name, t in bindings.items():
                if self._engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                    result[name] = t.float().cpu().numpy()
        if len(result) == 1:
            return {"joint_targets": next(iter(result.values()))}
        return result

    @staticmethod
    def _pick_input(name: str, inputs: Dict[str, Any], arrays: List[Any], idx: int) -> np.ndarray:
        if name in inputs:
            return np.asarray(inputs[name])
        if idx < len(arrays):
            return np.asarray(arrays[idx])
        raise KeyError(f"missing engine input {name!r}; got keys {sorted(inputs)}")
