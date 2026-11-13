"""ONNX Runtime backend.

Runs an exported ``.onnx`` model as a pure tensor-in / tensor-out forward pass.
``onnxruntime`` is imported lazily so the core package (and the echo-backend
tests) need no onnx runtime; install the ``onnx`` extra to use it. Inputs are
matched to the session's input names by key, with a single-input model accepting
the sole array regardless of key.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from hivla.runtimes.base import ModelBackend


class OnnxBackend(ModelBackend):
    """Run an ONNX checkpoint behind the backend contract.

    Args:
        checkpoint: Path to the ``.onnx`` file.
        device: ``"cpu"`` or ``"cuda"`` (selects execution providers).
    """

    def __init__(self, checkpoint: str, device: str = "cpu") -> None:
        self._checkpoint = checkpoint
        self._device = device
        self._session: Optional[Any] = None

    def load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "OnnxBackend needs onnxruntime; install the 'onnx' extra: "
                "pip install 'hivla[onnx]'."
            ) from exc
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self._session = ort.InferenceSession(self._checkpoint, providers=providers)

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self._session is None:
            raise RuntimeError("OnnxBackend.load() must be called before predict().")
        input_names = [i.name for i in self._session.get_inputs()]
        feed: Dict[str, np.ndarray] = {}
        if len(input_names) == 1:
            arr = next(iter(inputs.values()))
            feed[input_names[0]] = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
        else:
            for name in input_names:
                if name in inputs:
                    feed[name] = np.ascontiguousarray(np.asarray(inputs[name], dtype=np.float32))
                else:
                    raise KeyError(f"missing model input {name!r}; got keys {sorted(inputs)}")

        outputs = self._session.run(None, feed)
        if len(outputs) == 1:
            return {"joint_targets": outputs[0]}
        out_names = [o.name for o in self._session.get_outputs()]
        return {name: val for name, val in zip(out_names, outputs)}
