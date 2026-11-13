"""The base contract every model backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class ModelBackend(ABC):
    """Loads and runs a specific model format (PyTorch, ONNX, TensorRT, remote).

    A backend is intentionally low-level: it maps a dict of already-preprocessed
    input tensors to a dict of raw output tensors. Preprocessing (normalize /
    pad / tokenize) and action decoding (unnormalize / slice) are *not* the
    backend's job — they live in :class:`~hivla.core.transforms` and are run by
    :class:`~hivla.core.policy.VlaPolicy`, which wraps a backend. The network
    client (hiRobot) only ships a semi-raw observation and receives a decoded
    action chunk; the server owns everything in between.
    """

    @abstractmethod
    def load(self) -> None:
        """Load weights / establish the connection. Called once before inference."""

    @abstractmethod
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run a forward pass on preprocessed inputs and return raw outputs."""

    def predict_batch(self, batch: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """Run a batch of inputs; default is a sequential fallback.

        Backends that can stack tensors into one forward pass (e.g. torch) may
        override this for throughput; the default keeps single-request semantics.
        """
        return [self.predict(inputs) for inputs in batch]

    #: Whether the model keeps per-episode state between calls (an observation
    #: window, a temporal-aggregation buffer). A stateful backend gets
    #: :meth:`reset` at the start of each episode and, with ``observe ==
    #: "every_step"``, :meth:`observe` for every executed step of a chunk; the
    #: engine serves it one session at a time, without batching or caching.
    stateful: bool = False
    #: Which observations the model needs: ``"chunk"`` (only the one sent with
    #: each request for a chunk) or ``"every_step"`` (also one per executed
    #: step, sent with :meth:`observe`). Reported in ``/metadata``.
    observe_mode: str = "chunk"

    def reset(self) -> None:
        """Start a new episode: drop per-episode state. Default: nothing to drop."""

    def observe(self, inputs: Dict[str, Any]) -> None:
        """Take one observation without predicting. Default: ignore it."""

    def metadata(self) -> Dict[str, Any]:
        """Extra ``/metadata`` entries the backend knows (default: none)."""
        return {}

    def warmup(self) -> None:
        """Run one dummy forward pass to pay first-call JIT/alloc cost up front.

        Called once after :meth:`load`, before serving traffic, when the engine's
        ``warmup`` flag is on. The default is a no-op; backends that can cheaply
        build a representative input (they know their own schema) override it so
        the first real request does not eat lazy CUDA/compile init.
        """

    def unload(self) -> None:
        """Release resources. Override if the backend holds handles."""
