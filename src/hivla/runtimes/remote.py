"""A backend that proxies to another hiVLA server (or a compatible endpoint).

This is the delegation hook: instead of loading a model, forward the (decoded)
observation to a remote ``/act`` endpoint and return its decoded action chunk. It
lets hiVLA route a model family to another serving system — another hiVLA server,
or in the future an autoregressive backbone served by sglang/vLLM — without
changing the engine or transport above it.

Because the remote server owns the model (and its normalization), a remote-backed
policy should be configured with **no** ``norm_stats`` and only key renames, so
the observation reaches the remote unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from hivla.runtimes.base import ModelBackend
from hivla.client import VlaClient


class RemoteBackend(ModelBackend):
    """Proxy inference to a remote model server through :class:`~hivla.client.VlaClient`.

    Args:
        endpoint: Base URL of the remote server (e.g. ``http://host:8000``).
        timeout: Per-request timeout in seconds.
        api_key: Optional ``Api-Key`` sent in the ``Authorization`` header.
        wire: ``"json"`` (base64 arrays, any compatible server) or ``"msgpack"``
            (raw array bytes; needs a hiVLA server with msgpack support).
    """

    def __init__(self, endpoint: str, timeout: float = 30.0, api_key: Optional[str] = None,
                 wire: str = "json") -> None:
        # VlaClient checks wire and imports httpx only when first used.
        self._client = VlaClient(endpoint, timeout=timeout, api_key=api_key, wire=wire)

    def load(self) -> None:
        """Open the pooled HTTP client now (keep-alive saves a TCP setup per request)."""
        self._client._ensure_client()

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self._client.act(inputs)

    def unload(self) -> None:
        self._client.close()
