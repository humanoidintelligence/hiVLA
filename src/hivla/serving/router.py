"""A load-balancing gateway in front of a pool of ``hivla serve`` backends.

The router speaks the same ``POST /act`` wire format as a single server
(:mod:`hivla.serving.server`), so a client points at the router exactly as it would at
one backend. The router forwards each request to one backend chosen by a
routing policy (below), letting a pool of servers (one per GPU, across one
or many hosts) sit behind a single endpoint.

It is deliberately thin: it holds no model and runs no inference. For
round-robin and least-outstanding routing it relays the raw request body
untouched (no decode). Only consistent-hash routing unpacks the observation to
compute its signature, so a given observation sticks to one backend and keeps
that backend's repeat-observation cache warm.

Forwarding is async (``httpx.AsyncClient``, one keep-alive client per backend)
so one slow backend does not stall the others. This is separate from the sync
:class:`~hivla.runtimes.remote.RemoteBackend`; the two never share a
client. ``/ws`` routing is not handled yet (a long-lived connection pins a
client to one backend for its lifetime); clients needing ``/ws`` connect to a
backend directly. ``fastapi``/``httpx`` are imported lazily.

(No ``from __future__ import annotations`` here: FastAPI resolves the ``Request``
parameter from its live annotation, which stringized annotations would hide.)
"""

import json
import threading
import zlib
from typing import Any, List, Optional, Protocol, Sequence, Tuple

from hivla import __version__
from hivla.protocol import MSGPACK_MEDIA_TYPE, decode_observation, unpack

# --- routing policies -------------------------------------------------------
# A policy decides which backend server in a pool answers a given request.
# Round-robin ignores load but is free; least-outstanding tracks in-flight
# requests so a slow backend stops drawing work; consistent-hash pins each
# distinct observation to one backend, so that backend's repeat-observation
# cache (:class:`~hivla.serving.engine.ObservationCache`) keeps hitting.
# ``pick`` returns an index into the backend list. Only consistent-hash needs
# the observation content (``needs_signature``); the others route on the pool
# alone, so their requests are forwarded without unpacking the body. Counters
# are guarded by a lock: the gateway updates in-flight counts around ``await``
# points and from the threadpool.

class RoutingPolicy(Protocol):
    """Pick a backend index for each request.

    Implementations are constructed once per router and called concurrently.
    """

    name: str

    def needs_signature(self) -> bool:
        """True iff :meth:`pick` uses the observation signature.

        When False the router forwards the raw request body untouched (no
        unpack), which is the fast path for round-robin and least-outstanding.
        """

    def pick(self, n: int, signature: Optional[Tuple] = None) -> int:
        """Return a backend index in ``[0, n)``.

        Args:
            n: Number of backends in the pool (always >= 1).
            signature: The observation signature (from
                :func:`hivla.serving.engine._observation_signature`) when
                :meth:`needs_signature` is True, else ``None``.
        """


class RoundRobin:
    """Hand requests to backends in turn, ignoring load."""

    name = "round_robin"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._i = -1

    def needs_signature(self) -> bool:
        return False

    def pick(self, n: int, signature: Optional[Tuple] = None) -> int:
        with self._lock:
            self._i = (self._i + 1) % n
            return self._i


class LeastOutstanding:
    """Send to the backend with the fewest in-flight requests.

    The router owns the in-flight counts (incremented before dispatch,
    decremented on completion) and passes them in via :meth:`bind`. Ties go to
    the lowest index, so an idle pool still round-trips deterministically.
    """

    name = "least_outstanding"

    def __init__(self) -> None:
        self._inflight: Optional[list] = None

    def bind(self, inflight: list) -> None:
        """Attach the router's per-backend in-flight counter list."""
        self._inflight = inflight

    def needs_signature(self) -> bool:
        return False

    def pick(self, n: int, signature: Optional[Tuple] = None) -> int:
        counts = self._inflight
        if counts is None:
            return 0
        best, best_n = 0, counts[0]
        for i in range(1, n):
            if counts[i] < best_n:
                best, best_n = i, counts[i]
        return best


class ConsistentHash:
    """Pin each distinct observation to one backend by hashing its signature.

    Uses the same content signature the server-side cache keys on, so repeated
    observations land on the same backend and hit that backend's cache. Hashing
    is stable across processes (``zlib.crc32`` over the signature's repr), so a
    restarted router routes an identical observation the same way.
    """

    name = "consistent_hash"

    def needs_signature(self) -> bool:
        return True

    def pick(self, n: int, signature: Optional[Tuple] = None) -> int:
        if signature is None:
            return 0
        return zlib.crc32(repr(signature).encode()) % n


def make_policy(name: str) -> RoutingPolicy:
    """Construct a routing policy by name.

    Args:
        name: ``round_robin`` | ``least_outstanding`` | ``consistent_hash``.
    """
    policies = {
        "round_robin": RoundRobin,
        "least_outstanding": LeastOutstanding,
        "consistent_hash": ConsistentHash,
    }
    try:
        return policies[name]()
    except KeyError:
        raise ValueError(
            f"unknown routing policy {name!r}; choose from {sorted(policies)}"
        ) from None


# --- gateway ----------------------------------------------------------------


def _signature_from_body(body: bytes, binary: bool):
    """Compute the observation signature for consistent-hash routing.

    Unpacks just enough to key on content, reusing the server-side cache's
    signature so the router and the chosen backend agree on identity.
    """
    from hivla.serving.engine import _observation_signature

    payload = unpack(body) if binary else json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("observation payload must be a mapping")
    return _observation_signature(decode_observation(payload))


class BackendPool:
    """Keep-alive async HTTP clients for a fixed list of backend servers.

    One :class:`httpx.AsyncClient` per backend (connection reuse), an in-flight
    request counter per backend (for least-outstanding routing), and the
    forward call that dispatches one request. Clients are opened in
    :meth:`start` (needs a running event loop) and closed in :meth:`aclose`.
    """

    def __init__(self, backends: Sequence[str], api_key: Optional[str] = None,
                 timeout: float = 30.0) -> None:
        if not backends:
            raise ValueError("router needs at least one backend URL")
        self.urls: List[str] = [u.rstrip("/") for u in backends]
        self.inflight: List[int] = [0] * len(self.urls)
        self._api_key = api_key
        self._timeout = timeout
        self._clients: List[Any] = []

    def start(self) -> None:
        """Open one pooled async client per backend (call inside startup)."""
        import httpx

        headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else {}
        self._clients = [
            httpx.AsyncClient(base_url=url, timeout=self._timeout, headers=headers)
            for url in self.urls
        ]

    async def aclose(self) -> None:
        """Close every client (call inside shutdown)."""
        for client in self._clients:
            await client.aclose()
        self._clients = []

    async def forward(self, index: int, body: bytes, content_type: str, timings: str):
        """POST the raw ``body`` to backend ``index``; return its httpx response.

        The body is relayed byte-for-byte with its original content type, so no
        re-encode happens on the fast path. The in-flight counter brackets the
        call so least-outstanding sees true concurrency.
        """
        client = self._clients[index]
        params = {"timings": timings} if timings else None
        self.inflight[index] += 1
        try:
            return await client.post(
                "/act", content=body,
                headers={"Content-Type": content_type} if content_type else None,
                params=params,
            )
        finally:
            self.inflight[index] -= 1


def create_router_app(backends: Sequence[str], policy: str = "round_robin",
                      api_key: Optional[str] = None, timeout: float = 30.0):
    """Build the FastAPI app that routes ``/act`` across ``backends``.

    Args:
        backends: Base URLs of the backend servers (``http://host:port``).
        policy: Routing policy name (see :func:`hivla.serving.router.make_policy`).
        api_key: If set, required on the router's own ``/act`` (``Authorization:
            Api-Key <key>``) and forwarded to the backends.
        timeout: Per-request timeout to a backend, in seconds.
    """
    try:
        from contextlib import asynccontextmanager

        from fastapi import FastAPI, Header, HTTPException, Request, Response
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "the hiVLA router needs fastapi/uvicorn and httpx; install both extras: "
            "pip install 'hivla[server,client]'."
        ) from exc

    pool = BackendPool(backends, api_key=api_key, timeout=timeout)
    route: RoutingPolicy = make_policy(policy)
    if isinstance(route, LeastOutstanding):
        route.bind(pool.inflight)

    @asynccontextmanager
    async def lifespan(app):
        # Open the async clients once a loop is running, close them on shutdown.
        pool.start()
        yield
        await pool.aclose()

    app = FastAPI(title="hiVLA router", version=__version__, lifespan=lifespan)

    def _check_auth(authorization: Optional[str]) -> None:
        if api_key is not None and authorization != f"Api-Key {api_key}":
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "backends": pool.urls, "policy": route.name}

    @app.post("/act")
    async def act(request: Request, authorization: Optional[str] = Header(default=None)):
        _check_auth(authorization)
        body = await request.body()
        content_type = request.headers.get("content-type", "")
        binary = content_type.startswith(MSGPACK_MEDIA_TYPE)
        timings = request.query_params.get("timings", "")

        signature = None
        if route.needs_signature():
            try:
                signature = _signature_from_body(body, binary)
            except (ValueError, TypeError, KeyError) as exc:
                # Garbled payload is a client error; reject before forwarding so
                # a bad request never reaches a backend.
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        index = route.pick(len(pool.urls), signature)
        try:
            resp = await pool.forward(index, body, content_type, timings)
        except Exception as exc:  # httpx connect/read errors, etc.
            raise HTTPException(
                status_code=502, detail=f"backend {pool.urls[index]} failed: {exc}"
            ) from exc
        if resp.status_code != 200:
            # Surface the backend's own error (e.g. 400 for a bad observation).
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        media_type = resp.headers.get("content-type", "application/json")
        return Response(content=resp.content, media_type=media_type)

    return app


def run_router(backends: Sequence[str], host: str = "0.0.0.0", port: int = 8000,
               policy: str = "round_robin", api_key: Optional[str] = None,
               timeout: float = 30.0) -> None:
    """Build the router app and run it (blocking)."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "the hiVLA router needs uvicorn; install the 'server' extra: "
            "pip install 'hivla[server]'."
        ) from exc
    app = create_router_app(backends, policy=policy, api_key=api_key, timeout=timeout)
    uvicorn.run(app, host=host, port=port)
