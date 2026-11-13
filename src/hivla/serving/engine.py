"""The inference engine: a policy wrapped with the serving-side lifecycle.

An :class:`InferenceEngine` owns a :class:`~hivla.core.policy.VlaPolicy` (which in
turn owns a loaded :class:`~hivla.runtimes.base.ModelBackend`) and exposes a
single :meth:`infer` entry point taking a decoded observation and returning an
action chunk. This is the unit the network server sits in front of, and the
place serving-time optimizations live:

* :class:`ObservationCache` — replay the previous chunk when the same observation
  recurs (the common control-loop case: a fresh image/state has not arrived yet),
  saving a forward pass. LRU with a TTL.
* dynamic batching — when ``max_batch_size > 1``, concurrent :meth:`infer` calls
  are coalesced by a background :class:`_BatchExecutor` into batches dispatched to
  :meth:`VlaPolicy.infer_batch`. VLA control loops are latency-sensitive, so this
  is opt-in (default off) and uses a tiny accumulation window.
* sessions — a stateful model (``backend.stateful``: an observation window, a
  temporal-aggregation buffer) serves one session at a time. :meth:`reset`
  starts an episode and binds the model to the caller's session; :meth:`infer`
  and :meth:`observe` from any other session are refused with
  :class:`SessionError`, so two callers can never interleave one model's state.
  Stateful models run without the cache and without batching.
"""

from __future__ import annotations

import binascii
import concurrent.futures
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

import numpy as np

from hivla.core.policy import VlaPolicy

#: The bound session before the first :meth:`InferenceEngine.reset`.
_UNBOUND = object()


class SessionError(RuntimeError):
    """A request for a stateful model from a session that does not hold it."""


def _array_fingerprint(arr: np.ndarray) -> tuple:
    return (arr.shape, arr.dtype.str, binascii.crc32(arr.tobytes()))


def _observation_signature(observation: Dict[str, Any]) -> tuple:
    """A hashable, content-based key for an observation dict."""
    parts = []
    for key in sorted(observation):
        val = observation[key]
        if isinstance(val, np.ndarray):
            parts.append((key, _array_fingerprint(val)))
        elif isinstance(val, dict):
            sub = tuple(
                sorted((k, _array_fingerprint(v) if isinstance(v, np.ndarray) else v)
                       for k, v in val.items())
            )
            parts.append((key, sub))
        else:
            parts.append((key, val))
    return tuple(parts)


class ObservationCache:
    """LRU cache of ``observation signature -> action chunk`` with a TTL.

    Args:
        max_size: Maximum number of distinct observations to remember.
        ttl: Seconds a cached entry remains valid. A control loop at 30–50 Hz
            with a re-query only every ``n_action_steps`` rarely repeats an
            observation, so this mainly absorbs back-to-back identical frames.
    """

    def __init__(self, max_size: int = 16, ttl: float = 0.5) -> None:
        self._max_size = max_size
        self._ttl = ttl
        self._store: OrderedDict[tuple, tuple] = OrderedDict()

    def get(self, observation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        key = _observation_signature(observation)
        entry = self._store.get(key)
        if entry is None:
            return None
        chunk, expiry = entry
        if time.monotonic() > expiry:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return chunk

    def put(self, observation: Dict[str, Any], chunk: Dict[str, Any]) -> None:
        key = _observation_signature(observation)
        self._store[key] = (chunk, time.monotonic() + self._ttl)
        self._store.move_to_end(key)
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


class _BatchExecutor:
    """Coalesce concurrent :meth:`submit` calls into batches of up to ``max_size``.

    A single background thread accumulates pending requests for ``timeout``
    seconds (or until the batch is full), then dispatches them to ``worker`` and
    resolves each request's future. Exceptions from ``worker`` are propagated to
    every request in the failed batch.
    """

    def __init__(self, max_size: int, timeout: float, worker) -> None:
        self._max_size = max_size
        self._timeout = timeout
        self._worker = worker
        self._cond = threading.Condition()
        self._pending: list = []
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        self._thread.join(timeout=5)

    def submit(self, observation: Dict[str, Any]) -> concurrent.futures.Future:
        fut: concurrent.futures.Future = concurrent.futures.Future()
        with self._cond:
            self._pending.append((observation, fut))
            self._cond.notify()
        return fut

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._pending and not self._stop:
                    self._cond.wait()
                if self._stop and not self._pending:
                    return
                deadline = time.monotonic() + self._timeout
                while len(self._pending) < self._max_size and not self._stop:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cond.wait(timeout=remaining)
                batch = self._pending[: self._max_size]
                del self._pending[: self._max_size]
            try:
                chunks = self._worker([obs for obs, _ in batch])
            except Exception as exc:  # noqa: BLE001 - propagate to all waiters
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(exc)
                continue
            for (_, fut), chunk in zip(batch, chunks):
                if not fut.done():
                    fut.set_result(chunk)


class InferenceEngine:
    """Serving-side wrapper around a single VLA policy.

    Args:
        policy: The preprocess -> backend -> postprocess pipeline to serve.
        cache: Optional :class:`ObservationCache` for repeat-observation replay.
        max_batch_size: If > 1, enable dynamic request batching.
        batch_timeout: Seconds to accumulate a batch before dispatching.
    """

    def __init__(
        self,
        policy: VlaPolicy,
        cache: Optional[ObservationCache] = None,
        max_batch_size: int = 1,
        batch_timeout: float = 0.002,
        warmup: bool = True,
    ) -> None:
        self._policy = policy
        self._cache = cache
        self._max_batch_size = max(1, int(max_batch_size))
        self._batch_timeout = batch_timeout
        self._warmup = warmup
        self._loaded = False
        self._executor: Optional[_BatchExecutor] = None
        self._stateful = bool(getattr(policy._backend, "stateful", False))
        if self._stateful and (cache is not None or self._max_batch_size > 1):
            raise ValueError("a stateful model runs without the observation cache and "
                             "without batching (set runtime.cache: false, max_batch_size: 1)")
        self._session: Any = _UNBOUND
        self._session_lock = threading.Lock()

    @property
    def metadata(self) -> Dict[str, Any]:
        """Policy metadata surfaced through the server's ``/metadata`` handshake."""
        return self._policy.metadata

    @property
    def stateful(self) -> bool:
        """Whether the model keeps per-episode state (see :class:`SessionError`)."""
        return self._stateful

    def _check_session(self, session: Optional[str]) -> None:
        """Caller holds ``_session_lock``."""
        if self._session is _UNBOUND:
            raise SessionError("the model is stateful: call reset before the first request")
        if session != self._session:
            raise SessionError("the model is stateful and another session holds it; "
                               "call reset to take it over")

    def reset(self, session: Optional[str] = None) -> None:
        """Start a new episode for ``session``: reset the model and drop cached chunks.

        For a stateful model this also binds the model to ``session``; requests
        from any other session then raise :class:`SessionError` until the next
        reset. A stateless model ignores ``session``.
        """
        if not self._loaded:
            raise RuntimeError("InferenceEngine.start() must be called before reset().")
        with self._session_lock:
            self._policy.reset()
            if self._cache is not None:
                self._cache.clear()
            if self._stateful:
                self._session = session

    def observe(self, observation: Dict[str, Any], session: Optional[str] = None) -> None:
        """Hand one observation to a stateful model without asking for a chunk.

        For models that read an observation every executed step
        (``observe: every_step`` in ``/metadata``). A stateless model ignores it.
        """
        if not self._loaded:
            raise RuntimeError("InferenceEngine.start() must be called before observe().")
        if not self._stateful:
            return
        with self._session_lock:
            self._check_session(session)
            self._policy.observe(observation)

    def start(self) -> None:
        """Load the underlying backend. Called once before serving traffic."""
        if not self._loaded:
            self._policy._backend.load()
            self._loaded = True
            if self._warmup:
                # Pay first-call JIT/alloc (and torch.compile trace) before traffic.
                self._policy._backend.warmup()
            if self._max_batch_size > 1:
                self._executor = _BatchExecutor(
                    self._max_batch_size, self._batch_timeout, self._policy.infer_batch
                )
                self._executor.start()

    def infer(self, observation: Dict[str, Any], session: Optional[str] = None) -> Dict[str, Any]:
        """Run the policy on a decoded observation, returning an action chunk.

        Raises:
            RuntimeError: If called before :meth:`start`.
            SessionError: The model is stateful and ``session`` does not hold it.
        """
        if not self._loaded:
            raise RuntimeError("InferenceEngine.start() must be called before infer().")
        if self._stateful:
            with self._session_lock:
                self._check_session(session)
                return self._policy.infer(observation)

        if self._cache is not None:
            hit = self._cache.get(observation)
            if hit is not None:
                return hit

        if self._executor is not None:
            chunk = self._executor.submit(dict(observation)).result()
        else:
            chunk = self._policy.infer(observation)

        if self._cache is not None:
            self._cache.put(observation, chunk)
        return chunk

    def stop(self) -> None:
        """Stop batching and unload the backend, releasing resources."""
        if self._executor is not None:
            self._executor.stop()
            self._executor = None
        if self._loaded:
            self._policy._backend.unload()
            self._loaded = False


class AsyncInferenceEngine:
    """Run a VLA policy on a background worker so inference overlaps execution.

    The async counterpart of :class:`InferenceEngine`: instead of a blocking
    :meth:`infer`, the control loop calls :meth:`submit` to hand the newest
    observation to a worker thread (returning immediately) and :meth:`get_action`
    to read the most recently completed chunk without waiting. This is the
    double-buffering pattern lerobot's remote-inference engine uses — "inference
    can overlap execution of an earlier chunk": while the robot replays chunk N,
    the worker computes chunk N+1.

    The worker always computes the *newest* submitted observation and drops any
    observation superseded while it was busy (latest-wins), so a fast control
    loop never builds a backlog. The engine itself is library-level; exposing it
    over the network is a transport concern layered on top.
    """

    def __init__(self, policy: VlaPolicy) -> None:
        self._policy = policy
        self._cond = threading.Condition()
        self._latest_obs: Optional[Dict[str, Any]] = None
        self._obs_version = 0
        self._ready_chunk: Optional[Dict[str, Any]] = None
        self._stop = False
        self._worker: Optional[threading.Thread] = None
        self._loaded = False

    @property
    def metadata(self) -> Dict[str, Any]:
        """Policy metadata surfaced through the server's ``/metadata`` handshake."""
        return self._policy.metadata

    def start(self) -> None:
        """Load the backend and start the background worker."""
        if not self._loaded:
            self._policy._backend.load()
            self._loaded = True
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()

    def submit(self, observation: Dict[str, Any]) -> None:
        """Hand the newest observation to the worker and return immediately.

        Raises:
            RuntimeError: If called before :meth:`start`.
        """
        if not self._loaded:
            raise RuntimeError("AsyncInferenceEngine.start() must be called before submit().")
        with self._cond:
            self._latest_obs = dict(observation)
            self._obs_version += 1
            self._cond.notify_all()

    def get_action(self) -> Optional[Dict[str, Any]]:
        """Return the most recently completed chunk, or ``None`` if none is ready.

        Non-blocking. The caller replays the returned chunk while the worker
        computes the next one; a ``None`` means no chunk has finished yet (e.g.
        the very first tick).
        """
        with self._cond:
            return self._ready_chunk

    def _run(self) -> None:
        processed = 0
        while True:
            with self._cond:
                while not self._stop and self._obs_version == processed:
                    self._cond.wait()
                if self._stop:
                    return
                processed = self._obs_version
                obs = dict(self._latest_obs)
            # Compute outside the lock so submission/reads never block on the model.
            chunk = self._policy.infer(obs)
            with self._cond:
                self._ready_chunk = chunk

    def stop(self) -> None:
        """Signal the worker to exit and unload the backend."""
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=5)
        if self._loaded:
            self._policy._backend.unload()
            self._loaded = False
