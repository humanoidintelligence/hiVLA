"""Clients for a hiVLA server: :class:`VlaClient` (HTTP) and :class:`VlaWsClient` (WebSocket).

They only move data. An observation goes in as a plain dict of numpy arrays::

    {"joint_pos": array, "images": {"head": HxWx3 uint8}, "instruction": "..."}

and the action chunk comes back as a dict whose keys are hiRobot's
``ActionChunk`` fields (``mode``, ``joint_targets``, ``kp``, ``kd``,
``ee_targets``, ``gripper``, ``joint_names``, ``base_velocity``,
``base_target``, ``latent``, ``extra``, ``parts``; only the keys the server
sent, see :data:`hivla.protocol.CHUNK_FIELDS`). Caching
chunks, replaying them and cutting them to ``n_action_steps`` is the caller's
job. The model's suggested ``n_action_steps`` is in :attr:`metadata`.

Arrays travel as base64 JSON (``wire="json"``) or raw msgpack bytes
(``wire="msgpack"``, faster for camera frames); see :mod:`hivla.protocol`.
Neither format carries code, and object-dtype arrays are refused both ways.

Episodes: call :meth:`reset` before each one. It fetches ``/metadata`` and
tells the server to start an episode, which a stateful model (``stateful`` in
:attr:`metadata`) needs before it answers, and which binds the model to this
client's session. A model with ``observe: every_step`` also wants one
:meth:`observe` per executed step of a chunk.

This module imports only numpy and :mod:`hivla.protocol`; ``httpx`` /
``websockets`` / ``msgpack`` (the ``client`` extra) load when first used.
"""

from __future__ import annotations

import json
import threading
import uuid
from typing import Any, Dict, Optional

from hivla.protocol import (
    MSGPACK_MEDIA_TYPE,
    SESSION_HEADER,
    decode_action_chunk,
    encode_observation,
    pack,
    unpack,
)

#: Reply key with the server's own decode / infer / encode times (``timings=True``).
SERVER_TIMINGS_KEY = "server_timings_ms"

_WIRES = ("json", "msgpack")


class SessionLost(RuntimeError):
    """A stateful model's session ended with the connection; the episode must restart.

    :class:`VlaWsClient` raises it from the first :meth:`~VlaWsClient.act` or
    :meth:`~VlaWsClient.observe` after it reconnected to a stateful model. By then
    it has already sent ``reset`` on the new connection, so the next call is
    answered; but the model has forgotten the episode so far (its observation
    history, an open session upstream), so the caller should start the episode
    over, or at least take a fresh observation. The request that raised was not
    sent.
    """


def _decode_reply(payload: Dict[str, Any]) -> Dict[str, Any]:
    chunk = decode_action_chunk(payload)
    if SERVER_TIMINGS_KEY in payload:
        chunk[SERVER_TIMINGS_KEY] = payload[SERVER_TIMINGS_KEY]
    return chunk


def _auth_headers(api_key: Optional[str]) -> Dict[str, str]:
    return {"Authorization": f"Api-Key {api_key}"} if api_key else {}


class VlaClient:
    """One HTTP request per chunk: ``POST {endpoint}/act``, ``GET {endpoint}/metadata``.

    Args:
        endpoint: Base URL of the server, e.g. ``http://localhost:8000``.
        timeout: Per-request timeout in seconds.
        api_key: Sent as ``Authorization: Api-Key ...`` when set.
        wire: ``"json"`` (base64 arrays, any compatible server) or ``"msgpack"``
            (raw array bytes; prefer it when sending camera images).
        timings: Ask the server for its decode / infer / encode times; they come
            back under ``"server_timings_ms"`` in each chunk.

    Calls are serialized with a lock, so a prefetch thread and the control loop
    can share one client. Each client has its own session id (sent as
    ``X-Hivla-Session``), so two clients never share a stateful model's episode.
    """

    def __init__(
        self,
        endpoint: str,
        timeout: float = 10.0,
        api_key: Optional[str] = None,
        wire: str = "json",
        timings: bool = False,
    ) -> None:
        if wire not in _WIRES:
            raise ValueError(f"wire must be 'json' or 'msgpack', got {wire!r}")
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.wire = wire
        self.timings = timings
        #: This client's session id.
        self.session = uuid.uuid4().hex
        self._headers = {**_auth_headers(api_key), SESSION_HEADER: self.session}
        self._client: Any = None
        self._lock = threading.Lock()
        #: The server's ``/metadata`` (model, ``action_horizon``, ``n_action_steps``, ...);
        #: filled by :meth:`reset`, empty if the server has none or is down.
        self.metadata: Dict[str, Any] = {}

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "VlaClient needs httpx; install the 'client' extra: "
                    "pip install 'hivla[client]'"
                ) from exc
            self._client = httpx.Client(timeout=self.timeout, headers=self._headers)
        return self._client

    def reset(self) -> Dict[str, Any]:
        """Start an episode: fetch ``/metadata``, then ``POST /reset``; return the metadata.

        A server that is down leaves :attr:`metadata` empty and shows up on the
        next :meth:`act`. A server that is up but refuses the reset raises.
        """
        with self._lock:
            client = self._ensure_client()
            try:
                resp = client.get(f"{self.endpoint}/metadata")
            except Exception:
                # Not required to run; a server that is down shows up on act().
                self.metadata = {}
                return self.metadata
            self.metadata = resp.json() if resp.status_code == 200 else {}
            client.post(f"{self.endpoint}/reset").raise_for_status()
        return self.metadata

    def _post_observation(self, path: str, observation: Dict[str, Any], params=None) -> Any:
        """POST an encoded observation; return the raw response. Caller holds the lock."""
        client = self._ensure_client()
        url = f"{self.endpoint}{path}"
        if self.wire == "msgpack":
            body = pack(encode_observation(observation, binary=True))
            resp = client.post(url, content=body, params=params,
                               headers={"Content-Type": MSGPACK_MEDIA_TYPE})
        else:
            resp = client.post(url, json=encode_observation(observation), params=params)
        resp.raise_for_status()
        return resp

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Send one observation, return the decoded action chunk."""
        params = {"timings": "1"} if self.timings else None
        with self._lock:
            resp = self._post_observation("/act", observation, params)
        if self.wire == "msgpack":
            return _decode_reply(unpack(resp.content))
        return _decode_reply(resp.json())

    def observe(self, observation: Dict[str, Any]) -> None:
        """Send one observation without asking for a chunk (``POST /observe``)."""
        with self._lock:
            self._post_observation("/observe", observation)

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None


class VlaWsClient:
    """One long-lived WebSocket to ``{endpoint}/ws``.

    On connect the server sends a JSON ``{"type": "metadata"}`` handshake; each
    ``{"type": "act"}`` request then gets one ``{"type": "action"}`` (or
    ``{"type": "error"}``) reply. With ``wire="msgpack"`` requests are binary
    frames with raw array bytes and replies come back the same way. Keeping one
    connection skips per-request HTTP overhead (a real ACT request measured
    8.2 ms here vs 12.5 ms over HTTP msgpack).

    permessage-deflate is never offered: camera frames barely compress, and
    deflate added ~36 ms per 480x640 frame in measurements. Messages have no
    size limit, since a multi-camera observation easily exceeds 1 MiB.

    The connection opens on first use (or :meth:`reset`). After a timeout or a
    dropped connection it is thrown away, so a late reply can never be paired
    with the next request, and the following call reconnects. Calls are
    serialized with a lock. The connection is the session: a reconnect is a new
    session. If an episode was started (:meth:`reset`) on the lost connection and
    the model is stateful, the next :meth:`act` / :meth:`observe` reconnects,
    resets the new session and raises :class:`SessionLost` instead of sending
    its request; calls after that work normally.

    Args:
        endpoint: ``http(s)://`` or ``ws(s)://`` base URL; ``/ws`` is appended.
        timeout: Seconds to wait for connecting and for each reply.
        api_key: Sent as ``Authorization: Api-Key ...`` when set.
        wire: ``"msgpack"`` (default; raw array bytes) or ``"json"`` (base64).
        timings: Ask the server for its infer / encode times; they come back
            under ``"server_timings_ms"`` in each chunk.
    """

    def __init__(
        self,
        endpoint: str,
        timeout: float = 10.0,
        api_key: Optional[str] = None,
        wire: str = "msgpack",
        timings: bool = False,
    ) -> None:
        if wire not in _WIRES:
            raise ValueError(f"wire must be 'json' or 'msgpack', got {wire!r}")
        base = endpoint.rstrip("/")
        for http, ws in (("http://", "ws://"), ("https://", "wss://")):
            if base.startswith(http):
                base = ws + base[len(http):]
        if not base.startswith(("ws://", "wss://")):
            raise ValueError(f"endpoint must be http(s):// or ws(s)://, got {endpoint!r}")
        self.url = f"{base}/ws"
        self.timeout = timeout
        self.wire = wire
        self.timings = timings
        self._headers = _auth_headers(api_key)
        self._conn: Any = None
        self._lock = threading.Lock()
        #: An episode was reset on the current connection.
        self._in_episode = False
        #: The connection holding an episode was dropped; the next call re-resets.
        self._lost = False
        #: The server's metadata, from the connect handshake or :meth:`reset`.
        self.metadata: Dict[str, Any] = {}

    def _ensure_conn(self) -> Any:
        if self._conn is None:
            try:
                from websockets.sync.client import connect
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "VlaWsClient needs websockets; install the 'client' extra: "
                    "pip install 'hivla[client]'"
                ) from exc
            # Entering the result yields a plain ClientConnection on every
            # websockets>=12; calling connect() bare is deprecated from 17 on.
            conn = connect(
                self.url,
                additional_headers=self._headers,
                compression=None,
                max_size=None,
                open_timeout=self.timeout,
            ).__enter__()
            try:
                hello = json.loads(conn.recv(timeout=self.timeout))
            except BaseException:
                conn.close()
                raise
            self.metadata = hello.get("metadata", {}) if hello.get("type") == "metadata" else {}
            self._conn = conn
        return self._conn

    def _roundtrip(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Send one request and return its decoded reply. Caller holds the lock."""
        conn = self._ensure_conn()
        try:
            conn.send(pack(msg) if self.wire == "msgpack" else json.dumps(msg))
            frame = conn.recv(timeout=self.timeout)
        except BaseException:
            # The reply may still arrive later; drop the connection so it cannot
            # be read as the answer to the next request.
            self._drop(lost=True)
            raise
        if isinstance(frame, (bytes, bytearray)):
            return unpack(frame)
        return json.loads(frame)

    def reset(self) -> Dict[str, Any]:
        """Start an episode: (re)fetch metadata, then send ``reset``; return the metadata.

        A server that is down leaves :attr:`metadata` empty and shows up on the
        next :meth:`act`. A server that is up but refuses the reset raises.
        """
        with self._lock:
            try:
                if self._conn is None:
                    self._ensure_conn()  # the connect handshake carries metadata
                else:
                    self.metadata = self._roundtrip({"type": "metadata"}).get("metadata", {})
            except Exception:
                # A server that is down shows up on the next act() instead.
                self.metadata = {}
                return self.metadata
            reply = self._roundtrip({"type": "reset"})
            if reply.get("type") != "reset":
                raise RuntimeError(f"server refused reset: {reply.get('detail', reply)}")
            self._in_episode, self._lost = True, False
        return self.metadata

    def _recover(self) -> None:
        """After losing an episode's connection: reconnect, reset, raise :class:`SessionLost`.

        Caller holds the lock. A stateless model loses nothing, so it just goes on.
        """
        if not self._lost:
            return
        self._ensure_conn()  # raises (and stays lost) while the server is unreachable
        if not self.metadata.get("stateful"):
            self._lost = False
            return
        reply = self._roundtrip({"type": "reset"})
        if reply.get("type") != "reset":
            raise RuntimeError(f"server refused reset: {reply.get('detail', reply)}")
        self._in_episode, self._lost = True, False
        raise SessionLost(
            "lost the connection to the hiVLA server; the client reconnected and reset a new "
            "session, but the model's episode state is gone: restart the episode")

    def observe(self, observation: Dict[str, Any]) -> None:
        """Send one observation without asking for a chunk."""
        msg = {"type": "observe",
               "observation": encode_observation(observation, binary=self.wire == "msgpack")}
        with self._lock:
            self._recover()
            reply = self._roundtrip(msg)
        if reply.get("type") != "observed":
            raise RuntimeError(f"server rejected observation: {reply.get('detail', reply)}")

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Send one observation, return the decoded action chunk."""
        msg: Dict[str, Any] = {
            "type": "act",
            "observation": encode_observation(observation, binary=self.wire == "msgpack"),
        }
        if self.timings:
            msg["timings"] = True
        with self._lock:
            self._recover()
            reply = self._roundtrip(msg)
        if reply.get("type") == "error":
            raise RuntimeError(f"server rejected observation: {reply.get('detail')}")
        if reply.get("type") != "action":
            raise RuntimeError(f"unexpected reply type {reply.get('type')!r}")
        return _decode_reply(reply["chunk"])

    def _drop(self, lost: bool = False) -> None:
        """Throw the connection away; ``lost`` marks an episode on it as lost."""
        if lost and self._in_episode:
            self._lost = True
        self._in_episode = False
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass  # already broken; nothing left to release

    def close(self) -> None:
        with self._lock:
            self._drop()
            self._lost = False
