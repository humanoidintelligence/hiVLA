"""HTTP + WebSocket transport: exposes an :class:`InferenceEngine` over FastAPI.

Both transports share one engine and one wire format (JSON envelope, arrays as
base64 + dtype + shape via :mod:`hivla.protocol`, no pickle, object-dtype
refused). A transport is only the shell — swapping HTTP for WebSocket (or adding
binary msgpack later) does not touch the engine or policy.

* ``POST /act``          — body is an encoded observation; returns an encoded action chunk.
  JSON by default; a ``Content-Type: application/msgpack`` request is decoded
  as the binary envelope and answered in msgpack too (no base64 for images).
* ``WS   /ws``           — same contract over a long-lived connection (openpi-style).
  Text frames are JSON; binary frames are msgpack and get msgpack replies.
* ``POST /reset``        — start an episode (reset the model, bind it to the session).
* ``POST /observe``      — hand one observation to a stateful model, no chunk back.
* ``GET  /metadata``     — connect-time handshake so the client stays in sync.
* ``GET  /health``       — liveness probe.

Over HTTP the session is the ``X-Hivla-Session`` header (the client picks a
random id); over ``/ws`` each connection is one session. A stateful model
(``stateful: true`` in ``/metadata``) serves the session of the last reset and
refuses the others (HTTP 409, a WS ``error`` reply).

``fastapi``/``uvicorn`` are imported lazily so the core package stays
dependency-light; install the ``server`` extra to serve.
"""

import json
import time
import uuid
from typing import Any, Callable, Dict, Optional, Sequence

from hivla import __version__
from hivla.serving.engine import InferenceEngine, SessionError
from hivla.protocol import (
    MSGPACK_MEDIA_TYPE,
    SESSION_HEADER,
    decode_observation,
    encode_action_chunk,
    pack,
    unpack,
)


def create_app(
    engine: InferenceEngine,
    api_key: Optional[str] = None,
    enable_ws: bool = True,
    extensions: Optional[Sequence[Callable[[Any, InferenceEngine], None]]] = None,
):
    """Build the FastAPI app serving ``engine``.

    Args:
        engine: A started (or startable) inference engine.
        api_key: If set, require ``Authorization: Api-Key <key>`` on ``/act``,
            ``/reset`` and ``/observe``.
        enable_ws: Mount the ``/ws`` WebSocket endpoint (default True).
        extensions: Optional callables ``(app, engine) -> None`` run after the
            core routes are mounted, so an integration can add its own endpoints
            without the core package depending on it.
    """
    try:
        from fastapi import FastAPI, Header, HTTPException, Request, Response, WebSocket
        from starlette.concurrency import run_in_threadpool
        from starlette.websockets import WebSocketDisconnect
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "hiVLA serving needs fastapi/uvicorn; install the 'server' extra: "
            "pip install 'hivla[server]'."
        ) from exc

    app = FastAPI(title="hiVLA", version=__version__)

    def _check_auth(authorization: Optional[str]) -> None:
        if api_key is not None and authorization != f"Api-Key {api_key}":
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok"}

    @app.get("/metadata")
    def metadata() -> Dict[str, Any]:
        return engine.metadata

    def _decode_body(body: bytes, binary: bool) -> Dict[str, Any]:
        try:
            payload = unpack(body) if binary else json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("observation payload must be a mapping")
            return decode_observation(payload)
        except (ValueError, TypeError, KeyError) as exc:
            # Unsafe/garbled payload (e.g. object-dtype) is a client error.
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _act(body: bytes, binary: bool, timings: bool = False, session: Optional[str] = None):
        t0 = time.perf_counter()
        observation = _decode_body(body, binary)
        t1 = time.perf_counter()
        try:
            action = engine.infer(observation, session=session)
        except SessionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        t2 = time.perf_counter()
        chunk = encode_action_chunk(action, binary)
        t3 = time.perf_counter()
        if timings:
            # Reserved key the client ignores (action_chunk_from_response reads only
            # known keys); lets a caller split its server-wait into decode/infer/encode.
            chunk["server_timings_ms"] = {"decode": round(1000 * (t1 - t0), 2),
                                          "infer": round(1000 * (t2 - t1), 2),
                                          "encode": round(1000 * (t3 - t2), 2)}
        if binary:
            return Response(content=pack(chunk), media_type=MSGPACK_MEDIA_TYPE)
        return chunk

    # The body is read raw (not via a typed Body param) so msgpack requests skip
    # JSON/pydantic entirely; decode + inference run off the event loop.
    @app.post("/act")
    async def act(request: Request, authorization: Optional[str] = Header(default=None)):
        _check_auth(authorization)
        binary = request.headers.get("content-type", "").startswith(MSGPACK_MEDIA_TYPE)
        timings = request.query_params.get("timings") in ("1", "true")
        session = request.headers.get(SESSION_HEADER)
        return await run_in_threadpool(_act, await request.body(), binary, timings, session)

    @app.post("/reset")
    async def reset(request: Request, authorization: Optional[str] = Header(default=None)):
        _check_auth(authorization)
        await run_in_threadpool(engine.reset, request.headers.get(SESSION_HEADER))
        return {"ok": True}

    def _observe(body: bytes, binary: bool, session: Optional[str]) -> Dict[str, Any]:
        observation = _decode_body(body, binary)
        try:
            engine.observe(observation, session=session)
        except SessionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"ok": True}

    @app.post("/observe")
    async def observe(request: Request, authorization: Optional[str] = Header(default=None)):
        _check_auth(authorization)
        binary = request.headers.get("content-type", "").startswith(MSGPACK_MEDIA_TYPE)
        return await run_in_threadpool(_observe, await request.body(), binary,
                                       request.headers.get(SESSION_HEADER))


    if enable_ws:

        def _ws_reply(data: Any, binary: bool, session: str) -> Dict[str, Any]:
            """Handle one parsed WS message; returns the reply envelope (wire-encoded)."""
            try:
                msg = unpack(data) if binary else json.loads(data)
                if not isinstance(msg, dict):
                    raise ValueError("message must be a mapping")
                kind = msg.get("type")
                if kind == "metadata":
                    return {"type": "metadata", "metadata": engine.metadata}
                if kind == "reset":
                    engine.reset(session)
                    return {"type": "reset"}
                if kind not in ("act", "observe"):
                    return {"type": "error", "detail": f"unknown message type {kind!r}"}
                observation = decode_observation(msg.get("observation", {}))
                timings = bool(msg.get("timings"))
            except (ValueError, TypeError, KeyError) as exc:
                return {"type": "error", "detail": str(exc)}
            t0 = time.perf_counter()
            try:
                if kind == "observe":
                    engine.observe(observation, session=session)
                    return {"type": "observed"}
                action = engine.infer(observation, session=session)
            except SessionError as exc:
                return {"type": "error", "detail": str(exc)}
            t1 = time.perf_counter()
            chunk = encode_action_chunk(action, binary)
            t2 = time.perf_counter()
            if timings:
                chunk["server_timings_ms"] = {"infer": round(1000 * (t1 - t0), 2),
                                              "encode": round(1000 * (t2 - t1), 2)}
            return {"type": "action", "chunk": chunk}

        # Frame type picks the wire per message: binary frames are msgpack (raw
        # array bytes), text frames are JSON, and each reply mirrors its request.
        # The connect-time handshake is JSON text so any client can read it.
        @app.websocket("/ws")
        async def ws(websocket: WebSocket):
            await websocket.accept()
            session = uuid.uuid4().hex  # one connection is one session
            # On connect, push the handshake first (openpi sends metadata on connect).
            await websocket.send_json({"type": "metadata", "metadata": engine.metadata})
            try:
                while True:
                    frame = await websocket.receive()
                    if frame["type"] == "websocket.disconnect":
                        raise WebSocketDisconnect(frame.get("code", 1000))
                    binary = frame.get("bytes") is not None
                    data = frame["bytes"] if binary else frame.get("text", "")
                    # Decode + inference off the event loop so one slow request
                    # does not stall every other connection.
                    reply = await run_in_threadpool(_ws_reply, data, binary, session)
                    if binary:
                        await websocket.send_bytes(pack(reply))
                    else:
                        await websocket.send_json(reply)
            except WebSocketDisconnect:
                pass  # client closed the connection
            except Exception:
                await websocket.close(code=1011)

    for extension in extensions or ():
        extension(app, engine)

    return app


# uvicorn accepts permessage-deflate by default, and the websockets client offers
# it by default. Camera frames barely compress, so deflate only burns CPU: ~36 ms
# per 480x640 frame vs ~1 ms without.
# openpi disables it for the same reason. Refusing it server-side covers every client.
UVICORN_WS_KWARGS: Dict[str, Any] = {"ws_per_message_deflate": False}


def serve(engine: InferenceEngine, host: str = "0.0.0.0", port: int = 8000,
          api_key: Optional[str] = None, enable_ws: bool = True,
          extensions: Optional[Sequence[Callable[[Any, InferenceEngine], None]]] = None) -> None:
    """Start the backend and run the HTTP+WS server (blocking)."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "hiVLA serving needs uvicorn; install the 'server' extra: "
            "pip install 'hivla[server]'."
        ) from exc
    engine.start()
    app = create_app(engine, api_key=api_key, enable_ws=enable_ws, extensions=extensions)
    uvicorn.run(app, host=host, port=port, **UVICORN_WS_KWARGS)


def load_extensions(paths: Sequence[str]) -> list:
    """Resolve ``module:attr`` (or ``module.attr``) strings to callables.

    Lets a config mount extra endpoints by dotted path without the core package
    importing them. Each target is called as ``(app, engine) -> None``.
    """
    import importlib

    resolved = []
    for path in paths:
        module_name, _, attr = path.replace(":", ".").rpartition(".")
        if not module_name:
            raise ValueError(f"extension path must be 'module:attr', got {path!r}")
        resolved.append(getattr(importlib.import_module(module_name), attr))
    return resolved
