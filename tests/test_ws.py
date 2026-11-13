"""WebSocket transport: handshake on connect + /act over a long-lived connection."""

from __future__ import annotations

import numpy as np
import pytest

from hivla import build_engine
from hivla.protocol import decode_array, encode_array
from hivla.serving.server import create_app

pytest.importorskip("fastapi")
from starlette.testclient import TestClient  # noqa: E402


def _client():
    engine = build_engine({
        "model": {"backend": "echo", "action_horizon": 6},
        "policy": {"action_dim": 7, "control_mode": "pd"},
    })
    engine.start()
    return TestClient(create_app(engine))


def test_ws_sends_metadata_on_connect():
    with _client().websocket_connect("/ws") as ws:
        first = ws.receive_json()
        assert first["type"] == "metadata"
        assert first["metadata"]["action_horizon"] == 6
        assert first["metadata"]["control_mode"] == "pd"


def test_ws_act_roundtrip():
    with _client().websocket_connect("/ws") as ws:
        ws.receive_json()  # metadata
        ws.send_json({"type": "act", "observation": {"joint_pos": encode_array(np.zeros(7, np.float32))}})
        resp = ws.receive_json()
        assert resp["type"] == "action"
        assert resp["chunk"]["mode"] == "pd"
        assert decode_array(resp["chunk"]["joint_targets"]).shape == (6, 7)


def test_ws_msgpack_binary_frames():
    pytest.importorskip("msgpack")
    from hivla.protocol import encode_observation, pack, unpack

    obs = {"joint_pos": np.zeros(7, np.float32), "image": np.ones((4, 5, 3), np.uint8)}
    with _client().websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "metadata"  # handshake stays JSON
        for _ in range(2):  # several requests on one connection
            ws.send_bytes(pack({"type": "act", "observation": encode_observation(obs, binary=True)}))
            resp = unpack(ws.receive_bytes())
            assert resp["type"] == "action"
            assert isinstance(resp["chunk"]["joint_targets"]["__ndarray__"], bytes)
            assert decode_array(resp["chunk"]["joint_targets"]).shape == (6, 7)
        ws.send_bytes(pack({"type": "metadata"}))
        assert unpack(ws.receive_bytes())["metadata"]["action_horizon"] == 6
        # JSON text frames still work on the same connection.
        ws.send_json({"type": "act", "observation": {"joint_pos": encode_array(np.zeros(7, np.float32))}})
        assert ws.receive_json()["type"] == "action"


@pytest.mark.parametrize("frame", [b"\xc1", b"\x92\x01", b"\x01"])
def test_ws_bad_msgpack_returns_error_and_keeps_connection(frame):
    pytest.importorskip("msgpack")
    from hivla.protocol import pack, unpack

    with _client().websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_bytes(frame)
        assert unpack(ws.receive_bytes())["type"] == "error"
        ws.send_bytes(pack({"type": "metadata"}))
        assert unpack(ws.receive_bytes())["type"] == "metadata"


def test_ws_msgpack_rejects_object_dtype():
    pytest.importorskip("msgpack")
    from hivla.protocol import pack, unpack

    with _client().websocket_connect("/ws") as ws:
        ws.receive_json()
        bad = {"joint_pos": {"__ndarray__": b"", "dtype": "object", "shape": [0]}}
        ws.send_bytes(pack({"type": "act", "observation": bad}))
        assert unpack(ws.receive_bytes())["type"] == "error"


def test_ws_bad_json_text_returns_error():
    with _client().websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text("{not json")
        assert ws.receive_json()["type"] == "error"


def test_uvicorn_ws_refuses_permessage_deflate():
    """A client offering deflate must end up uncompressed (deflate costs ~36 ms/frame)."""
    import threading
    import time

    uvicorn = pytest.importorskip("uvicorn")
    pytest.importorskip("websockets")
    from websockets.sync.client import connect

    from hivla.serving.server import UVICORN_WS_KWARGS

    engine = build_engine({"model": {"backend": "echo", "action_horizon": 6},
                           "policy": {"action_dim": 7}})
    engine.start()
    server = uvicorn.Server(uvicorn.Config(create_app(engine), host="127.0.0.1", port=8125,
                                           log_level="warning", **UVICORN_WS_KWARGS))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    try:
        with connect("ws://127.0.0.1:8125/ws", compression="deflate") as ws:
            assert ws.response.headers.get("Sec-WebSocket-Extensions") is None
            assert ws.recv()  # handshake still arrives
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_ws_rejects_object_dtype_payload():
    with _client().websocket_connect("/ws") as ws:
        ws.receive_json()
        bad = {"type": "act", "observation": {"joint_pos": {"__ndarray__": "", "dtype": "object", "shape": [0]}}}
        ws.send_json(bad)
        resp = ws.receive_json()
        assert resp["type"] == "error"
