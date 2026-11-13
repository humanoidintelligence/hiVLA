"""hivla.client against a real hiVLA server (echo backend) and a fake /ws server."""

from __future__ import annotations

import json
import socket
import threading
import time

import numpy as np
import pytest

from hivla import build_engine
from hivla.client import SERVER_TIMINGS_KEY, VlaClient, VlaWsClient
from hivla.protocol import decode_array, encode_array

# hivla.client imports these on first use, so skipping here is enough.
httpx = pytest.importorskip("httpx")
pytest.importorskip("websockets")
msgpack = pytest.importorskip("msgpack")

HORIZON, DOF = 6, 7


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server_url():
    """A real hiVLA server (create_app) on the echo backend, with an API key."""
    import uvicorn

    from hivla.serving.server import create_app

    engine = build_engine({
        "model": {"backend": "echo", "action_horizon": HORIZON, "n_action_steps": 4},
        "policy": {"action_dim": DOF, "control_mode": "pd"},
    })
    engine.start()
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(engine, api_key="k"), host="127.0.0.1",
                                           port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


def _obs(value: float = 0.0) -> dict:
    return {
        "joint_pos": np.full(DOF, value, np.float32),
        "images": {"head": np.zeros((48, 64, 3), np.uint8)},
        "instruction": "pick up the cube",
    }


def _clients(url, **kw):
    return {
        ("http", "json"): lambda: VlaClient(url, wire="json", api_key="k", **kw),
        ("http", "msgpack"): lambda: VlaClient(url, wire="msgpack", api_key="k", **kw),
        ("ws", "json"): lambda: VlaWsClient(url, wire="json", api_key="k", **kw),
        ("ws", "msgpack"): lambda: VlaWsClient(url, wire="msgpack", api_key="k", **kw),
    }


COMBOS = [("http", "json"), ("http", "msgpack"), ("ws", "json"), ("ws", "msgpack")]


@pytest.mark.parametrize("combo", COMBOS, ids=lambda c: "-".join(c))
def test_act_round_trip(server_url, combo):
    client = _clients(server_url)[combo]()
    try:
        meta = client.reset()
        assert meta is client.metadata
        assert meta["action_horizon"] == HORIZON and meta["n_action_steps"] == 4
        chunk = client.act(_obs(1.0))
        assert chunk["mode"] == "pd"
        assert chunk["joint_targets"].shape == (HORIZON, DOF)
        assert SERVER_TIMINGS_KEY not in chunk
        # A second call on the same client works too (pooled / long-lived connection).
        assert client.act(_obs(2.0))["joint_targets"].shape == (HORIZON, DOF)
    finally:
        client.close()
        client.close()  # idempotent


@pytest.mark.parametrize("combo", COMBOS, ids=lambda c: "-".join(c))
def test_timings_come_back(server_url, combo):
    client = _clients(server_url, timings=True)[combo]()
    try:
        timings = client.act(_obs())[SERVER_TIMINGS_KEY]
        assert "infer" in timings and all(v >= 0 for v in timings.values())
    finally:
        client.close()


def test_ws_metadata_arrives_with_the_handshake(server_url):
    client = VlaWsClient(server_url)
    try:
        client.act(_obs())  # no reset(): the connect handshake fills metadata
        assert client.metadata["n_action_steps"] == 4
    finally:
        client.close()


def test_http_without_api_key_is_rejected(server_url):
    client = VlaClient(server_url)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            client.act(_obs())
    finally:
        client.close()


def test_object_dtype_is_refused_before_sending(server_url):
    client = VlaClient(server_url, api_key="k")
    try:
        with pytest.raises(ValueError, match="object-dtype"):
            client.act({"joint_pos": np.array([object()], dtype=object)})
    finally:
        client.close()


@pytest.mark.parametrize(
    ("endpoint", "url"),
    [("http://host:8000", "ws://host:8000/ws"), ("https://host/", "wss://host/ws"),
     ("ws://host:8000", "ws://host:8000/ws")],
)
def test_ws_endpoint_scheme_mapping(endpoint, url):
    assert VlaWsClient(endpoint).url == url


def test_bad_endpoint_and_wire_rejected():
    with pytest.raises(ValueError):
        VlaWsClient("host:8000")
    with pytest.raises(ValueError):
        VlaWsClient("http://host", wire="protobuf")
    with pytest.raises(ValueError):
        VlaClient("http://host", wire="protobuf")


@pytest.mark.parametrize("cls", [VlaClient, VlaWsClient])
def test_reset_tolerates_server_down(cls):
    client = cls(f"http://127.0.0.1:{_free_port()}", timeout=0.5)
    assert client.reset() == {}
    with pytest.raises((OSError, httpx.TransportError)):  # connection refused
        client.act(_obs())
    client.close()


# -- a fake /ws server for slow and failing replies ---------------------------------


def _send(conn, msg, binary):
    conn.send(msgpack.packb(msg, use_bin_type=True) if binary else json.dumps(msg))


@pytest.fixture
def fake_ws():
    """Echoes joint_pos[0] back in the chunk so replies are identifiable."""
    from websockets.sync.server import serve

    state = {"connections": 0, "delay": 0.0, "extensions": [], "resets": 0}

    def handler(conn):
        state["connections"] += 1
        state["extensions"].append(conn.response.headers.get("Sec-WebSocket-Extensions"))
        conn.send(json.dumps({"type": "metadata", "metadata": {"action_horizon": HORIZON}}))
        for frame in conn:
            binary = isinstance(frame, (bytes, bytearray))
            msg = msgpack.unpackb(frame, raw=False) if binary else json.loads(frame)
            if msg.get("type") == "metadata":
                _send(conn, {"type": "metadata", "metadata": {"action_horizon": HORIZON}}, binary)
                continue
            if msg.get("type") == "reset":
                state["resets"] += 1
                _send(conn, {"type": "reset"}, binary)
                continue
            obs = msg["observation"]
            if "joint_pos" not in obs:
                _send(conn, {"type": "error", "detail": "missing joint_pos"}, binary)
                continue
            if state["delay"]:
                time.sleep(state["delay"])
            value = float(decode_array(obs["joint_pos"])[0])
            targets = encode_array(np.full((HORIZON, DOF), value, np.float32), binary)
            _send(conn, {"type": "action", "chunk": {"mode": "pd", "joint_targets": targets}},
                  binary)

    server = serve(handler, "127.0.0.1", 0)  # offers deflate; the client must not accept
    port = server.socket.getsockname()[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}", state
    server.shutdown()
    thread.join(timeout=5)


def test_ws_reset_on_open_connection_reuses_it(fake_ws):
    url, state = fake_ws
    client = VlaWsClient(url)
    client.reset()
    client.reset()  # asks again over the same socket
    assert client.metadata["action_horizon"] == HORIZON
    client.close()
    assert state["connections"] == 1
    assert state["resets"] == 2
    assert state["extensions"] == [None]  # no permessage-deflate negotiated


def test_ws_server_error_raises_and_connection_survives(fake_ws):
    url, state = fake_ws
    client = VlaWsClient(url)
    with pytest.raises(RuntimeError, match="missing joint_pos"):
        client.act({"images": {"head": np.zeros((4, 4, 3), np.uint8)}})
    assert float(client.act(_obs(1.0))["joint_targets"][0, 0]) == 1.0
    client.close()
    assert state["connections"] == 1


def test_ws_timeout_reconnects_and_drops_the_late_reply(fake_ws):
    url, state = fake_ws
    client = VlaWsClient(url, timeout=0.2)
    state["delay"] = 0.5
    with pytest.raises(TimeoutError):
        client.act(_obs(7.0))
    state["delay"] = 0.0
    # A fresh connection answers the new request, not the stale 7.0 reply.
    assert float(client.act(_obs(8.0))["joint_targets"][0, 0]) == 8.0
    client.close()
    assert state["connections"] == 2


def test_ws_calls_from_two_threads_do_not_cross(fake_ws):
    url, _ = fake_ws
    client = VlaWsClient(url)
    results = {}

    def call(v):
        results[v] = float(client.act(_obs(v))["joint_targets"][0, 0])

    threads = [threading.Thread(target=call, args=(float(v),)) for v in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    client.close()
    assert results == {float(v): float(v) for v in range(8)}
