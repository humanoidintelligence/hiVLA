"""Backend registry: auto inference + remote proxy round-trip."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from hivla import build_engine, infer_backend
from hivla.runtimes.remote import RemoteBackend
from hivla.protocol import (
    decode_action_chunk,
    encode_action_chunk,
    encode_observation,
)


def test_infer_backend_by_extension():
    assert infer_backend("model.engine") == "tensorrt"
    assert infer_backend("model.plan") == "tensorrt"
    assert infer_backend("model.onnx") == "onnx"
    assert infer_backend("model.safetensors") == "torch"
    assert infer_backend("model.pt") == "torch"
    assert infer_backend(None) == "echo"


def test_auto_backend_resolves_to_echo_without_checkpoint():
    engine = build_engine({"model": {"backend": "auto", "action_horizon": 4}, "policy": {"action_dim": 3}})
    engine.start()
    chunk = engine.infer({"joint_pos": np.zeros(3, dtype=np.float32)})
    assert chunk["joint_targets"].shape == (4, 3)


def test_encode_observation_and_decode_chunk_roundtrip():
    obs = {"joint_pos": np.zeros(7, np.float32), "images": {"head": np.zeros((2, 2, 3), np.uint8)}}
    payload = encode_observation(obs)
    assert "joint_pos" in payload and "images" in payload

    chunk = {"mode": "pd", "joint_targets": np.zeros((4, 7)), "gripper": {"g": np.zeros(4)}}
    dec = decode_action_chunk(encode_action_chunk(chunk))
    assert dec["mode"] == "pd"
    assert dec["joint_targets"].shape == (4, 7)
    assert dec["gripper"]["g"].shape == (4,)


@pytest.fixture(scope="module")
def echo_server_url():
    import uvicorn

    from hivla.serving.server import create_app

    engine = build_engine({
        "model": {"backend": "echo", "action_horizon": 5},
        "policy": {"action_dim": 7, "control_mode": "pd"},
    })
    engine.start()
    config = uvicorn.Config(create_app(engine), host="127.0.0.1", port=8124, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started
    yield "http://127.0.0.1:8124"
    server.should_exit = True
    thread.join(timeout=5)


def test_remote_backend_proxies_to_echo(echo_server_url):
    backend = RemoteBackend(endpoint=echo_server_url)
    backend.load()
    out = backend.predict({"joint_pos": np.zeros(7, dtype=np.float32), "instruction": "go"})
    assert out["mode"] == "pd"
    assert out["joint_targets"].shape == (5, 7)


def test_remote_backend_msgpack_wire(echo_server_url):
    pytest.importorskip("msgpack")
    backend = RemoteBackend(endpoint=echo_server_url, wire="msgpack")
    backend.load()
    obs = {"image": np.zeros((8, 8, 3), dtype=np.uint8), "instruction": "go"}
    for _ in range(2):  # second call reuses the pooled connection
        out = backend.predict(obs)
        assert out["mode"] == "pd"
        assert out["joint_targets"].shape == (5, 7)
    backend.unload()


def test_binary_array_roundtrip():
    from hivla.protocol import decode_array, encode_array, pack, unpack

    pytest.importorskip("msgpack")
    arr = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    enc = encode_array(arr, binary=True)
    assert isinstance(enc["__ndarray__"], bytes)
    np.testing.assert_array_equal(decode_array(unpack(pack(enc))), arr)
