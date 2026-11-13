"""HTTP transport tests: /health, /metadata, /act, auth, bad payloads."""

from __future__ import annotations

import numpy as np
import pytest

from hivla import build_engine
from hivla.protocol import decode_array, encode_array
from hivla.serving.server import create_app

pytest.importorskip("fastapi")
from starlette.testclient import TestClient  # noqa: E402


def _client(api_key=None):
    engine = build_engine({
        "model": {"name": "m", "backend": "echo", "action_horizon": 6},
        "policy": {"action_dim": 7, "cameras": ["head"], "control_mode": "pd"},
    })
    engine.start()
    return TestClient(create_app(engine, api_key=api_key))


def test_health():
    assert _client().get("/health").json() == {"status": "ok"}


def test_app_version_is_package_version():
    import hivla

    assert _client().app.version == hivla.__version__


def test_metadata_handshake():
    meta = _client().get("/metadata").json()
    assert meta["action_horizon"] == 6
    assert meta["control_mode"] == "pd"


def test_act_returns_encoded_chunk():
    client = _client()
    payload = {"joint_pos": encode_array(np.zeros(7, np.float32)), "instruction": "go"}
    resp = client.post("/act", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "pd"
    assert decode_array(body["joint_targets"]).shape == (6, 7)
    assert "server_timings_ms" not in body  # off unless asked


def test_act_timings_opt_in():
    client = _client()
    payload = {"joint_pos": encode_array(np.zeros(7, np.float32)), "instruction": "go"}
    body = client.post("/act?timings=1", json=payload).json()
    assert set(body["server_timings_ms"]) == {"decode", "infer", "encode"}
    assert all(v >= 0 for v in body["server_timings_ms"].values())


def test_act_rejects_object_dtype_payload():
    client = _client()
    bad = {"joint_pos": {"__ndarray__": "", "dtype": "object", "shape": [0]}}
    assert client.post("/act", json=bad).status_code == 400


def test_act_msgpack_roundtrip():
    from hivla.protocol import MSGPACK_MEDIA_TYPE, encode_observation, pack, unpack

    pytest.importorskip("msgpack")
    client = _client()
    obs = {"joint_pos": np.zeros(7, np.float32), "image": np.ones((4, 5, 3), np.uint8),
           "instruction": "go"}
    resp = client.post("/act", content=pack(encode_observation(obs, binary=True)),
                       headers={"Content-Type": MSGPACK_MEDIA_TYPE})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith(MSGPACK_MEDIA_TYPE)
    body = unpack(resp.content)
    assert isinstance(body["joint_targets"]["__ndarray__"], bytes)  # no base64
    assert body["mode"] == "pd"
    assert decode_array(body["joint_targets"]).shape == (6, 7)


@pytest.mark.parametrize("body", [b"\xc1", b"\x92\x01", b"\x01"])
def test_act_rejects_bad_msgpack(body):
    from hivla.protocol import MSGPACK_MEDIA_TYPE

    pytest.importorskip("msgpack")
    resp = _client().post("/act", content=body, headers={"Content-Type": MSGPACK_MEDIA_TYPE})
    assert resp.status_code == 400


def test_act_msgpack_rejects_object_dtype():
    from hivla.protocol import MSGPACK_MEDIA_TYPE, pack

    pytest.importorskip("msgpack")
    bad = {"joint_pos": {"__ndarray__": b"", "dtype": "object", "shape": [0]}}
    resp = _client().post("/act", content=pack(bad), headers={"Content-Type": MSGPACK_MEDIA_TYPE})
    assert resp.status_code == 400


def test_act_rejects_bad_json():
    resp = _client().post("/act", content=b"{not json", headers={"Content-Type": "application/json"})
    assert resp.status_code == 400


def test_auth_enforced():
    client = _client(api_key="secret")
    payload = {"joint_pos": encode_array(np.zeros(7, np.float32))}
    assert client.post("/act", json=payload).status_code == 401
    ok = client.post("/act", json=payload, headers={"Authorization": "Api-Key secret"})
    assert ok.status_code == 200
