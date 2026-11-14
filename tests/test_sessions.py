"""Sessions: reset / observe and a stateful model held by one session at a time."""

from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest

from hivla import ModelSpec
from hivla.runtimes.base import ModelBackend
from hivla.core.build import build_engine_from_spec
from hivla.serving.engine import InferenceEngine, ObservationCache, SessionError
from hivla.core.policy import VlaPolicy

T, DOF = 4, 3


class Counting(ModelBackend):
    """A stateful model: each chunk is filled with how many observations it has seen."""

    stateful = True
    observe_mode = "every_step"

    def __init__(self) -> None:
        self.seen = 0
        self.resets = 0

    def load(self) -> None:
        pass

    def reset(self) -> None:
        self.resets += 1
        self.seen = 0

    def observe(self, inputs) -> None:
        self.seen += 1

    def predict(self, inputs):
        self.seen += 1
        return {"joint_targets": np.full((T, DOF), self.seen, np.float32)}


def _engine(backend=None, **kw) -> InferenceEngine:
    engine = InferenceEngine(VlaPolicy(backend or Counting(), action_horizon=T), **kw)
    engine.start()
    return engine


def _obs() -> dict:
    return {"joint_pos": np.zeros(DOF, np.float32)}


def test_reset_binds_the_model_to_one_session():
    engine = _engine()
    with pytest.raises(SessionError, match="call reset"):
        engine.infer(_obs(), session="a")
    engine.reset("a")
    assert engine.infer(_obs(), session="a")["joint_targets"][0, 0] == 1
    engine.observe(_obs(), session="a")
    assert engine.infer(_obs(), session="a")["joint_targets"][0, 0] == 3
    with pytest.raises(SessionError, match="another session"):
        engine.infer(_obs(), session="b")
    with pytest.raises(SessionError):
        engine.observe(_obs(), session="b")
    engine.reset("b")  # b takes the model over and starts from zero
    assert engine.infer(_obs(), session="b")["joint_targets"][0, 0] == 1
    with pytest.raises(SessionError):
        engine.infer(_obs(), session="a")


def test_stateful_model_refuses_cache_and_batching():
    with pytest.raises(ValueError, match="stateful"):
        InferenceEngine(VlaPolicy(Counting()), cache=ObservationCache())
    with pytest.raises(ValueError, match="stateful"):
        InferenceEngine(VlaPolicy(Counting()), max_batch_size=4)


def test_stateless_model_ignores_sessions():
    engine = InferenceEngine(VlaPolicy(_Echo(), action_horizon=T), cache=ObservationCache())
    engine.start()
    engine.infer(_obs())  # no reset needed
    engine.infer(_obs(), session="x")
    engine.observe(_obs(), session="y")  # ignored
    engine.reset("z")
    engine.infer(_obs(), session="w")


class _Echo(ModelBackend):
    def load(self) -> None:
        pass

    def predict(self, inputs):
        return {"joint_targets": np.zeros((T, DOF), np.float32)}


def test_metadata_reports_stateful_and_observe():
    from hivla import build_engine

    meta = build_engine({"model": {"backend": "echo"}}).metadata
    assert meta["stateful"] is False and meta["observe"] == "chunk"


def test_observe_runs_the_input_transforms():
    seen = []

    class Keep(Counting):
        def observe(self, inputs):
            seen.append(inputs)

    engine = InferenceEngine(
        VlaPolicy(Keep(), input_transforms=[lambda d: {**d, "tagged": True}], action_horizon=T))
    engine.start()
    engine.reset("s")
    engine.observe(_obs(), session="s")
    assert seen[0]["tagged"] is True


# --- over the wire ------------------------------------------------------------

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")
pytest.importorskip("websockets")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def served():
    import uvicorn

    from hivla.serving.server import create_app

    backend = Counting()
    engine = _engine(backend)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(engine), host="127.0.0.1", port=port,
                                           log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started
    yield f"http://127.0.0.1:{port}", backend
    server.should_exit = True
    thread.join(timeout=5)


@pytest.mark.parametrize("kind", ["http", "ws"])
@pytest.mark.parametrize("wire", ["json", "msgpack"])
def test_client_episode_over_the_wire(served, kind, wire):
    from hivla.client import VlaClient, VlaWsClient

    url, backend = served
    cls = VlaClient if kind == "http" else VlaWsClient
    a, b = cls(url, wire=wire), cls(url, wire=wire)
    try:
        meta = a.reset()
        assert meta["stateful"] is True and meta["observe"] == "every_step"
        assert backend.resets == 1
        assert a.act(_obs())["joint_targets"][0, 0] == 1
        a.observe(_obs())
        a.observe(_obs())
        assert a.act(_obs())["joint_targets"][0, 0] == 4
        # A second client may not use the model a holds ...
        if kind == "http":
            with pytest.raises(httpx.HTTPStatusError) as err:
                b.act(_obs())
            assert err.value.response.status_code == 409
        else:
            with pytest.raises(RuntimeError, match="another session|call reset"):
                b.act(_obs())
        # ... until it resets, which takes the model over.
        b.reset()
        assert b.act(_obs())["joint_targets"][0, 0] == 1
    finally:
        a.close()
        b.close()


def test_http_without_session_header_is_one_shared_session(served):
    url, _ = served
    with httpx.Client() as c:
        assert c.post(f"{url}/reset").json() == {"ok": True}
        from hivla.protocol import encode_observation

        assert c.post(f"{url}/act", json=encode_observation(_obs())).status_code == 200


def test_spec_with_stateful_backend_reports_it(monkeypatch):
    from hivla import families

    class Fam(families.FamilyAdapter):
        name = "counting_test"

        def build_backend(self, spec):
            return Counting()

        def build_transforms(self, spec):
            return [], []

    monkeypatch.setitem(families._REGISTRY, "counting_test", Fam())
    engine = build_engine_from_spec(ModelSpec(family="counting_test", action_horizon=T,
                                              action_dim=DOF))
    assert engine.stateful and engine.metadata["stateful"] is True


def test_ws_reconnect_to_a_stateful_model_resets_and_raises_session_lost(served):
    from websockets.exceptions import ConnectionClosed

    from hivla.client import SessionLost, VlaWsClient

    url, backend = served
    client = VlaWsClient(url)
    try:
        client.reset()
        assert client.act(_obs())["joint_targets"][0, 0] == 1
        client._conn.close()  # the network drops the connection
        with pytest.raises(ConnectionClosed):
            client.act(_obs())
        resets = backend.resets
        # The next call reconnects and resets the new session, but tells the caller.
        with pytest.raises(SessionLost, match="restart the episode"):
            client.act(_obs())
        assert backend.resets == resets + 1
        # The new session works and starts from zero.
        assert client.act(_obs())["joint_targets"][0, 0] == 1
        client.observe(_obs())
    finally:
        client.close()


def test_ws_reconnect_without_an_episode_does_not_raise(served):
    from hivla.client import VlaWsClient

    url, backend = served
    client = VlaWsClient(url)
    try:
        client.reset()
        client.close()  # a deliberate close is not a lost session
        client.reset()
        assert client.act(_obs())["joint_targets"][0, 0] == 1
    finally:
        client.close()
