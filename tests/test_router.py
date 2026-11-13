"""Router gateway tests: policy picks + end-to-end /act dispatch.

The policies are unit-tested directly (deterministic). The app is tested with a
fake ``httpx.AsyncClient`` installed over ``BackendPool.start``, so each backend
is a counter that echoes a fixed chunk back; this asserts distribution per
policy, auth, bad-payload rejection before forwarding, backend-failure 502, and
raw msgpack passthrough — all without real servers.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest

from hivla.serving.router import ConsistentHash, LeastOutstanding, RoundRobin, make_policy
from hivla.protocol import encode_array, encode_observation

pytest.importorskip("fastapi")
from starlette.testclient import TestClient  # noqa: E402


# --- policies (no app) -------------------------------------------------------

def test_round_robin_cycles():
    p = RoundRobin()
    assert [p.pick(3) for _ in range(7)] == [0, 1, 2, 0, 1, 2, 0]


def test_least_outstanding_prefers_idle():
    p = LeastOutstanding()
    inflight = [5, 0, 2]
    p.bind(inflight)
    assert p.pick(3) == 1          # fewest in-flight
    inflight[1] = 9
    assert p.pick(3) == 2          # now index 2 is lightest


def test_consistent_hash_is_stable_and_spreads():
    p = ConsistentHash()
    sig_a = (("k", (1,)),)
    assert p.pick(4, sig_a) == p.pick(4, sig_a)      # same obs -> same backend
    picks = {p.pick(8, (("k", (i,)),)) for i in range(32)}
    assert len(picks) > 1                            # distinct obs spread out


def test_make_policy_unknown():
    with pytest.raises(ValueError):
        make_policy("nope")


# --- fake backend transport --------------------------------------------------

class _FakeResp:
    def __init__(self, status, content, content_type):
        self.status_code = status
        self.content = content
        self.headers = {"content-type": content_type}

    @property
    def text(self):
        # Lazy + tolerant: the router only reads .text on an error status, and
        # binary (msgpack) bodies are not valid utf-8.
        return self.content.decode(errors="replace") if isinstance(self.content, bytes) \
            else str(self.content)


class _FakeClient:
    """Stand-in for one backend's httpx.AsyncClient; records every POST."""

    def __init__(self, tag, hits, fail=False):
        self._tag = tag
        self._hits = hits
        self._fail = fail

    async def post(self, path, content=None, headers=None, params=None):
        self._hits.append(self._tag)
        if self._fail:
            raise RuntimeError("backend down")
        ctype = (headers or {}).get("Content-Type", "application/json")
        # Echo the request body straight back, like a backend would a chunk.
        return _FakeResp(200, content if content is not None else b"{}", ctype)

    async def aclose(self):
        pass


@contextmanager
def _app(policy="round_robin", n=3, api_key=None, fail_index=None):
    """Yield ``(client, hits)`` with fake backend clients installed.

    Patches ``BackendPool.start`` to swap in counting fakes, then enters the
    ``TestClient`` as a context manager so FastAPI startup/shutdown fire (that
    is what installs and later closes the pool's clients).
    """
    from hivla.serving import router

    hits = []
    original_start = router.BackendPool.start

    def fake_start(self):
        self._clients = [
            _FakeClient(i, hits, fail=(i == fail_index)) for i in range(len(self.urls))
        ]

    router.BackendPool.start = fake_start
    try:
        app = router.create_router_app(
            [f"http://b{i}" for i in range(n)], policy=policy, api_key=api_key
        )
        with TestClient(app) as client:
            yield client, hits
    finally:
        router.BackendPool.start = original_start


def _obs_json():
    return {"joint_pos": encode_array(np.zeros(7, np.float32)), "instruction": "go"}


def test_app_version_is_package_version():
    import hivla

    with _app(n=1) as (client, _):
        assert client.app.version == hivla.__version__


def test_health_reports_pool():
    with _app(policy="least_outstanding", n=2) as (client, _):
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["policy"] == "least_outstanding"
    assert len(body["backends"]) == 2


def test_round_robin_distributes_evenly():
    with _app(policy="round_robin", n=3) as (client, hits):
        for _ in range(9):
            assert client.post("/act", json=_obs_json()).status_code == 200
    assert sorted(hits) == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_consistent_hash_pins_same_observation():
    with _app(policy="consistent_hash", n=4) as (client, hits):
        for _ in range(6):
            client.post("/act", json=_obs_json())
    assert len(set(hits)) == 1          # identical obs -> one backend every time


def test_consistent_hash_matches_engine_signature():
    # The backend the router picks must match hashing the engine's own signature.
    from hivla.serving.engine import _observation_signature
    from hivla.serving.router import ConsistentHash
    from hivla.protocol import decode_observation

    with _app(policy="consistent_hash", n=4) as (client, hits):
        client.post("/act", json=_obs_json())
    sig = _observation_signature(decode_observation(_obs_json()))
    assert hits[0] == ConsistentHash().pick(4, sig)


def test_bad_payload_rejected_before_forwarding():
    with _app(policy="consistent_hash", n=3) as (client, hits):
        bad = {"joint_pos": {"__ndarray__": "", "dtype": "object", "shape": [0]}}
        assert client.post("/act", json=bad).status_code == 400
    assert hits == []                   # never reached a backend


def test_backend_failure_returns_502():
    with _app(policy="round_robin", n=1, fail_index=0) as (client, _):
        assert client.post("/act", json=_obs_json()).status_code == 502


def test_auth_enforced():
    with _app(policy="round_robin", n=1, api_key="secret") as (client, _):
        assert client.post("/act", json=_obs_json()).status_code == 401
        ok = client.post("/act", json=_obs_json(),
                         headers={"Authorization": "Api-Key secret"})
        assert ok.status_code == 200


def test_msgpack_passthrough_unchanged():
    from hivla.protocol import MSGPACK_MEDIA_TYPE, pack

    pytest.importorskip("msgpack")
    with _app(policy="round_robin", n=1) as (client, hits):
        body = pack(encode_observation({"joint_pos": np.zeros(7, np.float32)}, binary=True))
        resp = client.post("/act", content=body, headers={"Content-Type": MSGPACK_MEDIA_TYPE})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith(MSGPACK_MEDIA_TYPE)
        assert resp.content == body     # relayed byte-for-byte, no re-encode
    assert hits == [0]
