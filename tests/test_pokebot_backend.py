"""Pokebot family: PokebotBackend against a fake RoboAPI (httpx.MockTransport)."""

from __future__ import annotations

import base64
import io
import json

import numpy as np
import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("PIL")

from hivla import build_engine_from_spec  # noqa: E402
from hivla.core.spec import ModelSpec  # noqa: E402
from hivla.families import available_families  # noqa: E402
from hivla.families.pokebot import PokebotBackend, PokebotError  # noqa: E402

ROWS = 5
HOME = [-0.13, -0.46, 0.49, 0.0, 1.0, 0.0, 0.0]
KEY = "sk-test"


class FakeRoboAPI:
    """Just enough of RoboAPI: models, schema, sessions, chat turns, turn lookup, close."""

    def __init__(self, rows=ROWS):
        self.rows = rows
        self.calls = []          # (method, path, json body or None)
        self.sessions = {}       # id -> {"next_turn", "status", "turns": {n: response}}
        self.fail_next_chat = None   # None | "drop_after" | "drop_before" | int status
        self.bad = None          # a key of response to break
        self.n = 0

    def handler(self, request):
        body = json.loads(request.content) if request.content else None
        self.calls.append((request.method, request.url.path, body))
        if request.headers.get("authorization") != f"Bearer {KEY}":
            return httpx.Response(401, json={"error": {"message": "Invalid token"}})
        path = request.url.path.removeprefix("/v1")
        if path == "/models":
            return httpx.Response(200, json={"data": [{"id": "poke-aurora-preview"}]})
        if path == "/models/poke-aurora-preview/schema":
            return httpx.Response(200, json={
                "model": "poke-aurora-gauss-1030",
                "inputs": {"properties": {"state": {"maxItems": 8}}},
                "image": {"max_pixels": 16000000}, "text": {"max_chars": 4096},
                "session": {"supported": True, "output": {"columns": 8}}})
        if path == "/sessions" and request.method == "POST":
            self.n += 1
            sid = f"sess_{self.n}"
            self.sessions[sid] = {"next_turn": 1, "status": "open", "turns": {}}
            return httpx.Response(200, json={"id": sid, "status": "open", "next_turn": 1,
                                             "messages": []})
        if path.endswith("/close"):
            sid = path.split("/")[2]
            self.sessions[sid]["status"] = "closed"
            return httpx.Response(200, json={"id": sid, "status": "closed", "messages": []})
        if "/turns/" in path:
            _, _, sid, _, turn = path.split("/")
            resp = self.sessions[sid]["turns"].get(int(turn))
            if resp is None:
                return httpx.Response(404, json={"error": {"message": "turn not found"}})
            return httpx.Response(200, json={"status": "succeeded", "response": resp})
        if path == "/chat/completions":
            return self._chat(body)
        return httpx.Response(404, json={"error": {"message": f"no route {path}"}})

    def _chat(self, body):
        fail, self.fail_next_chat = self.fail_next_chat, None
        if fail == "drop_before":
            raise httpx.ReadTimeout("timed out")
        if isinstance(fail, int):
            return httpx.Response(fail, json={"error": {"message": "boom"}})
        sid = body["metadata"]["session_id"]
        turn = int(body["metadata"]["turn"])
        sess = self.sessions[sid]
        assert sess["status"] == "open" and turn == sess["next_turn"]
        state = json.loads(body["messages"][-1]["content"][0]["text"])["state"]
        out = [[*state[:3], *state[3:7], state[7]] for _ in range(self.rows)]
        for i, row in enumerate(out):
            row[0] += 0.01 * i
        rid = f"{turn:032x}"
        resp = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
                    {"id": f"call_{rid}", "type": "function", "function": {
                        "name": "execute_actions",
                        "arguments": json.dumps({"output": out})}}]}}],
                "metadata": {"session_id": sid, "turn": str(turn), "next_turn": str(turn + 1),
                             "action_shape": json.dumps([self.rows, 8])}}
        if self.bad == "tool":
            resp["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = "other"
        elif self.bad == "turn":
            resp["metadata"]["turn"] = str(turn + 5)
        elif self.bad == "nan":
            out[0][0] = float("nan")
            resp["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = \
                json.dumps({"output": out})
        sess["turns"][turn] = resp
        sess["next_turn"] = turn + 1
        if fail == "drop_after":
            raise httpx.ReadTimeout("timed out")
        return httpx.Response(200, json=resp, headers={"x-request-id": rid})

    def chats(self):
        return [b for m, p, b in self.calls if p.endswith("/chat/completions")]

    def closes(self):
        return [(p, b) for m, p, b in self.calls if p.endswith("/close")]


def _spec(**kw):
    native = {"ee": "eef", "camera": "wrist", **kw.pop("native", {})}
    return ModelSpec(name="poke", family="pokebot", action_horizon=kw.pop("horizon", ROWS),
                     n_action_steps=kw.pop("n_action_steps", None), device="cpu",
                     control_mode="cartesian", warmup=False, native_kwargs=native, **kw)


def _obs(x=-0.13, executed=None, image=None):
    obs = {"ee_poses": {"eef": np.array([x, *HOME[1:]], np.float32)},
           "gripper": {"eef": 0.8},
           "images": {"wrist": np.full((720, 1280, 3), 100, np.uint8) if image is None else image},
           "instruction": "pick up the cup"}
    if executed is not None:
        obs["extra"] = {"executed_steps": executed}
    return obs


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("POKEBOT_API_KEY", KEY)
    return FakeRoboAPI()


def _backend(api, **kw):
    b = PokebotBackend(_spec(**kw), transport=httpx.MockTransport(api.handler))
    b.load()
    return b


def test_registered():
    assert "pokebot" in available_families()


def test_first_turn_message_and_chunk(api):
    b = _backend(api)
    chunk = b.predict(_obs())
    (body,) = api.chats()
    assert body["model"] == "poke-aurora-preview"
    assert body["metadata"] == {"session_id": "sess_1", "turn": "1"}
    (user,) = body["messages"]
    text, image = user["content"]
    assert json.loads(text["text"]) == {"instruction": "pick up the cup",
                                        "state": pytest.approx([*HOME, 0.8])}
    url = image["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    from PIL import Image
    w, h = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size
    assert (w, h) == (640, 360)  # 1280x720 scaled to max_pixels, same aspect
    assert chunk["mode"] == "cartesian"
    assert chunk["ee_targets"]["eef"].shape == (ROWS, 7)
    assert chunk["gripper"]["eef"].shape == (ROWS,)
    np.testing.assert_allclose(chunk["ee_targets"]["eef"][:, 0],
                               -0.13 + 0.01 * np.arange(ROWS), atol=1e-6)


def test_second_turn_sends_assistant_and_receipt(api):
    b = _backend(api, n_action_steps=3)
    b.predict(_obs())
    b.predict(_obs(x=-0.10))
    first, second = api.chats()
    assert second["metadata"]["turn"] == "2"
    assistant, tool, user = second["messages"]
    assert assistant == first_response_message(api)
    assert tool["role"] == "tool" and tool["tool_call_id"] == f"call_{1:032x}"
    assert json.loads(tool["content"]) == {"executed_steps": 3, "status": "interrupted"}
    assert user["role"] == "user"


def first_response_message(api):
    return api.sessions["sess_1"]["turns"][1]["choices"][0]["message"]


def test_receipt_from_observation_extra(api):
    b = _backend(api)
    b.predict(_obs())
    b.predict(_obs(executed=ROWS))
    tool = api.chats()[1]["messages"][1]
    assert json.loads(tool["content"]) == {"executed_steps": ROWS, "status": "completed"}


def test_reset_closes_with_receipt_and_opens_new_session(api):
    b = _backend(api)
    b.predict(_obs())
    b.reset()
    (path, body), = api.closes()
    assert path.endswith("/sessions/sess_1/close")
    assert body["reason"] == "stopped"
    assistant, tool = body["messages"]
    assert assistant["tool_calls"][0]["id"] == f"call_{1:032x}"
    assert json.loads(tool["content"])["executed_steps"] == ROWS
    b.predict(_obs())
    assert api.chats()[-1]["metadata"] == {"session_id": "sess_2", "turn": "1"}
    assert len(api.chats()[-1]["messages"]) == 1


def test_reset_without_session_is_a_noop(api):
    b = _backend(api)
    b.reset()
    assert api.closes() == []


def test_unload_closes_session(api):
    b = _backend(api)
    b.predict(_obs())
    b.unload()
    assert len(api.closes()) == 1


def test_timeout_recovers_stored_turn(api):
    b = _backend(api)
    api.fail_next_chat = "drop_after"   # server finished the turn, reply lost
    chunk = b.predict(_obs())
    assert chunk["ee_targets"]["eef"].shape == (ROWS, 7)
    assert len(api.chats()) == 1        # not sent again
    b.predict(_obs())
    assert api.chats()[-1]["metadata"]["turn"] == "2"


def test_timeout_without_result_raises_and_closes(api):
    b = _backend(api)
    api.fail_next_chat = "drop_before"
    with pytest.raises(PokebotError, match="no result"):
        b.predict(_obs())
    (path, body), = api.closes()
    assert body["reason"] == "failed"
    b.predict(_obs())
    assert api.chats()[-1]["metadata"]["session_id"] == "sess_2"


@pytest.mark.parametrize("status", [400, 429, 500])
def test_http_error_raises_with_status(api, status):
    b = _backend(api)
    api.fail_next_chat = status
    with pytest.raises(PokebotError, match=f"HTTP {status}"):
        b.predict(_obs())


@pytest.mark.parametrize("bad, match", [("tool", "execute_actions"), ("turn", "turn"),
                                        ("nan", "non-finite")])
def test_bad_response_rejected(api, bad, match):
    b = _backend(api)
    api.bad = bad
    with pytest.raises(PokebotError, match=match):
        b.predict(_obs())


def test_horizon_cut_and_too_short(api):
    b = _backend(api, horizon=3)
    assert b.predict(_obs())["ee_targets"]["eef"].shape == (3, 7)
    b2 = _backend(FakeRoboAPI(rows=2), horizon=3) if False else None
    api.rows = 2
    with pytest.raises(PokebotError, match="< action_horizon"):
        b.predict(_obs())
    assert b2 is None


def test_missing_key(monkeypatch):
    monkeypatch.delenv("POKEBOT_API_KEY", raising=False)
    b = PokebotBackend(_spec(), transport=httpx.MockTransport(FakeRoboAPI().handler))
    with pytest.raises(RuntimeError, match="POKEBOT_API_KEY"):
        b.load()


def test_wrong_key(monkeypatch):
    monkeypatch.setenv("POKEBOT_API_KEY", "sk-wrong")
    b = PokebotBackend(_spec(), transport=httpx.MockTransport(FakeRoboAPI().handler))
    with pytest.raises(PokebotError, match="HTTP 401"):
        b.load()


def test_unknown_model(api):
    with pytest.raises(PokebotError, match="not available"):
        _backend(api, native={"model": "nope"})


@pytest.mark.parametrize("change, match", [
    (lambda o: o.pop("instruction"), "instruction"),
    (lambda o: o["ee_poses"].clear(), "ee_poses"),
    (lambda o: o["gripper"].clear(), "gripper"),
    (lambda o: o["images"].clear(), "images"),
])
def test_bad_observation(api, change, match):
    b = _backend(api)
    obs = _obs()
    change(obs)
    with pytest.raises(ValueError, match=match):
        b.predict(obs)
    assert api.chats() == []


def test_png_and_config_errors(api):
    b = _backend(api, native={"image_format": "png"})
    b.predict(_obs(image=np.zeros((10, 20, 3), np.uint8)))
    assert api.chats()[0]["messages"][0]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,")
    with pytest.raises(ValueError, match="native_kwargs.ee"):
        PokebotBackend(ModelSpec(family="pokebot", native_kwargs={"camera": "wrist"}))
    with pytest.raises(ValueError, match="image_format"):
        PokebotBackend(_spec(native={"image_format": "bmp"}))


def test_key_not_in_metadata(api):
    b = _backend(api)
    assert KEY not in json.dumps(b.metadata())


def test_engine_end_to_end(api, monkeypatch):
    import hivla.families.pokebot as mod

    real = mod.PokebotBackend
    monkeypatch.setattr(mod, "PokebotBackend",
                        lambda spec: real(spec, transport=httpx.MockTransport(api.handler)))
    engine = build_engine_from_spec(_spec())
    engine.start()
    meta = engine.metadata
    assert meta["stateful"] is True
    assert meta["pokebot"]["model"] == "poke-aurora-preview"
    assert KEY not in json.dumps(meta)
    engine.reset()                       # stateful: the engine wants a reset first
    assert api.closes() == []            # no session yet, nothing to close
    chunk = engine.infer(_obs())
    assert chunk["mode"] == "cartesian" and chunk["ee_targets"]["eef"].shape == (ROWS, 7)
    engine.reset()
    assert len(api.closes()) == 1
