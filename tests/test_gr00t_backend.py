"""GR00T family: Gr00tBackend against a fake ``gr00t`` package (gr1 layout)."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from hivla import ModelSpec, build_engine_from_spec, get_adapter
from hivla.families import Gr00tFamily

GR1_STATE = [("left_arm", 7), ("right_arm", 7), ("left_hand", 6), ("right_hand", 6), ("waist", 3)]
VIDEO_KEY = "ego_view_bg_crop_pad_res256_freq20"


def _install_fake_gr00t(monkeypatch, horizon=16, state_steps=1):
    """Minimal stand-in for the Isaac-GR00T n1.6 API surface Gr00tBackend uses."""
    calls = {}
    keys = [k for k, _ in GR1_STATE]

    class Gr00tPolicy:
        def __init__(self, tag, path, *, device, strict=True):
            calls.update(tag=tag, path=path, device=device, strict=strict)
            params = {k: {"dim": np.array(d)} for k, d in GR1_STATE}
            self.processor = SimpleNamespace(
                state_action_processor=SimpleNamespace(norm_params={tag: {"state": params}})
            )

        def get_modality_config(self):
            return {
                "video": SimpleNamespace(modality_keys=[VIDEO_KEY], delta_indices=[0]),
                "state": SimpleNamespace(modality_keys=keys,
                                         delta_indices=list(range(state_steps))),
                "action": SimpleNamespace(modality_keys=keys,
                                          delta_indices=list(range(horizon))),
                "language": SimpleNamespace(modality_keys=["task"], delta_indices=[0]),
            }

        def get_action(self, obs):
            calls["obs"] = obs
            if obs["video"][VIDEO_KEY].dtype != np.uint8:
                raise AssertionError("video must be uint8")
            b = len(obs["language"]["task"])
            # Each action group echoes its state group's batch index + group offset.
            out = {}
            for j, (k, d) in enumerate(GR1_STATE):
                base = np.arange(b, dtype=np.float32)[:, None, None] * 100 + j
                out[k] = np.broadcast_to(base, (b, horizon, d)).astype(np.float32)
            return out, {}

    tags = types.ModuleType("gr00t.data.embodiment_tags")
    tags.EmbodimentTag = str
    policy_mod = types.ModuleType("gr00t.policy.gr00t_policy")
    policy_mod.Gr00tPolicy = Gr00tPolicy
    for name, mod in [("gr00t", types.ModuleType("gr00t")),
                      ("gr00t.data", types.ModuleType("gr00t.data")),
                      ("gr00t.data.embodiment_tags", tags),
                      ("gr00t.policy", types.ModuleType("gr00t.policy")),
                      ("gr00t.policy.gr00t_policy", policy_mod)]:
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


def _spec(**native):
    native = {"embodiment_tag": "gr1", **native}
    return ModelSpec(family="gr00t", checkpoint="ckpt", device="cpu", action_horizon=16,
                     action_dim=29, native_kwargs=native)


def _obs(**extra):
    obs = {
        "joint_pos": np.arange(29, dtype=np.float64),
        "images": {"ego_view": np.zeros((256, 256, 3), np.uint8)},
        "instruction": "pick the apple",
    }
    obs.update(extra)
    return obs


def test_gr00t_is_a_dedicated_family():
    assert isinstance(get_adapter("gr00t"), Gr00tFamily)


def test_gr00t_backend_requires_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "gr00t", None)
    backend = get_adapter("gr00t").build_backend(_spec())
    with pytest.raises(ImportError, match="gr00t"):
        backend.load()


def test_gr00t_backend_requires_embodiment_tag(monkeypatch):
    _install_fake_gr00t(monkeypatch)
    spec = ModelSpec(family="gr00t", checkpoint="ckpt", device="cpu")
    with pytest.raises(ValueError, match="embodiment_tag"):
        get_adapter("gr00t").build_backend(spec).load()


def test_gr00t_engine_splits_state_and_joins_actions(monkeypatch):
    calls = _install_fake_gr00t(monkeypatch)
    engine = build_engine_from_spec(_spec())
    engine.start()
    chunk = engine.infer(_obs())

    assert (calls["tag"], calls["path"], calls["device"], calls["strict"]) == (
        "gr1", "ckpt", "cpu", True)
    obs = calls["obs"]
    assert obs["video"][VIDEO_KEY].shape == (1, 1, 256, 256, 3)  # (B, T, H, W, C)
    assert obs["language"] == {"task": [["pick the apple"]]}
    np.testing.assert_array_equal(obs["state"]["left_arm"], np.arange(7).reshape(1, 1, 7))
    np.testing.assert_array_equal(obs["state"]["waist"], np.arange(26, 29).reshape(1, 1, 3))
    assert obs["state"]["waist"].dtype == np.float32

    jt = chunk["joint_targets"]
    assert jt.shape == (16, 29) and jt.dtype == np.float32
    # Groups are concatenated in action-key order: left_arm (0) ... waist (4).
    np.testing.assert_array_equal(jt[0, :7], 0)
    np.testing.assert_array_equal(jt[0, 26:], 4)
    assert chunk["mode"] == "position"


def test_gr00t_backend_accepts_state_groups_and_task_default(monkeypatch):
    calls = _install_fake_gr00t(monkeypatch)
    backend = get_adapter("gr00t").build_backend(_spec(task="default task"))
    backend.load()
    state = {k: np.full(d, i, np.float32) for i, (k, d) in enumerate(GR1_STATE)}
    out = backend.predict({"state": state,
                           "images": {VIDEO_KEY: np.zeros((8, 8, 3), np.uint8)}})
    assert out["joint_targets"].shape == (16, 29)
    assert calls["obs"]["language"]["task"] == [["default task"]]
    np.testing.assert_array_equal(calls["obs"]["state"]["right_hand"], 3)


def test_gr00t_backend_batches_in_one_call(monkeypatch):
    _install_fake_gr00t(monkeypatch)
    backend = get_adapter("gr00t").build_backend(_spec())
    backend.load()
    outs = backend.predict_batch([_obs(), _obs(instruction="second")])
    assert [o["joint_targets"][0, 0] for o in outs] == [0.0, 100.0]


def test_gr00t_backend_compiles_only_the_dit_head(monkeypatch):
    _install_fake_gr00t(monkeypatch)
    compiled = {}
    torch = types.ModuleType("torch")
    torch.compile = lambda fn, mode: compiled.update(fn=fn, mode=mode) or (lambda *a: "compiled")
    monkeypatch.setitem(sys.modules, "torch", torch)

    dit = SimpleNamespace(forward=lambda *a: "eager")
    # The fake Gr00tPolicy has no model; give it the action_head.model path.
    import gr00t.policy.gr00t_policy as policy_mod

    original = policy_mod.Gr00tPolicy.__init__

    def init(self, *a, **kw):
        original(self, *a, **kw)
        self.model = SimpleNamespace(action_head=SimpleNamespace(model=dit))

    monkeypatch.setattr(policy_mod.Gr00tPolicy, "__init__", init)
    get_adapter("gr00t").build_backend(_spec(compile="max-autotune")).load()
    assert compiled["mode"] == "max-autotune"
    assert dit.forward() == "compiled"

    compiled.clear()
    get_adapter("gr00t").build_backend(_spec()).load()
    assert compiled == {}  # off by default


def test_gr00t_backend_slices_to_action_horizon(monkeypatch):
    _install_fake_gr00t(monkeypatch)
    spec = _spec()
    spec.action_horizon = 8
    backend = get_adapter("gr00t").build_backend(spec)
    backend.load()
    assert backend.predict(_obs())["joint_targets"].shape == (8, 29)


@pytest.mark.parametrize(
    "obs, match",
    [
        (_obs(joint_pos=np.zeros(28)), "29"),
        (_obs(images={}), "video keys"),
        (_obs(instruction=None), "instruction"),
    ],
)
def test_gr00t_backend_rejects_bad_observations(monkeypatch, obs, match):
    _install_fake_gr00t(monkeypatch)
    backend = get_adapter("gr00t").build_backend(_spec(cameras={"ego_view": VIDEO_KEY}))
    backend.load()
    with pytest.raises(ValueError, match=match):
        backend.predict(obs)


def test_gr00t_warmup_runs_one_zeroed_forward(monkeypatch):
    calls = _install_fake_gr00t(monkeypatch)
    backend = get_adapter("gr00t").build_backend(_spec())
    backend.load()
    backend.warmup()
    # Warmup fed a valid observation through: state groups and the video key present.
    assert set(calls["obs"]["state"]) == {k for k, _ in GR1_STATE}
    assert VIDEO_KEY in calls["obs"]["video"]
    assert calls["obs"]["video"][VIDEO_KEY].dtype == np.uint8


def test_gr00t_warmup_fires_via_engine_start(monkeypatch):
    calls = _install_fake_gr00t(monkeypatch)
    engine = build_engine_from_spec(_spec())  # runtime.warmup defaults True
    assert "obs" not in calls
    engine.start()
    assert "obs" in calls  # start() ran warmup's forward before any request


def test_gr00t_strict_assertion_becomes_value_error(monkeypatch):
    """GR00T's strict checks assert; the server must answer 400, not 500."""
    _install_fake_gr00t(monkeypatch)
    backend = get_adapter("gr00t").build_backend(_spec())
    backend.load()

    def reject(obs):
        raise AssertionError("State key 'waist' must be float32")

    backend._policy.get_action = reject
    with pytest.raises(ValueError, match="rejected.*waist"):
        backend.predict(_obs())


def test_gr00t_backend_rejects_long_horizon_and_history(monkeypatch):
    _install_fake_gr00t(monkeypatch, horizon=8)
    with pytest.raises(ValueError, match="action_horizon"):
        get_adapter("gr00t").build_backend(_spec()).load()
    _install_fake_gr00t(monkeypatch, state_steps=2)
    with pytest.raises(NotImplementedError, match="single-step"):
        get_adapter("gr00t").build_backend(_spec()).load()
