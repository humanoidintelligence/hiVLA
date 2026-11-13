"""StarVLA family: StarvlaBackend against a fake ``starVLA`` package (GR1 layout)."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from hivla import ModelSpec, build_engine_from_spec, get_adapter
from hivla.families.starvla import minmax_unnormalize, sincos_encode
from hivla.families import StarvlaFamily

GROUPS = [7, 7, 6, 6, 3]
LOW = np.linspace(-2.0, -1.0, 29)
HIGH = np.linspace(1.0, 3.0, 29)
MASK = [True] * 28 + [False]  # last dim passes through unnormalized


class _Namespace(dict):
    """Attribute access over nested dicts, like the OmegaConf config StarVLA uses."""

    def __getattr__(self, name):
        value = self[name]
        return _Namespace(value) if isinstance(value, dict) else value

    def __setattr__(self, name, value):
        self[name] = value


def _install_fake_starvla(monkeypatch, n_stats=1, horizon=16):
    calls = {"predict": []}

    class Head:
        def forward(self, x):
            return x

    class Model:
        def __init__(self, cfg):
            calls["cfg"] = cfg
            self.action_model = types.SimpleNamespace(model=Head())

        def load_state_dict(self, sd, strict):
            calls["strict"] = strict

        def to(self, arg):
            calls.setdefault("to", []).append(arg)
            return self

        def eval(self):
            return self

        def predict_action(self, examples):
            calls["predict"].append(examples)
            b = len(examples)
            # Normalized output: row i is all i/10, with one out-of-range value.
            out = np.stack([np.full((horizon, 29), i / 10.0) for i in range(b)])
            out[:, 0, 0] = 5.0  # clipped to 1 -> HIGH[0]
            return {"normalized_actions": out}

    stats = {"action": {"min": LOW.tolist(), "max": HIGH.tolist(), "mask": MASK}}
    norm = {f"key{i}": stats for i in range(n_stats)} if n_stats > 1 else {"gr1": stats}

    def read_mode_config(path):
        calls["path"] = path
        cfg = {"framework": {"qwenvl": {"base_vlm": "./pt_models/rel"},
                             "action_model": {"action_horizon": horizon}},
               "trainer": {}}
        return cfg, norm

    framework = types.ModuleType("starVLA.model.framework")
    framework.build_framework = lambda cfg: Model(cfg)
    share = types.ModuleType("starVLA.model.framework.share_tools")
    share.read_mode_config = read_mode_config
    share.dict_to_namespace = _Namespace

    torch = types.ModuleType("torch")
    torch.bfloat16 = "bf16"
    torch.load = lambda path, map_location: {}
    torch.compile = lambda fn, mode: calls.update(compile=mode) or fn

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    torch.inference_mode = _NoGrad

    cv2 = types.ModuleType("cv2")
    cv2.INTER_AREA = 3
    cv2.resize = lambda img, size, interpolation: (
        calls.setdefault("resize", []).append((img.shape, size, interpolation))
        or np.zeros((size[1], size[0], 3), np.uint8))

    for name, mod in [("starVLA", types.ModuleType("starVLA")),
                      ("starVLA.model", types.ModuleType("starVLA.model")),
                      ("starVLA.model.framework", framework),
                      ("starVLA.model.framework.share_tools", share),
                      ("torch", torch), ("cv2", cv2)]:
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


def _spec(**native):
    native = {"state_groups": GROUPS, "state_encoding": "sincos", **native}
    return ModelSpec(family="starvla", checkpoint="run/checkpoints/steps_1.pt", device="cpu",
                     action_horizon=16, action_dim=29, cameras=["ego_view"],
                     native_kwargs=native)


def _obs(**extra):
    obs = {
        "joint_pos": np.linspace(-1.0, 1.0, 29),
        "images": {"ego_view": np.zeros((256, 256, 3), np.uint8)},
        "instruction": "pick the bottle",
    }
    obs.update(extra)
    return obs


def test_starvla_is_a_dedicated_family():
    assert isinstance(get_adapter("starvla"), StarvlaFamily)


def test_starvla_backend_requires_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "starVLA", None)
    with pytest.raises(ImportError, match="starVLA"):
        get_adapter("starvla").build_backend(_spec()).load()


def test_helpers_match_starvla_robocasa_client():
    groups = [np.full(d, 0.5) for d in GROUPS]
    enc = sincos_encode(groups)
    assert enc.shape == (58,)
    np.testing.assert_allclose(enc[:7], np.sin(0.5))  # left_arm sin
    np.testing.assert_allclose(enc[7:14], np.cos(0.5))  # left_arm cos
    np.testing.assert_allclose(enc[-3:], np.cos(0.5))  # waist cos

    stats = {"min": LOW, "max": HIGH, "mask": MASK}
    out = minmax_unnormalize(np.array([[-1.0] * 29, [5.0] * 29]), stats)
    np.testing.assert_allclose(out[0, :28], LOW[:28])
    np.testing.assert_allclose(out[1, :28], HIGH[:28])
    assert out[0, 28] == -1.0 and out[1, 28] == 1.0  # unmasked: clipped, not scaled


def test_engine_encodes_state_resizes_and_unnormalizes(monkeypatch):
    calls = _install_fake_starvla(monkeypatch)
    engine = build_engine_from_spec(_spec(base_vlm="/abs/vlm"))
    engine.start()
    chunk = engine.infer(_obs())

    assert calls["path"] == "run/checkpoints/steps_1.pt" and calls["strict"] is True
    assert calls["cfg"]["framework"]["qwenvl"]["base_vlm"] == "/abs/vlm"
    assert calls["to"] == ["bf16", "cpu"]
    (example,) = calls["predict"][0]
    assert example["lang"] == "pick the bottle"
    assert len(example["image"]) == 1 and example["image"][0].shape == (224, 224, 3)
    assert calls["resize"][0] == ((256, 256, 3), (224, 224), 3)
    assert example["state"].shape == (1, 58)
    np.testing.assert_allclose(example["state"][0, :7], np.sin(np.linspace(-1, 1, 29)[:7]))

    jt = chunk["joint_targets"]
    assert jt.shape == (16, 29) and jt.dtype == np.float32
    np.testing.assert_allclose(jt[0, 0], HIGH[0], rtol=1e-6)  # 5.0 clipped to 1
    np.testing.assert_allclose(jt[1, :28], (LOW + HIGH)[:28] / 2, rtol=1e-6)  # 0 -> midpoint
    assert chunk["mode"] == "position"


def test_batch_is_one_predict_call(monkeypatch):
    calls = _install_fake_starvla(monkeypatch)
    backend = get_adapter("starvla").build_backend(_spec())
    backend.load()
    out = backend.predict_batch([_obs(), _obs(), _obs()])
    assert len(calls["predict"]) == 1 and len(calls["predict"][0]) == 3
    # Row i was i/10 normalized; check one masked dim per row.
    for i, o in enumerate(out):
        np.testing.assert_allclose(o["joint_targets"][2, 5],
                                   (i / 10 + 1) / 2 * (HIGH[5] - LOW[5]) + LOW[5], rtol=1e-6)


def test_compile_wraps_action_head(monkeypatch):
    calls = _install_fake_starvla(monkeypatch)
    get_adapter("starvla").build_backend(_spec(compile="max-autotune")).load()
    assert calls["compile"] == "max-autotune"


@pytest.mark.parametrize(
    "spec_kwargs, obs_kwargs, match",
    [
        ({}, {"joint_pos": np.zeros(28)}, "29"),
        ({}, {"images": {}}, "cameras"),
        ({}, {"instruction": None}, "instruction"),
    ],
)
def test_bad_observations_are_rejected(monkeypatch, spec_kwargs, obs_kwargs, match):
    _install_fake_starvla(monkeypatch)
    backend = get_adapter("starvla").build_backend(_spec(**spec_kwargs))
    backend.load()
    with pytest.raises(ValueError, match=match):
        backend.predict(_obs(**obs_kwargs))


def test_load_checks_stats_and_dims(monkeypatch):
    _install_fake_starvla(monkeypatch, n_stats=2)
    with pytest.raises(ValueError, match="unnorm_key"):
        get_adapter("starvla").build_backend(_spec()).load()
    _install_fake_starvla(monkeypatch)
    spec = _spec()
    spec.action_dim = 7
    with pytest.raises(ValueError, match="action dim"):
        get_adapter("starvla").build_backend(spec).load()
    with pytest.raises(ValueError, match="state_encoding"):
        get_adapter("starvla").build_backend(_spec(state_encoding="quat"))


def test_no_state_groups_sends_no_state(monkeypatch):
    calls = _install_fake_starvla(monkeypatch)
    backend = get_adapter("starvla").build_backend(_spec(state_groups=[]))
    backend.load()
    backend.predict(_obs())
    assert "state" not in calls["predict"][0][0]
