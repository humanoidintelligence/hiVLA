"""Family adapters: registry, generic build, native delegation."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from hivla import ModelSpec, build_engine_from_spec, get_adapter
from hivla.families.native import NativePolicyBackend
from hivla.families import FamilyAdapter, available_families


def test_available_families_includes_all():
    names = available_families()
    for expected in ("generic", "openpi", "gr00t", "lerobot", "starvla", "rlinf"):
        assert expected in names


def test_generic_family_builds_echo_engine():
    spec = ModelSpec(family="generic", backend="echo", action_horizon=5, action_dim=7)
    engine = build_engine_from_spec(spec)
    engine.start()
    chunk = engine.infer({"joint_pos": np.zeros(7, dtype=np.float32)})
    assert chunk["joint_targets"].shape == (5, 7)


def test_generic_family_applies_rename_and_norm():
    spec = ModelSpec(
        family="generic", backend="echo", action_horizon=3, action_dim=2,
        rename_inputs={"joint_pos": "state"},
        norm_stats={"state": {"mean": [1.0, 2.0], "std": [2.0, 4.0]}},
    )
    engine = build_engine_from_spec(spec)
    engine.start()
    # Echo ignores inputs, so just assert it runs through the transformed pipeline.
    engine.infer({"joint_pos": np.array([3.0, 10.0], dtype=np.float32)})


def test_openpi_family_is_a_family_adapter(monkeypatch):
    adapter = get_adapter("openpi")
    assert isinstance(adapter, FamilyAdapter)
    spec = ModelSpec(family="openpi", checkpoint="x",
                     native_kwargs={"config": "pi05_libero", "cameras": {"cam": "image"}})
    backend = adapter.build_backend(spec)
    # Package missing -> a clear error on load. Blocked explicitly: the rlinf env ships
    # RLinf's openpi fork, and earlier tests may have cached openpi submodules.
    for name in [m for m in sys.modules if m == "openpi" or m.startswith("openpi.")]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "openpi", None)
    with pytest.raises(ImportError, match="openpi"):
        backend.load()


def test_native_policy_backend_wraps_factory():
    mod = types.ModuleType("hivla_tests_fake_family")
    mod.make_policy = lambda checkpoint=None: (lambda obs: {"joint_targets": np.zeros((2, 3), np.float32)})
    sys.modules["hivla_tests_fake_family"] = mod
    backend = NativePolicyBackend("hivla_tests_fake_family", "make_policy", kwargs={"checkpoint": "x"})
    backend.load()
    out = backend.predict({"joint_pos": np.zeros(3)})
    assert out["joint_targets"].shape == (2, 3)


def test_lerobot_family_backend_requires_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "lerobot", None)  # force ImportError even if installed
    spec = ModelSpec(family="lerobot", checkpoint="x")
    backend = get_adapter("lerobot").build_backend(spec)
    with pytest.raises(ImportError, match="lerobot"):
        backend.load()


def _install_fake_lerobot(monkeypatch, n_obs_steps=1, chunk=10, dof=14):
    """Minimal stand-in for the lerobot v0.6.2 API surface LerobotBackend uses."""
    torch = pytest.importorskip("torch")
    calls = {}

    class Cfg:
        type = "act"

    Cfg.n_obs_steps = n_obs_steps

    class PreTrainedConfig:
        @staticmethod
        def from_pretrained(path, revision=None):
            calls["cfg_path"] = path
            return Cfg()

    class Policy:
        def eval(self):
            return self

        def predict_action_chunk(self, batch):
            calls["batch_keys"] = sorted(batch)
            return torch.ones(1, chunk, dof)

    class PolicyCls:
        @staticmethod
        def from_pretrained(path, config=None, revision=None):
            calls["device"] = config.device
            return Policy()

    def prepare(obs, device, task, robot_type):
        out = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
        out["task"] = task or ""
        return out

    def load_processors(path, revision=None, preprocessor_overrides=None):
        calls["overrides"] = preprocessor_overrides
        return (lambda b: b), (lambda a: a * 2)  # postprocessor "unnormalizes"

    configs = types.ModuleType("lerobot.configs")
    configs.PreTrainedConfig = PreTrainedConfig
    policies = types.ModuleType("lerobot.policies")
    policies.get_policy_class = lambda name: PolicyCls
    policies.prepare_observation_for_inference = prepare
    processor = types.ModuleType("lerobot.processor")
    processor.load_pretrained_policy_processors = load_processors
    for name, mod in [("lerobot", types.ModuleType("lerobot")), ("lerobot.configs", configs),
                      ("lerobot.policies", policies), ("lerobot.processor", processor)]:
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


def test_lerobot_backend_runs_processor_chain(monkeypatch):
    calls = _install_fake_lerobot(monkeypatch)
    spec = ModelSpec(family="lerobot", checkpoint="ckpt", device="cpu",
                     action_horizon=4, action_dim=14,
                     rename_inputs={"state": "observation.state"})
    engine = build_engine_from_spec(spec)
    engine.start()
    chunk = engine.infer({
        "state": np.zeros(14, dtype=np.float64),
        "observation.images.top": np.zeros((8, 8, 3), dtype=np.uint8),
        "task": "pick the cube",
    })
    assert chunk["joint_targets"].shape == (4, 14)
    np.testing.assert_allclose(chunk["joint_targets"], 2.0)  # postprocessor applied
    assert calls["cfg_path"] == "ckpt" and calls["device"] == "cpu"
    assert calls["overrides"] == {"device_processor": {"device": "cpu"}}
    assert calls["batch_keys"] == ["observation.images.top", "observation.state", "task"]


def test_lerobot_backend_rejects_short_chunk(monkeypatch):
    _install_fake_lerobot(monkeypatch, chunk=3)
    spec = ModelSpec(family="lerobot", checkpoint="ckpt", device="cpu", action_horizon=4)
    backend = get_adapter("lerobot").build_backend(spec)
    backend.load()
    with pytest.raises(ValueError, match="action_horizon"):
        backend.predict({"observation.state": np.zeros(14, np.float32)})


def test_lerobot_backend_rejects_obs_history_policies(monkeypatch):
    _install_fake_lerobot(monkeypatch, n_obs_steps=2)
    backend = get_adapter("lerobot").build_backend(ModelSpec(family="lerobot", checkpoint="c"))
    with pytest.raises(NotImplementedError, match="n_obs_steps"):
        backend.load()


def test_native_backend_non_dict_result_wrapped():
    mod = types.ModuleType("hivla_tests_fake_family2")
    mod.make_policy = lambda: (lambda obs: np.zeros((2, 3), np.float32))
    sys.modules["hivla_tests_fake_family2"] = mod
    backend = NativePolicyBackend("hivla_tests_fake_family2", "make_policy", call_method="__call__")
    backend.load()
    out = backend.predict({})
    assert out["joint_targets"].shape == (2, 3)
