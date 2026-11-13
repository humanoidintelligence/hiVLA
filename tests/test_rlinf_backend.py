"""rlinf family: RlinfBackend against a fake ``rlinf`` package (LIBERO layout)."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from hivla import ModelSpec, build_engine_from_spec, get_adapter
from hivla.families import RlinfFamily

LIBERO_LAYOUT = [["ee_targets", "eef", 6], ["gripper", "gripper", 1]]
CAMERAS = {"agentview": "main_images", "wrist": "wrist_images"}


def _install_fake_rlinf(monkeypatch, dim=7):
    calls = {"predict": []}

    class Model:
        def to(self, device):
            calls["device"] = device
            return self

        def eval(self):
            return self

        def predict_action_batch(self, env_obs, mode="eval", rng=None):
            calls["predict"].append((env_obs, mode, rng))
            batch = len(env_obs["task_descriptions"])
            chunk = calls["cfg"]["num_action_chunks"]
            # Sample b, row t is all b + t/10, so batching and slicing are easy to check.
            rows = np.arange(chunk)[None, :, None] / 10.0 + np.arange(batch)[:, None, None]
            return np.tile(rows, (1, 1, dim)), {"model_actions": None}

    def get_model(cfg, torch_dtype=None):
        calls["cfg"] = cfg
        return Model()

    class Generator:
        def __init__(self, device="cpu"):
            self.device = device

        def manual_seed(self, seed):
            self.seed = seed
            return self

    torch = types.ModuleType("torch")
    torch.Generator = Generator
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    omegaconf = types.ModuleType("omegaconf")
    omegaconf.OmegaConf = types.SimpleNamespace(create=lambda d: d)
    openpi = types.ModuleType("rlinf.models.embodiment.openpi")
    openpi.get_model = get_model
    for name, mod in [("torch", torch), ("omegaconf", omegaconf),
                      ("rlinf", types.ModuleType("rlinf")),
                      ("rlinf.models", types.ModuleType("rlinf.models")),
                      ("rlinf.models.embodiment", types.ModuleType("rlinf.models.embodiment")),
                      ("rlinf.models.embodiment.openpi", openpi)]:
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


def _spec(horizon=5, **native):
    native = {"config": "pi05_libero", "cameras": CAMERAS, "action_layout": LIBERO_LAYOUT,
              **native}
    return ModelSpec(family="rlinf", checkpoint="cache/rlinf/RLinf-Pi05-LIBERO-SFT",
                     action_horizon=horizon, action_dim=7, control_mode="cartesian_delta",
                     cameras=list(CAMERAS), native_kwargs=native)


def _obs(i=0):
    return {
        "images": {"agentview": np.full((256, 256, 3), i, np.uint8),
                   "wrist": np.full((256, 256, 3), 100 + i, np.uint8)},
        "joint_pos": np.arange(8, dtype=np.float64) + i,
        "instruction": f"task {i}",
    }


def _loaded(monkeypatch, **native):
    calls = _install_fake_rlinf(monkeypatch)
    backend = get_adapter("rlinf").build_backend(_spec(**native))
    backend.load()
    return backend, calls


def test_rlinf_is_a_dedicated_family():
    assert isinstance(get_adapter("rlinf"), RlinfFamily)


def test_rlinf_backend_requires_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "rlinf", None)
    with pytest.raises(ImportError, match="rlinf"):
        get_adapter("rlinf").build_backend(_spec()).load()


def test_rlinf_backend_requires_config():
    with pytest.raises(ValueError, match="config"):
        get_adapter("rlinf").build_backend(_spec(config=None))


def test_get_model_cfg(monkeypatch):
    _, calls = _loaded(monkeypatch, num_steps=3, seed=0, openpi={"torch_compile": False})
    cfg = calls["cfg"]
    assert cfg["model_path"] == "cache/rlinf/RLinf-Pi05-LIBERO-SFT"
    assert cfg["pi05"] is True and cfg["precision"] is None
    assert cfg["num_action_chunks"] == 5 and cfg["action_dim"] == 7 and cfg["num_steps"] == 3
    openpi = cfg["openpi"]
    assert openpi["task"] == "eval" and openpi["config_name"] == "pi05_libero"
    assert openpi["action_chunk"] == 5 and openpi["action_env_dim"] == 7
    assert openpi["num_steps"] == 3 and openpi["model_action_dim"] == 32
    assert openpi["torch_compile"] is False  # native_kwargs.openpi overrides the defaults
    assert calls["device"] == "cpu"  # cuda requested, none available


def test_engine_maps_libero_keys_and_splits_actions(monkeypatch):
    calls = _install_fake_rlinf(monkeypatch)
    engine = build_engine_from_spec(_spec(seed=7))
    engine.start()
    chunk = engine.infer(_obs())
    env_obs, mode, rng = calls["predict"][0]
    assert mode == "eval" and rng.seed == 7
    assert env_obs["main_images"].shape == (1, 256, 256, 3)
    assert env_obs["main_images"].dtype == np.uint8 and env_obs["wrist_images"][0, 0, 0, 0] == 100
    assert env_obs["states"].dtype == np.float32
    np.testing.assert_array_equal(env_obs["states"][0], np.arange(8))
    assert env_obs["task_descriptions"] == ["task 0"]
    assert chunk["ee_targets"]["eef"].shape == (5, 6)
    assert chunk["gripper"]["gripper"].shape == (5,)
    np.testing.assert_allclose(chunk["gripper"]["gripper"], np.arange(5) / 10.0)
    assert chunk["mode"] == "cartesian_delta"


def test_predict_batch_is_one_forward_pass(monkeypatch):
    backend, calls = _loaded(monkeypatch)
    out = backend.predict_batch([_obs(0), _obs(1), _obs(2)])
    assert len(calls["predict"]) == 1
    env_obs = calls["predict"][0][0]
    assert env_obs["main_images"].shape == (3, 256, 256, 3)
    assert env_obs["task_descriptions"] == ["task 0", "task 1", "task 2"]
    for b, chunk in enumerate(out):
        np.testing.assert_allclose(chunk["gripper"]["gripper"], b + np.arange(5) / 10.0)
    assert calls["predict"][0][2] is None  # no seed: RLinf's default torch.randn


def test_openpi_wire_obs_passes_through(monkeypatch):
    backend, calls = _loaded(monkeypatch)
    wire = {"main_images": np.zeros((4, 4, 3), np.uint8),
            "wrist_images": np.zeros((4, 4, 3), np.uint8),
            "states": np.zeros(8, np.float32), "task_descriptions": "pick"}
    backend.predict_batch([wire, wire])
    env_obs = calls["predict"][0][0]
    assert env_obs["main_images"].shape == (2, 4, 4, 3)
    assert env_obs["states"].shape == (2, 8)
    assert env_obs["task_descriptions"] == ["pick", "pick"]


@pytest.mark.parametrize("obs_kwargs, match", [
    ({"images": {"agentview": np.zeros((4, 4, 3))}}, "wrist"),
    ({"joint_pos": None}, "joint_pos"),
])
def test_bad_observations_are_rejected(monkeypatch, obs_kwargs, match):
    backend, _ = _loaded(monkeypatch)
    obs = {**_obs(), **obs_kwargs}
    obs = {k: v for k, v in obs.items() if v is not None}
    with pytest.raises(ValueError, match=match):
        backend.predict(obs)
