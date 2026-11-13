"""openpi family: OpenpiBackend against a fake ``openpi`` package (LIBERO layout)."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from hivla import ModelSpec, build_engine_from_spec, get_adapter
from hivla.families.openpi import parse_action_layout, split_actions
from hivla.families import OpenpiFamily

LIBERO_LAYOUT = [["ee_targets", "eef", 6], ["gripper", "gripper", 1]]
CAMERAS = {"agentview": "observation/image", "wrist": "observation/wrist_image"}


def _install_fake_openpi(monkeypatch, horizon=10, dim=7):
    calls = {"infer": []}

    class Policy:
        _rng = "default"

        def infer(self, obs):
            calls["infer"].append(obs)
            # Row t is all t/10, so slicing is easy to check.
            return {"actions": np.tile(np.arange(horizon)[:, None] / 10.0, (1, dim)),
                    "policy_timing": {"infer_ms": 1.0}}

    def create_trained_policy(train_config, checkpoint_dir, *, default_prompt=None,
                              sample_kwargs=None):
        calls.update(config=train_config.name, checkpoint=checkpoint_dir,
                     default_prompt=default_prompt, sample_kwargs=sample_kwargs)
        calls["policy"] = Policy()
        return calls["policy"]

    def get_config(name):
        return types.SimpleNamespace(name=name,
                                     model=types.SimpleNamespace(action_horizon=horizon))

    policy_config = types.ModuleType("openpi.policies.policy_config")
    policy_config.create_trained_policy = create_trained_policy
    config = types.ModuleType("openpi.training.config")
    config.get_config = get_config

    image_tools = types.ModuleType("openpi_client.image_tools")

    def resize_with_pad(img, h, w):
        calls.setdefault("resize", []).append((img.shape, h, w))
        return np.full((h, w, 3), 0.5, np.float32)  # float, so convert_to_uint8 matters

    image_tools.resize_with_pad = resize_with_pad
    image_tools.convert_to_uint8 = lambda img: (255 * img).astype(np.uint8)
    client = types.ModuleType("openpi_client")
    client.image_tools = image_tools

    jax = types.ModuleType("jax")
    jax.random = types.SimpleNamespace(key=lambda seed: ("key", seed))

    for name, mod in [("openpi", types.ModuleType("openpi")),
                      ("openpi.policies", types.ModuleType("openpi.policies")),
                      ("openpi.policies.policy_config", policy_config),
                      ("openpi.training", types.ModuleType("openpi.training")),
                      ("openpi.training.config", config),
                      ("openpi_client", client), ("openpi_client.image_tools", image_tools),
                      ("jax", jax)]:
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


def _spec(horizon=10, **native):
    native = {"config": "pi05_libero", "cameras": CAMERAS, "action_layout": LIBERO_LAYOUT,
              **native}
    return ModelSpec(family="openpi", checkpoint="cache/openpi/pi05_libero",
                     action_horizon=horizon, action_dim=7, control_mode="cartesian_delta",
                     cameras=list(CAMERAS), native_kwargs=native)


def _obs(**extra):
    obs = {
        "joint_pos": np.arange(8, dtype=np.float64),
        "images": {"agentview": np.zeros((256, 256, 3), np.uint8),
                   "wrist": np.zeros((256, 256, 3), np.uint8)},
        "instruction": "put the bowl on the plate",
    }
    obs.update(extra)
    return obs


def test_openpi_is_a_dedicated_family():
    assert isinstance(get_adapter("openpi"), OpenpiFamily)


def test_openpi_backend_requires_package(monkeypatch):
    # Also block cached submodules (the rlinf env really has openpi installed).
    for name in [m for m in sys.modules if m == "openpi" or m.startswith("openpi.")]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "openpi", None)
    with pytest.raises(ImportError, match="openpi"):
        get_adapter("openpi").build_backend(_spec()).load()


@pytest.mark.parametrize("native, match", [({"config": None}, "config"),
                                           ({"cameras": {}}, "cameras")])
def test_openpi_backend_requires_config_and_cameras(native, match):
    with pytest.raises(ValueError, match=match):
        get_adapter("openpi").build_backend(_spec(**native))


def test_layout_helpers():
    layout = parse_action_layout(LIBERO_LAYOUT)
    assert layout == [("ee_targets", "eef", 6), ("gripper", "gripper", 1)]
    actions = np.arange(21, dtype=np.float64).reshape(3, 7)
    chunk = split_actions(actions, layout)
    np.testing.assert_array_equal(chunk["ee_targets"]["eef"], actions[:, :6])
    np.testing.assert_array_equal(chunk["gripper"]["gripper"], actions[:, 6])
    assert chunk["gripper"]["gripper"].shape == (3,)
    assert "joint_targets" not in chunk
    # No layout: everything is joint_targets (ALOHA-style configs).
    assert split_actions(actions, [])["joint_targets"].shape == (3, 7)
    with pytest.raises(ValueError, match="covers 6"):
        split_actions(actions, [("ee_targets", "eef", 6)])
    with pytest.raises(ValueError, match="field"):
        parse_action_layout([["bogus", "x", 1]])


def test_engine_maps_libero_keys_and_splits_actions(monkeypatch):
    calls = _install_fake_openpi(monkeypatch)
    engine = build_engine_from_spec(_spec(task="default task", seed=7,
                                          sample_kwargs={"num_steps": 5}))
    engine.start()
    chunk = engine.infer(_obs())

    assert calls["config"] == "pi05_libero" and calls["checkpoint"] == "cache/openpi/pi05_libero"
    assert calls["default_prompt"] == "default task"
    assert calls["sample_kwargs"] == {"num_steps": 5}
    assert calls["policy"]._rng == ("key", 7)
    (obs,) = calls["infer"]
    assert set(obs) == {"observation/image", "observation/wrist_image", "observation/state",
                        "prompt"}
    assert obs["observation/image"].shape == (224, 224, 3)
    assert obs["observation/image"].dtype == np.uint8 and obs["observation/image"][0, 0, 0] == 127
    assert calls["resize"][0] == ((256, 256, 3), 224, 224)
    np.testing.assert_array_equal(obs["observation/state"], np.arange(8, dtype=np.float32))
    assert obs["prompt"] == "put the bowl on the plate"

    assert chunk["mode"] == "cartesian_delta"
    assert chunk["ee_targets"]["eef"].shape == (10, 6)
    assert chunk["gripper"]["gripper"].shape == (10,)
    np.testing.assert_allclose(chunk["ee_targets"]["eef"][3], 0.3, rtol=1e-6)


def test_horizon_is_truncated_and_seed_is_optional(monkeypatch):
    calls = _install_fake_openpi(monkeypatch, horizon=10)
    backend = get_adapter("openpi").build_backend(_spec(horizon=5))
    backend.load()
    out = backend.predict(_obs(instruction=None))
    assert calls["policy"]._rng == "default"
    assert "prompt" not in calls["infer"][0]  # openpi injects its default_prompt
    assert out["ee_targets"]["eef"].shape == (5, 6)


def test_horizon_longer_than_checkpoint_is_rejected(monkeypatch):
    _install_fake_openpi(monkeypatch, horizon=10)
    with pytest.raises(ValueError, match="checkpoint horizon"):
        get_adapter("openpi").build_backend(_spec(horizon=50)).load()


@pytest.mark.parametrize("obs_kwargs, match", [
    ({"images": {"agentview": np.zeros((4, 4, 3), np.uint8)}}, "wrist"),
    ({"joint_pos": None}, "joint_pos"),
])
def test_bad_observations_are_rejected(monkeypatch, obs_kwargs, match):
    _install_fake_openpi(monkeypatch)
    backend = get_adapter("openpi").build_backend(_spec())
    backend.load()
    obs = {k: v for k, v in _obs(**obs_kwargs).items() if v is not None}
    with pytest.raises(ValueError, match=match):
        backend.predict(obs)
