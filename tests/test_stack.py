"""Unit tests for the serving stack: serialization, transforms, policy, engine."""

from __future__ import annotations

import numpy as np
import pytest

from hivla import EchoBackend, VlaPolicy, build_engine
from hivla.protocol import (
    decode_array,
    decode_observation,
    encode_action_chunk,
    encode_array,
)
from hivla.core.transforms import Normalize, NormStats, RenameKeys, Unnormalize, compose


def test_array_roundtrip():
    arr = np.random.rand(4, 7).astype(np.float32)
    assert np.array_equal(decode_array(encode_array(arr)), arr)


def test_object_dtype_refused():
    with pytest.raises(ValueError):
        encode_array(np.array([{"x": 1}], dtype=object))
    with pytest.raises(ValueError):
        decode_array({"__ndarray__": "", "dtype": "object", "shape": [0]})


def test_decode_observation_mixed():
    payload = {
        "joint_pos": encode_array(np.zeros(7, np.float32)),
        "images": {"head": encode_array(np.zeros((2, 2, 3), np.uint8))},
        "instruction": "pick up the cube",
    }
    obs = decode_observation(payload)
    assert obs["joint_pos"].shape == (7,)
    assert obs["images"]["head"].shape == (2, 2, 3)
    assert obs["instruction"] == "pick up the cube"


def test_encode_action_chunk_only_present_keys():
    chunk = {"mode": "pd", "joint_targets": np.zeros((3, 7)), "kp": np.ones(7)}
    enc = encode_action_chunk(chunk)
    assert enc["mode"] == "pd"
    assert decode_array(enc["joint_targets"]).shape == (3, 7)
    assert "kd" not in enc and "gripper" not in enc


def test_normalize_unnormalize_roundtrip():
    stats = {"state": NormStats(mean=np.array([1.0, 2.0]), std=np.array([2.0, 4.0]))}
    norm, unnorm = Normalize(stats), Unnormalize(stats)
    x = {"state": np.array([3.0, 10.0], dtype=np.float32)}
    back = unnorm(norm(dict(x)))
    assert np.allclose(back["state"], x["state"], atol=1e-4)


def test_rename_keys():
    out = RenameKeys({"joint_pos": "state"})({"joint_pos": np.zeros(3)})
    assert "state" in out and "joint_pos" not in out


def test_compose_order():
    f = compose([lambda d: {**d, "v": d["v"] + 1}, lambda d: {**d, "v": d["v"] * 2}])
    assert f({"v": 1})["v"] == 4  # (1+1)*2


def test_policy_validates_horizon():
    policy = VlaPolicy(EchoBackend(action_dim=7, action_horizon=5), action_horizon=5)
    chunk = policy.infer({"joint_pos": np.zeros(7)})
    assert chunk["joint_targets"].shape == (5, 7)
    assert chunk["mode"] == "position"


def test_policy_rejects_horizon_mismatch():
    # Backend emits horizon 4 but policy declares 5 -> contract violation.
    policy = VlaPolicy(EchoBackend(action_dim=7, action_horizon=4), action_horizon=5)
    with pytest.raises(ValueError, match="horizon"):
        policy.infer({"joint_pos": np.zeros(7)})


class _EeBackend(EchoBackend):
    """Emits a cartesian chunk: ee_targets (T, 6) + gripper (T,)."""

    def __init__(self, horizon, gripper_shape=None):
        super().__init__(action_dim=7, action_horizon=horizon)
        self._gripper_shape = gripper_shape or (horizon,)

    def predict(self, inputs):
        t = self._action_horizon
        return {"ee_targets": {"eef": np.zeros((t, 6))},
                "gripper": {"gripper": np.zeros(self._gripper_shape)}}


def test_policy_validates_ee_and_gripper_chunks():
    policy = VlaPolicy(_EeBackend(5), action_horizon=5, default_mode="cartesian_delta")
    chunk = policy.infer({})
    assert chunk["mode"] == "cartesian_delta"
    assert chunk["ee_targets"]["eef"].shape == (5, 6)
    with pytest.raises(ValueError, match=r"ee_targets\['eef'\] horizon 4"):
        VlaPolicy(_EeBackend(4), action_horizon=5).infer({})
    with pytest.raises(ValueError, match=r"gripper\['gripper'\] must be 1-D"):
        VlaPolicy(_EeBackend(5, gripper_shape=(5, 1)), action_horizon=5).infer({})


def test_engine_requires_start():
    engine = build_engine({"model": {"backend": "echo", "action_horizon": 5},
                           "policy": {"action_dim": 7}})
    with pytest.raises(RuntimeError, match="start"):
        engine.infer({"joint_pos": np.zeros(7)})
    engine.start()
    chunk = engine.infer({"joint_pos": np.zeros(7)})
    assert chunk["joint_targets"].shape == (5, 7)


def test_build_engine_metadata():
    engine = build_engine({
        "model": {"name": "m", "backend": "echo", "action_horizon": 8},
        "policy": {"action_dim": 7, "cameras": ["head"], "control_mode": "pd"},
    })
    assert engine.metadata["action_horizon"] == 8
    assert engine.metadata["cameras"] == ["head"]
