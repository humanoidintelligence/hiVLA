"""VlaPolicy refuses chunks that break hiRobot's ActionChunk rules; ModelSpec checks control_mode."""

from __future__ import annotations

import numpy as np
import pytest

from hivla import VlaPolicy
from hivla.runtimes.base import ModelBackend
from hivla.protocol import CONTROL_MODES
from hivla.core.spec import ModelSpec

T = 5


class _Fixed(ModelBackend):
    """Returns a fixed raw chunk."""

    def __init__(self, chunk):
        self._chunk = chunk

    def load(self) -> None:
        pass

    def predict(self, inputs):
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in self._chunk.items()}


def _infer(chunk, horizon=T, mode="position"):
    return VlaPolicy(_Fixed(chunk), action_horizon=horizon, default_mode=mode).infer({})


def test_new_fields_pass_and_become_arrays():
    out = _infer({"mode": "base_velocity", "base_velocity": [[0.1, 0, 0]] * T}, mode="pd")
    assert out["mode"] == "base_velocity"
    assert isinstance(out["base_velocity"], np.ndarray) and out["base_velocity"].shape == (T, 3)
    out = _infer({"mode": "motion_latent", "latent": np.zeros((T, 32))})
    assert out["latent"].shape == (T, 32)


@pytest.mark.parametrize("chunk, match", [
    ({"mode": "walk", "joint_targets": np.zeros((T, 3))}, "mode must be one of"),
    ({"mode": "base_velocity", "base_velocity": np.zeros((T, 2))}, r"base_velocity must be 2-D \(horizon, 3\)"),
    ({"mode": "base_pose", "base_target": np.zeros(3)}, "base_target must be"),
    ({"mode": "motion_latent", "latent": np.zeros((4, 8))}, "latent horizon 4"),
    ({"joint_targets": np.zeros((T, 3)), "kp": np.ones((T, 3))}, "kp is one gain per joint"),
    ({"mode": "cartesian", "ee_targets": {"tcp": np.zeros((T, 7))},
      "gripper": {"tcp": np.zeros(T)}, "base_velocity": np.zeros((T, 3))}, None),
])
def test_field_rules(chunk, match):
    if match is None:
        _infer(chunk)
        return
    with pytest.raises(ValueError, match=match):
        _infer(chunk)


def test_mixed_horizons_refused_without_declared_horizon():
    chunk = {"mode": "cartesian", "ee_targets": {"tcp": np.zeros((4, 7))}, "gripper": {"tcp": np.zeros(5)}}
    with pytest.raises(ValueError, match="different horizons"):
        _infer(chunk, horizon=None)


def test_joint_names_rules():
    jt = np.zeros((T, 2))
    _infer({"mode": "base_velocity", "base_velocity": np.zeros((T, 3)), "joint_targets": jt,
            "joint_names": ["waist_yaw", "waist_pitch"]})
    with pytest.raises(ValueError, match="non-joint modes"):
        _infer({"mode": "position", "joint_targets": jt, "joint_names": ["a", "b"]})
    with pytest.raises(ValueError, match="needs joint_targets"):
        _infer({"mode": "base_velocity", "base_velocity": np.zeros((T, 3)), "joint_names": ["a"]})
    with pytest.raises(ValueError, match="once"):
        _infer({"mode": "base_velocity", "joint_targets": jt, "joint_names": ["a", "a"]})
    with pytest.raises(ValueError, match="once"):
        _infer({"mode": "base_velocity", "joint_targets": jt, "joint_names": ["a"]})


def test_parts_are_checked_recursively():
    left = {"mode": "position", "joint_targets": np.zeros((T, 6))}
    base = {"mode": "base_velocity", "base_velocity": np.zeros((T, 3))}
    out = _infer({"parts": {"cell": {"parts": {"left": left}}, "base": base}, "extra": {"k": 1}})
    assert "mode" not in out  # a node with parts has no mode of its own
    with pytest.raises(ValueError, match=r"parts\['cell'\].parts\['left'\] mode"):
        _infer({"parts": {"cell": {"parts": {"left": dict(left, mode="fly")}}}})


def test_parts_exclude_other_fields_and_need_one_horizon():
    left = {"mode": "position", "joint_targets": np.zeros((T, 6))}
    with pytest.raises(ValueError, match="no other command fields"):
        _infer({"parts": {"left": left}, "joint_targets": np.zeros((T, 6))})
    short = {"mode": "position", "joint_targets": np.zeros((3, 6))}
    with pytest.raises(ValueError, match="different horizons"):
        _infer({"parts": {"left": left, "right": short}}, horizon=None)


def test_spec_control_mode_must_be_a_hirobot_mode():
    for mode in CONTROL_MODES:
        ModelSpec.from_config({"policy": {"control_mode": mode}})
    with pytest.raises(ValueError, match="control_mode"):
        ModelSpec.from_config({"policy": {"control_mode": "joint_position"}})
    with pytest.raises(ValueError, match="control_mode"):
        ModelSpec.from_config({"policy": {"control_mode": "motion_clip"}})
