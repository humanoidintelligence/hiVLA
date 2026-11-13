"""The action-chunk wire format carries every hiRobot ActionChunk field, parts included."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hivla.protocol import (
    CHUNK_FIELDS,
    PROTOCOL_VERSION,
    decode_action_chunk,
    decode_observation,
    encode_action_chunk,
    encode_observation,
)

T = 4


def _full_chunk() -> dict:
    return {
        "mode": "cartesian",
        "joint_targets": np.arange(T * 3, dtype=np.float32).reshape(T, 3),
        "kp": np.ones(3),
        "kd": np.full(3, 0.5),
        "joint_names": ["waist_yaw", "waist_pitch", "waist_roll"],
        "ee_targets": {"left_tcp": np.zeros((T, 7))},
        "gripper": {"left_tcp": np.linspace(0, 1, T)},
        "base_velocity": np.ones((T, 3)),
        "base_target": np.zeros((T, 3)),
        "latent": np.zeros((T, 16), dtype=np.float32),
        "extra": {"value": np.array([0.3]), "note": "x", "nested": {"a": np.zeros(2)}},
    }


def _assert_same(a: dict, b: dict) -> None:
    assert set(a) == set(b)
    for key, val in a.items():
        if isinstance(val, np.ndarray):
            np.testing.assert_array_equal(val, b[key])
        elif isinstance(val, dict):
            _assert_same(val, b[key])
        else:
            assert val == b[key]


@pytest.mark.parametrize("binary", [False, True])
def test_every_field_roundtrips(binary):
    chunk = _full_chunk()
    assert set(chunk) == set(CHUNK_FIELDS) - {"parts"}
    _assert_same(chunk, decode_action_chunk(encode_action_chunk(chunk, binary)))


@pytest.mark.parametrize("binary", [False, True])
def test_nested_parts_roundtrip(binary):
    left = {"mode": "position", "joint_targets": np.zeros((T, 6)), "gripper": {"tcp": np.ones(T)}}
    base = {"mode": "base_velocity", "base_velocity": np.ones((T, 3))}
    chunk = {"parts": {"cell": {"parts": {"left": left}}, "base": base}, "extra": {"k": 1}}
    back = decode_action_chunk(encode_action_chunk(chunk, binary))
    _assert_same(chunk, back)
    assert back["parts"]["cell"]["parts"]["left"]["gripper"]["tcp"].shape == (T,)


def test_json_wire_is_json_serializable():
    chunk = dict(_full_chunk(), parts={"a": {"joint_targets": np.zeros((T, 2))}})
    enc = encode_action_chunk(chunk)
    json.dumps(enc)  # no ndarray or numpy scalar left behind


def test_only_set_fields_are_sent():
    enc = encode_action_chunk({"mode": "pd", "joint_targets": np.zeros((T, 7)), "extra": {},
                               "parts": {}, "joint_names": None, "latent": None})
    assert set(enc) == {"mode", "joint_targets"}


def test_mode_enum_is_sent_as_its_value():
    class Mode:
        value = "base_pose"

    assert encode_action_chunk({"mode": Mode()})["mode"] == "base_pose"


def test_unknown_keys_are_not_sent():
    enc = encode_action_chunk({"joint_targets": np.zeros((T, 2)), "debug": np.zeros(3)})
    assert "debug" not in enc


def test_observation_extra_and_gripper_roundtrip():
    obs = {
        "joint_pos": np.zeros(7, np.float32),
        "ee_poses": {"tcp": np.array([0, 0, 0, 0, 0, 0, 1.0])},
        "gripper": {"tcp": 0.25},
        "depth": {"head": np.zeros((2, 2), np.float32)},
        "extra": {"arm_joint_pos": np.ones(7), "deep": {"x": np.zeros(1)}},
        "instruction": "open the drawer",
    }
    back = decode_observation(encode_observation(obs))
    _assert_same(obs, back)


def test_protocol_version_is_in_metadata():
    from hivla.core.spec import ModelSpec

    assert ModelSpec.from_config({}).metadata()["protocol_version"] == PROTOCOL_VERSION
