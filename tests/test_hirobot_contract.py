"""hiVLA's wire format against hiRobot's real types (skipped when hiRobot is not installed).

hiVLA does not import hiRobot; it follows hiRobot's data conventions. These
tests catch drift: run them with ``PYTHONPATH=../hiRobot/src python -m pytest
tests/test_hirobot_contract.py`` (or with hiRobot installed in the env).
"""

from __future__ import annotations

import numpy as np
import pytest

hirobot_types = pytest.importorskip("hirobot.core.types")

from hivla.runtimes.base import ModelBackend  # noqa: E402
from hivla.core.policy import VlaPolicy  # noqa: E402
from hivla.protocol import (  # noqa: E402
    CHUNK_FIELDS,
    CONTROL_MODES,
    JOINT_MODES,
    decode_action_chunk,
    decode_observation,
    encode_action_chunk,
    encode_observation,
)

ActionChunk = hirobot_types.ActionChunk
ControlMode = hirobot_types.ControlMode
Observation = hirobot_types.Observation
T = 4


def test_control_modes_match_hirobot():
    # Every hiRobot mode but motion_clip (a MotionClip is never in a chunk).
    assert set(CONTROL_MODES) == {m.value for m in ControlMode} - {"motion_clip"}
    assert set(JOINT_MODES) == {m.value for m in hirobot_types.JOINT_MODES}


def test_chunk_fields_match_hirobot():
    assert set(CHUNK_FIELDS) == set(ActionChunk.model_fields)


class _Fixed(ModelBackend):
    def __init__(self, chunk):
        self._chunk = chunk

    def load(self) -> None:
        pass

    def predict(self, inputs):
        return self._chunk


def _through_wire(raw: dict, binary: bool) -> dict:
    """Server validation -> wire -> client decode, as a client would receive it."""
    chunk = VlaPolicy(_Fixed(raw), action_horizon=T).infer({})
    return decode_action_chunk(encode_action_chunk(chunk, binary))


@pytest.mark.parametrize("binary", [False, True])
def test_flat_chunk_is_a_valid_hirobot_chunk(binary):
    raw = {
        "mode": "base_velocity",
        "base_velocity": np.ones((T, 3)),
        "joint_targets": np.zeros((T, 2)),
        "joint_names": ["waist_yaw", "waist_pitch"],
        "gripper": {"left_tcp": np.linspace(0, 1, T)},
        "extra": {"source": "test"},
    }
    chunk = ActionChunk(**_through_wire(raw, binary))
    assert chunk.mode is ControlMode.BASE_VELOCITY and chunk.horizon == T
    step = chunk.step(T - 1)
    np.testing.assert_array_equal(step.base_velocity, [1, 1, 1])
    assert step.gripper["left_tcp"] == pytest.approx(1.0)
    assert step.joint_names == ["waist_yaw", "waist_pitch"]


@pytest.mark.parametrize("binary", [False, True])
def test_parts_chunk_is_a_valid_hirobot_chunk(binary):
    left = {"mode": "position", "joint_targets": np.zeros((T, 6)), "gripper": {"tcp": np.ones(T)}}
    base = {"mode": "base_velocity", "base_velocity": np.zeros((T, 3))}
    decoded = _through_wire({"parts": {"cell": {"parts": {"left": left}}, "base": base}}, binary)

    def build(d):  # hiRobot's ActionChunk takes parts as ActionChunk objects
        if d.get("parts"):
            d = dict(d, parts={k: build(v) for k, v in d["parts"].items()})
        return ActionChunk(**d)

    chunk = build(decoded)
    assert chunk.horizon == T
    step = chunk.step(1)
    assert step.parts["cell"].parts["left"].mode is ControlMode.POSITION
    assert step.parts["cell"].parts["left"].gripper["tcp"] == pytest.approx(1.0)
    assert step.parts["base"].mode is ControlMode.BASE_VELOCITY


def _observation_dict(obs) -> dict:
    """The set fields of a hiRobot Observation as a plain dict (what a client sends)."""
    out = {}
    for name in Observation.model_fields:
        if name == "stamps":
            continue  # sent as image_timestamps by the client
        val = getattr(obs, name)
        if val is None or (isinstance(val, dict) and not val):
            continue
        out[name] = val
    return out


@pytest.mark.parametrize("binary", [False, True])
def test_hirobot_observation_survives_the_wire(binary):
    obs = Observation(
        joint_pos=np.zeros(7), joint_vel=np.ones(7),
        base_orientation=np.array([0, 0, 0, 1.0]),  # [x, y, z, w]
        ee_poses={"left_tcp": np.array([0.3, 0, 0.5, 0, 0, 0, 1.0])},
        gripper={"left_tcp": 0.4},
        images={"head": np.zeros((4, 4, 3), np.uint8)},
        depth={"head": np.zeros((4, 4), np.float32)},
        instruction="pick the cup",
        extra={"arm_joint_pos": np.ones(7)},
    )
    sent = _observation_dict(obs)
    got = decode_observation(encode_observation(sent, binary))
    assert set(got) == set(sent)
    np.testing.assert_array_equal(got["ee_poses"]["left_tcp"], obs.ee_poses["left_tcp"])
    assert got["gripper"]["left_tcp"] == pytest.approx(0.4)
    assert got["depth"]["head"].dtype == np.float32
    np.testing.assert_array_equal(got["extra"]["arm_joint_pos"], np.ones(7))
    Observation(**got)  # a server can rebuild the robot's observation
