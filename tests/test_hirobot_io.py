"""Family-independent edges for hiRobot's conventions: state_layout, gripper_range, /metadata."""

from __future__ import annotations

import numpy as np
import pytest

from hivla import build_engine
from hivla.runtimes.base import ModelBackend
from hivla.core.policy import VlaPolicy
from hivla.core.spec import ModelSpec
from hivla.core.transforms import (
    ConcatState,
    GripperToOpening,
    parse_gripper_range,
    parse_state_layout,
)

T = 4


def test_concat_state_order_and_flatten():
    obs = {
        "joint_pos": np.array([1.0, 2.0]),
        "ee_poses": {"eef": np.arange(7.0)},
        "gripper": {"eef": 0.5},
        "extra": {"qpos": np.array([[9.0, 8.0]])},
    }
    layout = parse_state_layout([["ee_poses", "eef"], "gripper.eef", ["extra", "qpos"], ["joint_pos"]])
    out = ConcatState(tuple(layout))(obs)
    np.testing.assert_array_equal(out["joint_pos"], [0, 1, 2, 3, 4, 5, 6, 0.5, 9, 8, 1, 2])
    assert out["joint_pos"].dtype == np.float32
    assert out["ee_poses"] is obs["ee_poses"]  # the other fields stay


def test_concat_state_missing_field_names_it():
    t = ConcatState(tuple(parse_state_layout([["ee_poses", "left_tcp"]])))
    with pytest.raises(ValueError, match=r"ee_poses\['left_tcp'\]"):
        t({"ee_poses": {"right_tcp": np.zeros(7)}})
    with pytest.raises(ValueError, match=r"ee_poses\['left_tcp'\]"):
        t({"joint_pos": np.zeros(3)})


@pytest.mark.parametrize("layout, match", [
    ([["images", "head"]], "field must be one of"),
    ([["ee_poses"]], "needs a name"),
    ([["joint_pos", "x"]], "takes no name"),
    ([["a", "b", "c"]], r"\[field\] or \[field, name\]"),
])
def test_state_layout_rejected(layout, match):
    with pytest.raises(ValueError, match=match):
        parse_state_layout(layout)
    with pytest.raises(ValueError, match=match):
        ModelSpec.from_config({"policy": {"state_layout": layout}})


def test_gripper_to_opening_libero_range():
    # openpi LIBERO: -1 open, +1 closed  ->  hiRobot: 0 closed, 1 open.
    t = GripperToOpening(parse_gripper_range({"gripper": [1, -1]}))
    out = t({"gripper": {"gripper": np.array([-1.0, 1.0, 0.0, -3.0]), "other": np.array([5.0])}})
    np.testing.assert_allclose(out["gripper"]["gripper"], [1.0, 0.0, 0.5, 1.0])
    np.testing.assert_array_equal(out["gripper"]["other"], [5.0])  # not in ranges: unchanged


def test_gripper_to_opening_recurses_into_parts():
    t = GripperToOpening({"tcp": (0.0, 0.08)})
    out = t({"parts": {"left": {"gripper": {"tcp": np.array([0.04])}}}})
    np.testing.assert_allclose(out["parts"]["left"]["gripper"]["tcp"], [0.5])


def test_gripper_range_rejects_equal_ends():
    with pytest.raises(ValueError, match="both"):
        ModelSpec.from_config({"policy": {"gripper_range": {"g": [1, 1]}}})


class _Libero(ModelBackend):
    """Echoes the state width it got and a LIBERO-style gripper command."""

    def load(self) -> None:
        pass

    def predict(self, inputs):
        width = np.asarray(inputs["joint_pos"]).shape[0]
        return {"ee_targets": {"eef": np.full((T, 6), width, np.float32)},
                "gripper": {"gripper": np.full(T, -1.0)}}


def test_build_engine_wires_both_edges(monkeypatch):
    import hivla.families as families

    monkeypatch.setattr(families.GenericFamily, "build_backend", lambda self, spec: _Libero())
    engine = build_engine({
        "model": {"action_horizon": T},
        "policy": {"control_mode": "cartesian_delta",
                   "state_layout": [["ee_poses", "eef"], ["extra", "gripper_qpos"]],
                   "gripper_range": {"gripper": [1, -1]}},
    })
    engine.start()
    chunk = engine.infer({"ee_poses": {"eef": np.zeros(7)}, "extra": {"gripper_qpos": np.zeros(2)}})
    assert chunk["ee_targets"]["eef"][0, 0] == 9  # 7 + 2 state dims reached the model
    np.testing.assert_allclose(chunk["gripper"]["gripper"], np.ones(T))  # -1 (open) -> 1


def test_without_layouts_nothing_changes():
    policy = VlaPolicy(_Libero(), action_horizon=T, default_mode="cartesian_delta")
    chunk = policy.infer({"joint_pos": np.zeros(8)})
    np.testing.assert_array_equal(chunk["gripper"]["gripper"], -np.ones(T))


def test_metadata_reports_body_requirements():
    meta = ModelSpec.from_config({"policy": {
        "state_layout": [["joint_pos"], "gripper.left_tcp"],
        "joint_names": ["a", "b"],
        "gripper_range": {"left_tcp": [0, 1]},
    }}).metadata()
    assert meta["state_layout"] == [["joint_pos"], ["gripper", "left_tcp"]]
    assert meta["joint_names"] == ["a", "b"]
    assert meta["gripper"] == "opening"
    plain = ModelSpec.from_config({}).metadata()
    assert plain["state_layout"] is None and plain["joint_names"] is None and plain["gripper"] == "raw"
