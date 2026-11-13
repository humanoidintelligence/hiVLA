"""XPolicyLab family: XPolicyLabBackend against a fake ``XPolicyLab`` package."""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from hivla import build_engine_from_spec
from hivla.families.xpolicylab import (
    XPolicyLabBackend,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from hivla.serving.engine import SessionError
from hivla.core.spec import ModelSpec

# ur5e_dual's shape: two 6-joint arms, left first, a 1-D gripper each.
DUAL_ARMS = [
    {"joints": [0, 6], "ee": "left/tcp", "gripper": "left/tcp"},
    {"joints": [6, 12], "ee": "right/tcp", "gripper": "right/tcp"},
]
STEPS = 3


def _install_fake_xpl(monkeypatch, tmp_path, steps=STEPS, deploy_yml="seed: 0\n"):
    """A minimal XPolicyLab: policy ``fake`` records calls, returns ``steps`` actions.

    Like the real ``Model.__init__``, the fake reads the robot's dims from
    ``<parent>/env_cfg/`` so the generated files are exercised.
    """
    root = tmp_path / "bench"
    xpl_dir = root / "XPolicyLab"
    (xpl_dir / "policy" / "fake").mkdir(parents=True)
    (xpl_dir / "policy" / "fake" / "deploy.yml").write_text(deploy_yml)
    calls = {"reset": 0, "obs": [], "cfg": None}

    class Model:
        def __init__(self, model_cfg):
            import yaml

            calls["cfg"] = dict(model_cfg)
            env = yaml.safe_load((root / "env_cfg" / f"{model_cfg['env_cfg_type']}.yml")
                                 .read_text())
            info = json.loads((root / "env_cfg" / "robot" / "_robot_info.json").read_text())
            self.dims = info[env["config"]["robot"]]
            self.action_type = model_cfg["action_type"]

        def reset(self):
            calls["reset"] += 1
            calls["obs"].clear()

        def update_obs(self, obs):
            calls["obs"].append(obs)

        def get_action(self):
            n = len(self.dims["arm_dim"])
            prefixes = [""] if n == 1 else ["left_", "right_"]
            seen = len(calls["obs"])
            out = []
            for t in range(steps):
                step = {}
                for i, p in enumerate(prefixes):
                    if self.action_type == "joint":
                        step[f"{p}arm_joint_state"] = np.full(self.dims["arm_dim"][i],
                                                              10 * i + t, np.float32)
                    else:  # [x, y, z, qw, qx, qy, qz]
                        step[f"{p}ee_pose"] = np.array([t, i, seen, 1, 0, 0, 0], np.float32)
                    step[f"{p}ee_joint_state"] = np.full(self.dims["ee_dim"][i], 0.5, np.float32)
                out.append(step)
            return out

    pkg = types.ModuleType("XPolicyLab")
    pkg.__path__ = [str(xpl_dir)]
    policy_pkg = types.ModuleType("XPolicyLab.policy")
    policy_pkg.__path__ = [str(xpl_dir / "policy")]
    fake_pkg = types.ModuleType("XPolicyLab.policy.fake")
    model_mod = types.ModuleType("XPolicyLab.policy.fake.model")
    model_mod.Model = Model
    for name, mod in [("XPolicyLab", pkg), ("XPolicyLab.policy", policy_pkg),
                      ("XPolicyLab.policy.fake", fake_pkg),
                      ("XPolicyLab.policy.fake.model", model_mod)]:
        monkeypatch.setitem(sys.modules, name, mod)
    return root, calls


def _spec(arms=DUAL_ARMS, action_type="joint", horizon=STEPS, **kw) -> ModelSpec:
    native = {"policy_name": "fake", "action_type": action_type, "arms": arms,
              "cameras": {"cam_head": "scene"}, **kw.pop("native", {})}
    cfg = {
        "model": {"name": "xpl-fake", "family": "xpolicylab", "action_horizon": horizon},
        "native_kwargs": native,
        "runtime": {"cache": False},
        "policy": {"control_mode": "position", **kw.pop("policy", {})},
    }
    return ModelSpec.from_config(cfg)


def _obs(dof=12):
    return {
        "joint_pos": np.arange(dof, dtype=np.float32),
        "gripper": {"left/tcp": 1.0, "right/tcp": 0.0},
        "ee_poses": {"left/tcp": np.array([1, 2, 3, 0.1, 0.2, 0.3, 0.9], np.float32)},
        "images": {"scene": np.zeros((8, 8, 3), np.uint8)},
        "instruction": "stack the cups",
    }


def _engine(spec):
    engine = build_engine_from_spec(spec)
    engine.start()
    engine.reset("s")
    return engine


def test_quaternion_order_round_trips():
    xyzw = np.array([1, 2, 3, 0.1, 0.2, 0.3, 0.9], np.float32)
    wxyz = xyzw_to_wxyz(xyzw)
    np.testing.assert_allclose(wxyz, [1, 2, 3, 0.9, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(wxyz_to_xyzw(wxyz), xyzw)


def test_dual_arm_joint_chunk_and_observation(monkeypatch, tmp_path):
    root, calls = _install_fake_xpl(monkeypatch, tmp_path)
    engine = _engine(_spec(policy={"gripper_range": {"left/tcp": [0.0, 1.0],
                                                     "right/tcp": [0.0, 1.0]}}))
    chunk = engine.infer(_obs(), session="s")

    targets = chunk["joint_targets"]
    assert targets.shape == (STEPS, 12)
    np.testing.assert_allclose(targets[:, :6], np.arange(STEPS)[:, None] * np.ones(6))
    np.testing.assert_allclose(targets[:, 6:], 10 + np.arange(STEPS)[:, None] * np.ones(6))
    assert chunk["mode"] == "position"
    assert set(chunk["gripper"]) == {"left/tcp", "right/tcp"}
    np.testing.assert_allclose(chunk["gripper"]["left/tcp"], np.full(STEPS, 0.5))

    obs = calls["obs"][-1]
    np.testing.assert_allclose(obs["state"]["left_arm_joint_state"], np.arange(6))
    np.testing.assert_allclose(obs["state"]["right_arm_joint_state"], np.arange(6, 12))
    np.testing.assert_allclose(obs["state"]["left_ee_joint_state"], [1.0])
    np.testing.assert_allclose(obs["state"]["right_ee_joint_state"], [0.0])
    np.testing.assert_allclose(obs["state"]["left_ee_pose"], [1, 2, 3, 0.9, 0.1, 0.2, 0.3])
    assert "right_ee_pose" not in obs["state"]  # robot sent no pose for it
    assert obs["vision"]["cam_head"]["color"].shape == (8, 8, 3)
    assert obs["instruction"] == "stack the cups"

    # deploy.yml is the base; hiVLA's keys win.
    assert calls["cfg"]["seed"] == 0
    assert calls["cfg"]["env_cfg_type"] == "hivla_xpl_fake"
    info = json.loads((root / "env_cfg" / "robot" / "_robot_info.json").read_text())
    assert info["hivla_xpl_fake"] == {"arm_dim": [6, 6], "ee_dim": [1, 1]}


def test_gripper_range_maps_both_ways(monkeypatch, tmp_path):
    _, calls = _install_fake_xpl(monkeypatch, tmp_path)
    # The model's gripper is 0.08 open, 0.0 closed (a joint in metres).
    engine = _engine(_spec(policy={"gripper_range": {"left/tcp": [0.0, 0.08]}}))
    chunk = engine.infer(_obs(), session="s")
    np.testing.assert_allclose(calls["obs"][-1]["state"]["left_ee_joint_state"], [0.08])
    np.testing.assert_allclose(chunk["gripper"]["left/tcp"], np.ones(STEPS))  # 0.5 > 0.08: clipped
    np.testing.assert_allclose(chunk["gripper"]["right/tcp"], np.full(STEPS, 0.5))  # unmapped


def test_ee_action_type_gives_cartesian_targets(monkeypatch, tmp_path):
    _install_fake_xpl(monkeypatch, tmp_path)
    engine = _engine(_spec(action_type="ee", policy={"control_mode": "cartesian"}))
    chunk = engine.infer(_obs(), session="s")
    assert chunk["mode"] == "cartesian"
    assert "joint_targets" not in chunk
    left = chunk["ee_targets"]["left/tcp"]
    assert left.shape == (STEPS, 7)
    np.testing.assert_allclose(left[1], [1, 0, 1, 0, 0, 0, 1])  # [x, y, z, qx, qy, qz, qw]


def test_single_arm_uses_unprefixed_keys(monkeypatch, tmp_path):
    _, calls = _install_fake_xpl(monkeypatch, tmp_path)
    engine = _engine(_spec(arms=[{"joints": [0, 6], "ee": "ee"}]))
    obs = {"joint_pos": np.arange(6, dtype=np.float32), "gripper": {"ee": 0.25}}
    chunk = engine.infer(obs, session="s")
    assert chunk["joint_targets"].shape == (STEPS, 6)
    assert set(chunk["gripper"]) == {"ee"}
    state = calls["obs"][-1]["state"]
    assert set(state) >= {"arm_joint_state", "joint_state", "ee_joint_state"}


def test_reset_observe_and_sessions(monkeypatch, tmp_path):
    _, calls = _install_fake_xpl(monkeypatch, tmp_path)
    engine = _engine(_spec(action_type="ee", policy={"control_mode": "cartesian"}))
    assert calls["reset"] == 1
    assert engine.metadata["stateful"] is True and engine.metadata["observe"] == "every_step"
    assert engine.metadata["xpolicylab"]["arm_dim"] == [6, 6]
    engine.infer(_obs(), session="s")
    engine.observe(_obs(), session="s")
    engine.observe(_obs(), session="s")
    chunk = engine.infer(_obs(), session="s")
    # The fake writes how many observations it holds into z.
    assert chunk["ee_targets"]["left/tcp"][0, 2] == 4
    with pytest.raises(SessionError):
        engine.infer(_obs(), session="other")
    engine.reset("other")
    assert calls["reset"] == 2
    assert engine.infer(_obs(), session="other")["ee_targets"]["left/tcp"][0, 2] == 1


def test_short_action_list_is_an_error(monkeypatch, tmp_path):
    _install_fake_xpl(monkeypatch, tmp_path, steps=2)
    engine = _engine(_spec(horizon=3))
    with pytest.raises(ValueError, match="2 steps < action_horizon 3"):
        engine.infer(_obs(), session="s")


def test_longer_action_list_is_cut_to_the_horizon(monkeypatch, tmp_path):
    _install_fake_xpl(monkeypatch, tmp_path, steps=5)
    engine = _engine(_spec(horizon=3))
    assert engine.infer(_obs(), session="s")["joint_targets"].shape == (3, 12)


def test_env_cfg_conflict_refuses(monkeypatch, tmp_path):
    root, _ = _install_fake_xpl(monkeypatch, tmp_path)
    (root / "env_cfg" / "robot").mkdir(parents=True)
    (root / "env_cfg" / "robot" / "_robot_info.json").write_text(
        json.dumps({"hivla_xpl_fake": {"arm_dim": [7], "ee_dim": [1]}}))
    engine = build_engine_from_spec(_spec())
    with pytest.raises(ValueError, match="already has"):
        engine.start()


def test_env_cfg_keeps_other_entries(monkeypatch, tmp_path):
    root, _ = _install_fake_xpl(monkeypatch, tmp_path)
    (root / "env_cfg" / "robot").mkdir(parents=True)
    (root / "env_cfg" / "robot" / "_robot_info.json").write_text(
        json.dumps({"piper": {"arm_dim": [6, 6], "ee_dim": [1, 1]}}))
    _engine(_spec())
    info = json.loads((root / "env_cfg" / "robot" / "_robot_info.json").read_text())
    assert set(info) == {"piper", "hivla_xpl_fake"}


@pytest.mark.parametrize("native, match", [
    ({"policy_name": "../x"}, "policy_name"),
    ({"arms": []}, "one or two arms"),
    ({"arms": [{"joints": [0, 6]}] * 3}, "one or two arms"),
    ({"arms": [{"joints": [3, 3]}]}, "empty"),
    ({"arms": [{"joints": [0, 6], "ee": "ee", "ee_dim": 6}]}, "ee_dim 1"),
    ({"action_type": "delta"}, "action_type"),
])
def test_bad_config_is_refused(native, match):
    spec = _spec()
    spec.model_extra["native_kwargs"].update(native)
    with pytest.raises(ValueError, match=match):
        XPolicyLabBackend(spec)


def test_ee_action_needs_ee_name():
    spec = _spec(arms=[{"joints": [0, 6]}], action_type="ee")
    with pytest.raises(ValueError, match="ee"):
        XPolicyLabBackend(spec)


def test_family_is_registered():
    from hivla import available_families

    assert "xpolicylab" in available_families()


# --- against hiRobot's real types (skipped without hiRobot) -----------------


@pytest.mark.parametrize("action_type", ["joint", "ee"])
@pytest.mark.parametrize("binary", [False, True])
def test_ur5e_dual_chunk_is_a_valid_hirobot_chunk(monkeypatch, tmp_path, action_type, binary):
    types_ = pytest.importorskip("hirobot.core.types")
    from hivla.protocol import (
        decode_action_chunk,
        decode_observation,
        encode_action_chunk,
        encode_observation,
    )

    _install_fake_xpl(monkeypatch, tmp_path)
    mode = "position" if action_type == "joint" else "cartesian"
    engine = _engine(_spec(action_type=action_type, policy={"control_mode": mode}))
    # ur5e_dual's observation: 12 joints, poses for left/tcp and right/tcp.
    robot_obs = types_.Observation(
        timestamp=0.0,
        joint_pos=np.zeros(12, np.float32),
        joint_vel=np.zeros(12, np.float32),
        ee_poses={n: np.array([0, 0, 0.5, 0, 0, 0, 1], np.float32)
                  for n in ("left/tcp", "right/tcp")},
        images={"scene": np.zeros((8, 8, 3), np.uint8)},
        instruction="stack the cups",
    )
    payload = {k: getattr(robot_obs, k) for k in
               ("joint_pos", "joint_vel", "ee_poses", "gripper", "images", "instruction")}
    obs = decode_observation(encode_observation(payload, binary))
    raw = engine.infer(obs, session="s")
    chunk = types_.ActionChunk(**decode_action_chunk(encode_action_chunk(raw, binary)))
    assert chunk.horizon == STEPS
    assert chunk.mode.value == mode


def test_joints_outside_the_arms_hold_their_observed_position(monkeypatch, tmp_path):
    # G1's shape: 29 joints, arms at 15:22 and 22:29, legs and waist before them.
    _install_fake_xpl(monkeypatch, tmp_path)
    arms = [{"joints": [15, 22], "ee": "left_hand", "gripper": None, "ee_dim": 0},
            {"joints": [22, 29], "ee": "right_hand", "gripper": None, "ee_dim": 0}]
    engine = _engine(_spec(arms=arms))
    q = np.linspace(-1, 1, 29).astype(np.float32)
    chunk = engine.infer({"joint_pos": q, "images": {}}, session="s")
    targets = chunk["joint_targets"]
    assert targets.shape == (STEPS, 29)
    np.testing.assert_allclose(targets[:, :15], np.repeat(q[None, :15], STEPS, 0))
    np.testing.assert_allclose(targets[:, 15:22], np.arange(STEPS)[:, None] * np.ones(7))
    np.testing.assert_allclose(targets[:, 22:], 10 + np.arange(STEPS)[:, None] * np.ones(7))
    assert "gripper" not in chunk
    # Later chunks keep the first observation's values; reset starts over.
    later = engine.infer({"joint_pos": q + 1, "images": {}}, session="s")["joint_targets"]
    np.testing.assert_allclose(later[:, :15], np.repeat(q[None, :15], STEPS, 0))
    engine.reset("s")
    fresh = engine.infer({"joint_pos": q + 1, "images": {}}, session="s")["joint_targets"]
    np.testing.assert_allclose(fresh[:, :15], np.repeat(q[None, :15] + 1, STEPS, 0))
