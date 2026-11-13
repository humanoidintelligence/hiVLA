"""hivla.robot: hiRobot Observation -> client dict, chunk dict -> ActionChunk, run_episode."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hivla.robot import observation_to_dict


def test_observation_to_dict_without_hirobot():
    obs = SimpleNamespace(
        joint_pos=np.zeros(3), joint_vel=None, ee_poses={"tcp": np.arange(7.0)},
        gripper={"tcp": 0.5}, images={"scene": np.zeros((2, 2, 3), np.uint8)},
        stamps={"scene": SimpleNamespace(t=1.25)}, instruction="pick", extra={"k": 1})
    out = observation_to_dict(obs)
    assert set(out) == {"joint_pos", "ee_poses", "gripper", "images", "image_timestamps",
                        "instruction", "extra"}
    assert out["image_timestamps"] == {"scene": 1.25}
    assert observation_to_dict(obs, instruction="place")["instruction"] == "place"


def test_import_does_not_pull_hirobot():
    code = "import sys, hivla.robot; print('hirobot' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True)
    assert out.stdout.strip() == "False"


# --- with hiRobot (skipped without it) ---------------------------------------


class _FakeClient:
    """Answers like a hiVLA server for ur5e_dual's 12 joints."""

    endpoint = "fake"

    def __init__(self, horizon=4, observe="chunk", target=0.1):
        self.horizon, self.observe_mode, self.target = horizon, observe, target
        self.calls = []

    def reset(self):
        self.calls.append("reset")
        return {"action_horizon": self.horizon, "observe": self.observe_mode}

    def act(self, obs):
        self.calls.append("act")
        assert obs["joint_pos"].shape == (12,) and obs["instruction"] == "go"
        return {"mode": "position",
                "joint_targets": np.full((self.horizon, 12), self.target, np.float32),
                "server_timings_ms": {"total": 1.0}}

    def observe(self, obs):
        self.calls.append("observe")


@pytest.fixture
def robot():
    hirobot = pytest.importorskip("hirobot")
    return hirobot.make_robot("ur5e_dual")  # toy driver, no simulator


def test_chunk_from_dict_is_a_hirobot_chunk():
    pytest.importorskip("hirobot")
    from hivla.robot import chunk_from_dict

    chunk = chunk_from_dict({"mode": "position", "joint_targets": np.zeros((3, 12)),
                             "gripper": {"left/tcp": np.ones(3)},
                             "server_timings_ms": {"total": 2.0}})
    assert chunk.horizon == 3 and chunk.step(1).gripper == {"left/tcp": 1.0}


def test_run_episode_executes_chunks(robot):
    from hivla.robot import run_episode

    client = _FakeClient(horizon=4)
    result = run_episode(robot, client, 10, instruction="go")
    assert (result.steps, result.chunks, result.observes) == (10, 3, 0)
    assert client.calls == ["reset", "act", "act", "act"]
    assert result.stopped_by == "max_steps" and len(result.act_ms) == 3


def test_run_episode_observes_every_step_and_honours_n_action_steps(robot):
    from hivla.robot import run_episode

    client = _FakeClient(horizon=4, observe="every_step")
    result = run_episode(robot, client, 6, instruction="go", n_action_steps=2)
    assert (result.steps, result.chunks, result.observes) == (6, 3, 3)
    assert client.calls == ["reset"] + ["act", "observe"] * 3


def test_run_episode_holds_on_error(robot, monkeypatch):
    from hivla.robot import run_episode

    held = []
    monkeypatch.setattr(robot, "hold", lambda: held.append(True))
    client = _FakeClient()
    client.act = lambda obs: {"mode": "position", "joint_targets": np.zeros((2, 5))}
    with pytest.raises(Exception):
        run_episode(robot, client, 4, instruction="go")
    assert held == [True]


def test_run_episode_needs_a_server(robot):
    from hivla.robot import run_episode

    client = _FakeClient()
    client.reset = lambda: {}
    with pytest.raises(RuntimeError, match="no /metadata"):
        run_episode(robot, client, 4)
