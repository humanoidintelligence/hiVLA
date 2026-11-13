"""Drive a hiRobot ``Robot`` with a hiVLA server, no other package in between.

hiRobot supplies the robot (``reset``, ``get_observation``, ``send_action``,
``hold``, ``simulated``, ``embodiment.control_hz``) and has no control loop or
model client. This module is the small piece that joins the two::

    from hirobot import make_robot
    from hivla.client import VlaClient
    from hivla.robot import run_episode

    robot = make_robot("unitree_g1_mujoco", sim=True)
    result = run_episode(robot, VlaClient("http://127.0.0.1:8011"), max_steps=200,
                         instruction="wave")

* :func:`observation_to_dict`: ``hirobot.Observation`` -> the dict
  ``hivla.client`` sends (arrays stay arrays).
* :func:`chunk_from_dict`: the dict ``act()`` returns -> ``hirobot.ActionChunk``.
* :func:`run_episode`: reset the robot and the server, then per chunk one
  ``act`` and ``send_action`` per executed step; for a server reporting
  ``observe: every_step`` (XPolicyLab policies) one ``observe`` before each
  further step. A real robot is paced at ``embodiment.control_hz``; a simulator
  steps as fast as it runs. On an exception the robot is told to ``hold()``.

``import hivla.robot`` does not import hirobot; :func:`chunk_from_dict` and
:func:`run_episode` do, so they need hiRobot installed in the caller's env.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

#: ``Observation`` array fields sent when the driver fills them.
ARRAY_FIELDS = (
    "joint_pos",
    "joint_vel",
    "joint_torque",
    "base_orientation",
    "base_angular_velocity",
    "base_linear_acceleration",
    "base_position",
    "base_linear_velocity",
)

#: Chunk keys that are not ``ActionChunk`` fields (the client's timings).
_CLIENT_KEYS = ("server_timings_ms",)


def observation_to_dict(obs: Any, instruction: Optional[str] = None) -> Dict[str, Any]:
    """The populated fields of a ``hirobot.Observation``, as ``hivla.client`` takes them.

    Proprioception and floating-base arrays, ``ee_poses``, ``gripper``,
    ``images`` with their capture times (``image_timestamps``, on the robot's
    clock, from ``Observation.stamps``), ``instruction`` (``instruction``
    overrides the observation's) and ``extra``.
    """
    out: Dict[str, Any] = {}
    for name in ARRAY_FIELDS:
        value = getattr(obs, name, None)
        if value is not None:
            out[name] = np.asarray(value)
    for name in ("ee_poses", "images"):
        value = getattr(obs, name, None) or {}
        if value:
            out[name] = {k: np.asarray(v) for k, v in value.items()}
    gripper = getattr(obs, "gripper", None) or {}
    if gripper:
        out["gripper"] = {k: float(v) for k, v in gripper.items()}
    stamps = getattr(obs, "stamps", None) or {}
    times = {k: float(stamps[k].t) for k in out.get("images", {}) if k in stamps}
    if times:
        out["image_timestamps"] = times
    text = instruction if instruction is not None else getattr(obs, "instruction", None)
    if text is not None:
        out["instruction"] = text
    extra = getattr(obs, "extra", None) or {}
    if extra:
        out["extra"] = dict(extra)
    return out


def chunk_from_dict(chunk: Dict[str, Any]) -> Any:
    """A ``hirobot.ActionChunk`` from the dict ``VlaClient.act`` returns.

    The server already sends ``ActionChunk``'s field names (``protocol.CHUNK_FIELDS``),
    so this is ``ActionChunk(**chunk)`` without the client-only keys; hiRobot's
    own validation applies (an unknown field raises).
    """
    from hirobot import ActionChunk

    return ActionChunk(**{k: v for k, v in chunk.items() if k not in _CLIENT_KEYS})


@dataclass
class EpisodeResult:
    """What :func:`run_episode` did."""

    steps: int = 0
    chunks: int = 0
    observes: int = 0
    #: ``"max_steps"``, ``"done"`` or ``"success"``.
    stopped_by: str = "max_steps"
    success: Optional[bool] = None
    act_ms: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def run_episode(robot: Any, client: Any, max_steps: int, *,
                instruction: Optional[str] = None, seed: Optional[int] = None,
                n_action_steps: Optional[int] = None, connect: bool = True,
                reset_robot: bool = True, realtime: Optional[bool] = None) -> EpisodeResult:
    """Run one episode of ``client``'s model on ``robot``.

    Args:
        robot: A hiRobot ``Robot`` (from ``make_robot``).
        client: ``hivla.client.VlaClient`` or ``VlaWsClient``.
        max_steps: Control ticks to run at most.
        instruction: Sent with every observation (overrides the robot's).
        seed: Passed to ``robot.reset`` (simulators).
        n_action_steps: Steps executed per chunk before asking again;
            default the server's ``n_action_steps``, else the whole chunk.
        connect: Connect the robot first and disconnect it at the end.
        reset_robot: Call ``robot.reset(seed)`` first.
        realtime: Pace at ``embodiment.control_hz``. Default: pace real robots,
            step simulators unpaced. ``False`` on a real robot raises.
    """
    simulated = bool(getattr(robot, "simulated", False))
    if realtime is False and not simulated:
        raise ValueError("realtime=False is only for simulators; a real robot is paced")
    pace = (not simulated) if realtime is None else realtime
    period = 1.0 / float(robot.embodiment.control_hz) if pace else 0.0

    if connect:
        robot.connect()
    try:
        if reset_robot:
            robot.reset(seed)
        meta = dict(client.reset() or {})
        if not meta:
            where = getattr(client, "endpoint", "?")
            raise RuntimeError(f"no /metadata from the server at {where}")
        every_step = meta.get("observe") == "every_step"
        window = n_action_steps or meta.get("n_action_steps")
        result = EpisodeResult(metadata=meta)
        next_tick = time.monotonic()
        while result.steps < max_steps:
            t = time.perf_counter()
            chunk = chunk_from_dict(client.act(observation_to_dict(robot.get_observation(),
                                                                   instruction)))
            result.act_ms.append((time.perf_counter() - t) * 1e3)
            result.chunks += 1
            n = min(chunk.horizon, window or chunk.horizon, max_steps - result.steps)
            for i in range(n):
                if i > 0 and every_step:
                    client.observe(observation_to_dict(robot.get_observation(), instruction))
                    result.observes += 1
                robot.send_action(chunk.step(i))
                result.steps += 1
                if period:
                    next_tick += period
                    time.sleep(max(0.0, next_tick - time.monotonic()))
                if getattr(robot, "success", None):
                    result.stopped_by, result.success = "success", True
                    return result
                if getattr(robot, "done", None):
                    result.stopped_by, result.success = "done", getattr(robot, "success", None)
                    return result
        return result
    except BaseException:
        try:
            robot.hold()
        except Exception:  # noqa: BLE001 - the original error matters more
            pass
        raise
    finally:
        if connect:
            robot.disconnect()
