"""XPolicyLab backend: any ``XPolicyLab.policy.<name>.model.Model`` behind one adapter.

XPolicyLab gives every policy the same ``ModelTemplate`` interface: ``reset()``
per episode, ``update_obs(obs)`` for each observation and ``get_action()`` (no
observation argument) returning a list with one dict per step. The policy keeps
its state between calls, so this backend is ``stateful`` and reads an
observation every executed step (``observe: every_step``).

What it translates, per arm (one or two, configured in ``native_kwargs.arms``):

* observation ``joint_pos[a:b]`` -> ``state/{p}arm_joint_state`` (single arm
  also ``state/joint_state``), ``gripper[name]`` (opening in [0, 1]) ->
  ``state/{p}ee_joint_state`` in the model's range (``policy.gripper_range``),
  ``ee_poses[name]`` -> ``state/{p}ee_pose``. ``{p}`` is ``""`` for one arm and
  ``left_`` / ``right_`` for two.
* ``images[robot cam]`` -> ``vision/<model cam>/color``; ``instruction`` as is.
* action ``{p}arm_joint_state`` -> those arms' columns of ``joint_targets``
  (``action_type: joint``; the other joints hold the episode's first ``joint_pos``),
  or ``{p}ee_pose`` -> ``ee_targets[name]``
  (``action_type: ee``, mode ``cartesian``); ``{p}ee_joint_state`` ->
  ``gripper[name]`` in the model's range (mapped to [0, 1] afterwards by
  ``GripperToOpening`` when ``gripper_range`` is set).

Poses: XPolicyLab is ``[x, y, z, qw, qx, qy, qz]``, hiRobot ``[x, y, z, qx, qy,
qz, qw]``; both directions are converted.

``Model.__init__`` reads the robot's dimensions from ``<parent of
XPolicyLab>/env_cfg/`` (``<type>.yml``, ``robot/_robot_info.json``,
``sim/<type>.yml``). Those files belong to a benchmark repo, so the backend
writes them from ``arms`` before loading, under its own ``env_cfg_type``
(default ``hivla_<model name>``). It refuses to overwrite an entry that
differs.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from hivla.families.base import FamilyAdapter, rename_transforms
from hivla.runtimes.base import ModelBackend
from hivla.core.spec import ModelSpec

ACTION_TYPES = ("joint", "ee")


def wxyz_to_xyzw(pose: np.ndarray) -> np.ndarray:
    """XPolicyLab ``[x, y, z, qw, qx, qy, qz]`` -> hiRobot ``[x, y, z, qx, qy, qz, qw]``."""
    pose = np.asarray(pose, dtype=np.float32)
    return np.concatenate([pose[..., :3], pose[..., 4:7], pose[..., 3:4]], axis=-1)


def xyzw_to_wxyz(pose: np.ndarray) -> np.ndarray:
    """hiRobot ``[x, y, z, qx, qy, qz, qw]`` -> XPolicyLab ``[x, y, z, qw, qx, qy, qz]``."""
    pose = np.asarray(pose, dtype=np.float32)
    return np.concatenate([pose[..., :3], pose[..., 6:7], pose[..., 3:6]], axis=-1)


class _Arm:
    """One arm of the configured layout (``native_kwargs.arms[i]``)."""

    def __init__(self, cfg: Dict[str, Any], prefix: str, index: int) -> None:
        joints = cfg.get("joints")
        if not (isinstance(joints, (list, tuple)) and len(joints) == 2):
            raise ValueError(f"xpolicylab arms[{index}].joints must be [start, stop] into "
                             f"joint_pos, got {joints!r}")
        self.start, self.stop = int(joints[0]), int(joints[1])
        if not 0 <= self.start < self.stop:
            raise ValueError(f"xpolicylab arms[{index}].joints {joints!r} is empty")
        self.prefix = prefix
        #: hiRobot end-effector name for poses / ee_targets.
        self.ee: Optional[str] = cfg.get("ee")
        #: hiRobot end-effector name for the gripper (default: ``ee``).
        self.gripper: Optional[str] = cfg.get("gripper", self.ee)
        self.ee_dim = int(cfg.get("ee_dim", 1))
        if self.ee_dim < 0:
            raise ValueError(f"xpolicylab arms[{index}].ee_dim must be >= 0")
        if self.gripper and self.ee_dim != 1:
            raise ValueError(f"xpolicylab arms[{index}]: a gripper takes ee_dim 1 (got "
                             f"{self.ee_dim}); hands with more joints are not supported yet")

    @property
    def arm_dim(self) -> int:
        return self.stop - self.start


class XPolicyLabBackend(ModelBackend):
    """Run one XPolicyLab policy in-process.

    Args:
        spec: The :class:`~hivla.core.spec.ModelSpec`. ``action_horizon`` is the
            chunk length (longer action lists are cut to it).
            ``spec.extra["native_kwargs"]`` holds:

            * ``policy_name`` (required): directory under ``XPolicyLab/policy``.
            * ``action_type``: ``joint`` (default) or ``ee``.
            * ``arms`` (required): one or two ``{joints: [start, stop], ee,
              gripper, ee_dim}`` in the model's arm order (left first).
            * ``cameras``: model camera name -> robot camera name (default:
              the same name).
            * ``env_cfg_type``: name for the generated ``env_cfg`` entry.
            * ``deploy``: extra ``deploy_cfg`` keys (``ckpt_name``, ...),
              merged over the policy's own ``deploy.yml``.
    """

    stateful = True
    observe_mode = "every_step"

    def __init__(self, spec: ModelSpec) -> None:
        self._spec = spec
        kw: Dict[str, Any] = dict((spec.model_extra or {}).get("native_kwargs", {}) or {})
        self._policy_name = kw.get("policy_name")
        if not self._policy_name or not re.fullmatch(r"[A-Za-z0-9_]+", str(self._policy_name)):
            raise ValueError("family 'xpolicylab' needs native_kwargs.policy_name (a directory "
                             f"under XPolicyLab/policy), got {self._policy_name!r}")
        self._action_type = kw.get("action_type", "joint")
        if self._action_type not in ACTION_TYPES:
            raise ValueError(f"xpolicylab action_type must be one of {ACTION_TYPES}, "
                             f"got {self._action_type!r}")
        arms = kw.get("arms") or []
        if len(arms) not in (1, 2):
            raise ValueError("xpolicylab native_kwargs.arms must list one or two arms")
        prefixes = [""] if len(arms) == 1 else ["left_", "right_"]
        self._arms = [_Arm(a, p, i) for i, (a, p) in enumerate(zip(arms, prefixes))]
        if self._action_type == "ee" and not all(a.ee for a in self._arms):
            raise ValueError("xpolicylab action_type 'ee' needs arms[i].ee (end-effector name)")
        self._cameras: Dict[str, str] = dict(kw.get("cameras") or {})
        if not self._cameras:
            self._cameras = {str(c): str(c) for c in (spec.cameras or [])}
        default_type = "hivla_" + re.sub(r"[^A-Za-z0-9_]", "_", spec.name)
        self._env_cfg_type = str(kw.get("env_cfg_type") or default_type)
        if not re.fullmatch(r"[A-Za-z0-9_]+", self._env_cfg_type):
            raise ValueError(f"xpolicylab env_cfg_type must be [A-Za-z0-9_]+, "
                             f"got {self._env_cfg_type!r}")
        self._deploy_overrides: Dict[str, Any] = dict(kw.get("deploy") or {})
        from hivla.core.transforms import parse_gripper_range

        self._ranges = parse_gripper_range(spec.gripper_range)
        self._model: Any = None
        # joint_pos of the episode's first act: what joints outside the arms hold.
        self._held: Any = None
        # One model, one episode: calls from the server's threadpool must not
        # interleave update_obs / get_action.
        self._lock = threading.Lock()

    # --- env_cfg ------------------------------------------------------------

    def robot_action_dim_info(self) -> Dict[str, List[int]]:
        """The ``_robot_info.json`` entry XPolicyLab reads for this layout."""
        return {"arm_dim": [a.arm_dim for a in self._arms],
                "ee_dim": [a.ee_dim for a in self._arms]}

    def write_env_cfg(self, root: Path) -> None:
        """Write ``<root>/env_cfg/`` for :attr:`_env_cfg_type` (``root``: XPolicyLab's parent)."""
        cfg_dir = root / "env_cfg"
        (cfg_dir / "robot").mkdir(parents=True, exist_ok=True)
        (cfg_dir / "sim").mkdir(parents=True, exist_ok=True)
        name = self._env_cfg_type
        _write_if_same(cfg_dir / f"{name}.yml",
                       f"# Written by hiVLA (xpolicylab family).\nconfig:\n"
                       f"  robot: {name}\n  sim: {name}\n")
        _write_if_same(cfg_dir / "sim" / f"{name}.yml",
                       "# Written by hiVLA (xpolicylab family).\nscene:\n  num_envs: 1\n")
        info_path = cfg_dir / "robot" / "_robot_info.json"
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        want = self.robot_action_dim_info()
        if name in info and info[name] != want:
            raise ValueError(f"{info_path} already has {name!r} = {info[name]}, not {want}; "
                             "set native_kwargs.env_cfg_type to another name")
        if info.get(name) != want:
            info[name] = want
            tmp = info_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")
            os.replace(tmp, info_path)

    # --- lifecycle ----------------------------------------------------------

    def load(self) -> None:
        try:
            xpl = importlib.import_module("XPolicyLab")
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "family 'xpolicylab' needs XPolicyLab importable (pip install -e <XPolicyLab> "
                "in the policy's env)") from exc
        xpl_dir = Path(list(xpl.__path__)[0]).resolve()
        self.write_env_cfg(xpl_dir.parent)
        deploy_cfg = _load_deploy_yml(xpl_dir / "policy" / self._policy_name / "deploy.yml")
        deploy_cfg.update(self._deploy_overrides)
        deploy_cfg.update(policy_name=self._policy_name, env_cfg_type=self._env_cfg_type,
                          action_type=self._action_type)
        module = importlib.import_module(f"XPolicyLab.policy.{self._policy_name}.model")
        self._model = module.Model(deploy_cfg)

    def unload(self) -> None:
        self._model = None

    def warmup(self) -> None:
        # A policy's get_action needs a real observation first; nothing generic to run.
        pass

    def metadata(self) -> Dict[str, Any]:
        return {"xpolicylab": {"policy_name": self._policy_name,
                               "action_type": self._action_type,
                               "env_cfg_type": self._env_cfg_type,
                               **self.robot_action_dim_info()}}

    def reset(self) -> None:
        with self._lock:
            self._require_model().reset()
            self._held = None

    def observe(self, inputs: Dict[str, Any]) -> None:
        obs = self.to_xpl_observation(inputs)
        with self._lock:
            self._require_model().update_obs(obs)

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        obs = self.to_xpl_observation(inputs)
        with self._lock:
            model = self._require_model()
            model.update_obs(obs)
            actions = model.get_action()
            if self._held is None and inputs.get("joint_pos") is not None:
                self._held = np.asarray(inputs["joint_pos"], dtype=np.float32).copy()
            held = self._held
        return self.to_chunk(actions, joint_pos=held)

    def _require_model(self) -> Any:
        if self._model is None:
            raise RuntimeError("XPolicyLabBackend.load() must be called first")
        return self._model

    # --- translation --------------------------------------------------------

    def to_xpl_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """hiRobot-shaped observation -> XPolicyLab ``{vision, state, instruction}``."""
        joint_pos = obs.get("joint_pos")
        if joint_pos is None:
            raise ValueError("xpolicylab needs joint_pos in the observation")
        joint_pos = np.asarray(joint_pos, dtype=np.float32)
        need = max(a.stop for a in self._arms)
        if joint_pos.shape[-1] < need:
            raise ValueError(f"joint_pos has {joint_pos.shape[-1]} joints; arms need {need}")
        grippers = obs.get("gripper") or {}
        poses = obs.get("ee_poses") or {}
        state: Dict[str, np.ndarray] = {}
        for arm in self._arms:
            p = arm.prefix
            q = joint_pos[arm.start:arm.stop]
            state[f"{p}arm_joint_state"] = q
            if not p:
                state["joint_state"] = q  # pack_robot_state's single-arm key
            state[f"{p}ee_joint_state"] = self._gripper_in(arm, grippers)
            if arm.ee and arm.ee in poses:
                state[f"{p}ee_pose"] = xyzw_to_wxyz(poses[arm.ee])
        images = obs.get("images") or {}
        vision = {}
        for model_cam, robot_cam in self._cameras.items():
            if robot_cam in images:
                vision[model_cam] = {"color": np.asarray(images[robot_cam])}
        out: Dict[str, Any] = {"vision": vision, "state": state, "env_idx": 0,
                               "data_format_version": "v1.0"}
        if obs.get("instruction") is not None:
            out["instruction"] = obs["instruction"]
        return out

    def _gripper_in(self, arm: _Arm, grippers: Dict[str, Any]) -> np.ndarray:
        if not arm.gripper or arm.gripper not in grippers:
            return np.zeros(arm.ee_dim, dtype=np.float32)
        opening = float(np.asarray(grippers[arm.gripper]).reshape(-1)[0])
        if arm.gripper in self._ranges:
            closed, opened = self._ranges[arm.gripper]
            opening = closed + opening * (opened - closed)
        return np.array([opening], dtype=np.float32)

    def to_chunk(self, actions: Any, joint_pos: Any = None) -> Dict[str, Any]:
        """XPolicyLab's per-step action list -> a hiRobot-shaped chunk dict.

        With ``joint_pos``, ``joint_targets`` covers every joint, as hiRobot's
        joint modes require: joints outside ``arms`` (legs, waist) get those
        values. :meth:`predict` passes the episode's first ``joint_pos`` (since
        :meth:`reset`), so they hold where the episode started; refilling them
        from each new observation lets a PD-controlled body sag under gravity
        (0.5 rad in 100 steps on the G1 in MuJoCo).
        """
        if isinstance(actions, dict):
            actions = [actions]
        actions = list(actions)
        horizon = self._spec.action_horizon
        if len(actions) < horizon:
            raise ValueError(f"policy {self._policy_name!r} returned {len(actions)} steps "
                             f"< action_horizon {horizon}")
        actions = actions[:horizon]

        def column(key: str, dim: int) -> np.ndarray:
            rows = []
            for i, step in enumerate(actions):
                if key not in step:
                    raise KeyError(f"policy {self._policy_name!r} step {i} has no {key!r} "
                                   f"(keys: {sorted(step)})")
                row = np.asarray(step[key], dtype=np.float32).reshape(-1)
                if row.shape[0] != dim:
                    raise ValueError(f"policy {self._policy_name!r} step {i} {key!r} has "
                                     f"{row.shape[0]} values, expected {dim}")
                rows.append(row)
            return np.stack(rows)

        chunk: Dict[str, Any] = {}
        if self._action_type == "joint":
            single = len(self._arms) == 1
            cols = []
            for arm in self._arms:
                key = f"{arm.prefix}arm_joint_state"
                if single and key not in actions[0] and "joint_state" in actions[0]:
                    key = "joint_state"
                cols.append(column(key, arm.arm_dim))
            if joint_pos is None:
                targets = np.concatenate(cols, axis=1)
            else:
                held = np.asarray(joint_pos, dtype=np.float32).reshape(-1)
                targets = np.repeat(held[None, :], len(actions), axis=0)
                for arm, col in zip(self._arms, cols):
                    targets[:, arm.start:arm.stop] = col
            chunk["joint_targets"] = targets
            chunk["mode"] = "position"
        else:
            chunk["ee_targets"] = {arm.ee: wxyz_to_xyzw(column(f"{arm.prefix}ee_pose", 7))
                                   for arm in self._arms}
            chunk["mode"] = "cartesian"
        grip = {arm.gripper: column(f"{arm.prefix}ee_joint_state", 1)[:, 0]
                for arm in self._arms if arm.gripper}
        if grip:
            chunk["gripper"] = grip
        return chunk


def _write_if_same(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` unless it holds something else (then raise)."""
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return
        raise ValueError(f"{path} exists with other content; set native_kwargs.env_cfg_type "
                         "to another name")
    path.write_text(text, encoding="utf-8")


def _load_deploy_yml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return dict(yaml.safe_load(f) or {})


class XPolicyLabFamily(FamilyAdapter):
    """Serve any XPolicyLab policy (``XPolicyLab.policy.<name>.model.Model``) in-process.

    One adapter for every policy: ``native_kwargs.policy_name`` picks it and
    ``native_kwargs.arms`` maps hiRobot's flat observation to XPolicyLab's
    per-arm state (see :class:`~hivla.families.xpolicylab.XPolicyLabBackend`).
    Gripper commands come back in the model's range; ``policy.gripper_range``
    maps them to hiRobot's opening.
    """

    name = "xpolicylab"

    def build_backend(self, spec: ModelSpec) -> ModelBackend:
        return XPolicyLabBackend(spec)

    def build_transforms(self, spec: ModelSpec):
        return rename_transforms(spec)
