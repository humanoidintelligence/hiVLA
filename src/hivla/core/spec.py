"""The serving manifest: a single source of truth describing one servable model.

Each surveyed VLA stack spreads "how to serve a model" across several loose
artifacts — openpi's ``TrainConfig`` + ``norm_stats.json`` + ``asset_id``, GR00T's
``config.json`` + ``experiment_cfg/metadata.json`` + ``DATA_CONFIG_MAP``, starVLA's
``OmegaConf`` + ``unnorm_key``. :class:`ModelSpec` collapses them into one explicit
object so a deployment is reproducible from a single YAML file and so one server
can serve multiple models/robots (which openpi cannot, because its norm stats are
implicitly bound to a checkpoint's ``asset_id``).

The flat fields mirror the nested ``model``/``runtime``/``policy``/``server``
sections of ``configs/example.yaml``; :meth:`from_config` flattens either shape.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hivla.core.transforms import NormStats


class ModelSpec(BaseModel):
    """Everything needed to build and serve one VLA model."""

    model_config = ConfigDict(extra="allow")

    # --- model ---
    name: str = "vla"
    family: str = "generic"  # generic | openpi | gr00t | lerobot | starvla
    checkpoint: Optional[str] = None
    backend: str = "auto"  # auto | echo | torch | onnx | tensorrt | remote
    action_horizon: int = 16
    # Steps of each chunk a client should execute before asking again; the
    # model's suggestion, advertised in /metadata. None means action_horizon.
    n_action_steps: Optional[int] = None

    # --- runtime ---
    device: str = "cuda"
    precision: str = "fp32"
    max_batch_size: int = 1
    batch_timeout: float = 0.002
    cache: bool = False  # enable repeat-observation replay
    cache_ttl: float = 0.5
    warmup: bool = True  # run one dummy forward at start() to pay JIT/alloc up front

    # --- policy ---
    action_dim: int = 7
    control_mode: str = "position"  # one of hivla.protocol.CONTROL_MODES (hiRobot ControlMode)
    cameras: Any = Field(default_factory=list)  # list or {name: [h, w]}
    norm_stats: Any = None  # str path | {key: {mean/std/q01/q99/min/max/mode}}
    rename_inputs: Dict[str, str] = Field(default_factory=dict)
    rename_outputs: Dict[str, str] = Field(default_factory=dict)
    # hiRobot observation fields concatenated, in this order, into joint_pos
    # before the family sees it: [[field], [field, name], ...] (ConcatState).
    state_layout: Any = None
    # {gripper name: [closed, open]}: the model's gripper value at each end,
    # mapped to hiRobot's opening in [0, 1] (GripperToOpening).
    gripper_range: Any = None
    # The joint order the model was trained on (Embodiment.joint_names), so a
    # client can check it matches the robot. Reported in /metadata only.
    joint_names: Optional[List[str]] = None

    # --- server ---
    transport: str = "http"  # http | websocket
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: Optional[str] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ModelSpec":
        """Build from a (possibly nested) config dict, e.g. ``configs/example.yaml``.

        Accepts either ``{"model": {...}, "runtime": {...}, ...}`` (the YAML shape)
        or an already-flat dict of ModelSpec fields.
        """
        model = config.get("model", {}) or {}
        runtime = config.get("runtime", {}) or {}
        policy = config.get("policy", {}) or {}
        server = config.get("server", {}) or {}

        # A flat dict (already ModelSpec-shaped) takes priority; nested sections
        # otherwise. Fields explicitly set at the top level win.
        flat = {
            k: v
            for k, v in config.items()
            if k not in ("model", "runtime", "policy", "server")
        }

        values: Dict[str, Any] = {
            "name": model.get("name", "vla"),
            "family": model.get("family", "generic"),
            "checkpoint": model.get("checkpoint"),
            "backend": model.get("backend", "auto"),
            "action_horizon": model.get("action_horizon", 16),
            "n_action_steps": model.get("n_action_steps"),
            "device": runtime.get("device", "cuda"),
            "precision": runtime.get("precision", "fp32"),
            "max_batch_size": runtime.get("max_batch_size", 1),
            "batch_timeout": runtime.get("batch_timeout", 0.002),
            "cache": runtime.get("cache", False),
            "cache_ttl": runtime.get("cache_ttl", 0.5),
            "warmup": runtime.get("warmup", True),
            "action_dim": policy.get("action_dim", 7),
            "control_mode": policy.get("control_mode", "position"),
            "cameras": policy.get("cameras", []),
            "norm_stats": policy.get("norm_stats"),
            "rename_inputs": policy.get("rename_inputs", {}),
            "rename_outputs": policy.get("rename_outputs", {}),
            "state_layout": policy.get("state_layout"),
            "gripper_range": policy.get("gripper_range"),
            "joint_names": policy.get("joint_names"),
            "transport": server.get("transport", "http"),
            "host": server.get("host", "0.0.0.0"),
            "port": server.get("port", 8000),
            "api_key": server.get("api_key"),
        }
        values.update(flat)
        return cls(**values)

    @model_validator(mode="after")
    def _check_control_mode(self) -> "ModelSpec":
        from hivla.protocol import CONTROL_MODES

        if self.control_mode not in CONTROL_MODES:
            raise ValueError(
                f"control_mode must be one of hiRobot's chunk modes {CONTROL_MODES}, "
                f"got {self.control_mode!r}"
            )
        return self

    @model_validator(mode="after")
    def _check_layouts(self) -> "ModelSpec":
        from hivla.core.transforms import parse_gripper_range, parse_state_layout

        parse_state_layout(self.state_layout)
        parse_gripper_range(self.gripper_range)
        return self

    @model_validator(mode="after")
    def _check_n_action_steps(self) -> "ModelSpec":
        n = self.n_action_steps
        if n is not None and not 1 <= n <= self.action_horizon:
            raise ValueError(
                f"n_action_steps must be between 1 and action_horizon "
                f"({self.action_horizon}), got {n}"
            )
        return self

    def load_norm_stats(self) -> Optional[Dict[str, NormStats]]:
        """Resolve ``norm_stats`` (path or inline dict) into key -> NormStats."""
        if self.norm_stats is None:
            return None
        if isinstance(self.norm_stats, str):
            with open(self.norm_stats, "r") as f:
                raw = json.load(f)
        else:
            raw = self.norm_stats
        return {key: NormStats.from_dict(val) for key, val in raw.items()}

    def metadata(self) -> Dict[str, Any]:
        """The connect-time handshake advertised via ``/metadata``.

        Mirrors what openpi sends on connect (action horizon, camera names, control
        mode) so the client stays in sync with the server instead of hard-coding it.
        """
        from hivla.protocol import PROTOCOL_VERSION
        from hivla.core.transforms import parse_state_layout

        layout = parse_state_layout(self.state_layout)
        return {
            "protocol_version": PROTOCOL_VERSION,
            "model": self.name,
            "family": self.family,
            "action_horizon": self.action_horizon,
            "n_action_steps": self.n_action_steps or self.action_horizon,
            "action_dim": self.action_dim,
            "control_mode": self.control_mode,
            "cameras": self.cameras,
            # What the model needs from the robot, so a client can check before it runs.
            "state_layout": [[f] if n is None else [f, n] for f, n in layout] or None,
            "joint_names": self.joint_names,
            "gripper": "opening" if self.gripper_range else "raw",
        }
