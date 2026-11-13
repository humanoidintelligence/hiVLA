"""Data transforms: the per-model preprocessing / postprocessing pipeline.

A VLA model does not consume a robot observation directly. It needs its inputs
normalized, padded, and tokenized, and its raw outputs unnormalized and sliced
back to the robot's real action dimension. Following openpi, we express each
step as a small frozen-dataclass transform — a pure ``__call__(dict) -> dict``
function — and ``compose`` them into an input pipeline and an output pipeline.

Supporting a new model/robot means writing (or configuring) transforms, not
subclassing the model or the backend. :class:`~hivla.core.policy.VlaPolicy` runs
``input transforms -> backend -> output transforms``.

The transforms here are deliberately *model-agnostic* (rename / select / pad /
slice / normalize). Model-specific tokenization and image encoding live in the
family adapters (``hivla.families``), which import the model family's native
preprocessing — the "keep the native transform chain, enter through one inference
interface" principle RLinf uses for openpi.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

DataDict = Dict[str, Any]


@runtime_checkable
class DataTransformFn(Protocol):
    """A pure function from one data dict to another."""

    def __call__(self, data: DataDict) -> DataDict: ...


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Chain transforms left-to-right into a single transform."""

    def _composed(data: DataDict) -> DataDict:
        for t in transforms:
            data = t(data)
        return data

    return _composed


def _as_float(x) -> Optional[np.ndarray]:
    return None if x is None else np.asarray(x, dtype=np.float32)


@dataclasses.dataclass(frozen=True)
class NormStats:
    """Normalization statistics for one array key.

    Exactly one of the statistic families is used, chosen by ``mode`` (or inferred
    from which fields are present): z-score needs ``mean``/``std``, quantile needs
    ``q01``/``q99``, min-max needs ``min``/``max``. This unifies the four surveyed
    schemes — openpi's ``norm_stats.json`` (mean/std or q01/q99), GR00T's
    ``DatasetStatistics`` (min/max/q01/q99), and starVLA's per-dataloader stats.
    """

    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    q01: Optional[np.ndarray] = None
    q99: Optional[np.ndarray] = None
    min: Optional[np.ndarray] = None
    max: Optional[np.ndarray] = None
    mode: Optional[str] = None  # "zscore" | "quantile" | "minmax"; None -> infer

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NormStats":
        return cls(
            mean=_as_float(d.get("mean")),
            std=_as_float(d.get("std")),
            q01=_as_float(d.get("q01")),
            q99=_as_float(d.get("q99")),
            min=_as_float(d.get("min")),
            max=_as_float(d.get("max")),
            mode=d.get("mode"),
        )

    def resolve_mode(self) -> str:
        """Return the effective mode (explicit -> inferred -> z-score default)."""
        if self.mode is not None:
            return self.mode
        if self.mean is not None and self.std is not None:
            return "zscore"
        if self.q01 is not None and self.q99 is not None:
            return "quantile"
        if self.min is not None and self.max is not None:
            return "minmax"
        return "zscore"


def _scale(x: np.ndarray, st: NormStats, mode: str, eps: float) -> np.ndarray:
    """Normalize one array (already float, last axis is the feature dim)."""
    dim = x.shape[-1]
    if mode == "zscore":
        if st.mean is None or st.std is None:
            return x
        return (x - st.mean[..., :dim]) / (st.std[..., :dim] + eps)
    if mode == "quantile":
        if st.q01 is None or st.q99 is None:
            return x
        lo, hi = st.q01[..., :dim], st.q99[..., :dim]
        return 2.0 * (x - lo) / (hi - lo + eps) - 1.0
    if mode == "minmax":
        if st.min is None or st.max is None:
            return x
        lo, hi = st.min[..., :dim], st.max[..., :dim]
        return 2.0 * (x - lo) / (hi - lo + eps) - 1.0
    raise ValueError(f"unknown norm mode {mode!r}")


def _unscale(x: np.ndarray, st: NormStats, mode: str, eps: float) -> np.ndarray:
    """Invert :func:`_scale`."""
    dim = x.shape[-1]
    if mode == "zscore":
        if st.mean is None or st.std is None:
            return x
        return x * (st.std[..., :dim] + eps) + st.mean[..., :dim]
    if mode == "quantile":
        if st.q01 is None or st.q99 is None:
            return x
        lo, hi = st.q01[..., :dim], st.q99[..., :dim]
        return (x + 1.0) / 2.0 * (hi - lo + eps) + lo
    if mode == "minmax":
        if st.min is None or st.max is None:
            return x
        lo, hi = st.min[..., :dim], st.max[..., :dim]
        return (x + 1.0) / 2.0 * (hi - lo + eps) + lo
    raise ValueError(f"unknown norm mode {mode!r}")


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    """Normalize selected keys using per-key :class:`NormStats`.

    Args:
        stats: key -> :class:`NormStats`.
        mode: Optional global mode override (ignores each stats' ``mode``).
        eps: Numerical guard for division.
    """

    stats: Dict[str, NormStats]
    mode: Optional[str] = None
    eps: float = 1e-6

    def __call__(self, data: DataDict) -> DataDict:
        data = dict(data)
        for key, st in self.stats.items():
            if key in data:
                data[key] = _scale(
                    np.asarray(data[key], dtype=np.float32),
                    st, self.mode or st.resolve_mode(), self.eps,
                )
        return data


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    """Invert :class:`Normalize` on selected keys."""

    stats: Dict[str, NormStats]
    mode: Optional[str] = None
    eps: float = 1e-6

    def __call__(self, data: DataDict) -> DataDict:
        data = dict(data)
        for key, st in self.stats.items():
            if key in data:
                data[key] = _unscale(
                    np.asarray(data[key], dtype=np.float32),
                    st, self.mode or st.resolve_mode(), self.eps,
                )
        return data


@dataclasses.dataclass(frozen=True)
class RenameKeys(DataTransformFn):
    """Rename keys (e.g. robot-facing ``joint_pos`` -> model-facing ``state``)."""

    mapping: Dict[str, str]

    def __call__(self, data: DataDict) -> DataDict:
        out = dict(data)
        for src, dst in self.mapping.items():
            if src in out:
                out[dst] = out.pop(src)
        return out


@dataclasses.dataclass(frozen=True)
class SelectKeys(DataTransformFn):
    """Drop every key not listed (whitelist)."""

    keys: Sequence[str]

    def __call__(self, data: DataDict) -> DataDict:
        keep = set(self.keys)
        return {k: v for k, v in data.items() if k in keep}


@dataclasses.dataclass(frozen=True)
class Slice(DataTransformFn):
    """Slice the trailing feature axis of a key to ``dim`` entries.

    VLA models often emit a fixed-width vector (e.g. action_dim 32) that must be
    sliced back to the robot's real DoF (openpi's ``AlohaOutputs``/``DroidOutputs``
    take the first N dims).
    """

    key: str
    dim: int

    def __call__(self, data: DataDict) -> DataDict:
        out = dict(data)
        if self.key in out:
            out[self.key] = np.asarray(out[self.key])[..., : self.dim]
        return out


@dataclasses.dataclass(frozen=True)
class PadToDim(DataTransformFn):
    """Pad the trailing feature axis of a key to ``dim`` entries (GR00T pads state
    to ``max_state_dim`` and actions to ``max_action_dim``)."""

    key: str
    dim: int
    value: float = 0.0

    def __call__(self, data: DataDict) -> DataDict:
        out = dict(data)
        if self.key in out:
            x = np.asarray(out[self.key], dtype=np.float32)
            if x.shape[-1] < self.dim:
                pad_width = [(0, 0)] * (x.ndim - 1) + [(0, self.dim - x.shape[-1])]
                x = np.pad(x, pad_width, mode="constant", constant_values=self.value)
            out[self.key] = x[..., : self.dim]
        return out


@dataclasses.dataclass(frozen=True)
class SinCos(DataTransformFn):
    """Encode the trailing feature axis of a key as ``[sin, cos]`` pairs.

    GR00T's ``StateActionSinCosTransform`` uses this for rotational joint state.
    Output width doubles (each angle -> two values).
    """

    key: str

    def __call__(self, data: DataDict) -> DataDict:
        out = dict(data)
        if self.key in out:
            x = np.asarray(out[self.key], dtype=np.float32)
            out[self.key] = np.concatenate([np.sin(x), np.cos(x)], axis=-1)
        return out


@dataclasses.dataclass(frozen=True)
class AsFloat(DataTransformFn):
    """Convert a key to float32 with an optional affine map ``x * scale + offset``.

    openpi converts uint8 images to float in ``[-1, 1]`` via ``x / 255 * 2 - 1``
    (i.e. ``scale = 2/255, offset = -1``).
    """

    key: str
    scale: float = 1.0
    offset: float = 0.0

    def __call__(self, data: DataDict) -> DataDict:
        out = dict(data)
        if self.key in out:
            out[self.key] = np.asarray(out[self.key], dtype=np.float32) * self.scale + self.offset
        return out


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    """Resize image array(s) to ``(size, size)``.

    Accepts either a single array under ``key`` (``HxWxC``) or a dict of
    ``name -> HxWxC`` arrays (a whole camera bank). Uses PIL lazily; the core
    package stays dependency-light and this transform is skipped until Pillow is
    installed. VLA backbones are uniformly 224x224 (openpi, GR00T).
    """

    key: str
    size: int

    def __call__(self, data: DataDict) -> DataDict:
        if self.key not in data:
            return data
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "ResizeImages needs Pillow; install the 'vision' extra: "
                "pip install 'hivla[vision]'."
            ) from exc

        def resize(img: np.ndarray) -> np.ndarray:
            arr = np.asarray(img)
            h, w = arr.shape[:2]
            if (h, w) == (self.size, self.size):
                return arr
            pil = Image.fromarray(arr)
            resized = pil.resize((self.size, self.size), Image.BILINEAR)
            return np.asarray(resized, dtype=arr.dtype)

        val = data[self.key]
        out = dict(data)
        if isinstance(val, dict):
            out[self.key] = {k: resize(v) for k, v in val.items()}
        else:
            out[self.key] = resize(val)
        return out


#: Observation fields :class:`ConcatState` can read: ``(name required, value shape)``.
#: The ``name -> value`` maps are hiRobot's ``Observation.ee_poses`` (7-D pose
#: ``[x, y, z, qx, qy, qz, qw]``) and ``Observation.gripper`` (opening, scalar).
STATE_FIELDS: Dict[str, bool] = {
    "joint_pos": False, "joint_vel": False, "joint_torque": False,
    "base_orientation": False, "base_angular_velocity": False,
    "base_linear_acceleration": False, "base_position": False, "base_linear_velocity": False,
    "ee_poses": True, "gripper": True, "extra": True,
}


def parse_state_layout(layout: Any) -> List[Tuple[str, Optional[str]]]:
    """Validate ``[[field], [field, name], ...]`` (or ``"field"`` / ``"field.name"``) entries."""
    out: List[Tuple[str, Optional[str]]] = []
    for entry in layout or []:
        if isinstance(entry, str):
            entry = entry.split(".", 1)
        entry = list(entry)
        if not 1 <= len(entry) <= 2:
            raise ValueError(f"state_layout entry must be [field] or [field, name], got {entry!r}")
        field, name = str(entry[0]), (str(entry[1]) if len(entry) == 2 else None)
        if field not in STATE_FIELDS:
            raise ValueError(f"state_layout field must be one of {sorted(STATE_FIELDS)}, got {field!r}")
        if STATE_FIELDS[field] != (name is not None):
            raise ValueError(f"state_layout field {field!r} "
                             f"{'needs a name' if STATE_FIELDS[field] else 'takes no name'}: {entry!r}")
        out.append((field, name))
    return out


@dataclasses.dataclass(frozen=True)
class ConcatState(DataTransformFn):
    """Build one state vector from hiRobot's structured observation fields.

    Families read their state from one key (``joint_pos`` by default). A robot
    reports it in several fields: joint positions, ``ee_poses[name]``,
    ``gripper[name]``, values in ``extra``. ``layout`` lists them in model
    order; each is flattened, cast to float32 and concatenated into ``key``.
    LIBERO's 8-D state, for example, is ``[[ee_poses, eef], [extra, gripper_qpos]]``
    when the driver reports a 7-D pose (the axis-angle conversion stays in the
    family). A missing field raises with its name instead of feeding the model
    a short vector.
    """

    layout: Tuple[Tuple[str, Optional[str]], ...]
    key: str = "joint_pos"

    def __call__(self, data: DataDict) -> DataDict:
        pieces = []
        for field, name in self.layout:
            val = data.get(field)
            if name is not None:
                val = val.get(name) if isinstance(val, dict) else None
            if val is None:
                label = field if name is None else f"{field}[{name!r}]"
                raise ValueError(f"state_layout needs observation {label}, which is missing; "
                                 f"observation has {sorted(data)}")
            pieces.append(np.asarray(val, dtype=np.float32).reshape(-1))
        out = dict(data)
        out[self.key] = np.concatenate(pieces) if pieces else np.zeros(0, np.float32)
        return out


@dataclasses.dataclass(frozen=True)
class GripperToOpening(DataTransformFn):
    """Map gripper commands from a model's range to hiRobot's opening in [0, 1].

    hiRobot's ``Action.gripper[name]`` is an opening: 0 closed, 1 open. A model
    may be trained on another range (openpi LIBERO: -1 open, +1 closed).
    ``ranges[name] = (closed, open)`` gives the model's value for each end;
    the command is mapped linearly and clipped to [0, 1]. Names not in
    ``ranges`` pass through unchanged. Recurses into ``parts``.
    """

    ranges: Dict[str, Tuple[float, float]]

    def __call__(self, data: DataDict) -> DataDict:
        out = dict(data)
        if out.get("gripper"):
            grip = dict(out["gripper"])
            for name, (closed, opened) in self.ranges.items():
                if name in grip:
                    x = np.asarray(grip[name], dtype=np.float32)
                    grip[name] = np.clip((x - closed) / (opened - closed), 0.0, 1.0)
            out["gripper"] = grip
        if out.get("parts"):
            out["parts"] = {k: self(v) for k, v in out["parts"].items()}
        return out


def parse_gripper_range(ranges: Any) -> Dict[str, Tuple[float, float]]:
    """Validate ``{name: [closed, open]}`` (the model's value at each end)."""
    out: Dict[str, Tuple[float, float]] = {}
    for name, pair in (ranges or {}).items():
        closed, opened = (float(v) for v in pair)
        if closed == opened:
            raise ValueError(f"gripper_range[{name!r}]: closed and open are both {closed}")
        out[str(name)] = (closed, opened)
    return out
