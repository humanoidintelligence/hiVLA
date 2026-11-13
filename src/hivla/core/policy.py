"""The VLA policy: the preprocessing/postprocessing pipeline around a backend.

A :class:`VlaPolicy` is hiVLA's equivalent of openpi's ``Policy`` and GR00T's
``Gr00tPolicy``: it turns a decoded robot observation into model inputs, runs the
:class:`~hivla.runtimes.base.ModelBackend`, and turns raw model outputs back
into an action chunk the robot can execute. This is where the "fat server, thin
client" split lives — the client only ships a semi-raw observation.

The pipeline is::

    observation --input_transforms--> model inputs
    model inputs --backend.predict--> raw outputs
    raw outputs --output_transforms--> action chunk

Supporting a new model is a matter of supplying transforms and a backend, not
subclassing this policy.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from hivla.runtimes.base import ModelBackend
from hivla.core.transforms import DataTransformFn, compose


class VlaPolicy:
    """Preprocess → backend → postprocess, producing an action chunk.

    Args:
        backend: The loaded model backend (tensor-in, tensor-out).
        input_transforms: Transforms applied to the decoded observation before
            the backend (normalize, pad, tokenize, rename).
        output_transforms: Transforms applied to raw model outputs (unnormalize,
            slice back to the robot's real action dimension).
        action_horizon: Length of the predicted action chunk; surfaced in
            metadata so the client can size its execution horizon.
        default_mode: Control mode label attached to the chunk when the output
            transforms do not set one (e.g. ``"position"`` / ``"pd"``).
        metadata: Extra static metadata exposed via the ``/metadata`` handshake.
    """

    def __init__(
        self,
        backend: ModelBackend,
        input_transforms: Sequence[DataTransformFn] = (),
        output_transforms: Sequence[DataTransformFn] = (),
        action_horizon: Optional[int] = None,
        default_mode: str = "position",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._backend = backend
        self._input_transform = compose(input_transforms)
        self._output_transform = compose(output_transforms)
        self._action_horizon = action_horizon
        self._default_mode = default_mode
        self._metadata = dict(metadata or {})
        if action_horizon is not None:
            self._metadata.setdefault("action_horizon", action_horizon)
        # What the backend knows of itself: whether it keeps per-episode state
        # (so a client must reset first) and which observations it reads.
        self._metadata["stateful"] = bool(getattr(backend, "stateful", False))
        self._metadata["observe"] = getattr(backend, "observe_mode", "chunk")
        self._metadata.update(backend.metadata())

    @property
    def metadata(self) -> Dict[str, Any]:
        """Static descriptor returned by the ``/metadata`` handshake."""
        return self._metadata

    def infer(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full pipeline on a decoded observation.

        Returns an action-chunk dict with (at least) ``mode`` and
        ``joint_targets``; optionally ``kp`` / ``kd`` / ``ee_targets`` /
        ``gripper``. Arrays are plain ndarrays — the server encodes them.
        """
        inputs = self._input_transform(dict(observation))
        return self._decode(self._backend.predict(inputs))

    def reset(self) -> None:
        """Start a new episode: the backend drops its per-episode state."""
        self._backend.reset()

    def observe(self, observation: Dict[str, Any]) -> None:
        """Pass one observation through the input transforms to the backend, no chunk back."""
        self._backend.observe(self._input_transform(dict(observation)))

    def infer_batch(self, observations: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """Run the pipeline on a batch of observations.

        Input transforms are applied per sample, then the batch is dispatched to
        :meth:`ModelBackend.predict_batch` (one forward pass for backends that
        support it) and each raw output is decoded.
        """
        inputs = [self._input_transform(dict(obs)) for obs in observations]
        raws = self._backend.predict_batch(inputs)
        return [self._decode(raw) for raw in raws]

    def _decode(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        chunk = self._output_transform(dict(raw))
        if not chunk.get("parts"):
            chunk.setdefault("mode", self._default_mode)
        self._validate_chunk(chunk)
        return chunk

    def _validate_chunk(self, chunk: Dict[str, Any], label: str = "chunk") -> int:
        """Enforce hiRobot's ``ActionChunk`` rules before the chunk leaves the server.

        A chunk that breaks them would fail later in the client, or worse, be
        cut wrong while it is replayed. Checked here (``label`` names the part
        in errors), arrays normalised to ndarrays in place:

        * ``mode`` is one of :data:`~hivla.protocol.CONTROL_MODES`.
        * Per-step arrays have the horizon as their leading axis:
          ``joint_targets`` ``(T, n)``, ``ee_targets[name]`` ``(T, d)``,
          ``gripper[name]`` ``(T,)``, ``base_velocity`` / ``base_target``
          ``(T, 3)``, ``latent`` ``(T, D)``. ``kp`` / ``kd`` are 1-D, shared by
          every step. All per-step arrays have the same ``T``; if
          ``action_horizon`` is declared, ``T`` equals it (the
          continuous-window contract GR00T enforces).
        * ``joint_names`` is only for the non-joint modes, needs
          ``joint_targets`` and names each of its columns once.
        * ``parts`` (part name -> chunk, may nest): with it, every other
          field but ``extra`` stays empty, and every part has the same horizon.

        Returns the horizon ``T`` (0 for a chunk with nothing per step).
        """
        from hivla.protocol import CHUNK_FIELDS, CONTROL_MODES, JOINT_MODES

        parts = chunk.get("parts")
        if parts:
            used = [k for k in CHUNK_FIELDS if k not in ("parts", "extra") and _is_set(chunk.get(k))]
            if used:
                raise ValueError(f"{label} has parts, so it takes no other command fields; got {used}")
            horizons = {name: self._validate_chunk(sub, f"{label}.parts[{name!r}]")
                        for name, sub in parts.items()}
            if len(set(horizons.values())) > 1:
                raise ValueError(f"{label}: part chunks have different horizons: {horizons}")
            return next(iter(horizons.values()))

        mode = chunk.get("mode")
        mode = getattr(mode, "value", mode)
        if mode is not None and mode not in CONTROL_MODES:
            raise ValueError(f"{label} mode must be one of {CONTROL_MODES}, got {mode!r}")

        steps: Dict[str, int] = {}

        def per_step(name: str, val: Any, tail: tuple) -> np.ndarray:
            arr = np.asarray(val)
            want = "(horizon" + "".join(f", {t if t else 'n'}" for t in tail) + ")"
            if arr.ndim != 1 + len(tail) or any(t and arr.shape[1 + i] != t for i, t in enumerate(tail)):
                raise ValueError(f"{label} {name} must be {1 + len(tail)}-D {want} with a leading "
                                 f"horizon axis; "
                                 f"got shape {arr.shape}")
            self._check_horizon(f"{label} {name}", arr)
            steps[name] = arr.shape[0]
            return arr

        for key, tail in (("joint_targets", (None,)), ("base_velocity", (3,)),
                          ("base_target", (3,)), ("latent", (None,))):
            if chunk.get(key) is not None:
                chunk[key] = per_step(key, chunk[key], tail)
        for key, tail in (("ee_targets", (None,)), ("gripper", ())):
            for name, val in (chunk.get(key) or {}).items():
                chunk[key][name] = per_step(f"{key}[{name!r}]", val, tail)
        for key in ("kp", "kd"):
            if chunk.get(key) is not None:
                chunk[key] = np.asarray(chunk[key])
                if chunk[key].ndim != 1:
                    raise ValueError(f"{label} {key} is one gain per joint, shared by every step "
                                     f"(1-D); got shape {chunk[key].shape}")
        if len(set(steps.values())) > 1:
            raise ValueError(f"{label} arrays have different horizons: {steps}")

        names = chunk.get("joint_names")
        if names is not None:
            if mode in JOINT_MODES:
                raise ValueError(f"{label} joint_names is for the non-joint modes; {mode!r} takes "
                                 "one joint_targets column per joint")
            jt = chunk.get("joint_targets")
            if jt is None:
                raise ValueError(f"{label} joint_names needs joint_targets")
            if len(names) != jt.shape[1] or len(set(names)) != len(names):
                raise ValueError(f"{label} joint_names must name each of the {jt.shape[1]} "
                                 f"joint_targets columns once; got {list(names)}")
        return next(iter(steps.values()), 0)

    def _check_horizon(self, label: str, arr: np.ndarray) -> None:
        if self._action_horizon is not None and arr.shape[0] != self._action_horizon:
            raise ValueError(
                f"{label} horizon {arr.shape[0]} != declared action_horizon "
                f"{self._action_horizon}"
            )


def _is_set(value: Any) -> bool:
    """hiRobot's test for a command field that is in use (not None / empty / False)."""
    if value is None or value is False:
        return False
    if isinstance(value, (dict, list, tuple)) and not value:
        return False
    return True
