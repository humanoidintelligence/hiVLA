"""Build a serving stack (backend -> policy -> engine) from a :class:`ModelSpec`.

Maps a hiVLA config (see ``configs/example.yaml``) to a ready
:class:`~hivla.serving.engine.InferenceEngine`. A checkpoint is described once — which
model family, which backend runs it, what transforms wrap it, its norm stats and
horizon — and this assembles the pieces:

* :func:`build_engine` accepts a raw config dict (nested ``model``/``runtime``/
  ``policy``/``server`` sections or flat), converts it to a
  :class:`~hivla.core.spec.ModelSpec`, and delegates to :func:`build_engine_from_spec`.
* :func:`build_engine_from_spec` resolves the family adapter, which contributes
  the backend and the input/output transforms.
"""

from __future__ import annotations

from typing import Any, Dict

from hivla.serving.engine import InferenceEngine, ObservationCache
from hivla.families import get_adapter
from hivla.core.policy import VlaPolicy
from hivla.core.spec import ModelSpec
from hivla.core.transforms import (
    ConcatState,
    GripperToOpening,
    parse_gripper_range,
    parse_state_layout,
)


def build_engine_from_spec(spec: ModelSpec) -> InferenceEngine:
    """Assemble an :class:`InferenceEngine` from a :class:`ModelSpec`."""
    adapter = get_adapter(spec.family)
    backend = adapter.build_backend(spec)
    input_transforms, output_transforms = adapter.build_transforms(spec)
    # Family-independent edges, from hiRobot's conventions: build the state
    # vector from structured fields first, and map gripper commands to an
    # opening in [0, 1] last, after the family has decoded its actions.
    layout = parse_state_layout(spec.state_layout)
    if layout:
        input_transforms = [ConcatState(tuple(layout)), *input_transforms]
    ranges = parse_gripper_range(spec.gripper_range)
    if ranges:
        output_transforms = [*output_transforms, GripperToOpening(ranges)]

    policy = VlaPolicy(
        backend,
        input_transforms=input_transforms,
        output_transforms=output_transforms,
        action_horizon=spec.action_horizon,
        default_mode=spec.control_mode,
        metadata=spec.metadata(),
    )

    cache = ObservationCache(ttl=spec.cache_ttl) if spec.cache else None
    return InferenceEngine(
        policy,
        cache=cache,
        max_batch_size=spec.max_batch_size,
        batch_timeout=spec.batch_timeout,
        warmup=spec.warmup,
    )


def build_engine(config: Dict[str, Any]) -> InferenceEngine:
    """Assemble an :class:`InferenceEngine` from a parsed config dict."""
    return build_engine_from_spec(ModelSpec.from_config(config))
