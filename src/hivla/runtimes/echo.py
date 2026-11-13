"""A dependency-free echo backend for testing the serving path end to end.

:class:`EchoBackend` runs no model. It synthesizes a zero action chunk shaped to
a configured action dimension and horizon, so the transport, policy pipeline,
and hiRobot client can be exercised without a checkpoint or torch/onnx. Use it
in tests and for wiring up a new deployment before the real model is ready.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from hivla.runtimes.base import ModelBackend


class EchoBackend(ModelBackend):
    """Returns a zero ``(horizon, action_dim)`` action chunk for any input.

    Args:
        action_dim: Width of the synthesized action vector.
        action_horizon: Number of timesteps in the returned chunk.
    """

    def __init__(self, action_dim: int = 7, action_horizon: int = 16) -> None:
        self._action_dim = action_dim
        self._action_horizon = action_horizon

    def load(self) -> None:
        """Nothing to load."""

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"joint_targets": np.zeros((self._action_horizon, self._action_dim), dtype=np.float32)}
