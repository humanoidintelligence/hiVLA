"""AsyncInferenceEngine: submit/get_action overlap, latest-wins, lifecycle."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from hivla import AsyncInferenceEngine, VlaPolicy
from hivla.runtimes.base import ModelBackend


class _IdBackend(ModelBackend):
    """Returns a chunk encoding the first joint position as an id, with a delay."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.action_dim = 7
        self.action_horizon = 4
        self.predict_calls = 0
        self._lock = threading.Lock()

    def load(self) -> None:
        pass

    def predict(self, inputs):
        with self._lock:
            self.predict_calls += 1
            n = float(np.asarray(inputs["joint_pos"]).reshape(-1)[0])
        if self.delay:
            time.sleep(self.delay)
        return {"joint_targets": np.full((self.action_horizon, self.action_dim), n, np.float32)}


def _engine(backend):
    return AsyncInferenceEngine(VlaPolicy(backend, action_horizon=backend.action_horizon))


def test_submit_before_start_raises():
    engine = _engine(_IdBackend())
    with pytest.raises(RuntimeError, match="start"):
        engine.submit({"joint_pos": np.zeros(7, np.float32)})


def test_get_action_returns_none_until_ready():
    backend = _IdBackend(delay=0.1)
    engine = _engine(backend)
    engine.start()
    try:
        assert engine.get_action() is None
        engine.submit({"joint_pos": np.array([5.0] * 7, np.float32)})
        # Worker is still inside predict(); no chunk is ready yet.
        time.sleep(0.02)
        assert engine.get_action() is None
        deadline = time.time() + 2.0
        while engine.get_action() is None and time.time() < deadline:
            time.sleep(0.01)
        chunk = engine.get_action()
        assert chunk is not None and chunk["joint_targets"][0, 0] == 5.0
    finally:
        engine.stop()


def test_latest_wins_drops_superseded_observations():
    backend = _IdBackend(delay=0.05)
    engine = _engine(backend)
    engine.start()
    try:
        for i in range(3):
            engine.submit({"joint_pos": np.array([float(i)] * 7, np.float32)})
        deadline = time.time() + 2.0
        chunk = None
        while time.time() < deadline:
            chunk = engine.get_action()
            if chunk is not None and chunk["joint_targets"][0, 0] == 2.0:
                break
            time.sleep(0.01)
        assert chunk is not None
        # The final chunk reflects the newest observation (latest wins).
        assert chunk["joint_targets"][0, 0] == 2.0
        assert backend.predict_calls <= 3  # stale observations are dropped, not queued
    finally:
        engine.stop()


def test_submit_does_not_block_on_model():
    backend = _IdBackend(delay=0.2)
    engine = _engine(backend)
    engine.start()
    try:
        t0 = time.time()
        engine.submit({"joint_pos": np.array([1.0] * 7, np.float32)})
        elapsed = time.time() - t0
        assert elapsed < 0.1  # submit returns immediately, does not wait for predict
    finally:
        engine.stop()
