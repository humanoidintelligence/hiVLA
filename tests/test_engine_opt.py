"""InferenceEngine optimizations: observation cache + dynamic batching."""

from __future__ import annotations

import threading
import time

import numpy as np

from hivla import InferenceEngine, ObservationCache, VlaPolicy
from hivla.runtimes.base import ModelBackend


class CountingBackend(ModelBackend):
    """Returns a chunk encoding the first joint position as an id; counts batches."""

    def __init__(self, action_dim=7, action_horizon=4):
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.predict_calls = 0
        self.batch_calls = 0
        self._lock = threading.Lock()

    def load(self):
        pass

    def predict(self, inputs):
        with self._lock:
            self.predict_calls += 1
        n = float(np.asarray(inputs["joint_pos"]).reshape(-1)[0])
        return {"joint_targets": np.full((self.action_horizon, self.action_dim), n, np.float32)}

    def predict_batch(self, batch):
        with self._lock:
            self.batch_calls += 1
        return [self.predict(x) for x in batch]


def _engine(backend, max_batch_size=1, cache=None, warmup=False):
    return InferenceEngine(VlaPolicy(backend, action_horizon=backend.action_horizon),
                           cache=cache, max_batch_size=max_batch_size, batch_timeout=0.05,
                           warmup=warmup)


class WarmupBackend(CountingBackend):
    """CountingBackend that runs one dummy predict in warmup()."""

    def warmup(self):
        self.predict({"joint_pos": np.zeros(self.action_dim, np.float32)})


def test_warmup_runs_one_forward_before_traffic():
    backend = WarmupBackend()
    engine = _engine(backend, warmup=True)
    engine.start()
    assert backend.predict_calls == 1  # warmup fired before any infer()
    engine.infer({"joint_pos": np.array([2.0] * 7, np.float32)})
    assert backend.predict_calls == 2


def test_warmup_skipped_when_disabled():
    backend = WarmupBackend()
    engine = _engine(backend, warmup=False)
    engine.start()
    assert backend.predict_calls == 0


def test_observation_cache_replays_identical_obs():
    backend = CountingBackend()
    engine = _engine(backend, cache=ObservationCache(ttl=10.0))
    engine.start()
    obs = {"joint_pos": np.array([1.0] * 7, np.float32)}
    first = engine.infer(obs)
    second = engine.infer(dict(obs))
    assert np.array_equal(first["joint_targets"], second["joint_targets"])
    assert backend.predict_calls == 1  # second was served from cache


def test_observation_cache_ttl_expiry():
    backend = CountingBackend()
    engine = _engine(backend, cache=ObservationCache(ttl=0.01))
    engine.start()
    obs = {"joint_pos": np.array([1.0] * 7, np.float32)}
    engine.infer(obs)
    time.sleep(0.05)
    engine.infer(obs)
    assert backend.predict_calls == 2  # cached entry expired


def test_dynamic_batching_coalesces_concurrent_requests():
    backend = CountingBackend()
    engine = _engine(backend, max_batch_size=4)
    engine.start()

    n = 8
    barrier = threading.Barrier(n)
    results = [None] * n

    def worker(i):
        barrier.wait()
        results[i] = engine.infer({"joint_pos": np.array([float(i)] * 7, np.float32)})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    for i, chunk in enumerate(results):
        assert chunk["joint_targets"][0, 0] == float(i)  # correct id returned
    assert backend.batch_calls < n  # at least one coalescing happened
