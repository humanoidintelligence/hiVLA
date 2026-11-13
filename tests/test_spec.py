"""ModelSpec: nested/flat parsing, norm-stats resolution, metadata."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hivla.core.spec import ModelSpec


def test_from_config_nested():
    spec = ModelSpec.from_config({
        "model": {"name": "m", "family": "gr00t", "backend": "auto", "action_horizon": 16},
        "runtime": {"device": "cpu", "max_batch_size": 4, "cache": True},
        "policy": {"action_dim": 7, "control_mode": "pd", "cameras": ["head"]},
        "server": {"transport": "websocket", "port": 9000},
    })
    assert spec.family == "gr00t"
    assert spec.backend == "auto"
    assert spec.max_batch_size == 4
    assert spec.cache is True
    assert spec.control_mode == "pd"
    assert spec.transport == "websocket"


def test_from_config_flat_overrides_sections():
    spec = ModelSpec.from_config({
        "model": {"name": "nested"},
        "name": "flat",  # flat key wins
    })
    assert spec.name == "flat"


def test_metadata_handshake_fields():
    spec = ModelSpec.from_config({
        "model": {"name": "m", "action_horizon": 10},
        "policy": {"action_dim": 7, "control_mode": "pd", "cameras": ["head"]},
    })
    meta = spec.metadata()
    assert meta["action_horizon"] == 10
    assert meta["action_dim"] == 7
    assert meta["control_mode"] == "pd"
    assert meta["cameras"] == ["head"]


def test_load_norm_stats_inline():
    spec = ModelSpec(
        norm_stats={"state": {"mean": [0.0, 1.0], "std": [2.0, 3.0]}},
    )
    stats = spec.load_norm_stats()
    assert "state" in stats
    assert np.allclose(stats["state"].mean, [0.0, 1.0])


def test_load_norm_stats_from_file(tmp_path):
    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps({"state": {"min": [-1.0], "max": [1.0], "mode": "minmax"}}))
    spec = ModelSpec(norm_stats=str(path))
    stats = spec.load_norm_stats()
    assert stats["state"].resolve_mode() == "minmax"


def test_n_action_steps_defaults_to_action_horizon_in_metadata():
    spec = ModelSpec.from_config({"model": {"action_horizon": 10}})
    assert spec.n_action_steps is None
    assert spec.metadata()["n_action_steps"] == 10


def test_n_action_steps_from_config():
    spec = ModelSpec.from_config({"model": {"action_horizon": 10, "n_action_steps": 5}})
    assert spec.metadata()["n_action_steps"] == 5


@pytest.mark.parametrize("n", [0, 11])
def test_n_action_steps_out_of_range_rejected(n):
    with pytest.raises(ValueError, match="n_action_steps"):
        ModelSpec.from_config({"model": {"action_horizon": 10, "n_action_steps": n}})


@pytest.mark.parametrize("path, n", [("gr00t_n1d6_gr1", 8),
                                      ("openpi_pi05_libero", 5)])
def test_shipped_configs_set_n_action_steps(path, n, monkeypatch):
    from hivla import load_config

    for var in ("HIVLA_MODELS", "OPENPI_DATA_HOME"):
        monkeypatch.setenv(var, "/tmp")
    assert ModelSpec.from_config(load_config(path)).metadata()["n_action_steps"] == n
