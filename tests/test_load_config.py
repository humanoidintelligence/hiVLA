"""load_config expands ~ and environment variables in config string values."""

from __future__ import annotations

import os

import pytest
import yaml

from hivla import load_config


def _write(tmp_path, data):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def test_expands_var_with_default_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("HIVLA_MODELS", raising=False)
    cfg = _write(tmp_path, {"model": {"checkpoint": "${HIVLA_MODELS:-cache/models}/m"}})
    assert load_config(cfg)["model"]["checkpoint"] == "cache/models/m"


def test_env_overrides_default(tmp_path, monkeypatch):
    monkeypatch.setenv("HIVLA_MODELS", "/data/models")
    cfg = _write(tmp_path, {"model": {"checkpoint": "${HIVLA_MODELS:-cache/models}/m"}})
    assert load_config(cfg)["model"]["checkpoint"] == "/data/models/m"


def test_bare_var_forms(tmp_path, monkeypatch):
    monkeypatch.setenv("CKPT", "/w")
    cfg = _write(tmp_path, {"a": "$CKPT/x", "b": "${CKPT}/y"})
    out = load_config(cfg)
    assert out["a"] == "/w/x"
    assert out["b"] == "/w/y"


def test_unset_var_without_default_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("MISSING", raising=False)
    cfg = _write(tmp_path, {"model": {"checkpoint": "${MISSING}/x"}})
    with pytest.raises(KeyError, match="MISSING"):
        load_config(cfg)


def test_unset_bare_var_without_default_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("MISSING", raising=False)
    cfg = _write(tmp_path, {"a": "$MISSING/x"})
    with pytest.raises(KeyError, match="MISSING"):
        load_config(cfg)


def test_expands_user_home(tmp_path):
    cfg = _write(tmp_path, {"a": "~/models"})
    assert load_config(cfg)["a"] == os.path.join(os.path.expanduser("~"), "models")


def test_expands_nested_dicts_and_lists(tmp_path, monkeypatch):
    monkeypatch.setenv("ROOT", "/r")
    cfg = _write(tmp_path, {
        "native_kwargs": {"base_vlm": "${ROOT}/vlm", "cameras": ["$ROOT/cam"]},
        "port": 8000,
    })
    out = load_config(cfg)
    assert out["native_kwargs"]["base_vlm"] == "/r/vlm"
    assert out["native_kwargs"]["cameras"] == ["/r/cam"]
    assert out["port"] == 8000  # non-strings untouched


def test_builtin_name_resolves(monkeypatch):
    from hivla.core.config import builtin_configs, config_root, resolve_config

    assert "gr00t_n1d6_gr1" in builtin_configs()
    assert resolve_config("echo_example") == config_root() / "echo_example.yaml"
    assert resolve_config("echo_example.yaml") == config_root() / "echo_example.yaml"
    assert load_config("echo_example")["model"]["backend"] == "echo"


def test_path_wins_over_builtin(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "echo_example.yaml").write_text("model: {backend: torch}\n")
    assert load_config("echo_example.yaml")["model"]["backend"] == "torch"
    assert load_config("echo_example")["model"]["backend"] == "echo"


def test_unknown_config_lists_builtins():
    with pytest.raises(FileNotFoundError, match="echo_example"):
        load_config("no_such_config_xyz")


def test_cli_lists_configs(capsys):
    from hivla import cli

    assert cli.main(["configs"]) == 0
    assert "gr00t_n1d6_gr1" in capsys.readouterr().out.split()
