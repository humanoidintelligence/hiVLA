"""hivla.serving.launch: start hivla serve in a given interpreter and wait for /health."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import urllib.request

import pytest

pytest.importorskip("uvicorn")
pytest.importorskip("fastapi")

from hivla import cli  # noqa: E402
from hivla.serving.launch import LaunchError, launch, resolve_python  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _config(tmp_path, port, extra=""):
    path = tmp_path / "echo.yaml"
    path.write_text(
        "model: {backend: echo, action_horizon: 4}\n"
        "policy: {action_dim: 3, control_mode: pd}\n"
        f"server: {{host: 127.0.0.1, port: {port}}}\n" + extra)
    return path


def test_launch_starts_worker_and_waits_for_health(tmp_path):
    port = _free_port()
    worker = launch(_config(tmp_path, port, f"launch: {{env: {sys.executable}}}\n"),
                    ready_timeout=60, log=tmp_path / "w.log")
    with worker:
        assert worker.url == f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(f"{worker.url}/metadata", timeout=5) as resp:
            assert json.load(resp)["action_horizon"] == 4
    assert not worker.alive()
    assert (tmp_path / "w.log").exists()


def test_launch_refuses_a_taken_port(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        with pytest.raises(LaunchError, match="already in use"):
            launch(_config(tmp_path, port), ready_timeout=5)


def test_launch_reports_a_worker_that_dies(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(f"model: {{family: no_such_family}}\nserver: {{port: {_free_port()}}}\n")
    with pytest.raises(LaunchError, match="exited with code") as err:
        launch(path, ready_timeout=60, log=tmp_path / "bad.log")
    assert "no_such_family" in str(err.value)  # log tail is included


def test_resolve_python_forms(tmp_path):
    assert resolve_python(None) == sys.executable
    assert resolve_python(sys.executable) == sys.executable
    prefix = tmp_path / "venv"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "python").symlink_to(sys.executable)
    assert resolve_python(str(prefix)) == str(prefix / "bin" / "python")
    with pytest.raises(LaunchError, match="without bin/python"):
        resolve_python(str(tmp_path))
    with pytest.raises(LaunchError, match="does not exist"):
        resolve_python(str(tmp_path / "missing" / "python"))
    with pytest.raises(LaunchError, match="no conda env"):
        resolve_python("hivla-no-such-env-xyz")


def test_launch_env_without_hivla(tmp_path):
    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\necho 'ModuleNotFoundError: No module named hivla' >&2\nexit 1\n")
    fake.chmod(0o755)
    with pytest.raises(LaunchError, match="does not import"):
        launch(_config(tmp_path, _free_port()), env=str(fake), ready_timeout=5)


def test_cli_launch_error_exits_1(tmp_path, capsys):
    code = cli.main(["launch", "--config", str(_config(tmp_path, _free_port())),
                     "--env", "hivla-no-such-env-xyz"])
    assert code == 1
    assert "no conda env" in capsys.readouterr().err


def test_launch_import_is_light():
    code = ("import sys, hivla.serving.launch; "
            "print(','.join(m for m in ('hivla.serving.engine', 'pydantic', 'fastapi', 'torch') "
            "if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True)
    assert out.stdout.strip() == ""
