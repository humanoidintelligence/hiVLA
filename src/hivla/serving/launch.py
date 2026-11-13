"""Start a model's serving worker in the model's own environment and wait until it is up.

hiVLA is installed into each model family's own env, so a
model has to be served by that env's interpreter. A config names it::

    launch:
      env: xpolicylab        # conda env name, a venv / conda prefix, or a python path
      ready_timeout: 600     # seconds to wait for /health (default 300)

:func:`launch` resolves the interpreter, checks hiVLA imports there, refuses a
port that is already taken (another server's ``/health`` must not pass for
ours), runs ``python -m hivla.cli serve --config <config>`` as a child and polls
``/health`` until it answers. A child that exits first raises
:class:`LaunchError` with the tail of its log.

This module stays light: standard library plus :mod:`hivla.core.config` (pyyaml).
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Union

DEFAULT_READY_TIMEOUT = 300.0


class LaunchError(RuntimeError):
    """The worker could not be started or did not become ready."""


def resolve_python(env: Optional[str]) -> str:
    """Return the Python executable for ``env``.

    ``env`` is, in order: ``None`` (this interpreter), a path to an executable,
    a directory holding ``bin/python`` (venv or conda prefix), or a conda env
    name looked up with ``conda env list``.
    """
    if not env:
        return sys.executable
    path = Path(os.path.expanduser(env))
    if path.is_file() and os.access(path, os.X_OK):
        return str(path)
    if path.is_dir():
        python = path / "bin" / "python"
        if python.is_file():
            return str(python)
        raise LaunchError(f"launch.env {env!r} is a directory without bin/python")
    if os.sep in env:
        raise LaunchError(f"launch.env {env!r} does not exist")
    for prefix in _conda_envs():
        if Path(prefix).name == env:
            python = Path(prefix) / "bin" / "python"
            if python.is_file():
                return str(python)
    raise LaunchError(f"no conda env named {env!r} (and it is not a path)")


def _conda_envs() -> list:
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda:
        return []
    try:
        out = subprocess.run([conda, "env", "list", "--json"], capture_output=True,
                             text=True, timeout=60, check=True)
    except (OSError, subprocess.SubprocessError):
        return []
    return list(json.loads(out.stdout).get("envs", []))


def _check_hivla(python: str) -> None:
    probe = subprocess.run([python, "-c", "import hivla.cli"], capture_output=True, text=True)
    if probe.returncode != 0:
        tail = probe.stderr.strip().splitlines()[-1:] or ["?"]
        raise LaunchError(f"hivla does not import in {python}: {tail[0]}  "
                          "(install it there: pip install -e '<hiVLA>[server]')")


def _probe_host(host: str) -> str:
    # A server bound to every interface is reached on loopback.
    return "127.0.0.1" if host in ("0.0.0.0", "", "::") else host


def _port_taken(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


class Worker:
    """A running ``hivla serve`` child. Stop it with :meth:`stop` (or use ``with``)."""

    def __init__(self, process: subprocess.Popen, url: str, log_path: Optional[str]) -> None:
        self.process = process
        self.url = url
        self.log_path = log_path

    @property
    def pid(self) -> int:
        return self.process.pid

    def alive(self) -> bool:
        return self.process.poll() is None

    def stop(self, timeout: float = 10.0) -> None:
        """SIGTERM, then SIGKILL after ``timeout`` seconds."""
        if not self.alive():
            return
        self.process.terminate()
        try:
            self.process.wait(timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

    def __enter__(self) -> "Worker":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def launch(config: Union[str, os.PathLike], *, env: Optional[str] = None,
           host: Optional[str] = None, port: Optional[int] = None,
           ready_timeout: Optional[float] = None,
           log: Union[None, str, os.PathLike, bool] = True) -> Worker:
    """Start ``hivla serve`` for ``config`` in the model's env; return once ``/health`` is ok.

    Args:
        config: Serving config path. Its ``launch`` and ``server`` sections
            give the defaults for the other arguments.
        env: Overrides ``launch.env``.
        host, port: Override ``server.host`` / ``server.port``.
        ready_timeout: Overrides ``launch.ready_timeout``.
        log: Where the worker's output goes: ``True`` (a temp file, path in
            :attr:`Worker.log_path`), a path, or ``False`` / ``None`` to
            inherit this process's stdout / stderr.
    """
    from hivla.core.config import load_config, resolve_config

    # The worker gets a file path: a built-in name resolves here, in this
    # install, so both processes read the same file.
    config = os.path.abspath(resolve_config(config))
    cfg = load_config(config)
    launch_cfg: Dict[str, Any] = cfg.get("launch") or {}
    server_cfg: Dict[str, Any] = cfg.get("server") or {}
    host = host or server_cfg.get("host", "0.0.0.0")
    port = int(port or server_cfg.get("port", 8000))
    timeout = float(ready_timeout or launch_cfg.get("ready_timeout", DEFAULT_READY_TIMEOUT))

    python = resolve_python(env or launch_cfg.get("env"))
    _check_hivla(python)
    probe = _probe_host(host)
    if _port_taken(probe, port):
        raise LaunchError(f"{probe}:{port} is already in use; stop that server or pick "
                          "another port")

    if log is True:
        fd, log_path = tempfile.mkstemp(prefix="hivla-worker-", suffix=".log")
        os.close(fd)
    elif log:
        log_path = os.fspath(log)
    else:
        log_path = None
    out = open(log_path, "ab") if log_path else None
    try:
        process = subprocess.Popen(
            [python, "-m", "hivla.cli", "serve", "--config", config,
             "--host", host, "--port", str(port)],
            stdout=out, stderr=subprocess.STDOUT if out else None)
    finally:
        if out:
            out.close()

    worker = Worker(process, f"http://{probe}:{port}", log_path)
    try:
        _wait_ready(worker, timeout)
    except BaseException:
        worker.stop()
        raise
    return worker


def _wait_ready(worker: Worker, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not worker.alive():
            raise LaunchError(f"worker exited with code {worker.process.returncode} before "
                              f"it was ready{_log_tail(worker.log_path)}")
        try:
            with urllib.request.urlopen(f"{worker.url}/health", timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.25)
    raise LaunchError(f"worker not ready after {timeout:.0f} s{_log_tail(worker.log_path)}")


def _log_tail(path: Optional[str], lines: int = 20) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "r", errors="replace") as f:
        tail = f.read().splitlines()[-lines:]
    return f"; log {path}:\n" + "\n".join(tail) if tail else f"; log {path} is empty"
