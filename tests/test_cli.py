"""hivla serve takes host / port from the config unless the flags override them."""

from __future__ import annotations

import pytest

from hivla import cli


@pytest.fixture
def captured(monkeypatch, tmp_path):
    seen = {}
    import hivla.serving.server as server

    monkeypatch.setattr(server, "serve", lambda engine, **kw: seen.update(kw))
    path = tmp_path / "c.yaml"
    path.write_text("model: {backend: echo}\nserver: {host: 127.0.0.1, port: 8011}\n")
    return str(path), seen


def test_serve_uses_config_host_and_port(captured):
    path, seen = captured
    assert cli.main(["serve", "--config", path]) == 0
    assert (seen["host"], seen["port"]) == ("127.0.0.1", 8011)


def test_serve_flags_override_config(captured):
    path, seen = captured
    cli.main(["serve", "--config", path, "--host", "0.0.0.0", "--port", "9000"])
    assert (seen["host"], seen["port"]) == ("0.0.0.0", 9000)
