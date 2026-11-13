"""``import hivla.client`` must not pull in the model side or the server stack."""

from __future__ import annotations

import subprocess
import sys

HEAVY = ("hivla.runtimes", "hivla.families", "hivla.serving.engine", "hivla.core.spec",
         "pydantic", "torch", "fastapi", "yaml")


def _loaded_after(statement: str) -> list:
    code = (
        f"import sys; {statement}; "
        f"print(','.join(m for m in {HEAVY!r} if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return [m for m in out.stdout.strip().split(",") if m]


def test_client_import_is_light():
    assert _loaded_after("import hivla.client") == []


def test_protocol_import_is_light():
    assert _loaded_after("import hivla.protocol") == []


def test_top_level_names_still_import():
    loaded = _loaded_after("from hivla import build_engine, ModelSpec, load_config")
    assert "hivla.serving.engine" in loaded and "pydantic" in loaded
