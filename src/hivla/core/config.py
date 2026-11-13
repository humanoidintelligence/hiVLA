"""Reading hiVLA serving configs (YAML with environment expansion).

A config is named either by a path or by the name of a built-in config shipped
in the package (``src/hivla/configs/<name>.yaml``), e.g. ``gr00t_n1d6_gr1``.
"""

from __future__ import annotations

import importlib.resources
import os
import re
from pathlib import Path
from typing import List, Union

_SUFFIXES = (".yaml", ".yml")


def config_root() -> Path:
    """Directory of the built-in configs (inside the installed package)."""
    return Path(str(importlib.resources.files("hivla").joinpath("configs")))


def builtin_configs() -> List[str]:
    """Names of the built-in configs, sorted."""
    return sorted(p.stem for p in config_root().glob("*.yaml"))


def resolve_config(ref: Union[str, "os.PathLike[str]"]) -> Path:
    """The file a config reference names: an existing path, else a built-in name.

    Raises:
        FileNotFoundError: Neither a file nor a built-in config of that name.
    """
    ref = os.fspath(ref)
    path = Path(os.path.expanduser(ref))
    if path.is_file():
        return path
    name = ref[: -len(Path(ref).suffix)] if ref.endswith(_SUFFIXES) else ref
    if os.sep not in name and "/" not in name:
        builtin = config_root() / f"{name}.yaml"
        if builtin.is_file():
            return builtin
    raise FileNotFoundError(
        f"no config file {ref!r} and no built-in config of that name; "
        f"built-in: {', '.join(builtin_configs())}"
    )


def load_config(config_path: Union[str, "os.PathLike[str]"]) -> dict:
    """Load a hiVLA serving config from a YAML file or a built-in config name.

    String values are expanded against the environment so configs stay free of
    machine-specific absolute paths: ``~`` becomes the home directory, ``$VAR`` /
    ``${VAR}`` the environment variable, and ``${VAR:-default}`` the variable or
    ``default`` when it is unset. A reference to an unset variable with no default
    raises :class:`KeyError`, so a required path must be supplied explicitly
    rather than resolving to something unintended. Relative paths resolve against
    the process working directory as before.
    """
    import yaml

    with open(resolve_config(config_path), "r") as f:
        return _expand(yaml.safe_load(f) or {})


_VAR_RE = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}|\$([A-Za-z_][A-Za-z0-9_]*)"
)


def _expand(value):
    """Recursively expand ``~`` and environment variables in config strings."""
    if isinstance(value, str):
        def sub(match):
            name = match.group(1) or match.group(3)
            default = match.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise KeyError(
                f"config references unset environment variable ${{{name}}} "
                f"(no default); set it, or write ${{{name}:-<default>}}"
            )

        return os.path.expanduser(_VAR_RE.sub(sub, value))
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value
