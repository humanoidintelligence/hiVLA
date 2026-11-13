"""Model families: one module per family, each holding its adapter and backend.

A :class:`FamilyAdapter` (``families/base.py``) builds what is specific to one
model family: the backend that loads and runs its checkpoint, and the
transforms that map hiRobot's fields to the model's and back. hiVLA owns only
the family-agnostic machinery (wire codec, batching, caching, transport).

* ``generic``: hiVLA loads a raw checkpoint through a :mod:`hivla.runtimes`
  runtime (torch / ONNX / TensorRT / echo / remote) and normalizes itself.
* ``lerobot``, ``openpi``, ``gr00t``, ``starvla``, ``xpolicylab``, ``rlinf``:
  the model's own package runs the whole pipeline; the module next to the
  adapter (``families/<name>.py``) maps the fields.
* :class:`NativeFamily` (``families/native.py``): an adapter for any package
  whose factory returns an ``obs -> action`` policy.

Family packages are imported only when a backend loads, so importing this
module needs numpy and pydantic, none of the model stacks.
"""

from __future__ import annotations

from typing import Dict, List

from hivla.families.base import FamilyAdapter, rename_transforms
from hivla.families.generic import GenericFamily
from hivla.families.gr00t import Gr00tFamily
from hivla.families.lerobot import LerobotFamily
from hivla.families.native import NativeFamily
from hivla.families.openpi import OpenpiFamily
from hivla.families.rlinf import RlinfFamily
from hivla.families.starvla import StarvlaFamily
from hivla.families.xpolicylab import XPolicyLabFamily

__all__ = [
    "FamilyAdapter", "rename_transforms", "GenericFamily", "NativeFamily",
    "LerobotFamily", "OpenpiFamily", "Gr00tFamily", "StarvlaFamily",
    "XPolicyLabFamily", "RlinfFamily",
    "register_family", "get_adapter", "available_families",
]

_REGISTRY: Dict[str, FamilyAdapter] = {}


def register_family(adapter: FamilyAdapter) -> FamilyAdapter:
    _REGISTRY[adapter.name] = adapter
    return adapter


def get_adapter(family: str) -> FamilyAdapter:
    if family not in _REGISTRY:
        raise ValueError(
            f"unknown model family {family!r}; available: {list(_REGISTRY)}"
        )
    return _REGISTRY[family]


def available_families() -> List[str]:
    """Built-in family names in canonical registration order."""
    return list(_REGISTRY)


# --- built-in families (canonical order) ------------------------------------
register_family(GenericFamily())

register_family(LerobotFamily())
register_family(OpenpiFamily())
register_family(Gr00tFamily())
register_family(StarvlaFamily())
register_family(XPolicyLabFamily())
register_family(RlinfFamily())
