"""A backend that delegates to a model family's native inference entry point.

For model families whose own package already ships the full observation -> action
pipeline (openpi's ``Policy.infer``, GR00T's ``Gr00tPolicy.get_action``, starVLA's
``framework.predict_action``, lerobot's policy call), hiVLA does **not**
reimplement their tokenization/normalization. It reuses the family's native
transform chain and enters through a single inference interface — the same
principle RLinf documents for openpi ("keep the native transform chain, enter the
engine through its L1 inference interface").

:class:`NativePolicyBackend` wraps such a native callable (``obs dict -> action
chunk dict``) behind the standard :class:`ModelBackend` contract. The module and
factory are imported lazily so hiVLA itself has no dependency on the family
packages; a clear error is raised only when a native family is actually served
without its package installed.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Optional

from hivla.core.spec import ModelSpec
from hivla.families.base import FamilyAdapter, rename_transforms
from hivla.runtimes.base import ModelBackend


class NativePolicyBackend(ModelBackend):
    """Wrap a family-provided ``obs -> action chunk`` callable as a backend.

    Args:
        module: Dotted import path of the module holding the factory (e.g.
            ``openpi.policies.policy_config``).
        factory: Name of the factory callable in ``module`` that builds the
            policy (e.g. ``create_trained_policy``), called with ``**kwargs``.
        kwargs: Keyword arguments forwarded to the factory. ``spec`` is made
            available as a ``spec=`` kwarg if the factory declares it.
        call_method: Name of the method on the built policy to invoke for
            inference (default ``"infer"``); GR00T uses ``get_action``.
    """

    def __init__(
        self,
        module: str,
        factory: str,
        kwargs: Optional[Dict[str, Any]] = None,
        call_method: str = "infer",
    ) -> None:
        self._module_path = module
        self._factory_name = factory
        self._kwargs = dict(kwargs or {})
        self._call_method = call_method
        self._policy: Optional[Any] = None

    def load(self) -> None:
        try:
            module = importlib.import_module(self._module_path)
        except ImportError as exc:
            raise ImportError(
                f"Native family backend needs the '{self._module_path.split('.')[0]}' "
                "package; install it into the serving env to serve this family."
            ) from exc
        factory: Callable = getattr(module, self._factory_name)
        self._policy = factory(**self._kwargs)

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self._policy is None:
            raise RuntimeError("NativePolicyBackend.load() must be called before predict().")
        if callable(self._policy):
            # The factory returned the callable itself (e.g. a closure or a
            # policy with __call__); invoke it directly.
            out = self._policy(inputs)
        else:
            call = getattr(self._policy, self._call_method)
            out = call(inputs)
        return out if isinstance(out, dict) else {"joint_targets": out}

    def unload(self) -> None:
        self._policy = None


class NativeFamily(FamilyAdapter):
    """Delegate to a family's native ``obs -> action`` pipeline.

    Args:
        name: Registry key (also ``ModelSpec.family``).
        module: Dotted import path of the module holding the factory.
        factory: Name of the callable that builds the policy from ``**kwargs``.
        call_method: Inference method on the built policy.
        default_kwargs: Fixed kwargs merged into the factory call. ``checkpoint``
            is always forwarded from the spec; extra per-family kwargs are read
            from ``spec.extra["native_kwargs"]``.
        doc: One-line summary of the family's native entry point.
    """

    def __init__(
        self,
        name: str,
        module: str,
        factory: str,
        call_method: str = "infer",
        default_kwargs: Dict[str, Any] | None = None,
        doc: str = "",
    ) -> None:
        self.name = name
        self._module = module
        self._factory = factory
        self._call_method = call_method
        self._default_kwargs = dict(default_kwargs or {})
        self.__doc__ = doc

    def build_backend(self, spec: ModelSpec) -> ModelBackend:
        kwargs = dict(self._default_kwargs)
        kwargs.setdefault("checkpoint", spec.checkpoint)
        kwargs.update((spec.model_extra or {}).get("native_kwargs", {}))
        return NativePolicyBackend(
            self._module, self._factory, kwargs=kwargs, call_method=self._call_method
        )

    def build_transforms(self, spec: ModelSpec):
        # A native policy already normalizes/decodes; hiVLA only renames keys
        # between the robot schema and the family's expected schema.
        return rename_transforms(spec)
