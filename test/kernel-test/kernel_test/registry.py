# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Kernel discovery and registry loading for kernel-test."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import importlib
import pkgutil

from .backends import BackendAdapter
from .results import CaseResult


class RegistryError(RuntimeError):
    """Raised when registry discovery fails."""


@dataclass(frozen=True)
class OperatorSpec:
    """Framework contract for one registered kernel operator."""

    name: str
    default_backend: str
    backend_names: tuple[str, ...]
    create_backend: Callable[[str], BackendAdapter]
    list_cases: Callable[[str], Mapping[str, object]]
    verify: Callable[[str, object, object], CaseResult]
    cycle_fields: Callable[[str, object, BackendAdapter], Mapping[str, object]]
    summary: str = ""


def _empty_verify(case_id: str, case: object, output: object) -> CaseResult:
    del case_id, case, output
    raise NotImplementedError("operator spec does not provide a verifier")


def _empty_cycle_fields(
    case_id: str,
    case: object,
    backend: BackendAdapter,
) -> Mapping[str, object]:
    del case_id, case, backend
    return {}


def make_operator_spec(
    *,
    name: str,
    default_backend: str = "cce",
    backend_names: Iterable[str] = ("cce",),
    create_backend: Callable[[str], BackendAdapter],
    list_cases: Callable[[str], Mapping[str, object]],
    verify: Callable[[str, object, object], CaseResult] = _empty_verify,
    cycle_fields: Callable[[str, object, BackendAdapter], Mapping[str, object]] = _empty_cycle_fields,
    summary: str = "",
) -> OperatorSpec:
    """Build a normalized operator spec with immutable backend names."""

    return OperatorSpec(
        name=name,
        default_backend=default_backend,
        backend_names=tuple(backend_names),
        create_backend=create_backend,
        list_cases=list_cases,
        verify=verify,
        cycle_fields=cycle_fields,
        summary=summary,
    )


class KernelRegistry:
    """In-memory mapping of kernel name to discovered spec."""

    def __init__(self) -> None:
        self._entries: dict[str, OperatorSpec] = {}

    def register(self, spec: OperatorSpec) -> None:
        existing = self._entries.get(spec.name)
        if existing is not None:
            raise RegistryError(f"duplicate kernel registration: {spec.name}")
        self._entries[spec.name] = spec

    def get(self, name: str) -> OperatorSpec | None:
        return self._entries.get(name)

    def list_names(self) -> list[str]:
        return sorted(self._entries.keys())


def _load_from_module(module_name: str, registry: KernelRegistry) -> None:
    module = importlib.import_module(module_name)

    if hasattr(module, "register"):
        module.register(registry)
        return

    if hasattr(module, "get_operator_spec"):
        registry.register(module.get_operator_spec())
        return

    if hasattr(module, "OPERATOR_SPEC"):
        registry.register(module.OPERATOR_SPEC)
        return

    if hasattr(module, "get_kernel_spec"):
        registry.register(module.get_kernel_spec())
        return

    if hasattr(module, "KERNEL_SPEC"):
        registry.register(module.KERNEL_SPEC)
        return

    raise RegistryError(
        f"kernel module {module_name!r} must expose register(), get_kernel_spec(), or KERNEL_SPEC"
    )


def load_registry() -> KernelRegistry:
    try:
        import kernels
    except ImportError as exc:
        raise RegistryError("failed to import kernel adapters package 'kernels'") from exc

    registry = KernelRegistry()
    prefix = f"{kernels.__name__}."

    for module_info in pkgutil.iter_modules(kernels.__path__, prefix):
        short_name = module_info.name.rsplit(".", 1)[-1]
        if short_name.startswith("_"):
            continue
        _load_from_module(module_info.name, registry)

    return registry
