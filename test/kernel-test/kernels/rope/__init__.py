"""Rope kernel adapter for the kernel-test framework."""

from __future__ import annotations

from kernel_test.registry import OperatorSpec, make_operator_spec

from .backends import create_backend
from .spec import cycle_fields, list_cases, verify_case


def get_operator_spec() -> OperatorSpec:
    """Return the rope operator registration for the shared registry."""

    return make_operator_spec(
        name="rope",
        default_backend="cce",
        backend_names=("cce", "mi", "vmi"),
        create_backend=create_backend,
        list_cases=list_cases,
        verify=verify_case,
        cycle_fields=cycle_fields,
        summary="VF sim rope kernel adapter",
    )
