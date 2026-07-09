"""Shared Python package for the kernel-test framework."""

from .backends import BackendAdapter
from .registry import KernelRegistry, OperatorSpec, RegistryError, load_registry
from .results import CaseResult, RunSummary

__all__ = [
    "BackendAdapter",
    "CaseResult",
    "KernelRegistry",
    "OperatorSpec",
    "RegistryError",
    "RunSummary",
    "__version__",
    "load_registry",
]

__version__ = "0.1.0"
