"""Backend factory for the rope kernel."""

from __future__ import annotations

from kernel_test.backends import BackendAdapter


def create_backend(name: str) -> BackendAdapter:
    """Create one rope backend adapter from the local backend packages."""

    if name == "cce":
        try:
            from .cce.backend import RopeCceBackend
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "rope cce backend requires the rope runtime dependencies; use the "
                "`pto` conda environment or install numpy/torch/torch_npu first"
            ) from exc

        return RopeCceBackend()
    if name == "vmi":
        try:
            from .vmi.backend import RopeVmiBackend
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "rope vmi backend requires the rope runtime dependencies; use the "
                "`pto` conda environment or install ptodsl/torch/torch_npu first"
            ) from exc

        return RopeVmiBackend()
    if name == "mi":
        try:
            from .mi.backend import RopeMiBackend
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "rope mi backend requires the rope runtime dependencies; use the "
                "`pto` conda environment or install ptodsl/torch/torch_npu first"
            ) from exc

        return RopeMiBackend()
    raise ValueError(f"unknown rope backend: {name}")
