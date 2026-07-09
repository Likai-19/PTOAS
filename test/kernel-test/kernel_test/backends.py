"""Backend interfaces for the kernel-test framework."""

from __future__ import annotations

from typing import Literal, Protocol

RunPurpose = Literal["correctness", "cycle"]


class BackendAdapter(Protocol):
    """Stable interface shared by all framework backends."""

    name: str

    def is_supported(self, case: object, *, purpose: RunPurpose) -> tuple[bool, str | None]:
        """Return support status and an optional human-readable reason."""

    def launch(self, case: object, *, purpose: RunPurpose) -> object:
        """Launch one case and return backend-specific outputs."""
