"""Result models shared by framework runners."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CaseResult:
    """Normalized result for one case execution."""

    ok: bool
    message: str
    skipped: bool = False


@dataclass(frozen=True)
class RunSummary:
    """Aggregate summary for one correctness run."""

    total: int
    passed: int
    failed: int
    skipped: int

    @property
    def all_passed(self) -> bool:
        return self.failed == 0
