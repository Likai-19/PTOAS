"""Case selection helpers for framework runners."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeVar

CaseT = TypeVar("CaseT")


def select_cases(
    cases: Mapping[str, CaseT],
    *,
    case_ids: Sequence[str],
    case_filter: str | None,
    require_single: bool = False,
) -> dict[str, CaseT]:
    """Resolve exact-case and substring filters into a concrete selection."""

    if case_ids:
        selected: dict[str, CaseT] = {}
        missing: list[str] = []
        for case_id in case_ids:
            if case_id not in cases:
                missing.append(case_id)
                continue
            selected[case_id] = cases[case_id]
    else:
        selected = {case_id: cases[case_id] for case_id in sorted(cases.keys())}
        missing = []

    if missing:
        raise ValueError(f"unknown case ids: {', '.join(missing)}")

    if case_filter:
        selected = {
            case_id: case
            for case_id, case in selected.items()
            if case_filter in case_id
        }

    if not selected:
        raise ValueError("no cases matched the requested selection")

    if require_single and len(selected) != 1:
        raise ValueError(
            f"cycle workflow requires exactly one case, but resolved {len(selected)} cases"
        )

    return selected
