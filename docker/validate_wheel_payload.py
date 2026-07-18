#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate the packaged PTOAS wheel payload and launcher contract."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


REQUIRED_FILES = {
    "ptoas/__init__.py",
    "ptoas/_launcher.py",
    "ptoas/_runtime_entry.py",
    "ptoas/_runtime/lib/ptoas.so",
}
FORBIDDEN_FILES = {
    "ptoas/_runtime/bin/ptoas",
}
ENTRYPOINT_SNIPPET = "ptoas=ptoas._launcher:main"
WHEEL_GLOB = "ptoas*.whl"


def _resolve_wheel(candidate: str) -> Path:
    path = Path(candidate)
    if path.is_file():
        return path
    if path.is_dir():
        wheels = sorted(path.glob(WHEEL_GLOB))
        if len(wheels) != 1:
            raise SystemExit(
                f"expected exactly one wheel in {path}, found {len(wheels)}"
            )
        return wheels[0]
    matches = sorted(path.parent.glob(path.name))
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one wheel matching {candidate!r}, found {len(matches)}"
        )
    return matches[0]


def validate_wheel_payload(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
        missing = sorted(REQUIRED_FILES - names)
        if missing:
            raise SystemExit(f"wheel is missing required payload files: {missing}")

        present_forbidden = sorted(FORBIDDEN_FILES & names)
        if present_forbidden:
            raise SystemExit(
                "wheel unexpectedly contains forbidden payload files: "
                f"{present_forbidden}"
            )

        entry_points_names = sorted(
            name
            for name in names
            if name.endswith(".dist-info/entry_points.txt")
        )
        if len(entry_points_names) != 1:
            raise SystemExit(
                "wheel must contain exactly one dist-info/entry_points.txt, "
                f"found {len(entry_points_names)}"
            )
        entry_points_name = entry_points_names[0]

        entry_points = zf.read(entry_points_name).decode("utf-8")
        if ENTRYPOINT_SNIPPET not in entry_points:
            raise SystemExit(
                "wheel entry points do not route ptoas through "
                "ptoas._launcher:main"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the packaged PTOAS wheel payload."
    )
    parser.add_argument(
        "wheel",
        help="Wheel file, directory containing exactly one ptoas*.whl, or glob.",
    )
    args = parser.parse_args()

    wheel = _resolve_wheel(args.wheel)
    validate_wheel_payload(wheel)
    print(f"validated wheel payload and launcher contract: {wheel.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
