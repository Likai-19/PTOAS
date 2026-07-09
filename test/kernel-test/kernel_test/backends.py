# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

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
