<!--
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
-->

# kernel-test scripts

This directory is split into two groups:

## User-facing entrypoints

- `run_cycle.sh`
  - Recommended command for cycle collection.
  - Expands selected cases, invokes `kernel-test/run.py` once per case, and prints a cycle report.
- `run_sim.sh`
  - Generic `cannsim` transport for one Python entrypoint.
  - Use this when you want to run a specific script under cannsim directly.
- `run_msprof.sh`
  - Generic `msprof` transport for one Python entrypoint.
  - Use this when you want direct `msprof` artifacts for a specific script.

## Internal helpers

- `helpers/common.sh`
- `helpers/run_sim_entry.sh`
- `helpers/report_cycles.py`

These helper scripts are framework internals. They are called by the user-facing
entrypoints above and are not intended to be run directly.
