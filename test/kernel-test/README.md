<!--
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
-->

# kernel-test

Shared test framework skeleton for multi-kernel, multi-backend validation.

Current `rope` adapter validation assumes the shell is already running inside the
`pto` environment, because the copied rope implementation depends on packages that
are not available in the base environment.

## Layout

- `run.py`: unified CLI entry point
- `kernel_test/`: shared framework package
- `kernels/`: one directory per kernel, each discovered by the registry
  and free to keep local `cce/`, `vmi/`, and `mi/` backend directories
- `scripts/`: shell transport wrappers for cannsim and other external runners
  - see `scripts/README.md` for the entrypoint/helper split

## Usage

List registered kernels:

```bash
python kernel-test/run.py --list-ops
```

List cases for one kernel after its adapter is added:

```bash
python kernel-test/run.py --op rope --list-cases
```

Run direct Python correctness:

```bash
python kernel-test/run.py --op rope --workflow correctness --backend cce
```

Run one cycle case directly through Python:

```bash
python kernel-test/run.py --op rope --workflow cycle --backend cce --case f16_half
```

Run one cycle case through the generic cannsim transport:

```bash
kernel-test/scripts/run_sim.sh \
  --output kernel-test/sim_outputs/manual/rope-cce-f16-half \
  kernel-test/run.py \
  -- \
  --op rope --workflow cycle --backend cce --case f16_half
```

Run multiple cycle cases, optionally in parallel, and store outputs under
`kernel-test/sim_outputs/<op>/<backend>/<case>/`. The default cycle engine is
`msprof`, with `cannsim` still available as a fallback:

```bash
kernel-test/scripts/run_cycle.sh --op rope --backend cce --parallel-sim 1 --jobs 4
```

If you need a non-default interpreter for cannsim transport, `run_sim.sh` still
accepts `--python-cmd`.

`run_cycle.sh` now runs a kernel-local cycle analysis step after successful jobs.
For `rope`, the primary VF cycle number now comes from `msprof` `RVECEX` pipe
cycles when available; `trace`, `instr_log`, and coarse SoC cycles are reported
only as supporting references.

## Status

This directory currently contains the framework foundation from tasks 1 through 3:

- package structure
- registry-based kernel discovery
- unified CLI entry point
- shared runtime types
- case selection helpers
- correctness and cycle runners
- one self-contained rope kernel directory with `cce/`, `vmi/`, and `mi/` backends
- shared rope runtime that prepares launch arguments before backend execution

Simulator transport scripts now live under `scripts/`. `run_sim.sh` owns one
single-case cannsim run, while `run_cycle.sh` expands a kernel's selected case
set into one sim job per case.

The first concrete adapter lives under `kernels/rope/`. Future kernels should follow
the same per-kernel directory pattern so backend-specific files can stay local to the
kernel they belong to. Rope now keeps its copied CCE source directly under `cce/`,
its copied backend PTO files directly under `vmi/` and `mi/`, and its CCE build
configuration in `cce/CMakeLists.txt`.
