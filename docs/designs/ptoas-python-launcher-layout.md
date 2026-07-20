// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

# PTOAS Python Launcher Layout Design

## Purpose

This document records the internal launcher contract for the Python-backed
`ptoas` command. The user-facing README should describe how to install and run
PTOAS, but it should not expose this implementation detail.

The wheel launcher intentionally uses a Python wrapper instead of packaging the
native `ptoas` executable as the console entry. This avoids auditwheel /
manylinux handling issues for native executables while still letting the wheel
ship the native compiler implementation as a shared library.

## Goals

- Keep wheel, build-tree, and install-tree startup deterministic.
- Resolve Python packages and runtime payloads from the current distribution
  layout only.
- Avoid falling back to unrelated source checkouts, editable installs, or
  ambient `PYTHONPATH` entries.
- Avoid loading duplicated LLVM/MLIR shared libraries inside one Python process.
- Keep `ptoas._launcher` importable temporarily as a compatibility shim, but do
  not treat it as a stable public API.

## Entry Model

The launcher has three layers:

```text
distribution-specific entry
  -> layout resolver
  -> ptoas._runtime_entry.launch()
       -> ctypes loads ptoas.so
       -> calls ptoas_entrypoint(argc, argv)
```

`ptoas._runtime_entry` is the shared execution layer. It should not guess which
distribution layout is active; callers pass a resolved `PTOASRuntimeLayout`.

## Wheel Layout

Wheel console scripts point to the top-level `ptoas_wheel_bootstrap:main`
module, not to `ptoas._launcher:main`.

The bootstrap module lives outside the `ptoas` package so that the console entry
can identify the installed wheel root before importing `ptoas`. This prevents a
polluted `PYTHONPATH` from resolving a shadow `ptoas` package from a source
checkout or another environment.

Wheel startup is a two-stage process:

1. Stage 1 resolves the wheel-owned package root and runtime root.
2. Stage 1 `execve`s the same console script with a sanitized environment.
3. Stage 2 imports the wheel-owned `ptoas._runtime_entry`.
4. Stage 2 calls the shared `ptoas_entrypoint` from
   `ptoas/_runtime/lib/ptoas.so`.

In stage 2, the environment is isolated:

- `PYTHONPATH` is set to the wheel site-packages root.
- `LD_LIBRARY_PATH` is set to `ptoas/_runtime/lib`.
- `DYLD_LIBRARY_PATH` is set to `ptoas/_runtime/lib`.

Wheel mode must not preload every shared library under `ptoas/_runtime/lib`
before loading `ptoas.so`. Auditwheel may rewrite `ptoas.so` dependencies to
hashed libraries under `<distribution>.libs`. If the launcher preloads the
unhashed runtime copies first and `ptoas.so` then loads the auditwheel-rewritten
copies, LLVM/MLIR command-line options can be registered twice in the same
process.

Therefore wheel mode lets the dynamic loader resolve `ptoas.so` dependencies
from the already-sanitized process environment and the auditwheel RPATH.

## Build-Tree Layout

The build-tree wrapper is generated for one build directory and resolves only
that tree's staged payload:

- wrapper: `<build>/tools/ptoas/ptoas`
- Python root: `<build>/python`
- runtime root: `<build>/runtime-staging`
- shared entry: `<build>/runtime-staging/lib/ptoas.so`

If the staged Python package or runtime payload is missing, the wrapper fails
with a layout error. It must not repair startup by searching unrelated Python
roots or source checkouts.

Build-tree mode prepends the tree-owned Python and library paths to the current
environment. Because changing `LD_LIBRARY_PATH` after process startup is not a
reliable way to affect subsequent `dlopen` resolution on glibc, build-tree mode
may preload runtime libraries before loading `ptoas.so`.

## Install-Tree Layout

The install-tree wrapper resolves only files installed under the same prefix:

- wrapper: `<prefix>/bin/ptoas`
- Python root: the installed PTOAS Python package root under that prefix
- runtime root: the installed PTOAS runtime payload under that prefix
- shared entry: `<runtime-root>/lib/ptoas.so`

As with build-tree mode, missing installed payloads are hard layout errors, not
signals to search another checkout or wheel.

Install-tree mode follows the same dynamic-library policy as build-tree mode:
it may preload runtime libraries before loading `ptoas.so` because the process
was not re-execed with a fully isolated library environment.

## Compatibility Shim

`ptoas._launcher` remains importable for a transition period and forwards to the
new launcher path. New code should invoke the `ptoas` command instead of
importing `ptoas._launcher` directly.

The shim exists only to keep older internal scripts from breaking immediately.
It should not grow new layout-resolution logic.

## Non-Goals

- Support installing `ptoas` and `ptoas-vmi` into the same Python environment.
  Those distributions share the same import package and console script and are
  intentionally mutually exclusive.
- Support arbitrary `PTOAS_PYTHON_ROOT` fallback search in normal launcher
  operation.
- Replace the wheel Python wrapper with a native executable console entry.
