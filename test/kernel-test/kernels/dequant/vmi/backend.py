# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Runtime VMI backend for dequant."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

from kernel_test.backends import RunPurpose
from kernel_test.npu_runtime import ensure_runtime, stream_ptr, sync

from .anti_mx_quant_tail_axis_vmi import dequant_vmi_bf16, dequant_vmi_f16, dequant_vmi_f32
from ..runtime import artifact_case_dir, prepare_compile_args, prepare_launch_args

_VMI_ROOT = Path(__file__).resolve().parent
_KERNEL_ROOT = _VMI_ROOT.parent
_GENERATED_DIR = _KERNEL_ROOT / "generated" / "dequant"
_REPO_ROOT = Path(__file__).resolve().parents[5]
_PTOAS_BIN = Path(
    os.environ.get(
        "PTOAS_BIN",
        str(_REPO_ROOT / "build" / "tools" / "ptoas" / "ptoas"),
    )
)
_COMPILED: dict[tuple[str, str, int, int], object] = {}


def _kernel_for_dst(dst_fmt: str):
    if dst_fmt == "f32":
        return dequant_vmi_f32
    if dst_fmt == "bf16":
        return dequant_vmi_bf16
    if dst_fmt == "f16":
        return dequant_vmi_f16
    raise ValueError(f"unsupported dst_fmt: {dst_fmt}")


def _prepare_runtime_kernel(case: dict) -> object:
    compile_args = prepare_compile_args(case)
    cache_key = (
        compile_args.src_fmt,
        compile_args.dst_fmt,
        compile_args.row_block_num,
        compile_args.col_block_num,
    )
    compiled = _COMPILED.get(cache_key)
    if compiled is None:
        compiled = _kernel_for_dst(compile_args.dst_fmt).compile(
            SRC_FMT=compile_args.src_fmt,
            ROW_BLOCK_NUM=compile_args.row_block_num,
            COL_BLOCK_NUM=compile_args.col_block_num,
        )
        _COMPILED[cache_key] = compiled
    return compiled


def _run_lowering(vmi_path: Path, mi_path: Path) -> str:
    ptoas_bin = _PTOAS_BIN
    if not ptoas_bin.is_file():
        fallback = _REPO_ROOT / "install" / "bin" / "ptoas"
        if fallback.is_file():
            ptoas_bin = fallback
        else:
            raise FileNotFoundError(f"ptoas not found: {_PTOAS_BIN}")

    result = subprocess.run(
        [
            str(ptoas_bin),
            "--pto-arch=a5",
            "--pto-backend=vpto",
            "--enable-vmi",
            "--pto-level=level3",
            "--emit-vpto",
            "-o",
            str(mi_path),
            str(vmi_path),
        ],
        cwd=_REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "failed to lower VMI artifact with ptoas:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return mi_path.read_text(encoding="utf-8")


class AntiMxQuantTailAxisVmiBackend:
    """Runtime VMI backend for dequant."""

    name = "vmi"

    def is_supported(self, case: object, *, purpose: RunPurpose) -> tuple[bool, str | None]:
        if purpose == "cycle":
            return False, "backend=vmi has correctness launch only; no cycle probe yet"
        supported = case["src_fmt"] in {"e4m3", "e5m2"} and case["dst_fmt"] in {"f32", "bf16", "f16"}
        if supported:
            return True, None
        return False, "backend=vmi not wired for this case"

    def launch(self, case: object, *, purpose: RunPurpose) -> object:
        if purpose != "correctness":
            raise ValueError(f"unsupported purpose for dequant vmi backend: {purpose}")

        ensure_runtime("dequant")
        compile_args = prepare_compile_args(case)
        launch_args = prepare_launch_args(case)
        compiled = _prepare_runtime_kernel(case)

        case_dir = artifact_case_dir(_GENERATED_DIR, case)
        case_dir.mkdir(parents=True, exist_ok=True)
        vmi_path = case_dir / "vmi.pto"
        mi_path = case_dir / "mi.pto"

        vmi_text = compiled.mlir_text()
        vmi_path.write_text(vmi_text, encoding="utf-8")
        mi_text = _run_lowering(vmi_path, mi_path)

        named_vmi_path = case_dir / f"{compile_args.named_stem}.vmi.pto"
        named_mi_path = case_dir / f"{compile_args.named_stem}.mi.pto"
        shutil.copyfile(vmi_path, named_vmi_path)
        shutil.copyfile(mi_path, named_mi_path)

        root_aliases: dict[str, str] = {}
        if compile_args.default_alias:
            _GENERATED_DIR.mkdir(parents=True, exist_ok=True)
            alias_targets = {
                "root_vmi": _GENERATED_DIR / "vmi.pto",
                "root_mi": _GENERATED_DIR / "mi.pto",
                "root_named_vmi": _GENERATED_DIR / f"{compile_args.named_stem}.vmi.pto",
                "root_named_mi": _GENERATED_DIR / f"{compile_args.named_stem}.mi.pto",
            }
            shutil.copyfile(vmi_path, alias_targets["root_vmi"])
            shutil.copyfile(mi_path, alias_targets["root_mi"])
            shutil.copyfile(vmi_path, alias_targets["root_named_vmi"])
            shutil.copyfile(mi_path, alias_targets["root_named_mi"])
            root_aliases = {name: str(path) for name, path in alias_targets.items()}

        compiled[1, stream_ptr()](
            launch_args.x.data_ptr(),
            launch_args.scale.data_ptr(),
            launch_args.y.data_ptr(),
        )
        sync()

        return {
            "y": launch_args.y,
            "case_dir": str(case_dir),
            "vmi_path": str(vmi_path),
            "mi_path": str(mi_path),
            "named_vmi_path": str(named_vmi_path),
            "named_mi_path": str(named_mi_path),
            "vmi_text": vmi_text,
            "mi_text": mi_text,
            "root_aliases": root_aliases,
        }

    def cache_tag(self) -> str:
        backend_py = _VMI_ROOT / "backend.py"
        kernel_py = _VMI_ROOT / "anti_mx_quant_tail_axis_vmi.py"
        return (
            f"vmi:{backend_py}:{os.path.getmtime(backend_py):.0f}:"
            f"{kernel_py}:{os.path.getmtime(kernel_py):.0f}"
        )
