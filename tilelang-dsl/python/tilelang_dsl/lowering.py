# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Authoring-form VPTO lowering skeleton for TileLang DSL v1."""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass

from .semantic import (
    SemanticAlignStoreStmt,
    SemanticAlignType,
    SemanticAssignStmt,
    SemanticAttributeAccess,
    SemanticBinaryExpr,
    SemanticBindingRef,
    SemanticCallExpr,
    SemanticDmaConfigStmt,
    SemanticDmaUnaryConfigStmt,
    SemanticExpr,
    SemanticExprStmt,
    SemanticForStmt,
    SemanticGroupedMteDmaStmt,
    SemanticGetBufStmt,
    SemanticIndexCastExpr,
    SemanticIfStmt,
    SemanticIndexType,
    SemanticIfResult,
    SemanticKernel,
    SemanticLiteralExpr,
    SemanticMemBarStmt,
    SemanticLowLevelCopyStmt,
    SemanticMaskType,
    SemanticMetaType,
    SemanticPadValueType,
    SemanticPipeBarrierStmt,
    SemanticPredicateStoreStmt,
    SemanticPtrType,
    SemanticReturnStmt,
    SemanticRlsBufStmt,
    SemanticScalarStoreStmt,
    SemanticScalarType,
    SemanticSetCrossCoreStmt,
    SemanticSetFlagStmt,
    SemanticSetIntraBlockStmt,
    SemanticSetIntraCoreStmt,
    SemanticShapeType,
    SemanticStmt,
    SemanticVecscopeStmt,
    SemanticStrictVecscopeStmt,
    SemanticSubscriptAccess,
    SemanticSymbolExpr,
    SemanticTensorSliceExpr,
    SemanticTensorViewType,
    SemanticPartitionTensorViewType,
    SemanticTileType,
    SemanticType,
    SemanticTupleExpr,
    SemanticTupleType,
    SemanticVScatterStmt,
    SemanticVRegType,
    SemanticVectorType,
    SemanticVectorPairStoreStmt,
    SemanticVectorStoreStmt,
    SemanticWaitFlagDevStmt,
    SemanticWaitFlagStmt,
    SemanticWaitIntraCoreStmt,
)
from .types import (
    CompactMode,
    MaskPattern,
    PadValue,
    ScalarType,
    TileConfig,
    bytewidth,
    get_lanes,
    integer_bitwidth,
    integer_signedness,
    is_float_dtype,
    is_integer_dtype,
)


_I1_TYPE = SemanticScalarType(dtype=ScalarType("i1"))
_I32_TYPE = SemanticScalarType(dtype=ScalarType("i32"))
_I64_TYPE = SemanticScalarType(dtype=ScalarType("i64"))


def _signless_mov_pad_scalar_type(dtype: ScalarType) -> SemanticScalarType | None:
    bitwidth = integer_bitwidth(dtype)
    if bitwidth == 8:
        return SemanticScalarType(dtype=ScalarType("i8"))
    if bitwidth == 16:
        return SemanticScalarType(dtype=ScalarType("i16"))
    if bitwidth == 32:
        return SemanticScalarType(dtype=ScalarType("i32"))
    return None


def _format_symbol_name(symbol_name: str) -> str:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$.]*", symbol_name):
        return f"@{symbol_name}"
    escaped = symbol_name.replace("\\", "\\\\").replace('"', '\\"')
    return f'@"{escaped}"'


@dataclass(frozen=True)
class AuthoringModule:
    """Lowering result that owns authoring-form VPTO text emission."""

    kernel: SemanticKernel

    def render(self) -> str:
        kernel_text = _AuthoringRenderer(self.kernel).render()
        if not self.kernel.inline_helpers:
            return kernel_text

        base_lines = kernel_text.splitlines()
        module_close_index = max(
            (index for index, line in enumerate(base_lines) if line == "}"),
            default=-1,
        )
        if module_close_index < 0:
            return kernel_text

        merged_lines = base_lines[:module_close_index]
        for helper in self.kernel.inline_helpers:
            helper_lines = _extract_single_function_lines(
                _AuthoringRenderer(helper).render()
            )
            if not helper_lines:
                continue
            helper_lines[0] = _rewrite_inline_helper_attrs(helper_lines[0])
            merged_lines.extend(helper_lines)

        merged_lines.append("}")
        merged_lines.append("")
        return "\n".join(merged_lines)


def _extract_single_function_lines(rendered_text: str) -> list[str]:
    lines = rendered_text.splitlines()
    try:
        function_start = next(
            index for index, line in enumerate(lines) if line.lstrip().startswith("func.func ")
        )
    except StopIteration:
        return []
    module_close_index = max(
        (index for index, line in enumerate(lines) if line == "}"),
        default=-1,
    )
    if module_close_index <= function_start:
        return []
    return lines[function_start:module_close_index]


def _rewrite_inline_helper_attrs(function_line: str) -> str:
    helper_attr = "private "
    helper_marker_attr = "pto.tilelang.inline_proc"
    if function_line.lstrip().startswith("func.func "):
        after_keyword = function_line.lstrip()[len("func.func ") :]
        if not after_keyword.startswith(("private ", "public ")):
            function_line = function_line.replace("func.func ", f"func.func {helper_attr}", 1)
    if "pto.tilelang.instance" in function_line:
        return function_line.replace("pto.tilelang.instance", helper_marker_attr)
    if "attributes {" in function_line:
        return function_line
    if function_line.rstrip().endswith("{"):
        stripped = function_line.rstrip()
        if stripped.lstrip().startswith("func.func "):
            after_keyword = stripped.lstrip()[len("func.func ") :]
            if not after_keyword.startswith(("private ", "public ")):
                stripped = stripped.replace("func.func ", f"func.func {helper_attr}", 1)
        return stripped[:-1] + f" attributes {{ {helper_marker_attr} }} {{"
    return function_line


@dataclass(frozen=True)
class _RenderedValue:
    name: str
    type: SemanticType


@dataclass(frozen=True)
class _RenderedTextualType(SemanticType):
    text: str


class _AuthoringRenderer:
    def __init__(self, kernel: SemanticKernel):
        self.kernel = kernel
        self._constant_lines: list[str] = []
        self._constant_cache: dict[tuple[str, object], str] = {}
        self._castptr_cache: dict[tuple[str, str], str] = {}
        self._tile_memref_cache: dict[str, _RenderedValue] = {}
        self._tile_ptr_cache: dict[str, _RenderedValue] = {}
        self._tile_valid_dim_cache: dict[tuple[str, int], _RenderedValue] = {}
        self._used_tile_buffers = self._collect_used_tile_buffers(kernel.body)
        self._temp_counter = 0
        self._loop_counter = 0

    def render(self) -> str:
        parameter_list = ", ".join(
            f"{param.ssa_name}: {self._render_type(param.type)}"
            for param in self.kernel.parameters
            if param.kind != "tile_valid_shape" and self._should_materialize_function_boundary_type(param.type)
        )
        result_sig = ""
        if self.kernel.body and isinstance(self.kernel.body[-1], SemanticReturnStmt):
            return_value = self.kernel.body[-1].value
            if return_value is not None:
                result_sig = f" -> {self._render_type(return_value.type)}"
        env = {
            param.name: _RenderedValue(name=param.ssa_name, type=param.type)
            for param in self.kernel.parameters
            if param.kind != "tile_valid_shape"
        }
        entry_lines: list[str] = []
        for param in self.kernel.parameters:
            if param.kind != "tile":
                continue
            if param.name in self._used_tile_buffers:
                self._materialize_tile_memref(
                    env[param.name],
                    indent=4,
                    into=entry_lines,
                )
        body_lines = self._render_block(self.kernel.body, env, indent=4)

        lines = [
            f"// tilelang.target = {self.kernel.target}",
            f"// tilelang.op = {self.kernel.op}",
            f"// tilelang.dtypes = {self.kernel.dtype_signature}",
            f"// tilelang.verify = {self.kernel.verify_enabled}",
            f"// tilelang.advanced = {self.kernel.advanced_enabled}",
        ]
        for binding in self.kernel.tile_bindings:
            valid_shape = ""
            if binding.valid_shape is not None:
                valid_shape = f" valid_shape={self._format_shape_tuple(binding.valid_shape)}"
            lines.append(
                "// tilelang.specialize "
                f"{binding.name} shape={binding.shape} memory_space={binding.memory_space} "
                f"config={binding.config}{valid_shape}"
            )
        kernel_kind = "cube" if self.kernel.kernel_family == "cube" else "vector"
        function_attrs = (
            "attributes { "
            f"pto.entry, pto.tilelang.instance, pto.kernel_kind = #pto.kernel_kind<{kernel_kind}> "
            "}"
        )
        lines.append(f'module attributes {{pto.target_arch = "{self.kernel.target}"}} {{')
        lines.append(
            "  func.func "
            f"{_format_symbol_name(self.kernel.symbol_name)}({parameter_list}){result_sig} "
            f"{function_attrs} {{"
        )
        lines.extend(self._constant_lines)
        lines.extend(entry_lines)
        lines.extend(body_lines)
        lines.append("  }")
        lines.append("}")
        lines.append("")
        return "\n".join(lines)

    def _should_materialize_function_boundary_type(self, ty: SemanticType) -> bool:
        return not isinstance(ty, (SemanticMetaType, SemanticPadValueType))

    def _collect_used_tile_buffers(
        self,
        statements: tuple[SemanticStmt, ...],
    ) -> set[str]:
        used: set[str] = set()
        for stmt in statements:
            self._collect_used_tile_buffers_from_stmt(stmt, used)
        return used

    def _collect_used_tile_buffers_from_stmt(
        self,
        stmt: SemanticStmt,
        used: set[str],
    ) -> None:
        if isinstance(stmt, SemanticAssignStmt):
            self._collect_used_tile_buffers_from_expr(stmt.value, used)
            return
        if isinstance(stmt, SemanticExprStmt):
            self._collect_used_tile_buffers_from_expr(stmt.expr, used)
            return
        if isinstance(stmt, SemanticVectorStoreStmt):
            self._collect_used_tile_buffers_from_expr(stmt.value, used)
            self._record_tile_buffer_use(stmt.destination, used)
            for index in stmt.indices:
                self._collect_used_tile_buffers_from_expr(index, used)
            self._collect_used_tile_buffers_from_expr(stmt.mask, used)
            return
        if isinstance(stmt, SemanticVScatterStmt):
            self._collect_used_tile_buffers_from_expr(stmt.value, used)
            self._record_tile_buffer_use(stmt.destination, used)
            self._collect_used_tile_buffers_from_expr(stmt.offsets, used)
            self._collect_used_tile_buffers_from_expr(stmt.mask, used)
            return
        if isinstance(stmt, SemanticPredicateStoreStmt):
            self._collect_used_tile_buffers_from_expr(stmt.value, used)
            self._record_tile_buffer_use(stmt.destination, used)
            for index in stmt.indices:
                self._collect_used_tile_buffers_from_expr(index, used)
            self._collect_used_tile_buffers_from_expr(stmt.dist, used)
            return
        if isinstance(stmt, SemanticAlignStoreStmt):
            self._collect_used_tile_buffers_from_expr(stmt.value, used)
            self._record_tile_buffer_use(stmt.destination, used)
            for index in stmt.indices:
                self._collect_used_tile_buffers_from_expr(index, used)
            if stmt.offset is not None:
                self._collect_used_tile_buffers_from_expr(stmt.offset, used)
            return
        if isinstance(stmt, SemanticVecscopeStmt):
            for nested in stmt.body:
                self._collect_used_tile_buffers_from_stmt(nested, used)
            return
        if isinstance(stmt, SemanticStrictVecscopeStmt):
            for capture in stmt.captures:
                self._record_tile_buffer_use(capture, used)
                self._collect_used_tile_buffers_from_expr(capture, used)
            for nested in stmt.body:
                self._collect_used_tile_buffers_from_stmt(nested, used)
            return
        if isinstance(stmt, SemanticForStmt):
            self._collect_used_tile_buffers_from_expr(stmt.lower_bound, used)
            self._collect_used_tile_buffers_from_expr(stmt.upper_bound, used)
            self._collect_used_tile_buffers_from_expr(stmt.step, used)
            for nested in stmt.body:
                self._collect_used_tile_buffers_from_stmt(nested, used)
            return
        if isinstance(stmt, SemanticIfStmt):
            self._collect_used_tile_buffers_from_expr(stmt.condition, used)
            for nested in stmt.then_body:
                self._collect_used_tile_buffers_from_stmt(nested, used)
            for nested in stmt.else_body:
                self._collect_used_tile_buffers_from_stmt(nested, used)
            return
        if isinstance(stmt, SemanticReturnStmt) and stmt.value is not None:
            self._collect_used_tile_buffers_from_expr(stmt.value, used)

    def _collect_used_tile_buffers_from_expr(
        self,
        expr: SemanticExpr,
        used: set[str],
    ) -> None:
        if isinstance(expr, SemanticCallExpr):
            if expr.namespace == "pto" and expr.name in {"vlds", "vldas", "vldus"} and expr.args:
                self._record_tile_buffer_use(expr.args[0], used)
            for arg in expr.args:
                self._collect_used_tile_buffers_from_expr(arg, used)
            return
        if isinstance(expr, SemanticBinaryExpr):
            self._collect_used_tile_buffers_from_expr(expr.lhs, used)
            self._collect_used_tile_buffers_from_expr(expr.rhs, used)
            return
        if isinstance(expr, SemanticTupleExpr):
            for element in expr.elements:
                self._collect_used_tile_buffers_from_expr(element, used)
            return
        if isinstance(expr, SemanticTensorSliceExpr):
            self._collect_used_tile_buffers_from_expr(expr.base, used)
            for slice_expr in expr.slices:
                if slice_expr.start is not None:
                    self._collect_used_tile_buffers_from_expr(slice_expr.start, used)
                if slice_expr.stop is not None:
                    self._collect_used_tile_buffers_from_expr(slice_expr.stop, used)
                if slice_expr.step is not None:
                    self._collect_used_tile_buffers_from_expr(slice_expr.step, used)
            return
        if isinstance(expr, SemanticAttributeAccess):
            if expr.attr not in {"shape", "valid_shape", "strides", "element_type"}:
                self._collect_used_tile_buffers_from_expr(expr.base, used)
            return
        if isinstance(expr, SemanticSubscriptAccess):
            self._collect_used_tile_buffers_from_expr(expr.base, used)
            self._collect_used_tile_buffers_from_expr(expr.index, used)

    def _record_tile_buffer_use(
        self,
        expr: SemanticExpr,
        used: set[str],
    ) -> None:
        if isinstance(expr, SemanticBindingRef) and isinstance(expr.type, SemanticTileType):
            used.add(expr.binding.name)

    def _render_block(
        self,
        statements: tuple[SemanticStmt, ...],
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        for stmt in statements:
            lines.extend(self._render_stmt(stmt, env, indent=indent))
        return lines

    def _render_stmt(
        self,
        stmt: SemanticStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        if isinstance(stmt, SemanticAssignStmt):
            return self._render_assign(stmt, env, indent=indent)
        if isinstance(stmt, SemanticExprStmt):
            lines: list[str] = []
            self._lower_expr(stmt.expr, env, indent=indent, into=lines)
            return lines
        if isinstance(stmt, SemanticVectorStoreStmt):
            return self._render_vector_store(stmt, env, indent=indent)
        if isinstance(stmt, SemanticVScatterStmt):
            return self._render_vscatter(stmt, env, indent=indent)
        if isinstance(stmt, SemanticVectorPairStoreStmt):
            return self._render_vector_pair_store(stmt, env, indent=indent)
        if isinstance(stmt, SemanticPredicateStoreStmt):
            return self._render_predicate_store(stmt, env, indent=indent)
        if isinstance(stmt, SemanticAlignStoreStmt):
            return self._render_align_store(stmt, env, indent=indent)
        if isinstance(stmt, SemanticScalarStoreStmt):
            return self._render_scalar_store(stmt, env, indent=indent)
        if isinstance(stmt, SemanticSetFlagStmt):
            return [
                self._indent(indent)
                + f'pto.set_flag["{stmt.src_pipe}", "{stmt.dst_pipe}", "{stmt.event}"]'
            ]
        if isinstance(stmt, SemanticWaitFlagStmt):
            return [
                self._indent(indent)
                + f'pto.wait_flag["{stmt.src_pipe}", "{stmt.dst_pipe}", "{stmt.event}"]'
            ]
        if isinstance(stmt, SemanticPipeBarrierStmt):
            return [self._indent(indent) + f"pto.barrier #pto.pipe<{stmt.pipe}>"]
        if isinstance(stmt, SemanticGetBufStmt):
            return self._render_buffer_sync_stmt("get_buf", stmt.pipe, stmt.buf_id, stmt.mode, env, indent=indent)
        if isinstance(stmt, SemanticRlsBufStmt):
            return self._render_buffer_sync_stmt("rls_buf", stmt.pipe, stmt.buf_id, stmt.mode, env, indent=indent)
        if isinstance(stmt, SemanticMemBarStmt):
            return [self._indent(indent) + f'pto.mem_bar "{stmt.barrier_type}"']
        if isinstance(stmt, SemanticSetCrossCoreStmt):
            return self._render_i64_pair_stmt("set_cross_core", stmt.core_id, stmt.event_id, env, indent=indent)
        if isinstance(stmt, SemanticSetIntraBlockStmt):
            return self._render_i64_pair_stmt("set_intra_block", stmt.block_id, stmt.event_id, env, indent=indent)
        if isinstance(stmt, SemanticSetIntraCoreStmt):
            return self._render_i32_stmt("set_intra_core", stmt.config, env, indent=indent)
        if isinstance(stmt, SemanticWaitFlagDevStmt):
            return self._render_i64_pair_stmt("wait_flag_dev", stmt.core_id, stmt.event_id, env, indent=indent)
        if isinstance(stmt, SemanticWaitIntraCoreStmt):
            return self._render_i64_pair_stmt("wait_intra_core", stmt.block_id, stmt.event_id, env, indent=indent)
        if isinstance(stmt, SemanticDmaUnaryConfigStmt):
            return self._render_dma_unary_config(stmt, env, indent=indent)
        if isinstance(stmt, SemanticDmaConfigStmt):
            return self._render_dma_config(stmt, env, indent=indent)
        if isinstance(stmt, SemanticLowLevelCopyStmt):
            return self._render_low_level_copy(stmt, env, indent=indent)
        if isinstance(stmt, SemanticGroupedMteDmaStmt):
            return self._render_grouped_mte_dma(stmt, env, indent=indent)
        if isinstance(stmt, SemanticReturnStmt):
            lines: list[str] = []
            if stmt.value is None:
                return [self._indent(indent) + "return"]
            value = self._lower_expr(stmt.value, env, indent=indent, into=lines)
            lines.append(self._indent(indent) + f"return {value.name} : {self._render_type(value.type)}")
            return lines
        if isinstance(stmt, SemanticVecscopeStmt):
            return self._render_vecscope(stmt, env, indent=indent)
        if isinstance(stmt, SemanticStrictVecscopeStmt):
            return self._render_strict_vecscope(stmt, env, indent=indent)
        if isinstance(stmt, SemanticForStmt):
            return self._render_for(stmt, env, indent=indent)
        if isinstance(stmt, SemanticIfStmt):
            return self._render_if(stmt, env, indent=indent)
        raise ValueError(f"unsupported semantic statement {type(stmt).__name__}")

    def _render_dma_config(
        self,
        stmt: SemanticDmaConfigStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        first = self._lower_to_i64(stmt.first, env, indent=indent, into=lines)
        second = self._lower_to_i64(stmt.second, env, indent=indent, into=lines)
        lines.append(
            self._indent(indent)
            + f"pto.{stmt.name} {first.name}, {second.name} : i64, i64"
        )
        return lines

    def _render_dma_unary_config(
        self,
        stmt: SemanticDmaUnaryConfigStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        value = self._lower_expr(stmt.value, env, indent=indent, into=lines)
        if (
            stmt.name == "set_mov_pad_val"
            and isinstance(value.type, SemanticScalarType)
            and is_integer_dtype(value.type.dtype)
        ):
            signless_type = _signless_mov_pad_scalar_type(value.type.dtype)
            if signless_type is not None:
                value = self._coerce_rendered_value(
                    value,
                    signless_type,
                    indent=indent,
                    into=lines,
                )
        lines.append(
            self._indent(indent)
            + f"pto.{stmt.name} {value.name} : {self._render_type(value.type)}"
        )
        return lines

    def _render_buffer_sync_stmt(
        self,
        name: str,
        pipe: str,
        buf_id: SemanticExpr,
        mode: SemanticExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        rendered_buf_id = self._lower_to_i64(buf_id, env, indent=indent, into=lines)
        rendered_mode = self._lower_to_i64(mode, env, indent=indent, into=lines)
        lines.append(
            self._indent(indent)
            + f'pto.{name} "{pipe}", {rendered_buf_id.name}, {rendered_mode.name} : i64, i64'
        )
        return lines

    def _render_i64_pair_stmt(
        self,
        name: str,
        first: SemanticExpr,
        second: SemanticExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        rendered_first = self._lower_to_i64(first, env, indent=indent, into=lines)
        rendered_second = self._lower_to_i64(second, env, indent=indent, into=lines)
        lines.append(
            self._indent(indent)
            + f"pto.{name} {rendered_first.name}, {rendered_second.name} : i64, i64"
        )
        return lines

    def _render_i32_stmt(
        self,
        name: str,
        value: SemanticExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        rendered_value = self._lower_to_i32(value, env, indent=indent, into=lines)
        lines.append(
            self._indent(indent)
            + f"pto.{name} {rendered_value.name} : i32"
        )
        return lines

    def _render_low_level_copy(
        self,
        stmt: SemanticLowLevelCopyStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        source = self._lower_expr(stmt.source, env, indent=indent, into=lines)
        destination = self._lower_expr(stmt.destination, env, indent=indent, into=lines)

        rendered_operands = []
        rendered_types = []
        for index, operand in enumerate(stmt.operands):
            if stmt.name == "copy_gm_to_ubuf" and index == 5:
                lowered = self._lower_to_i1(operand, env, indent=indent, into=lines)
            else:
                lowered = self._lower_to_i64(operand, env, indent=indent, into=lines)
            rendered_operands.append(lowered.name)
            rendered_types.append(self._render_type(lowered.type))

        operand_text = ", ".join([source.name, destination.name, *rendered_operands])
        type_text = ", ".join(
            [self._render_type(source.type), self._render_type(destination.type), *rendered_types]
        )
        lines.append(
            self._indent(indent)
            + f"pto.{stmt.name} {operand_text} : {type_text}"
        )
        return lines

    def _render_grouped_mte_dma(
        self,
        stmt: SemanticGroupedMteDmaStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        expr = SemanticCallExpr(namespace="pto", name=stmt.name, args=stmt.args, type=None)
        if stmt.name == "mte_gm_ub":
            self._render_mte_gm_ub(expr, env, indent=indent, into=lines)
            return lines
        if stmt.name == "mte_ub_gm":
            self._render_mte_ub_gm(expr, env, indent=indent, into=lines)
            return lines
        if stmt.name in {"mte_ub_ub", "mte_ub_l1"}:
            self._render_mte_ub_copy(expr, env, indent=indent, into=lines)
            return lines
        raise ValueError(f"unsupported grouped MTE DMA stmt pto.{stmt.name}")

    def _render_assign(
        self,
        stmt: SemanticAssignStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        if len(stmt.targets) != 1:
            if isinstance(stmt.value, SemanticTupleExpr) or (
                isinstance(stmt.value, SemanticAttributeAccess)
                and isinstance(stmt.value.type, SemanticShapeType)
            ):
                return self._render_tuple_expr_assign(stmt, env, indent=indent)
            return self._render_multi_result_assign(stmt, env, indent=indent)
        target = stmt.targets[0]
        if isinstance(target.type, (SemanticMetaType, SemanticPadValueType)):
            env[target.name] = _RenderedValue(name=target.ssa_name, type=target.type)
            return []
        lines: list[str] = []
        lowered = self._lower_expr(
            stmt.value,
            env,
            indent=indent,
            desired_name=target.ssa_name,
            into=lines,
        )
        env[target.name] = lowered
        return lines

    def _render_tuple_expr_assign(
        self,
        stmt: SemanticAssignStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        if isinstance(stmt.value, SemanticTupleExpr):
            elements = stmt.value.elements
        elif isinstance(stmt.value, SemanticAttributeAccess) and isinstance(stmt.value.type, SemanticShapeType):
            elements = tuple(
                SemanticSubscriptAccess(
                    base=stmt.value,
                    index=SemanticLiteralExpr(value=axis, type=SemanticIndexType()),
                    type=SemanticIndexType(),
                )
                for axis in range(stmt.value.type.rank)
            )
        else:
            raise NotImplementedError(
                "tuple expression assignment expects a SemanticTupleExpr or shape-like attribute value"
            )
        if len(stmt.targets) != len(elements):
            raise NotImplementedError("tuple expression assignment arity mismatch")

        lines: list[str] = []
        for target, element in zip(stmt.targets, elements):
            lowered = self._lower_expr(
                element,
                env,
                indent=indent,
                desired_name=target.ssa_name,
                into=lines,
            )
            env[target.name] = lowered
        return lines

    def _render_multi_result_assign(
        self,
        stmt: SemanticAssignStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        if not isinstance(stmt.value, SemanticCallExpr):
            raise NotImplementedError("multi-result assignment expects a call expression in TileLang DSL v1")
        if stmt.value.namespace != "pto":
            raise NotImplementedError(
                f"multi-result assignment for `pto.{stmt.value.name}` is not supported in TileLang DSL v1"
            )
        if not isinstance(stmt.value.type, SemanticTupleType):
            raise NotImplementedError("multi-result lowering expects a tuple result type")
        if len(stmt.targets) != len(stmt.value.type.elements):
            raise NotImplementedError("multi-result lowering expects assignment arity to match result arity")

        if stmt.value.name in {"make_mask", "plt_b8", "plt_b16", "plt_b32"}:
            if len(stmt.targets) != 2 or len(stmt.value.type.elements) != 2:
                raise NotImplementedError("mask multi-result lowering expects exactly two results")
            lines: list[str] = []
            if stmt.value.name == "make_mask":
                dtype_expr, remaining_expr = stmt.value.args
                if not self._is_dtype_meta_expr(dtype_expr):
                    raise NotImplementedError("make_mask dtype lowering expects a dtype symbol")
                remaining = self._lower_remaining_to_i32(remaining_expr, env, indent=indent, into=lines)
                op_name = None
            else:
                remaining = self._lower_remaining_to_i32(stmt.value.args[0], env, indent=indent, into=lines)
                op_name = stmt.value.name
            mask_target, remaining_target = stmt.targets
            mask_type, remaining_type = stmt.value.type.elements
            suffix = self._mask_suffix(mask_type)
            lowered_op = op_name or f"plt_{suffix}"
            lines.append(
                self._indent(indent)
                + f"{mask_target.ssa_name}, {remaining_target.ssa_name} = pto.{lowered_op} {remaining.name} : "
                + f"i32 -> {self._render_type(mask_type)}, {self._render_type(remaining_type)}"
            )
            env[mask_target.name] = _RenderedValue(name=mask_target.ssa_name, type=mask_type)
            env[remaining_target.name] = _RenderedValue(name=remaining_target.ssa_name, type=remaining_type)
            return lines

        if stmt.value.name in {"vaddc", "vsubc"}:
            lines = []
            lhs = self._lower_expr(stmt.value.args[0], env, indent=indent, into=lines)
            rhs = self._lower_expr(stmt.value.args[1], env, indent=indent, into=lines)
            mask = self._lower_expr(stmt.value.args[2], env, indent=indent, into=lines)
            result_target, carry_target = stmt.targets
            result_type, carry_type = stmt.value.type.elements
            lines.append(
                self._indent(indent)
                + f"{result_target.ssa_name}, {carry_target.ssa_name} = pto.{stmt.value.name} "
                + f"{lhs.name}, {rhs.name}, {mask.name} : "
                + f"{self._render_type(lhs.type)}, {self._render_type(rhs.type)}, {self._render_type(mask.type)} "
                + f"-> {self._render_type(result_type)}, {self._render_type(carry_type)}"
            )
            env[result_target.name] = _RenderedValue(name=result_target.ssa_name, type=result_type)
            env[carry_target.name] = _RenderedValue(name=carry_target.ssa_name, type=carry_type)
            return lines

        if stmt.value.name in {"vaddcs", "vsubcs"}:
            lines = []
            lhs = self._lower_expr(stmt.value.args[0], env, indent=indent, into=lines)
            rhs = self._lower_expr(stmt.value.args[1], env, indent=indent, into=lines)
            carry_in = self._lower_expr(stmt.value.args[2], env, indent=indent, into=lines)
            mask = self._lower_expr(stmt.value.args[3], env, indent=indent, into=lines)
            result_target, carry_target = stmt.targets
            result_type, carry_type = stmt.value.type.elements
            lines.append(
                self._indent(indent)
                + f"{result_target.ssa_name}, {carry_target.ssa_name} = pto.{stmt.value.name} "
                + f"{lhs.name}, {rhs.name}, {carry_in.name}, {mask.name} : "
                + f"{self._render_type(lhs.type)}, {self._render_type(rhs.type)}, "
                + f"{self._render_type(carry_in.type)}, {self._render_type(mask.type)} "
                + f"-> {self._render_type(result_type)}, {self._render_type(carry_type)}"
            )
            env[result_target.name] = _RenderedValue(name=result_target.ssa_name, type=result_type)
            env[carry_target.name] = _RenderedValue(name=carry_target.ssa_name, type=carry_type)
            return lines

        if stmt.value.name in {"vintlv", "vdintlv"}:
            lines = []
            lhs = self._lower_expr(stmt.value.args[0], env, indent=indent, into=lines)
            rhs = self._lower_expr(stmt.value.args[1], env, indent=indent, into=lines)
            low_target, high_target = stmt.targets
            low_type, high_type = stmt.value.type.elements
            lines.append(
                self._indent(indent)
                + f"{low_target.ssa_name}, {high_target.ssa_name} = pto.{stmt.value.name} "
                + f"{lhs.name}, {rhs.name} : {self._render_type(lhs.type)}, {self._render_type(rhs.type)} "
                + f"-> {self._render_type(low_type)}, {self._render_type(high_type)}"
            )
            env[low_target.name] = _RenderedValue(name=low_target.ssa_name, type=low_type)
            env[high_target.name] = _RenderedValue(name=high_target.ssa_name, type=high_type)
            return lines

        if stmt.value.name in {"pdintlv_b8", "pdintlv_b16", "pdintlv_b32", "pintlv_b8", "pintlv_b16", "pintlv_b32"}:
            lines = []
            lhs = self._lower_expr(stmt.value.args[0], env, indent=indent, into=lines)
            rhs = self._lower_expr(stmt.value.args[1], env, indent=indent, into=lines)
            low_target, high_target = stmt.targets
            low_type, high_type = stmt.value.type.elements
            lines.append(
                self._indent(indent)
                + f"{low_target.ssa_name}, {high_target.ssa_name} = pto.{stmt.value.name} "
                + f"{lhs.name}, {rhs.name} : {self._render_type(lhs.type)}, {self._render_type(rhs.type)} "
                + f"-> {self._render_type(low_type)}, {self._render_type(high_type)}"
            )
            env[low_target.name] = _RenderedValue(name=low_target.ssa_name, type=low_type)
            env[high_target.name] = _RenderedValue(name=high_target.ssa_name, type=high_type)
            return lines

        if stmt.value.name == "vmull":
            lines = []
            lhs = self._lower_expr(stmt.value.args[0], env, indent=indent, into=lines)
            rhs = self._lower_expr(stmt.value.args[1], env, indent=indent, into=lines)
            mask = self._lower_expr(stmt.value.args[2], env, indent=indent, into=lines)
            low_target, high_target = stmt.targets
            low_type, high_type = stmt.value.type.elements
            lines.append(
                self._indent(indent)
                + f"{low_target.ssa_name}, {high_target.ssa_name} = pto.vmull "
                + f"{lhs.name}, {rhs.name}, {mask.name} : "
                + f"{self._render_type(lhs.type)}, {self._render_type(rhs.type)}, {self._render_type(mask.type)} "
                + f"-> {self._render_type(low_type)}, {self._render_type(high_type)}"
            )
            env[low_target.name] = _RenderedValue(name=low_target.ssa_name, type=low_type)
            env[high_target.name] = _RenderedValue(name=high_target.ssa_name, type=high_type)
            return lines

        if stmt.value.name == "vldsx2":
            lines = []
            source = self._lower_expr(stmt.value.args[0], env, indent=indent, into=lines)
            if isinstance(source.type, SemanticTileType):
                source = self._materialize_tile_memref(source, indent=indent, into=lines)
            index_args = stmt.value.args[1:-1]
            if (
                isinstance(stmt.value.args[0].type, SemanticTileType)
                and stmt.value.args[0].type.rank == 2
                and len(index_args) == 2
            ):
                source = self._materialize_rank2_tile_subview(
                    source,
                    stmt.value.args[0].type,
                    index_args,
                    env,
                    indent=indent,
                    into=lines,
                )
                rendered_indices = self._materialize_constant(0, SemanticIndexType())
            else:
                rendered_indices = self._render_index_list(index_args, env, indent=indent, into=lines)
            dist = self._render_string_literal(stmt.value.args[-1])
            low_target, high_target = stmt.targets
            low_type, high_type = stmt.value.type.elements
            lines.append(
                self._indent(indent)
                + f"{low_target.ssa_name}, {high_target.ssa_name} = pto.vldsx2 "
                + f"{source.name}[{rendered_indices}], {dist} : "
                + f"{self._render_type(source.type)}, index -> "
                + f"{self._render_type(low_type)}, {self._render_type(high_type)}"
            )
            env[low_target.name] = _RenderedValue(name=low_target.ssa_name, type=low_type)
            env[high_target.name] = _RenderedValue(name=high_target.ssa_name, type=high_type)
            return lines

        if stmt.value.name == "vldus":
            lines = []
            source = self._lower_expr(stmt.value.args[0], env, indent=indent, into=lines)
            index_args = stmt.value.args[1:-1]

            # Convert tile directly to ptr (not memref) — same as vldas.
            if isinstance(source.type, SemanticTileType):
                source = self._materialize_tile_ptr(source, indent=indent, into=lines)

            if (
                isinstance(stmt.value.args[0].type, SemanticTileType)
                and stmt.value.args[0].type.rank == 2
                and len(index_args) == 2
            ):
                orig_tile_type = stmt.value.args[0].type
                row_index, col_index = index_args
                row_value = self._lower_expr(row_index, env, indent=indent, into=lines)
                col_value = self._lower_expr(col_index, env, indent=indent, into=lines)
                total_cols = orig_tile_type.shape[1]
                total_cols_const = self._materialize_constant(
                    total_cols, SemanticIndexType()
                )
                # offset = row * total_cols + col
                row_offset = self._new_temp()
                lines.append(
                    self._indent(indent)
                    + f"{row_offset} = arith.muli {row_value.name}, {total_cols_const} : index"
                )
                elem_offset = self._new_temp()
                lines.append(
                    self._indent(indent)
                    + f"{elem_offset} = arith.addi {row_offset}, {col_value.name} : index"
                )
                ptr_name = self._new_temp()
                lines.append(
                    self._indent(indent)
                    + f"{ptr_name} = pto.addptr {source.name}, {elem_offset} : "
                    + f"{self._render_type(source.type)} -> {self._render_type(source.type)}"
                )
                source = _RenderedValue(name=ptr_name, type=source.type)
            align = self._lower_expr(stmt.value.args[-1], env, indent=indent, into=lines)
            result_target, align_target = stmt.targets
            result_type, align_type = stmt.value.type.elements
            lines.append(
                self._indent(indent)
                + f"{result_target.ssa_name}, {align_target.ssa_name} = pto.vldus "
                + f"{source.name}, {align.name} : "
                + f"{self._render_type(source.type)}, {self._render_type(align.type)} -> "
                + f"{self._render_type(result_type)}, {self._render_type(align_type)}"
            )
            env[result_target.name] = _RenderedValue(name=result_target.ssa_name, type=result_type)
            env[align_target.name] = _RenderedValue(name=align_target.ssa_name, type=align_type)
            return lines

        if stmt.value.name == "pstu":
            lines = []
            align_in = self._lower_expr(stmt.value.args[0], env, indent=indent, into=lines)
            value = self._lower_expr(stmt.value.args[1], env, indent=indent, into=lines)
            base = self._lower_expr(stmt.value.args[2], env, indent=indent, into=lines)
            align_target, base_target = stmt.targets
            align_type, base_type = stmt.value.type.elements
            lines.append(
                self._indent(indent)
                + f"{align_target.ssa_name}, {base_target.ssa_name} = pto.pstu "
                + f"{align_in.name}, {value.name}, {base.name} : "
                + f"{self._render_type(align_in.type)}, {self._render_type(value.type)}, {self._render_type(base.type)} "
                + f"-> {self._render_type(align_type)}, {self._render_type(base_type)}"
            )
            env[align_target.name] = _RenderedValue(name=align_target.ssa_name, type=align_type)
            env[base_target.name] = _RenderedValue(name=base_target.ssa_name, type=base_type)
            return lines

        if stmt.value.name == "get_vms4_sr":
            lines = []
            result_names = ", ".join(target.ssa_name for target in stmt.targets)
            result_types = ", ".join(self._render_type(result_type) for result_type in stmt.value.type.elements)
            lines.append(
                self._indent(indent)
                + f"{result_names} = pto.get_vms4_sr : {result_types}"
            )
            for target, result_type in zip(stmt.targets, stmt.value.type.elements):
                env[target.name] = _RenderedValue(name=target.ssa_name, type=result_type)
            return lines

        raise NotImplementedError(
            f"multi-result assignment for `pto.{stmt.value.name}` is not supported in TileLang DSL v1"
        )

    def _render_vector_store(
        self,
        stmt: SemanticVectorStoreStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        value = self._lower_expr(stmt.value, env, indent=indent, into=lines)
        destination = self._lower_expr(stmt.destination, env, indent=indent, into=lines)
        if isinstance(destination.type, SemanticTileType):
            destination = self._materialize_tile_memref(destination, indent=indent, into=lines)
        if (
            isinstance(stmt.destination.type, SemanticTileType)
            and stmt.destination.type.rank == 2
            and len(stmt.indices) == 2
        ):
            destination = self._materialize_rank2_tile_subview(
                destination,
                stmt.destination.type,
                stmt.indices,
                env,
                indent=indent,
                into=lines,
            )
            rendered_indices = self._materialize_constant(0, SemanticIndexType())
        else:
            rendered_indices = self._render_index_list(stmt.indices, env, indent=indent, into=lines)
        mask = self._lower_expr(stmt.mask, env, indent=indent, into=lines)
        attrs = ""
        if stmt.dist is not None:
            dist = self._render_string_literal(stmt.dist)
            attrs = f" {{dist = {dist}}}"
        lines.append(
            self._indent(indent)
            + "pto.vsts "
            + f"{value.name}, {destination.name}[{rendered_indices}], {mask.name}{attrs} : "
            + f"{self._render_type(value.type)}, {self._render_type(destination.type)}, {self._render_type(mask.type)}"
        )
        return lines

    def _render_vector_pair_store(
        self,
        stmt: SemanticVectorPairStoreStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        low = self._lower_expr(stmt.low, env, indent=indent, into=lines)
        high = self._lower_expr(stmt.high, env, indent=indent, into=lines)
        destination = self._lower_expr(stmt.destination, env, indent=indent, into=lines)
        if isinstance(destination.type, SemanticTileType):
            destination = self._materialize_tile_memref(destination, indent=indent, into=lines)
        if (
            isinstance(stmt.destination.type, SemanticTileType)
            and stmt.destination.type.rank == 2
            and len(stmt.indices) == 2
        ):
            destination = self._materialize_rank2_tile_subview(
                destination,
                stmt.destination.type,
                stmt.indices,
                env,
                indent=indent,
                into=lines,
            )
            rendered_indices = self._materialize_constant(0, SemanticIndexType())
        else:
            rendered_indices = self._render_index_list(stmt.indices, env, indent=indent, into=lines)
        dist = self._render_string_literal(stmt.dist)
        mask = self._lower_expr(stmt.mask, env, indent=indent, into=lines)
        lines.append(
            self._indent(indent)
            + "pto.vstsx2 "
            + f"{low.name}, {high.name}, {destination.name}[{rendered_indices}], {dist}, {mask.name} : "
            + f"{self._render_type(low.type)}, {self._render_type(high.type)}, "
            + f"{self._render_type(destination.type)}, {self._render_type(mask.type)}"
        )
        return lines

    def _render_vscatter(
        self,
        stmt: SemanticVScatterStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        value = self._lower_expr(stmt.value, env, indent=indent, into=lines)
        destination = self._lower_expr(stmt.destination, env, indent=indent, into=lines)
        offsets = self._lower_expr(stmt.offsets, env, indent=indent, into=lines)
        mask = self._lower_expr(stmt.mask, env, indent=indent, into=lines)
        lines.append(
            self._indent(indent)
            + "pto.vscatter "
            + f"{value.name}, {destination.name}, {offsets.name}, {mask.name} : "
            + f"{self._render_type(value.type)}, {self._render_type(destination.type)}, "
            + f"{self._render_type(offsets.type)}, {self._render_type(mask.type)}"
        )
        return lines

    def _render_predicate_store(
        self,
        stmt: SemanticPredicateStoreStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        value = self._lower_expr(stmt.value, env, indent=indent, into=lines)
        destination = self._lower_expr(stmt.destination, env, indent=indent, into=lines)
        if isinstance(destination.type, SemanticTileType):
            destination = self._materialize_tile_memref(destination, indent=indent, into=lines)
        if (
            isinstance(stmt.destination.type, SemanticTileType)
            and stmt.destination.type.rank == 2
            and len(stmt.indices) == 2
        ):
            destination = self._materialize_rank2_tile_subview(
                destination,
                stmt.destination.type,
                stmt.indices,
                env,
                indent=indent,
                into=lines,
            )
            rendered_offset = self._materialize_constant(0, SemanticIndexType())
        else:
            if stmt.op_name == "psti":
                rendered_offset = self._lower_to_index(stmt.indices[0], env, indent=indent, into=lines)
            else:
                rendered_offset = self._lower_expr(stmt.indices[0], env, indent=indent, into=lines)
        dist = self._render_string_literal(stmt.dist)
        lines.append(
            self._indent(indent)
            + f"pto.{stmt.op_name} {value.name}, {destination.name}[{rendered_offset.name}], {dist} : "
            + f"{self._render_type(value.type)}, {self._render_type(destination.type)}, {self._render_type(rendered_offset.type)}"
        )
        return lines

    def _render_align_store(
        self,
        stmt: SemanticAlignStoreStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        value = self._lower_expr(stmt.value, env, indent=indent, into=lines)
        destination = self._lower_expr(stmt.destination, env, indent=indent, into=lines)
        if isinstance(destination.type, SemanticTileType):
            destination = self._materialize_tile_memref(destination, indent=indent, into=lines)
        if (
            isinstance(stmt.destination.type, SemanticTileType)
            and stmt.destination.type.rank == 2
            and len(stmt.indices) == 2
        ):
            destination = self._materialize_rank2_tile_subview(
                destination,
                stmt.destination.type,
                stmt.indices,
                env,
                indent=indent,
                into=lines,
            )
        if stmt.op_name == "vstar":
            lines.append(
                self._indent(indent)
                + f"pto.vstar {value.name}, {destination.name} : "
                + f"{self._render_type(value.type)}, {self._render_type(destination.type)}"
            )
            return lines
        if stmt.offset is None:
            raise NotImplementedError("vstas lowering expects an explicit offset operand")
        offset = self._lower_expr(stmt.offset, env, indent=indent, into=lines)
        offset = self._coerce_rendered_value(offset, _I32_TYPE, indent=indent, into=lines)
        lines.append(
            self._indent(indent)
            + f"pto.vstas {value.name}, {destination.name}, {offset.name} : "
            + f"{self._render_type(value.type)}, {self._render_type(destination.type)}, {self._render_type(offset.type)}"
        )
        return lines

    def _render_scalar_store(
        self,
        stmt: SemanticScalarStoreStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        value = self._lower_expr(stmt.value, env, indent=indent, into=lines)
        destination = self._lower_expr(stmt.destination, env, indent=indent, into=lines)
        offset = self._lower_expr(stmt.offset, env, indent=indent, into=lines)
        lines.append(
            self._indent(indent)
            + f"pto.store_scalar {value.name}, {destination.name}[{offset.name}] : "
            + f"{self._render_type(destination.type)}, {self._render_type(value.type)}"
        )
        return lines

    def _render_index_list(
        self,
        indices: tuple[SemanticExpr, ...],
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> str:
        rendered = [
            self._lower_expr(index, env, indent=indent, into=into).name for index in indices
        ]
        return ", ".join(rendered)

    def _render_rank2_subview_result_type(
        self,
        *,
        element_dtype: str,
        memory_space: str,
    ) -> _RenderedTextualType:
        return _RenderedTextualType(
            f"memref<?x?x{element_dtype}, strided<[?, ?], offset: ?>, "
            f"{self._render_memref_memory_space(memory_space)}>"
        )

    def _materialize_rank2_tile_subview(
        self,
        base: _RenderedValue,
        tile_type: SemanticTileType,
        indices: tuple[SemanticExpr, ...],
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        row_index, col_index = indices
        row_value = self._lower_expr(row_index, env, indent=indent, into=into)
        col_value = self._lower_expr(col_index, env, indent=indent, into=into)
        one = self._materialize_constant(1, SemanticIndexType())
        total_cols = self._materialize_constant(tile_type.shape[1], SemanticIndexType())
        remaining_cols = self._new_temp()
        into.append(
            self._indent(indent)
            + f"{remaining_cols} = arith.subi {total_cols}, {col_value.name} : index"
        )
        subview_type = self._render_rank2_subview_result_type(
            element_dtype=self._render_dtype_name(tile_type.element_dtype),
            memory_space=tile_type.memory_space or "ub",
        )
        subview_name = self._new_temp()
        into.append(
            self._indent(indent)
            + f"{subview_name} = memref.subview {base.name}[{row_value.name}, {col_value.name}] "
            + f"[{one}, {remaining_cols}] [{one}, {one}] : "
            + f"{self._render_type(base.type)} to {self._render_type(subview_type)}"
        )
        return _RenderedValue(name=subview_name, type=subview_type)

    def _tensor_slice_extents(self, expr: SemanticTensorSliceExpr) -> tuple[int, int]:
        if expr.type.rank != 2 or len(expr.type.extents) != 2:
            raise NotImplementedError("TileLang DSL v1 DMA lowering currently only supports rank-2 TensorView slices")
        return expr.type.extents

    def _materialize_tensor_slice_axis_size(
        self,
        slice_axis: object,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        if slice_axis.extent is not None:
            return _RenderedValue(
                name=self._materialize_constant(slice_axis.extent, SemanticIndexType()),
                type=SemanticIndexType(),
            )
        distance = self._emit_binary_value(
            "sub",
            self._lower_expr(slice_axis.stop, env, indent=indent, into=into),
            self._lower_expr(slice_axis.start, env, indent=indent, into=into),
            SemanticIndexType(),
            indent=indent,
            into=into,
        )
        step_value = self._static_expr_value(slice_axis.step, default=1)
        if not isinstance(step_value, int) or step_value <= 0:
            raise NotImplementedError(
                "partition_view lowering currently expects a static positive slice step in TileLang DSL v1"
            )
        if step_value == 1:
            return distance
        numerator = self._emit_binary_value(
            "add",
            distance,
            _RenderedValue(
                name=self._materialize_constant(step_value - 1, SemanticIndexType()),
                type=SemanticIndexType(),
            ),
            SemanticIndexType(),
            indent=indent,
            into=into,
        )
        return self._emit_binary_value(
            "floordiv",
            numerator,
            _RenderedValue(
                name=self._materialize_constant(step_value, SemanticIndexType()),
                type=SemanticIndexType(),
            ),
            SemanticIndexType(),
            indent=indent,
            into=into,
        )

    def _lower_tensor_slice_expr(
        self,
        expr: SemanticTensorSliceExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        desired_name: str | None,
        into: list[str] | None,
    ) -> _RenderedValue:
        if into is None:
            into = []
        tensor_base = self._lower_expr(expr.base, env, indent=indent, into=into)
        if not isinstance(tensor_base.type, (SemanticTensorViewType, SemanticPartitionTensorViewType)):
            raise NotImplementedError("partition_view lowering expects a TensorView/PartitionTensorView source")

        offsets: list[_RenderedValue] = []
        sizes: list[_RenderedValue] = []
        for axis_slice in expr.slices:
            offsets.append(self._lower_expr(axis_slice.start, env, indent=indent, into=into))
            sizes.append(
                self._materialize_tensor_slice_axis_size(
                    axis_slice,
                    env,
                    indent=indent,
                    into=into,
                )
            )

        result_name = desired_name or self._new_temp()
        result_type_text = self._render_partition_tensor_view_type(
            element_dtype=self._render_dtype_name(expr.type.element_dtype),
            shape=tuple("?" if dim is None else dim for dim in expr.type.extents),
        )
        into.append(
            self._indent(indent)
            + f"{result_name} = pto.partition_view {tensor_base.name}, "
            + f"offsets = [{', '.join(value.name for value in offsets)}], "
            + f"sizes = [{', '.join(value.name for value in sizes)}] : "
            + f"{self._render_type(tensor_base.type)} -> {result_type_text}"
        )
        return _RenderedValue(name=result_name, type=_RenderedTextualType(result_type_text))

    def _static_expr_value(self, expr: SemanticExpr | None, *, default: object = None) -> object:
        if expr is None:
            return default
        if isinstance(expr, SemanticLiteralExpr):
            return expr.value
        if isinstance(expr, SemanticSymbolExpr):
            return expr.value
        if isinstance(expr, SemanticBindingRef):
            return expr.binding.value
        return None

    def _materialize_tensor_dim(
        self,
        tensor_base: _RenderedValue,
        *,
        axis: int,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        dim_index = self._new_temp()
        axis_value = self._materialize_constant(axis, SemanticIndexType())
        if isinstance(tensor_base.type, (SemanticTensorViewType, SemanticPartitionTensorViewType)):
            into.append(
                self._indent(indent)
                + f"{dim_index} = pto.get_tensor_view_dim {tensor_base.name}, {axis_value} : "
                + f"{self._render_type(tensor_base.type)} -> index"
            )
        else:
            into.append(
                self._indent(indent)
                + f"{dim_index} = memref.dim {tensor_base.name}, {axis_value} : {self._render_type(tensor_base.type)}"
            )
        return _RenderedValue(name=dim_index, type=SemanticIndexType())

    def _materialize_tensor_axis_stride_elems(
        self,
        tensor_base: _RenderedValue,
        axis: int,
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        stride = _RenderedValue(
            name=self._materialize_constant(1, SemanticIndexType()),
            type=SemanticIndexType(),
        )
        for dim_axis in range(axis + 1, tensor_base.type.rank):
            dim_value = self._materialize_tensor_dim(
                tensor_base,
                axis=dim_axis,
                indent=indent,
                into=into,
            )
            stride = self._emit_binary_value(
                "mul",
                stride,
                dim_value,
                SemanticIndexType(),
                indent=indent,
                into=into,
            )
        return stride

    def _emit_binary_value(
        self,
        op: str,
        lhs: _RenderedValue,
        rhs: _RenderedValue,
        result_type: SemanticType,
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        result_name = self._new_temp()
        into.append(
            self._indent(indent)
            + f"{result_name} = {self._render_binary_op(op, result_type)} "
            f"{lhs.name}, {rhs.name} : {self._render_type(result_type)}"
        )
        return _RenderedValue(name=result_name, type=result_type)

    def _render_strict_vecscope(
        self,
        stmt: SemanticStrictVecscopeStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        capture_values = []
        block_argument_values = []
        for expr, binding in zip(stmt.captures, stmt.block_arguments):
            capture = self._lower_expr(expr, env, indent=indent, into=lines)
            capture, block_arg = self._materialize_strict_vecscope_capture(
                capture,
                binding,
                indent=indent,
                into=lines,
            )
            capture_values.append(capture)
            block_argument_values.append(block_arg)
        capture_names = ", ".join(value.name for value in capture_values)
        block_args = ", ".join(
            f"{binding.ssa_name}: {self._render_type(value.type)}"
            for binding, value in zip(stmt.block_arguments, block_argument_values)
        )
        function_type = ", ".join(
            self._render_type(value.type) for value in block_argument_values
        )

        scope_env = {
            binding.name: _RenderedValue(name=binding.ssa_name, type=value.type)
            for binding, value in zip(stmt.block_arguments, block_argument_values)
        }

        lines.append(self._indent(indent) + f"pto.strict_vecscope({capture_names}) {{")
        lines.append(self._indent(indent) + f"^bb0({block_args}):")
        lines.extend(self._render_block(stmt.body, scope_env, indent=indent + 2))
        lines.append(self._indent(indent) + f"}} : ({function_type}) -> ()")
        return lines

    def _render_vecscope(
        self,
        stmt: SemanticVecscopeStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        scope_env = dict(env)
        lines = [self._indent(indent) + "pto.vecscope {"]
        lines.extend(self._render_block(stmt.body, scope_env, indent=indent + 2))
        lines.append(self._indent(indent) + "}")
        return lines

    def _render_for(
        self,
        stmt: SemanticForStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        lower_bound = self._lower_expr(stmt.lower_bound, env, indent=indent, into=lines)
        upper_bound = self._lower_expr(stmt.upper_bound, env, indent=indent, into=lines)
        step = self._lower_expr(stmt.step, env, indent=indent, into=lines)

        body_env = dict(env)
        body_env[stmt.induction_variable.name] = _RenderedValue(
            name=stmt.induction_variable.ssa_name,
            type=stmt.induction_variable.type,
        )

        if not stmt.loop_carried:
            lines.append(
                self._indent(indent)
                + f"scf.for {stmt.induction_variable.ssa_name} = {lower_bound.name} "
                f"to {upper_bound.name} step {step.name} {{"
            )
            lines.extend(self._render_block(stmt.body, body_env, indent=indent + 2))
            lines.append(self._indent(indent) + "}")
            return lines

        carried_bindings = tuple(stmt.loop_carried)
        if len(carried_bindings) == 1:
            carried_binding = carried_bindings[0]
            initial_value = self._coerce_rendered_value(
                env[carried_binding.name],
                carried_binding.type,
                indent=indent,
                into=lines,
            )
            iter_arg_name = f"%{carried_binding.name}_iter_{self._loop_counter}"
            self._loop_counter += 1
            body_env[carried_binding.name] = _RenderedValue(
                name=iter_arg_name,
                type=carried_binding.type,
            )

            lines.append(
                self._indent(indent)
                + f"{carried_binding.ssa_name}:1 = scf.for {stmt.induction_variable.ssa_name} = "
                f"{lower_bound.name} to {upper_bound.name} step {step.name} "
                f"iter_args({iter_arg_name} = {initial_value.name}) -> "
                f"({self._render_type(carried_binding.type)}) {{"
            )
            lines.extend(self._render_block(stmt.body, body_env, indent=indent + 2))
            yielded_value = self._coerce_rendered_value(
                body_env[carried_binding.name],
                carried_binding.type,
                indent=indent + 2,
                into=lines,
            )
            lines.append(
                self._indent(indent + 2)
                + f"scf.yield {yielded_value.name} : {self._render_type(yielded_value.type)}"
            )
            lines.append(self._indent(indent) + "}")
            env[carried_binding.name] = _RenderedValue(
                name=carried_binding.ssa_name,
                type=carried_binding.type,
            )
            return lines

        loop_id = self._loop_counter
        self._loop_counter += 1

        initial_values: list[_RenderedValue] = []
        iter_arg_names: list[str] = []
        for index, binding in enumerate(carried_bindings):
            initial_values.append(
                self._coerce_rendered_value(
                    env[binding.name],
                    binding.type,
                    indent=indent,
                    into=lines,
                )
            )
            iter_arg_names.append(f"%{binding.name}_iter_{loop_id}_{index}")
            body_env[binding.name] = _RenderedValue(
                name=iter_arg_names[-1],
                type=binding.type,
            )

        result_names = ", ".join(binding.ssa_name for binding in carried_bindings)
        iter_args = ", ".join(
            f"{iter_name} = {initial.name}"
            for iter_name, initial in zip(iter_arg_names, initial_values)
        )
        result_types = ", ".join(self._render_type(binding.type) for binding in carried_bindings)

        lines.append(
            self._indent(indent)
            + f"{result_names} = scf.for {stmt.induction_variable.ssa_name} = "
            f"{lower_bound.name} to {upper_bound.name} step {step.name} "
            f"iter_args({iter_args}) -> ({result_types}) {{"
        )
        lines.extend(self._render_block(stmt.body, body_env, indent=indent + 2))
        yielded_values = [
            self._coerce_rendered_value(
                body_env[binding.name],
                binding.type,
                indent=indent + 2,
                into=lines,
            )
            for binding in carried_bindings
        ]
        yielded_names = ", ".join(value.name for value in yielded_values)
        yielded_types = ", ".join(self._render_type(value.type) for value in yielded_values)
        lines.append(
            self._indent(indent + 2)
            + f"scf.yield {yielded_names} : {yielded_types}"
        )
        lines.append(self._indent(indent) + "}")
        for binding in carried_bindings:
            env[binding.name] = _RenderedValue(
                name=binding.ssa_name,
                type=binding.type,
            )
        return lines

    def _render_if(
        self,
        stmt: SemanticIfStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        cond_lines: list[str] = []
        condition = self._lower_condition(stmt.condition, env, indent=indent, into=cond_lines)
        then_env = dict(env)
        else_env = dict(env)

        if not stmt.results:
            lines = list(cond_lines)
            lines.append(self._indent(indent) + f"scf.if {condition.name} {{")
            lines.extend(self._render_block(stmt.then_body, then_env, indent=indent + 2))
            if stmt.else_body:
                lines.append(self._indent(indent) + "} else {")
                lines.extend(self._render_block(stmt.else_body, else_env, indent=indent + 2))
            lines.append(self._indent(indent) + "}")
            return lines

        lines = list(cond_lines)
        result_names = ", ".join(result.result_binding.ssa_name for result in stmt.results)
        result_types = ", ".join(
            self._render_type(result.result_binding.type) for result in stmt.results
        )
        lines.append(
            self._indent(indent)
            + f"{result_names} = scf.if {condition.name} -> ({result_types}) {{"
        )
        lines.extend(self._render_block(stmt.then_body, then_env, indent=indent + 2))
        then_values = []
        for result in stmt.results:
            then_value = then_env.get(
                result.result_binding.name,
                then_env.get(result.then_binding.name),
            )
            if then_value is None:
                then_value = _RenderedValue(result.then_binding.ssa_name, result.then_binding.type)
            then_values.append(then_value)
        lines.append(
            self._indent(indent + 2)
            + "scf.yield "
            + ", ".join(value.name for value in then_values)
            + " : "
            + ", ".join(self._render_type(value.type) for value in then_values)
        )
        lines.append(self._indent(indent) + "} else {")
        lines.extend(self._render_block(stmt.else_body, else_env, indent=indent + 2))
        else_values = []
        for result in stmt.results:
            else_value = else_env.get(
                result.result_binding.name,
                else_env.get(result.else_binding.name),
            )
            if else_value is None:
                else_value = _RenderedValue(result.else_binding.ssa_name, result.else_binding.type)
            else_values.append(else_value)
        lines.append(
            self._indent(indent + 2)
            + "scf.yield "
            + ", ".join(value.name for value in else_values)
            + " : "
            + ", ".join(self._render_type(value.type) for value in else_values)
        )
        lines.append(self._indent(indent) + "}")
        for result in stmt.results:
            env[result.result_binding.name] = _RenderedValue(
                name=result.result_binding.ssa_name,
                type=result.result_binding.type,
            )
        return lines

    def _lower_condition(
        self,
        expr: SemanticExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        value = self._lower_expr(expr, env, indent=indent, into=into)
        if isinstance(value.type, SemanticScalarType) and value.type.dtype.name == "i1":
            return value

        zero_type: SemanticType
        predicate: str
        if isinstance(value.type, SemanticIndexType):
            zero_type = SemanticIndexType()
            predicate = "arith.cmpi ne"
        elif isinstance(value.type, SemanticScalarType):
            zero_type = value.type
            if value.type.dtype.name in {"f16", "bf16", "f32"}:
                predicate = "arith.cmpf une"
            else:
                predicate = "arith.cmpi ne"
        else:
            raise NotImplementedError(f"unsupported if condition type {value.type!r}")

        zero = self._materialize_constant(0, zero_type)
        result_name = self._new_temp()
        into.append(
            self._indent(indent)
            + f"{result_name} = {predicate}, {value.name}, {zero} : {self._render_type(value.type)}"
        )
        return _RenderedValue(name=result_name, type=_I1_TYPE)

    def _lower_expr(
        self,
        expr: SemanticExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        desired_name: str | None = None,
        into: list[str] | None = None,
    ) -> _RenderedValue:
        if isinstance(expr, SemanticBindingRef):
            return env.get(expr.binding.name, _RenderedValue(expr.binding.ssa_name, expr.type))
        if isinstance(expr, SemanticLiteralExpr):
            return self._lower_literal_expr(
                expr.value,
                expr.type,
                indent=indent,
                desired_name=desired_name,
                into=into,
            )
        if isinstance(expr, SemanticIndexCastExpr):
            if into is None:
                into = []
            value = self._lower_expr(expr.value, env, indent=indent, into=into)
            return self._coerce_rendered_to_index(value, indent=indent, into=into)
        if isinstance(expr, SemanticSubscriptAccess):
            return self._lower_subscript_access(
                expr,
                env,
                indent=indent,
                desired_name=desired_name,
                into=into,
            )
        if isinstance(expr, SemanticBinaryExpr):
            if into is None:
                into = []
            if expr.op in {"and", "or"}:
                return self._lower_bool_expr(
                    expr.op,
                    expr.lhs,
                    expr.rhs,
                    env,
                    indent=indent,
                    desired_name=desired_name,
                    into=into,
                )
            lhs = self._lower_expr(expr.lhs, env, indent=indent, into=into)
            rhs = self._lower_expr(expr.rhs, env, indent=indent, into=into)
            if expr.op in {"eq", "ne", "gt", "lt", "ge", "le"}:
                return self._lower_compare_expr(
                    expr.op,
                    lhs,
                    rhs,
                    indent=indent,
                    desired_name=desired_name,
                    into=into,
                )
            if isinstance(expr.type, SemanticScalarType):
                lhs = self._coerce_rendered_value(lhs, expr.type, indent=indent, into=into)
                rhs = self._coerce_rendered_value(rhs, expr.type, indent=indent, into=into)
            result_name = desired_name or self._new_temp()
            into.append(
                self._indent(indent)
                + f"{result_name} = {self._render_binary_op(expr.op, expr.type)} "
                f"{lhs.name}, {rhs.name} : {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)
        if isinstance(expr, SemanticCallExpr):
            return self._lower_call_expr(expr, env, indent=indent, desired_name=desired_name, into=into)
        if isinstance(expr, SemanticAttributeAccess):
            raise NotImplementedError("bare shape attribute values are not materialized directly")
        if isinstance(expr, SemanticTensorSliceExpr):
            return self._lower_tensor_slice_expr(
                expr,
                env,
                indent=indent,
                desired_name=desired_name,
                into=into,
            )
        if isinstance(expr, SemanticSymbolExpr):
            raise NotImplementedError("symbol expressions are only lowered through specialized TileLang DSL ops")
        raise NotImplementedError(f"unsupported semantic expression {type(expr).__name__}")

    def _lower_call_expr(
        self,
        expr: SemanticCallExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        desired_name: str | None,
        into: list[str] | None,
    ) -> _RenderedValue:
        if expr.namespace is None:
            if into is None:
                into = []
            rendered_args = [
                self._lower_expr(arg, env, indent=indent, into=into)
                for arg in expr.args
                if self._should_materialize_function_boundary_type(arg.type)
            ]
            rendered_arg_names = ", ".join(arg.name for arg in rendered_args)
            rendered_arg_types = ", ".join(self._render_type(arg.type) for arg in rendered_args)
            if not rendered_arg_types:
                rendered_arg_types = ""
            if expr.type is None:
                into.append(
                    self._indent(indent)
                    + f"func.call {_format_symbol_name(expr.name)}({rendered_arg_names}) : "
                    + f"({rendered_arg_types}) -> ()"
                )
                return _RenderedValue(name="__void_call__", type=SemanticMetaType(kind="void"))
            result_name = desired_name or self._new_temp()
            into.append(
                self._indent(indent)
                + f"{result_name} = func.call {_format_symbol_name(expr.name)}({rendered_arg_names}) : "
                + f"({rendered_arg_types}) -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.namespace != "pto":
            raise NotImplementedError(f"unsupported call namespace {expr.namespace!r}")
        if isinstance(expr.type, SemanticTupleType):
            raise NotImplementedError("multi-result call values must be assigned directly in TileLang DSL v1")
        if into is None:
            into = []
        result_name = desired_name or self._new_temp()

        if expr.name == "make_mask":
            dtype_expr, pattern_expr = expr.args
            if not self._is_dtype_meta_expr(dtype_expr):
                raise NotImplementedError("make_mask dtype lowering expects a dtype symbol")
            if not isinstance(pattern_expr, SemanticSymbolExpr) or not isinstance(pattern_expr.value, MaskPattern):
                raise NotImplementedError("make_mask pattern lowering expects a MaskPattern symbol")
            suffix = expr.type.granularity
            into.append(
                self._indent(indent)
                + f'{result_name} = pto.pset_{suffix} "{pattern_expr.value.value}" : {self._render_type(expr.type)}'
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name in {"pset_b8", "pset_b16", "pset_b32", "pge_b8", "pge_b16", "pge_b32"}:
            if not isinstance(expr.args[0], SemanticSymbolExpr) or not isinstance(expr.args[0].value, MaskPattern):
                raise NotImplementedError(f"{expr.name} lowering expects a MaskPattern symbol")
            pattern_token = expr.args[0].value.value.replace("\\", "\\\\").replace('"', '\\"')
            pattern = f'"{pattern_token}"'
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.{expr.name} {pattern} : {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "init_align":
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.init_align : {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vlds":
            source = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            if isinstance(source.type, SemanticTileType):
                source = self._materialize_tile_memref(source, indent=indent, into=into)
            index_args = expr.args[1:]
            dist_suffix = ""
            if index_args and self._has_optional_string_literal(index_args[-1]):
                dist_suffix = f" {{dist = {self._render_string_literal(index_args[-1])}}}"
                index_args = index_args[:-1]
            if (
                isinstance(expr.args[0].type, SemanticTileType)
                and expr.args[0].type.rank == 2
                and len(index_args) == 2
            ):
                source = self._materialize_rank2_tile_subview(
                    source,
                    expr.args[0].type,
                    index_args,
                    env,
                    indent=indent,
                    into=into,
                )
                rendered_indices = self._materialize_constant(0, SemanticIndexType())
            else:
                rendered_indices = self._render_index_list(index_args, env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vlds {source.name}[{rendered_indices}]{dist_suffix} : "
                + f"{self._render_type(source.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name in {"plds", "pldi"}:
            source = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            if expr.name == "pldi":
                offset = self._lower_to_index(expr.args[1], env, indent=indent, into=into)
            else:
                offset = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            dist = self._render_string_literal(expr.args[2])
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.{expr.name} {source.name}[{offset.name}], {dist} : "
                + f"{self._render_type(source.type)}, {self._render_type(offset.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vldas":
            source = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            index_args = expr.args[1:]

            # Convert tile directly to ptr (not memref) for vldas.
            # vldas only accepts !pto.ptr, and the memref→subview→castptr
            # path is broken.  Instead: tile → ptr via tile_buf_addr, then
            # pto.addptr with the element offset computed from the indices.
            if isinstance(source.type, SemanticTileType):
                source = self._materialize_tile_ptr(source, indent=indent, into=into)

            if (
                isinstance(expr.args[0].type, SemanticTileType)
                and expr.args[0].type.rank == 2
                and len(index_args) == 2
            ):
                orig_tile_type = expr.args[0].type
                row_index, col_index = index_args
                row_value = self._lower_expr(row_index, env, indent=indent, into=into)
                col_value = self._lower_expr(col_index, env, indent=indent, into=into)
                # uniform offset formula: row * shape[1] + col.
                # Works for row_major [M,N] and col_major [M,1] (col=0).
                total_cols = orig_tile_type.shape[1]
                total_cols_const = self._materialize_constant(
                    total_cols, SemanticIndexType()
                )
                # offset = row * total_cols + col
                row_offset = self._new_temp()
                into.append(
                    self._indent(indent)
                    + f"{row_offset} = arith.muli {row_value.name}, {total_cols_const} : index"
                )
                elem_offset = self._new_temp()
                into.append(
                    self._indent(indent)
                    + f"{elem_offset} = arith.addi {row_offset}, {col_value.name} : index"
                )
                ptr_name = self._new_temp()
                into.append(
                    self._indent(indent)
                    + f"{ptr_name} = pto.addptr {source.name}, {elem_offset} : "
                    + f"{self._render_type(source.type)} -> {self._render_type(source.type)}"
                )
                source = _RenderedValue(name=ptr_name, type=source.type)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vldas {source.name} : "
                + f"{self._render_type(source.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "load_scalar":
            source = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            offset = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.load_scalar {source.name}[{offset.name}] : "
                + f"{self._render_type(source.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vstus":
            align_in = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            offset = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            offset = self._coerce_rendered_value(offset, _I32_TYPE, indent=indent, into=into)
            value = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            base = self._lower_expr(expr.args[3], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vstus {align_in.name}, {offset.name}, {value.name}, {base.name} : "
                + f"{self._render_type(align_in.type)}, {self._render_type(offset.type)}, {self._render_type(value.type)}, {self._render_type(base.type)} "
                + f"-> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vstur":
            align_in = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            value = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            base = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            mode = self._render_string_literal(expr.args[3])
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vstur {align_in.name}, {value.name}, {base.name}, {mode} : "
                + f"{self._render_type(align_in.type)}, {self._render_type(value.type)}, {self._render_type(base.type)} "
                + f"-> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name in {
            "get_block_idx",
            "get_subblock_idx",
            "get_block_num",
            "get_subblock_num",
        }:
            into.append(self._indent(indent) + f"{result_name} = pto.{expr.name}")
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vbr":
            scalar = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vbr {scalar.name} : "
                + f"{self._render_type(scalar.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vdup":
            value = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            mask = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            if len(expr.args) == 3:
                position = self._render_string_literal(expr.args[2])
                into.append(
                    self._indent(indent)
                    + f"{result_name} = pto.vdup {value.name}, {mask.name} {{position = {position}}} : "
                    + f"{self._render_type(value.type)}, {self._render_type(mask.type)} -> {self._render_type(expr.type)}"
                )
            else:
                into.append(
                    self._indent(indent)
                    + f"{result_name} = pto.vdup {value.name}, {mask.name} : "
                    + f"{self._render_type(value.type)}, {self._render_type(mask.type)} -> {self._render_type(expr.type)}"
                )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vci":
            index = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            order = self._render_string_literal(expr.args[1])
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vci {index.name} {{order = {order}}} : "
                + f"{self._render_type(index.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "tensor_view_as_ptr":
            source = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.tensor_view_addr {source.name} : "
                + f"{self._render_type(source.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "tile_as_ptr":
            source = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.tile_buf_addr {source.name} : "
                + f"{self._render_type(source.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "alloc_tile":
            if not isinstance(expr.type, SemanticTileType):
                raise NotImplementedError("pto.alloc_tile lowering expects a SemanticTileType result")
            attrs: list[str] = []
            if len(expr.args) >= 1:
                valid_shape_expr = expr.args[0]
                valid_shape = expr.type.valid_shape
                if valid_shape is None:
                    raise NotImplementedError("pto.alloc_tile lowering requires known tile valid_shape metadata")
                if expr.type.rank >= 1 and valid_shape[0] is None:
                    valid_row = self._lower_expr(valid_shape_expr, env, indent=indent, into=into)
                    attrs.append(f"valid_row = {valid_row.name}")
                if expr.type.rank >= 2 and valid_shape[1] is None:
                    if not isinstance(valid_shape_expr, SemanticTupleExpr):
                        raise NotImplementedError(
                            "pto.alloc_tile lowering expects tuple valid_shape when valid_col is dynamic"
                        )
                    valid_col = self._lower_expr(valid_shape_expr.elements[1], env, indent=indent, into=into)
                    attrs.append(f"valid_col = {valid_col.name}")
            if len(expr.args) >= 2:
                addr = self._lower_expr(expr.args[1], env, indent=indent, into=into)
                attrs.append(f"addr = {addr.name}")
            op_text = f"{result_name} = pto.alloc_tile"
            if attrs:
                op_text += " " + " ".join(attrs)
            into.append(
                self._indent(indent)
                + f"{op_text} : {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "castptr":
            value = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            if isinstance(expr.type, SemanticPtrType) and isinstance(value.type, SemanticIndexType):
                value = self._coerce_rendered_to_i64(value, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.castptr {value.name} : "
                + f"{self._render_type(value.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name in {
            "i1",
            "i8",
            "si8",
            "ui8",
            "i16",
            "si16",
            "ui16",
            "i32",
            "si32",
            "ui32",
            "i64",
            "si64",
            "ui64",
            "f16",
            "bf16",
            "f32",
        }:
            value = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            return self._coerce_rendered_value(value, expr.type, indent=indent, into=into)

        if expr.name == "addptr":
            pointer = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            offset = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.addptr {pointer.name}, {offset.name} : "
                + f"{self._render_type(pointer.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name in {"mad", "mad_acc", "mad_bias", "mad_mx", "mad_mx_acc", "mad_mx_bias"}:
            self._render_cube_mad_like(expr, env, indent=indent, into=into)
            return _RenderedValue(name="__void_call__", type=SemanticMetaType(kind="void"))

        if expr.name in {"mte_gm_l1", "mte_l1_ub"}:
            self._render_mte_load_store(expr, env, indent=indent, into=into)
            return _RenderedValue(name="__void_call__", type=SemanticMetaType(kind="void"))

        if expr.name == "mte_gm_l1_frac":
            self._render_mte_gm_l1_frac(expr, env, indent=indent, into=into)
            return _RenderedValue(name="__void_call__", type=SemanticMetaType(kind="void"))

        if expr.name == "mte_l1_bt":
            self._render_mte_l1_bt(expr, env, indent=indent, into=into)
            return _RenderedValue(name="__void_call__", type=SemanticMetaType(kind="void"))

        if expr.name == "mte_l1_fb":
            self._render_mte_l1_fb(expr, env, indent=indent, into=into)
            return _RenderedValue(name="__void_call__", type=SemanticMetaType(kind="void"))

        if expr.name in {"mte_l1_l0a", "mte_l1_l0b", "mte_l1_l0a_mx", "mte_l1_l0b_mx"}:
            self._render_mte_stage_load(expr, env, indent=indent, into=into)
            return _RenderedValue(name="__void_call__", type=SemanticMetaType(kind="void"))

        if expr.name in {"mte_l0c_l1", "mte_l0c_gm", "mte_l0c_ub"}:
            self._render_mte_l0c_store(expr, env, indent=indent, into=into)
            return _RenderedValue(name="__void_call__", type=SemanticMetaType(kind="void"))

        if expr.name in {"ppack", "punpack"}:
            value = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            part = self._render_string_literal(expr.args[1])
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.{expr.name} {value.name}, {part} : "
                + f"{self._render_type(value.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "pnot":
            value = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            mask = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.pnot {value.name}, {mask.name} : "
                + f"{self._render_type(value.type)}, {self._render_type(mask.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name in {"psel", "pand", "por", "pxor"}:
            src0 = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            src1 = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            mask = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.{expr.name} {src0.name}, {src1.name}, {mask.name} : "
                + f"{self._render_type(src0.type)}, {self._render_type(src1.type)}, {self._render_type(mask.type)} "
                + f"-> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vcmp":
            lhs = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            rhs = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            seed = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            cmp_mode = self._render_string_literal(expr.args[3])
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vcmp {lhs.name}, {rhs.name}, {seed.name}, {cmp_mode} : "
                + f"{self._render_type(lhs.type)}, {self._render_type(rhs.type)}, {self._render_type(seed.type)} "
                + f"-> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vcmps":
            vector = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            scalar = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            seed = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            cmp_mode = self._render_string_literal(expr.args[3])
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vcmps {vector.name}, {scalar.name}, {seed.name}, {cmp_mode} : "
                + f"{self._render_type(vector.type)}, {self._render_type(scalar.type)}, {self._render_type(seed.type)} "
                + f"-> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vsel":
            src0 = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            src1 = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            mask = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vsel {src0.name}, {src1.name}, {mask.name} : "
                + f"{self._render_type(src0.type)}, {self._render_type(src1.type)}, {self._render_type(mask.type)} "
                + f"-> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name in {"vselr", "vselrv2"}:
            src0 = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            src1 = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.{expr.name} {src0.name}, {src1.name} : "
                + f"{self._render_type(src0.type)}, {self._render_type(src1.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name in {"vintlvv2", "vdintlvv2"}:
            lhs = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            rhs = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            part = self._render_string_literal(expr.args[2])
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.{expr.name} {lhs.name}, {rhs.name}, {part} : "
                + f"{self._render_type(lhs.type)}, {self._render_type(rhs.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vcvt":
            value = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            mask = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            attr_parts: list[str] = []
            if self._has_optional_string_literal(expr.args[3]):
                attr_parts.append(f"rnd = {self._render_string_literal(expr.args[3])}")
            if self._has_optional_string_literal(expr.args[4]):
                attr_parts.append(f"sat = {self._render_string_literal(expr.args[4])}")
            if self._has_optional_string_literal(expr.args[5]):
                attr_parts.append(f"part = {self._render_string_literal(expr.args[5])}")
            attr_suffix = f" {{{', '.join(attr_parts)}}}" if attr_parts else ""
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vcvt {value.name}, {mask.name}{attr_suffix} : "
                + f"{self._render_type(value.type)}, {self._render_type(mask.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vbitcast":
            value = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vbitcast {value.name} : "
                + f"{self._render_type(value.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "pbitcast":
            value = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.pbitcast {value.name} : "
                + f"{self._render_type(value.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vbitsort":
            destination = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            source = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            indices = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            repeat_times = self._lower_expr(expr.args[3], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"pto.vbitsort {destination.name}, {source.name}, {indices.name}, {repeat_times.name} : "
                + f"{self._render_type(destination.type)}, {self._render_type(source.type)}, "
                + f"{self._render_type(indices.type)}, {self._render_type(repeat_times.type)}"
            )
            return _RenderedValue(name="__void_call__", type=SemanticMetaType(kind="void"))

        if expr.name == "vmrgsort4":
            destination = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            source0 = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            source1 = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            source2 = self._lower_expr(expr.args[3], env, indent=indent, into=into)
            source3 = self._lower_expr(expr.args[4], env, indent=indent, into=into)
            count = self._lower_expr(expr.args[5], env, indent=indent, into=into)
            config = self._lower_expr(expr.args[6], env, indent=indent, into=into)
            count = self._coerce_rendered_value(count, _I64_TYPE, indent=indent, into=into)
            config = self._coerce_rendered_value(config, _I64_TYPE, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"pto.vmrgsort4 {destination.name}, {source0.name}, {source1.name}, {source2.name}, {source3.name}, "
                + f"{count.name}, {config.name} : {self._render_type(destination.type)}, {self._render_type(source0.type)}, "
                + f"{self._render_type(source1.type)}, {self._render_type(source2.type)}, {self._render_type(source3.type)}, "
                + f"{self._render_type(count.type)}, {self._render_type(config.type)}"
            )
            return _RenderedValue(name="__void_call__", type=SemanticMetaType(kind="void"))

        if expr.name == "vtrc":
            value = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            mask = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            rnd = self._render_string_literal(expr.args[2])
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vtrc {value.name}, {mask.name}, {rnd} : "
                + f"{self._render_type(value.type)}, {self._render_type(mask.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name in {
            "vabs",
            "vrelu",
            "vexp",
            "vln",
            "vsqrt",
            "vrec",
            "vnot",
            "vcadd",
            "vcmax",
            "vbcnt",
            "vneg",
            "vcls",
            "vcmin",
            "vrsqrt",
            "vmov",
            "vsunpack",
            "vzunpack",
            "vusqz",
            "vsqz",
            "vcgadd",
            "vcgmax",
            "vcgmin",
            "vcpadd",
            "vsort32",
        }:
            if expr.name in {"vsunpack", "vzunpack"}:
                value = self._lower_expr(expr.args[0], env, indent=indent, into=into)
                part = self._lower_to_index(expr.args[1], env, indent=indent, into=into)
                into.append(
                    self._indent(indent)
                    + f"{result_name} = pto.{expr.name} {value.name}, {part.name} : "
                    + f"{self._render_type(value.type)} -> {self._render_type(expr.type)}"
                )
                return _RenderedValue(name=result_name, type=expr.type)
            value = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            mask = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.{expr.name} {value.name}, {mask.name} : "
                + f"{self._render_type(value.type)}, {self._render_type(mask.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vexpdif":
            lhs = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            rhs = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            mask = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            part = self._render_string_literal(expr.args[3])
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vexpdif {lhs.name}, {rhs.name}, {mask.name}, {part} : "
                + f"{self._render_type(lhs.type)}, {self._render_type(rhs.type)}, {self._render_type(mask.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name in {
            "vadd",
            "vsub",
            "vmul",
            "vdiv",
            "vmax",
            "vmin",
            "vand",
            "vor",
            "vxor",
            "vaddrelu",
            "vaddreluconv",
            "vsubrelu",
            "vmulconv",
            "vshl",
            "vshr",
            "vprelu",
            "vperm",
            "vmrgsort",
        }:
            lhs = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            rhs = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            mask = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.{expr.name} {lhs.name}, {rhs.name}, {mask.name} : "
                + f"{self._render_type(lhs.type)}, {self._render_type(rhs.type)}, {self._render_type(mask.type)} "
                + f"-> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vpack":
            vector = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            part = self._render_string_literal(expr.args[1])
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vpack {vector.name}, {part} : "
                + f"{self._render_type(vector.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name in {"vshift", "vslide"}:
            vector = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            immediate = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            mask = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.{expr.name} {vector.name}, {immediate.name}, {mask.name} : "
                + f"{self._render_type(vector.type)}, {self._render_type(immediate.type)}, {self._render_type(mask.type)} "
                + f"-> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name in {"vadds", "vsubs", "vmuls", "vdivs", "vmaxs", "vmins", "vlrelu", "vshls", "vshrs", "vands", "vors", "vxors"}:
            value = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            scalar = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            mask = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.{expr.name} {value.name}, {scalar.name}, {mask.name} : "
                + f"{self._render_type(value.type)}, {self._render_type(scalar.type)}, {self._render_type(mask.type)} "
                + f"-> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name in {"vaxpy", "vmula"}:
            vec0 = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            vec1 = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            vec2 = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            mask = self._lower_expr(expr.args[3], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.{expr.name} {vec0.name}, {vec1.name}, {vec2.name}, {mask.name} : "
                + f"{self._render_type(vec0.type)}, {self._render_type(vec1.type)}, {self._render_type(vec2.type)}, {self._render_type(mask.type)} "
                + f"-> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        raise NotImplementedError(f"unsupported pto call `{expr.name}` in lowering")

    def _render_cube_mad_like(
        self,
        expr: SemanticCallExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> None:
        has_bias = "bias" in expr.name
        lhs = self._lower_expr(expr.args[0], env, indent=indent, into=into)
        rhs = self._lower_expr(expr.args[1], env, indent=indent, into=into)
        dst = self._lower_expr(expr.args[2], env, indent=indent, into=into)
        bias = None
        dim_start = 3
        if has_bias:
            bias = self._lower_expr(expr.args[3], env, indent=indent, into=into)
            dim_start = 4
        m = self._lower_to_i64(expr.args[dim_start], env, indent=indent, into=into)
        n = self._lower_to_i64(expr.args[dim_start + 1], env, indent=indent, into=into)
        k = self._lower_to_i64(expr.args[dim_start + 2], env, indent=indent, into=into)
        unit_flag = self._extract_optional_static_string(expr.args[dim_start + 3], context=f"pto.{expr.name} unit_flag")
        disable_gemv = self._extract_static_bool(expr.args[dim_start + 4], context=f"pto.{expr.name} disable_gemv")
        sat = self._extract_optional_static_string(expr.args[dim_start + 5], context=f"pto.{expr.name} sat")
        tf32_mode = self._extract_optional_static_string(expr.args[dim_start + 6], context=f"pto.{expr.name} tf32_mode")
        n_dir = self._extract_static_bool(expr.args[dim_start + 7], context=f"pto.{expr.name} n_dir")

        operands = [lhs.name, rhs.name, dst.name]
        operand_types = [self._render_type(lhs.type), self._render_type(rhs.type), self._render_type(dst.type)]
        if bias is not None:
            operands.append(bias.name)
            operand_types.append(self._render_type(bias.type))
        operands.extend([m.name, n.name, k.name])
        operand_types.extend([self._render_type(m.type), self._render_type(n.type), self._render_type(k.type)])
        clause_parts: list[str] = []
        if unit_flag is not None:
            clause_parts.append(f"unit_flag({unit_flag})")
        if disable_gemv:
            clause_parts.append("disable_gemv")
        if sat is not None:
            clause_parts.append(sat)
        if tf32_mode is not None:
            clause_parts.append(f"tf32_mode({tf32_mode})")
        if n_dir:
            clause_parts.append("n_dir")
        clause_suffix = (" " + " ".join(clause_parts)) if clause_parts else ""
        into.append(
            self._indent(indent)
            + f"pto.{expr.name} "
            + ", ".join(operands)
            + clause_suffix
            + " : "
            + ", ".join(operand_types)
        )

    def _render_mte_load_store(
        self,
        expr: SemanticCallExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> None:
        source = self._lower_expr(expr.args[0], env, indent=indent, into=into)
        destination = self._lower_expr(expr.args[1], env, indent=indent, into=into)
        len_burst = self._lower_to_i64(expr.args[2], env, indent=indent, into=into)
        nburst = self._lower_cube_i64_tuple(expr.args[3], env, indent=indent, into=into, expected_len=3)
        loop_groups = self._lower_cube_loop_groups(expr.args[4], env, indent=indent, into=into)

        op_text = (
            f"pto.{expr.name} {source.name}, {destination.name}, {len_burst.name}"
            f" nburst({nburst[0].name}, {nburst[1].name}, {nburst[2].name})"
        )
        type_text = (
            f"{self._render_type(source.type)}, {self._render_type(destination.type)}, "
            f"{self._render_type(len_burst.type)}, {self._render_type(nburst[0].type)}, "
            f"{self._render_type(nburst[1].type)}, {self._render_type(nburst[2].type)}"
        )
        for count, src_stride, dst_stride in loop_groups:
            op_text += f" loop({count.name}, {src_stride.name}, {dst_stride.name})"
            type_text += (
                f", loop {self._render_type(count.type)}, {self._render_type(src_stride.type)}, "
                f"{self._render_type(dst_stride.type)}"
            )
        into.append(self._indent(indent) + op_text + " : " + type_text)

    def _render_mte_gm_ub(
        self,
        expr: SemanticCallExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> None:
        source = self._lower_expr(expr.args[0], env, indent=indent, into=into)
        destination = self._lower_expr(expr.args[1], env, indent=indent, into=into)
        l2_cache_ctl = self._lower_to_i64(expr.args[2], env, indent=indent, into=into)
        len_burst = self._lower_to_i64(expr.args[3], env, indent=indent, into=into)
        nburst = self._lower_cube_i64_tuple(expr.args[4], env, indent=indent, into=into, expected_len=3)
        loop_groups = self._lower_cube_loop_groups(expr.args[5], env, indent=indent, into=into)
        op_text = (
            f"pto.{expr.name} {source.name}, {destination.name}, {l2_cache_ctl.name}, {len_burst.name}"
            f" nburst({nburst[0].name}, {nburst[1].name}, {nburst[2].name})"
        )
        type_text = (
            f"{self._render_type(source.type)}, {self._render_type(destination.type)}, "
            f"{self._render_type(l2_cache_ctl.type)}, {self._render_type(len_burst.type)}, "
            f"{self._render_type(nburst[0].type)}, {self._render_type(nburst[1].type)}, {self._render_type(nburst[2].type)}"
        )
        for count, src_stride, dst_stride in loop_groups:
            op_text += f" loop({count.name}, {src_stride.name}, {dst_stride.name})"
            type_text += (
                f", loop {self._render_type(count.type)}, {self._render_type(src_stride.type)}, "
                f"{self._render_type(dst_stride.type)}"
            )
        if isinstance(expr.args[6], SemanticTupleExpr):
            pad_values = self._lower_cube_tuple_elements(expr.args[6], env, indent=indent, into=into, expected_len=None)
            op_text += f" pad({pad_values[0].name}"
            type_text += f", pad {self._render_type(pad_values[0].type)}"
            if len(pad_values) == 3:
                op_text += f", {pad_values[1].name}, {pad_values[2].name}"
                type_text += f", {self._render_type(pad_values[1].type)}, {self._render_type(pad_values[2].type)}"
            op_text += ")"
        into.append(self._indent(indent) + op_text + " : " + type_text)

    def _render_mte_ub_gm(
        self,
        expr: SemanticCallExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> None:
        source = self._lower_expr(expr.args[0], env, indent=indent, into=into)
        destination = self._lower_expr(expr.args[1], env, indent=indent, into=into)
        len_burst = self._lower_to_i64(expr.args[2], env, indent=indent, into=into)
        nburst = self._lower_cube_i64_tuple(expr.args[3], env, indent=indent, into=into, expected_len=3)
        loop_groups = self._lower_cube_loop_groups(expr.args[4], env, indent=indent, into=into)
        op_text = (
            f"pto.{expr.name} {source.name}, {destination.name}, {len_burst.name}"
            f" nburst({nburst[0].name}, {nburst[1].name}, {nburst[2].name})"
        )
        type_text = (
            f"{self._render_type(source.type)}, {self._render_type(destination.type)}, "
            f"{self._render_type(len_burst.type)}, {self._render_type(nburst[0].type)}, "
            f"{self._render_type(nburst[1].type)}, {self._render_type(nburst[2].type)}"
        )
        for count, src_stride, dst_stride in loop_groups:
            op_text += f" loop({count.name}, {src_stride.name}, {dst_stride.name})"
            type_text += (
                f", loop {self._render_type(count.type)}, {self._render_type(src_stride.type)}, "
                f"{self._render_type(dst_stride.type)}"
            )
        into.append(self._indent(indent) + op_text + " : " + type_text)

    def _render_mte_ub_copy(
        self,
        expr: SemanticCallExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> None:
        source = self._lower_expr(expr.args[0], env, indent=indent, into=into)
        destination = self._lower_expr(expr.args[1], env, indent=indent, into=into)
        len_burst = self._lower_to_i64(expr.args[2], env, indent=indent, into=into)
        nburst = self._lower_cube_i64_tuple(expr.args[3], env, indent=indent, into=into, expected_len=3)
        into.append(
            self._indent(indent)
            + f"pto.{expr.name} {source.name}, {destination.name}, {len_burst.name}"
            + f" nburst({nburst[0].name}, {nburst[1].name}, {nburst[2].name}) : "
            + f"{self._render_type(source.type)}, {self._render_type(destination.type)}, "
            + f"{self._render_type(nburst[0].type)}, {self._render_type(len_burst.type)}, "
            + f"{self._render_type(nburst[1].type)}, {self._render_type(nburst[2].type)}"
        )

    def _render_mte_gm_l1_frac(
        self,
        expr: SemanticCallExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> None:
        source = self._lower_expr(expr.args[0], env, indent=indent, into=into)
        destination = self._lower_expr(expr.args[1], env, indent=indent, into=into)
        mode = self._extract_static_string(expr.args[2], context=f"pto.{expr.name} mode")
        shape = self._lower_cube_i64_tuple(expr.args[3], env, indent=indent, into=into, expected_len=2)
        src_layout = self._lower_cube_i64_tuple(expr.args[4], env, indent=indent, into=into, min_len=1, max_len=2)
        dst_group = self._lower_cube_i64_tuple(expr.args[5], env, indent=indent, into=into, expected_len=4)
        ctrl = self._lower_cube_tuple_elements(expr.args[6], env, indent=indent, into=into, expected_len=2)
        l2_cache_ctrl = self._coerce_rendered_to_i64(ctrl[0], indent=indent, into=into)
        smallc0_en = self._coerce_rendered_value(ctrl[1], _I1_TYPE, indent=indent, into=into)

        src_layout_operands = ", ".join(value.name for value in src_layout)
        src_layout_types = ", ".join(self._render_type(value.type) for value in src_layout)
        into.append(
            self._indent(indent)
            + f"pto.{expr.name} {source.name}, {destination.name}, {mode}, "
            + f"shape({shape[0].name}, {shape[1].name}), "
            + f"src_layout({src_layout_operands}), "
            + f"dst_group({dst_group[0].name}, {dst_group[1].name}, {dst_group[2].name}, {dst_group[3].name}), "
            + f"ctrl({l2_cache_ctrl.name}, {smallc0_en.name}) : "
            + f"{self._render_type(source.type)}, {self._render_type(destination.type)}, {mode}, "
            + f"shape {self._render_type(shape[0].type)}, {self._render_type(shape[1].type)}, "
            + f"src_layout({src_layout_types}), "
            + f"dst_group {self._render_type(dst_group[0].type)}, {self._render_type(dst_group[1].type)}, "
            + f"{self._render_type(dst_group[2].type)}, {self._render_type(dst_group[3].type)}, "
            + f"ctrl {self._render_type(l2_cache_ctrl.type)}, {self._render_type(smallc0_en.type)}"
        )

    def _render_mte_l1_bt(
        self,
        expr: SemanticCallExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> None:
        source = self._lower_expr(expr.args[0], env, indent=indent, into=into)
        destination = self._lower_expr(expr.args[1], env, indent=indent, into=into)
        len_burst = self._lower_to_i64(expr.args[2], env, indent=indent, into=into)
        nburst = self._lower_cube_i64_tuple(expr.args[3], env, indent=indent, into=into, expected_len=3)
        into.append(
            self._indent(indent)
            + f"pto.{expr.name} {source.name}, {destination.name}, {len_burst.name}"
            + f" nburst({nburst[0].name}, {nburst[1].name}, {nburst[2].name}) : "
            + f"{self._render_type(source.type)}, {self._render_type(destination.type)}, "
            + f"{self._render_type(len_burst.type)}, {self._render_type(nburst[0].type)}, "
            + f"{self._render_type(nburst[1].type)}, {self._render_type(nburst[2].type)}"
        )

    def _render_mte_l1_fb(
        self,
        expr: SemanticCallExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> None:
        source = self._lower_expr(expr.args[0], env, indent=indent, into=into)
        destination = self._lower_expr(expr.args[1], env, indent=indent, into=into)
        len_burst = self._lower_to_i64(expr.args[2], env, indent=indent, into=into)
        nburst = self._lower_cube_i64_tuple(expr.args[3], env, indent=indent, into=into, expected_len=3)
        into.append(
            self._indent(indent)
            + f"pto.{expr.name} {source.name}, {destination.name}, {len_burst.name}"
            + f" nburst({nburst[0].name}, {nburst[1].name}, {nburst[2].name}) : "
            + f"{self._render_type(source.type)}, {self._render_type(destination.type)}, "
            + f"{self._render_type(len_burst.type)}, {self._render_type(nburst[0].type)}, "
            + f"{self._render_type(nburst[1].type)}, {self._render_type(nburst[2].type)}"
        )

    def _render_mte_stage_load(
        self,
        expr: SemanticCallExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> None:
        source = self._lower_expr(expr.args[0], env, indent=indent, into=into)
        destination = self._lower_expr(expr.args[1], env, indent=indent, into=into)
        first = self._lower_to_i64(expr.args[2], env, indent=indent, into=into)
        second = self._lower_to_i64(expr.args[3], env, indent=indent, into=into)
        cursor = 4
        start_suffix = ""
        start_type_suffix = ""
        attr_entries: list[str] = []
        if expr.name in {"mte_l1_l0a", "mte_l1_l0b", "mte_l1_l0a_mx", "mte_l1_l0b_mx"}:
            if len(expr.args) < 7:
                if expr.name in {"mte_l1_l0a", "mte_l1_l0b"}:
                    raise ValueError(f"pto.{expr.name} requires normalized start_row and start_col operands")
                if len(expr.args) < 6:
                    raise ValueError(f"pto.{expr.name} requires normalized start_row and start_col operands")
            start_row_expr = expr.args[cursor]
            start_col_expr = expr.args[cursor + 1]
            cursor += 2
            start_row = self._lower_to_i64(start_row_expr, env, indent=indent, into=into)
            start_col = self._lower_to_i64(start_col_expr, env, indent=indent, into=into)
            start_suffix = f", {start_row.name}, {start_col.name}"
            start_type_suffix = f", {self._render_type(start_row.type)}, {self._render_type(start_col.type)}"
        if expr.name in {"mte_l1_l0a", "mte_l1_l0b"} and len(expr.args) > cursor and self._extract_static_bool(
            expr.args[cursor], context=f"pto.{expr.name} transpose"
        ):
            attr_entries.append("transpose = true")
        attr_suffix = f" {{{', '.join(attr_entries)}}}" if attr_entries else ""
        into.append(
            self._indent(indent)
            + f"pto.{expr.name} {source.name}, {destination.name}, {first.name}, {second.name}{start_suffix}{attr_suffix} : "
            + f"{self._render_type(source.type)}, {self._render_type(destination.type)}, "
            + f"{self._render_type(first.type)}, {self._render_type(second.type)}{start_type_suffix}"
        )

    def _render_mte_l0c_store(
        self,
        expr: SemanticCallExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> None:
        source = self._lower_expr(expr.args[0], env, indent=indent, into=into)
        destination = self._lower_expr(expr.args[1], env, indent=indent, into=into)
        m = self._lower_to_i64(expr.args[2], env, indent=indent, into=into)
        n = self._lower_to_i64(expr.args[3], env, indent=indent, into=into)
        src_stride = self._lower_to_i64(expr.args[4], env, indent=indent, into=into)
        dst_stride = self._lower_to_i64(expr.args[5], env, indent=indent, into=into)
        cursor = 6

        operands = [source, destination, m, n, src_stride, dst_stride]
        type_parts = [self._render_type(value.type) for value in operands]
        clause_parts: list[str] = []

        if expr.name == "mte_l0c_gm":
            sid = self._lower_to_i64(expr.args[cursor], env, indent=indent, into=into)
            l2_cache_ctrl = self._lower_to_i64(expr.args[cursor + 1], env, indent=indent, into=into)
            operands.extend([sid, l2_cache_ctrl])
            type_parts.extend([self._render_type(sid.type), self._render_type(l2_cache_ctrl.type)])
            cursor += 2
        elif expr.name == "mte_l0c_ub":
            dst_mode_kind = self._extract_static_string(expr.args[cursor], context="pto.mte_l0c_ub dst_mode kind")
            dst_mode_value_expr = expr.args[cursor + 1]
            if dst_mode_kind == "sub_blockid":
                if self._is_none_meta_expr(dst_mode_value_expr):
                    clause_parts.append("dst_mode(sub_blockid)")
                else:
                    dst_mode_value = self._lower_to_i64(dst_mode_value_expr, env, indent=indent, into=into)
                    clause_parts.append(f"dst_mode({dst_mode_value.name})")
                    type_parts.append(self._render_type(dst_mode_value.type))
            elif dst_mode_kind in {"split_m", "split_n"}:
                clause_parts.append(f"dst_mode({dst_mode_kind})")
            else:
                raise NotImplementedError(f"unsupported pto.mte_l0c_ub dst_mode kind `{dst_mode_kind}` in lowering")
            cursor += 2

        unit_flag = self._extract_optional_static_string(expr.args[cursor], context=f"pto.{expr.name} unit_flag")
        pre_quant_payload_expr = expr.args[cursor + 1]
        pre_quant_mode = self._extract_optional_static_string(expr.args[cursor + 2], context=f"pto.{expr.name} pre_quant mode")
        pre_relu_payload_expr = expr.args[cursor + 3]
        pre_relu_mode = self._extract_optional_static_string(expr.args[cursor + 4], context=f"pto.{expr.name} pre_relu mode")
        clip_expr = expr.args[cursor + 5]
        layout_mode = self._extract_optional_static_string(expr.args[cursor + 6], context=f"pto.{expr.name} layout")
        layout_arg_expr = expr.args[cursor + 7]
        loop3_expr = expr.args[cursor + 8]
        sat = self._extract_optional_static_string(expr.args[cursor + 9], context=f"pto.{expr.name} sat")
        atomic_type = self._extract_optional_static_string(expr.args[cursor + 10], context=f"pto.{expr.name} atomic type")
        atomic_op = self._extract_optional_static_string(expr.args[cursor + 11], context=f"pto.{expr.name} atomic op")

        if pre_quant_mode is not None:
            source = self._bridge_rendered_ptr_element_to_signless_integer(
                source,
                indent=indent,
                into=into,
            )
            operands[0] = source
            type_parts[0] = self._render_type(source.type)

        if unit_flag is not None:
            clause_parts.append(f"unit_flag({unit_flag})")

        if pre_quant_mode is not None:
            pre_quant_payload = self._lower_expr(pre_quant_payload_expr, env, indent=indent, into=into)
            clause_parts.append(f"pre_quant({pre_quant_payload.name}, mode = {pre_quant_mode})")
            type_parts.append(self._render_type(pre_quant_payload.type))

        if pre_relu_mode is not None:
            pre_relu_clause_parts: list[str] = []
            if not self._is_none_meta_expr(pre_relu_payload_expr):
                pre_relu_payload = self._lower_expr(pre_relu_payload_expr, env, indent=indent, into=into)
                pre_relu_clause_parts.append(pre_relu_payload.name)
                type_parts.append(self._render_type(pre_relu_payload.type))
            pre_relu_clause_parts.append(f"mode = {pre_relu_mode}")
            if not self._is_none_meta_expr(clip_expr):
                clip = self._lower_expr(clip_expr, env, indent=indent, into=into)
                pre_relu_clause_parts.append(f"clip = {clip.name}")
                type_parts.append(self._render_type(clip.type))
            clause_parts.append(f"pre_relu({', '.join(pre_relu_clause_parts)})")

        if layout_mode is not None:
            if layout_mode == "nz2nd":
                clause_parts.append("nz2nd")
            elif layout_mode in {"nz2dn", "nz2nz"}:
                layout_arg = self._lower_to_i64(layout_arg_expr, env, indent=indent, into=into)
                clause_parts.append(f"{layout_mode}({layout_arg.name})")
                type_parts.append(self._render_type(layout_arg.type))
            else:
                raise NotImplementedError(f"unsupported pto.{expr.name} layout `{layout_mode}` in lowering")

        if not self._is_none_meta_expr(loop3_expr):
            loop3 = self._lower_cube_i64_tuple(loop3_expr, env, indent=indent, into=into, expected_len=3)
            clause_parts.append(f"loop3({loop3[0].name}, {loop3[1].name}, {loop3[2].name})")
            type_parts.extend(self._render_type(value.type) for value in loop3)

        if sat is not None:
            clause_parts.append(sat)

        if atomic_type is not None or atomic_op is not None:
            if atomic_type is None or atomic_op is None:
                raise NotImplementedError(f"pto.{expr.name} lowering requires both atomic type and atomic op")
            clause_parts.append(f"atomic(type = {atomic_type}, op = {atomic_op})")

        op_text = f"pto.{expr.name} " + ", ".join(value.name for value in operands)
        if clause_parts:
            op_text += ", " + ", ".join(clause_parts)
        into.append(self._indent(indent) + op_text + " : " + ", ".join(type_parts))

    def _lower_cube_loop_groups(
        self,
        expr: SemanticExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> list[tuple[_RenderedValue, _RenderedValue, _RenderedValue]]:
        if self._is_none_meta_expr(expr):
            return []
        if not isinstance(expr, SemanticTupleExpr):
            raise NotImplementedError("cube loop lowering expects a tuple of loop triples")
        loop_groups: list[tuple[_RenderedValue, _RenderedValue, _RenderedValue]] = []
        for loop_expr in expr.elements:
            lowered_loop = self._lower_cube_i64_tuple(loop_expr, env, indent=indent, into=into, expected_len=3)
            loop_groups.append((lowered_loop[0], lowered_loop[1], lowered_loop[2]))
        return loop_groups

    def _lower_cube_i64_tuple(
        self,
        expr: SemanticExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
        expected_len: int | None = None,
        min_len: int | None = None,
        max_len: int | None = None,
    ) -> tuple[_RenderedValue, ...]:
        lowered = self._lower_cube_tuple_elements(expr, env, indent=indent, into=into, expected_len=expected_len, min_len=min_len, max_len=max_len)
        return tuple(self._coerce_rendered_to_i64(value, indent=indent, into=into) for value in lowered)

    def _lower_cube_tuple_elements(
        self,
        expr: SemanticExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
        expected_len: int | None = None,
        min_len: int | None = None,
        max_len: int | None = None,
    ) -> tuple[_RenderedValue, ...]:
        if not isinstance(expr, SemanticTupleExpr):
            raise NotImplementedError("cube structured lowering expects a tuple expression")
        elements = expr.elements
        if expected_len is not None and len(elements) != expected_len:
            raise NotImplementedError(f"cube structured lowering expects exactly {expected_len} tuple elements")
        if min_len is not None and len(elements) < min_len:
            raise NotImplementedError(f"cube structured lowering expects at least {min_len} tuple elements")
        if max_len is not None and len(elements) > max_len:
            raise NotImplementedError(f"cube structured lowering expects at most {max_len} tuple elements")
        return tuple(self._lower_expr(element, env, indent=indent, into=into) for element in elements)

    def _extract_static_value(self, expr: SemanticExpr, *, context: str) -> object:
        if isinstance(expr, SemanticLiteralExpr):
            return expr.value
        if isinstance(expr, SemanticBindingRef):
            return expr.binding.value
        raise NotImplementedError(f"{context} must be a compile-time constant in TileLang DSL v1 lowering")

    def _extract_static_int(self, expr: SemanticExpr, *, context: str) -> int:
        value = self._extract_static_value(expr, context=context)
        if isinstance(value, bool) or not isinstance(value, int):
            raise NotImplementedError(f"{context} must be an integer constant in TileLang DSL v1 lowering")
        return value

    def _extract_static_bool(self, expr: SemanticExpr, *, context: str) -> bool:
        value = self._extract_static_value(expr, context=context)
        if not isinstance(value, bool):
            raise NotImplementedError(f"{context} must be a boolean constant in TileLang DSL v1 lowering")
        return value

    def _extract_static_string(self, expr: SemanticExpr, *, context: str) -> str:
        value = self._extract_static_value(expr, context=context)
        if not isinstance(value, str):
            raise NotImplementedError(f"{context} must be a string constant in TileLang DSL v1 lowering")
        return value

    def _extract_optional_static_string(self, expr: SemanticExpr, *, context: str) -> str | None:
        if self._is_none_meta_expr(expr):
            return None
        return self._extract_static_string(expr, context=context)

    def _is_none_meta_expr(self, expr: SemanticExpr | None) -> bool:
        return isinstance(expr, SemanticLiteralExpr) and expr.value is None

    def _lower_compare_expr(
        self,
        op: str,
        lhs: _RenderedValue,
        rhs: _RenderedValue,
        *,
        indent: int,
        desired_name: str | None,
        into: list[str],
    ) -> _RenderedValue:
        result_name = desired_name or self._new_temp()
        if isinstance(lhs.type, SemanticIndexType) and isinstance(rhs.type, SemanticIndexType):
            index_predicates = {
                "eq": "eq",
                "ne": "ne",
                "gt": "sgt",
                "lt": "slt",
                "ge": "sge",
                "le": "sle",
            }
            predicate = index_predicates[op]
            cmp_name = "arith.cmpi"
        elif (
            isinstance(lhs.type, SemanticIndexType)
            and isinstance(rhs.type, SemanticScalarType)
            and is_integer_dtype(rhs.type.dtype)
            and integer_bitwidth(rhs.type.dtype) in {8, 16, 32, 64}
        ):
            lhs = self._coerce_rendered_value(lhs, rhs.type, indent=indent, into=into)
        elif (
            isinstance(rhs.type, SemanticIndexType)
            and isinstance(lhs.type, SemanticScalarType)
            and is_integer_dtype(lhs.type.dtype)
            and integer_bitwidth(lhs.type.dtype) in {8, 16, 32, 64}
        ):
            rhs = self._coerce_rendered_value(rhs, lhs.type, indent=indent, into=into)

        if isinstance(lhs.type, SemanticScalarType) and lhs.type == rhs.type:
            if lhs.type.dtype.name in {"f16", "bf16", "f32"}:
                float_predicates = {
                    "eq": "oeq",
                    "ne": "une",
                    "gt": "ogt",
                    "lt": "olt",
                    "ge": "oge",
                    "le": "ole",
                }
                predicate = float_predicates[op]
                cmp_name = "arith.cmpf"
            else:
                int_sign = integer_signedness(lhs.type.dtype)
                int_predicates = {
                    "eq": "eq",
                    "ne": "ne",
                    "gt": "ugt" if int_sign == "unsigned" else "sgt",
                    "lt": "ult" if int_sign == "unsigned" else "slt",
                    "ge": "uge" if int_sign == "unsigned" else "sge",
                    "le": "ule" if int_sign == "unsigned" else "sle",
                }
                predicate = int_predicates[op]
                cmp_name = "arith.cmpi"
            into.append(
                self._indent(indent)
                + f"{result_name} = {cmp_name} {predicate}, {lhs.name}, {rhs.name} : "
                f"{self._render_type(lhs.type)}"
            )
            return _RenderedValue(name=result_name, type=_I1_TYPE)

        if isinstance(lhs.type, SemanticIndexType) and isinstance(rhs.type, SemanticIndexType):
            into.append(
                self._indent(indent)
                + f"{result_name} = {cmp_name} {predicate}, {lhs.name}, {rhs.name} : {self._render_type(lhs.type)}"
            )
            return _RenderedValue(name=result_name, type=_I1_TYPE)

        raise NotImplementedError(
            f"comparison lowering requires matching scalar types or index operands, got {lhs.type!r} and {rhs.type!r}"
        )

    def _lower_bool_expr(
        self,
        op: str,
        lhs_expr: SemanticExpr,
        rhs_expr: SemanticExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        desired_name: str | None,
        into: list[str],
    ) -> _RenderedValue:
        lhs = self._lower_condition(lhs_expr, env, indent=indent, into=into)
        rhs = self._lower_condition(rhs_expr, env, indent=indent, into=into)
        result_name = desired_name or self._new_temp()
        arith_op = "arith.andi" if op == "and" else "arith.ori"
        into.append(
            self._indent(indent)
            + f"{result_name} = {arith_op} {lhs.name}, {rhs.name} : i1"
        )
        return _RenderedValue(name=result_name, type=_I1_TYPE)

    def _render_string_literal(self, expr: SemanticExpr) -> str:
        if isinstance(expr, SemanticLiteralExpr) and isinstance(expr.value, str):
            escaped = expr.value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        if isinstance(expr, SemanticBindingRef) and isinstance(expr.binding.value, str):
            escaped = expr.binding.value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        raise NotImplementedError("expected a string literal for TileLang DSL advanced-family lowering")

    def _has_optional_string_literal(self, expr: SemanticExpr) -> bool:
        if isinstance(expr, SemanticLiteralExpr):
            return isinstance(expr.value, str)
        if isinstance(expr, SemanticBindingRef):
            return isinstance(expr.binding.value, str)
        return False

    def _render_dtype_symbol(self, expr: SemanticExpr, *, context: str) -> str:
        if isinstance(expr, SemanticSymbolExpr) and isinstance(expr.value, ScalarType):
            return expr.value.name
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "dtype"
            and isinstance(expr.binding.value, ScalarType)
        ):
            return expr.binding.value.name
        raise NotImplementedError(f"{context} expects a dtype symbol in TileLang DSL v1 lowering")

    def _lower_to_i1(
        self,
        expr: SemanticExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        value = self._lower_expr(expr, env, indent=indent, into=into)
        if isinstance(value.type, SemanticScalarType) and value.type.dtype.name == "i1":
            return value
        raise NotImplementedError("expected an i1 operand during TileLang DSL v1 lowering")

    def _lower_to_i64(
        self,
        expr: SemanticExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        value = self._lower_expr(expr, env, indent=indent, into=into)
        return self._coerce_rendered_to_i64(value, indent=indent, into=into)

    def _lower_to_i32(
        self,
        expr: SemanticExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        value = self._lower_expr(expr, env, indent=indent, into=into)
        if isinstance(value.type, SemanticIndexType) or (
            isinstance(value.type, SemanticScalarType) and is_integer_dtype(value.type.dtype)
        ):
            return self._coerce_rendered_value(value, _I32_TYPE, indent=indent, into=into)
        raise NotImplementedError("expected an i32 or index operand during TileLang DSL v1 lowering")

    def _lower_to_index(
        self,
        expr: SemanticExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        value = self._lower_expr(expr, env, indent=indent, into=into)
        return self._coerce_rendered_to_index(value, indent=indent, into=into)

    def _coerce_rendered_to_index(
        self,
        value: _RenderedValue,
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        if isinstance(value.type, SemanticIndexType):
            return value
        if isinstance(value.type, SemanticScalarType) and is_integer_dtype(value.type.dtype):
            bits = integer_bitwidth(value.type.dtype)
            if bits in {8, 16}:
                value = self._coerce_rendered_value(value, _I32_TYPE, indent=indent, into=into)
            elif bits in {32, 64}:
                value = self._bridge_rendered_to_signless_integer(value, indent=indent, into=into)
            else:
                value = None
            if value is not None:
                cast_name = self._new_temp()
                into.append(
                    self._indent(indent)
                    + f"{cast_name} = arith.index_cast {value.name} : {value.type.dtype.name} to index"
                )
                return _RenderedValue(name=cast_name, type=SemanticIndexType())
        raise NotImplementedError("expected an integer scalar or index operand during TileLang DSL v1 lowering")

    def _coerce_rendered_to_i64(
        self,
        value: _RenderedValue,
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        if isinstance(value.type, SemanticIndexType) or (
            isinstance(value.type, SemanticScalarType) and is_integer_dtype(value.type.dtype)
        ):
            return self._coerce_rendered_value(value, _I64_TYPE, indent=indent, into=into)
        raise NotImplementedError("expected an i64 or index operand during TileLang DSL v1 lowering")

    def _lower_remaining_to_i32(
        self,
        expr: SemanticExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        value = self._lower_expr(expr, env, indent=indent, into=into)
        if isinstance(value.type, SemanticIndexType) or (
            isinstance(value.type, SemanticScalarType) and is_integer_dtype(value.type.dtype)
        ):
            return self._coerce_rendered_value(value, _I32_TYPE, indent=indent, into=into)
        raise NotImplementedError("tail make_mask lowering expects an i32 or index remaining operand")

    def _bridge_rendered_to_signless_integer(
        self,
        value: _RenderedValue,
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        if not isinstance(value.type, SemanticScalarType) or not is_integer_dtype(value.type.dtype):
            return value
        raw_type = self._signless_integer_scalar_type(value.type)
        if raw_type is None:
            return value
        cast_name = self._new_temp()
        into.append(
            self._indent(indent)
            + f"{cast_name} = builtin.unrealized_conversion_cast {value.name} : "
            f"{self._render_type(value.type)} to {self._render_type(raw_type)}"
        )
        return _RenderedValue(name=cast_name, type=raw_type)

    def _bridge_rendered_ptr_element_to_signless_integer(
        self,
        value: _RenderedValue,
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        if not isinstance(value.type, SemanticPtrType) or not is_integer_dtype(value.type.element_dtype):
            return value
        signless_element_type = self._signless_integer_scalar_type(
            SemanticScalarType(dtype=value.type.element_dtype)
        )
        if signless_element_type is None:
            return value
        target_type = SemanticPtrType(
            element_dtype=signless_element_type.dtype,
            memory_space=value.type.memory_space,
        )
        if value.type == target_type:
            return value
        rendered_target_type = self._render_type(target_type)
        cache_key = (value.name, rendered_target_type)
        existing = self._castptr_cache.get(cache_key)
        if existing is not None:
            return _RenderedValue(name=existing, type=target_type)
        cast_name = self._new_temp()
        into.append(
            self._indent(indent)
            + f"{cast_name} = pto.castptr {value.name} : "
            f"{self._render_type(value.type)} -> {rendered_target_type}"
        )
        self._castptr_cache[cache_key] = cast_name
        return _RenderedValue(name=cast_name, type=target_type)

    def _bridge_rendered_integer_to_target(
        self,
        value: _RenderedValue,
        target_type: SemanticScalarType,
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        if value.type == target_type:
            return value
        cast_name = self._new_temp()
        into.append(
            self._indent(indent)
            + f"{cast_name} = builtin.unrealized_conversion_cast {value.name} : "
            f"{self._render_type(value.type)} to {self._render_type(target_type)}"
        )
        return _RenderedValue(name=cast_name, type=target_type)

    def _materialize_copy_buffer_ptr(
        self,
        value: _RenderedValue,
        *,
        indent: int,
        into: list[str],
    ) -> tuple[str, str]:
        ptr_type = self._render_copy_buffer_type(value.type)
        cache_key = (value.name, ptr_type)
        existing = self._castptr_cache.get(cache_key)
        if existing is not None:
            return existing, ptr_type

        if isinstance(value.type, SemanticTileType):
            value = self._materialize_tile_memref(value, indent=indent, into=into)

        if self._is_memref_like_type(value.type):
            cast_name = self._new_temp()
            into.append(
                self._indent(indent)
                + f"{cast_name} = pto.castptr {value.name} : {self._render_type(value.type)} -> {ptr_type}"
            )
            self._castptr_cache[cache_key] = cast_name
            return cast_name, ptr_type

        return value.name, ptr_type

    def _materialize_tile_ptr(
        self,
        value: _RenderedValue,
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        """Convert a SemanticTileType value directly to a typed !pto.ptr.

        Unlike _materialize_tile_memref which produces a memref, this emits
        pto.tile_buf_addr with a !pto.ptr<...> result type so the ptr can be
        used directly with pto.addptr / pto.vldas / pto.vldus.
        """
        existing = self._tile_ptr_cache.get(value.name)
        if existing is not None:
            return existing
        if not isinstance(value.type, SemanticTileType):
            return value
        ptr_type = self._render_copy_buffer_type(value.type)
        ptr_name = self._new_temp()
        into.append(
            self._indent(indent)
            + f"{ptr_name} = pto.tile_buf_addr {value.name} : "
            + f"{self._render_type(value.type)} -> {ptr_type}"
        )
        rendered = _RenderedValue(name=ptr_name, type=_RenderedTextualType(ptr_type))
        self._tile_ptr_cache[value.name] = rendered
        return rendered

    def _coerce_rendered_value(
        self,
        value: _RenderedValue,
        target_type: SemanticType,
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        def _scalar_int_bits(dtype: ScalarType) -> int | None:
            if dtype.name == "i1":
                return 1
            return integer_bitwidth(dtype)

        def _scalar_int_sign(dtype: ScalarType) -> str:
            sign = integer_signedness(dtype)
            return "signless" if sign is None else sign

        def _signless_int_type_for_bits(bits: int) -> SemanticScalarType:
            if bits not in {8, 16, 32, 64}:
                raise NotImplementedError(
                    f"unsupported integer bitwidth {bits!r} for signless coercion in TileLang DSL v1 lowering"
                )
            return SemanticScalarType(dtype=ScalarType(f"i{bits}"))

        if type(value.type) is type(target_type) and value.type == target_type:
            return value
        if isinstance(value.type, SemanticIndexType) and isinstance(target_type, SemanticScalarType):
            target_int_bits = _scalar_int_bits(target_type.dtype)
            target_sign = _scalar_int_sign(target_type.dtype)
            signless_target_type = self._signless_integer_scalar_type(target_type)
            if signless_target_type is None and target_int_bits in {8, 16, 32, 64}:
                signless_target_type = target_type
            if target_int_bits in {8, 16, 32, 64} and signless_target_type is not None:
                carrier_type = (
                    _signless_int_type_for_bits(32)
                    if target_int_bits in {8, 16}
                    else _signless_int_type_for_bits(target_int_bits)
                )
                op = "arith.index_castui" if target_sign == "unsigned" else "arith.index_cast"
                cast_name = self._new_temp()
                into.append(
                    self._indent(indent)
                    + f"{cast_name} = {op} {value.name} : index to {carrier_type.dtype.name}"
                )
                lowered_value = _RenderedValue(name=cast_name, type=carrier_type)
                if target_int_bits in {8, 16}:
                    lowered_value = self._coerce_rendered_value(
                        lowered_value,
                        signless_target_type,
                        indent=indent,
                        into=into,
                    )
                if signless_target_type != target_type:
                    return self._coerce_rendered_value(
                        lowered_value,
                        target_type,
                        indent=indent,
                        into=into,
                    )
                return lowered_value
            if target_type.dtype.name in {"f16", "bf16", "f32"}:
                index_to_int_name = self._new_temp()
                index_to_int_op = "arith.index_castui"
                into.append(
                    self._indent(indent)
                    + f"{index_to_int_name} = {index_to_int_op} {value.name} : index to i64"
                )
                cast_name = self._new_temp()
                into.append(
                    self._indent(indent)
                    + f"{cast_name} = arith.uitofp {index_to_int_name} : i64 to {target_type.dtype.name}"
                )
                return _RenderedValue(name=cast_name, type=target_type)
        if isinstance(value.type, SemanticScalarType) and isinstance(target_type, SemanticScalarType):
            src = value.type.dtype.name
            dst = target_type.dtype.name
            if src == dst:
                return value
            src_bits = _scalar_int_bits(value.type.dtype)
            dst_bits = _scalar_int_bits(target_type.dtype)
            if src_bits is not None and dst_bits is not None:
                src_sign = _scalar_int_sign(value.type.dtype)
                signless_value = self._bridge_rendered_to_signless_integer(value, indent=indent, into=into)
                signless_target_type = self._signless_integer_scalar_type(target_type) or target_type
                if signless_value.type == signless_target_type:
                    if signless_target_type != target_type:
                        return self._bridge_rendered_integer_to_target(
                            signless_value,
                            target_type,
                            indent=indent,
                            into=into,
                        )
                    return signless_value
                cast_name = self._new_temp()
                if src_bits < dst_bits:
                    op = "arith.extui" if _scalar_int_sign(value.type.dtype) == "unsigned" else "arith.extsi"
                elif src_bits > dst_bits:
                    op = "arith.trunci"
                else:
                    raise NotImplementedError(
                        f"unsupported same-width integer coercion from {value.type!r} to {target_type!r} "
                        "in TileLang DSL v1 lowering"
                    )
                into.append(
                    self._indent(indent)
                    + f"{cast_name} = {op} {signless_value.name} : "
                    f"{self._render_type(signless_value.type)} to {self._render_type(signless_target_type)}"
                )
                lowered_value = _RenderedValue(name=cast_name, type=signless_target_type)
                if signless_target_type != target_type:
                    return self._bridge_rendered_integer_to_target(
                        lowered_value,
                        target_type,
                        indent=indent,
                        into=into,
                    )
                return lowered_value
            if src_bits is not None and dst in {"f16", "bf16", "f32"}:
                signless_value = self._bridge_rendered_to_signless_integer(value, indent=indent, into=into)
                cast_name = self._new_temp()
                op = "arith.uitofp" if _scalar_int_sign(value.type.dtype) == "unsigned" else "arith.sitofp"
                into.append(
                    self._indent(indent)
                    + f"{cast_name} = {op} {signless_value.name} : "
                    f"{self._render_type(signless_value.type)} to {dst}"
                )
                return _RenderedValue(name=cast_name, type=target_type)
            if src in {"f16", "bf16", "f32"} and dst_bits is not None:
                signless_target_type = self._signless_integer_scalar_type(target_type) or target_type
                cast_name = self._new_temp()
                op = "arith.fptoui" if _scalar_int_sign(target_type.dtype) == "unsigned" else "arith.fptosi"
                into.append(
                    self._indent(indent)
                    + f"{cast_name} = {op} {value.name} : {src} to {self._render_type(signless_target_type)}"
                )
                lowered_value = _RenderedValue(name=cast_name, type=signless_target_type)
                if signless_target_type != target_type:
                    return self._bridge_rendered_integer_to_target(
                        lowered_value,
                        target_type,
                        indent=indent,
                        into=into,
                    )
                return lowered_value
            cast_name = self._new_temp()
            if src in {"f16", "bf16", "f32"} and dst in {"f16", "bf16", "f32"}:
                op = "arith.extf" if src in {"f16", "bf16"} and dst == "f32" else "arith.truncf"
                into.append(
                    self._indent(indent)
                    + f"{cast_name} = {op} {value.name} : {src} to {dst}"
                )
                return _RenderedValue(name=cast_name, type=target_type)
        raise NotImplementedError(
            f"unsupported value coercion from {value.type!r} to {target_type!r} in TileLang DSL v1 lowering"
        )

    def _materialize_strict_vecscope_capture(
        self,
        capture: _RenderedValue,
        binding: SemanticBinding,
        *,
        indent: int,
        into: list[str],
    ) -> tuple[_RenderedValue, _RenderedValue]:
        if not self._is_memref_like_type(capture.type):
            return capture, _RenderedValue(name=binding.ssa_name, type=binding.type)

        ptr_name, ptr_type = self._materialize_copy_buffer_ptr(
            capture,
            indent=indent,
            into=into,
        )
        rendered_ptr_type = _RenderedTextualType(ptr_type)
        return (
            _RenderedValue(name=ptr_name, type=rendered_ptr_type),
            _RenderedValue(name=binding.ssa_name, type=rendered_ptr_type),
        )

    def _mask_suffix(self, ty: SemanticType) -> str:
        if not isinstance(ty, SemanticMaskType):
            raise NotImplementedError("tail make_mask lowering expects a mask result type")
        return ty.granularity

    def _is_dtype_meta_expr(self, expr: SemanticExpr) -> bool:
        if isinstance(expr, SemanticSymbolExpr):
            return isinstance(expr.value, ScalarType) and expr.type.kind == "dtype"
        if isinstance(expr, SemanticBindingRef):
            return (
                isinstance(expr.type, SemanticMetaType)
                and expr.type.kind == "dtype"
                and isinstance(expr.binding.value, ScalarType)
            )
        return False

    def _lower_subscript_access(
        self,
        expr: SemanticSubscriptAccess,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        desired_name: str | None,
        into: list[str] | None,
    ) -> _RenderedValue:
        if isinstance(expr.base, SemanticTupleExpr):
            if not isinstance(expr.index, SemanticLiteralExpr) or not isinstance(expr.index.value, int):
                raise NotImplementedError("tuple indices must be integer literals in TileLang DSL v1 lowering")
            if expr.index.value < 0 or expr.index.value >= len(expr.base.elements):
                raise NotImplementedError(
                    f"tuple subscript index {expr.index.value} is out of bounds for tuple length {len(expr.base.elements)}"
                )
            return self._lower_expr(
                expr.base.elements[expr.index.value],
                env,
                indent=indent,
                desired_name=desired_name,
                into=into,
            )
        if (
            into is not None
            and isinstance(expr.base, SemanticAttributeAccess)
            and expr.base.attr == "valid_shape"
            and isinstance(expr.base.base, SemanticBindingRef)
            and isinstance(expr.base.base.type, SemanticTileType)
            and isinstance(expr.index, SemanticLiteralExpr)
            and isinstance(expr.index.value, int)
        ):
            return self._materialize_tile_valid_dim(
                expr.base.base.binding,
                expr.index.value,
                indent=indent,
                into=into,
                desired_name=desired_name,
            )
        if (
            into is not None
            and isinstance(expr.base, SemanticAttributeAccess)
            and expr.base.attr in {"shape", "valid_shape", "strides"}
            and isinstance(expr.base.base, SemanticBindingRef)
            and isinstance(
                expr.base.base.type,
                (SemanticTensorViewType, SemanticPartitionTensorViewType),
            )
            and isinstance(expr.index, SemanticLiteralExpr)
            and isinstance(expr.index.value, int)
        ):
            tensor_value = env.get(
                expr.base.base.binding.name,
                _RenderedValue(expr.base.base.binding.ssa_name, expr.base.base.type),
            )
            result_name = desired_name or self._new_temp()
            axis_value = self._materialize_constant(expr.index.value, SemanticIndexType())
            op_name = (
                "pto.get_tensor_view_stride"
                if expr.base.attr == "strides"
                else "pto.get_tensor_view_dim"
            )
            into.append(
                self._indent(indent)
                + f"{result_name} = {op_name} {tensor_value.name}, {axis_value} : "
                + f"{self._render_type(tensor_value.type)} -> index"
            )
            return _RenderedValue(name=result_name, type=SemanticIndexType())
        value = self._extract_shape_subscript_value(expr, env)
        if isinstance(value, _RenderedValue):
            return value
        if desired_name is not None and into is not None:
            into.append(
                self._indent(indent)
                + f"{desired_name} = arith.constant {self._format_constant(value, expr.type)} : "
                f"{self._render_arith_constant_type(expr.type)}"
            )
            return _RenderedValue(name=desired_name, type=expr.type)
        return _RenderedValue(
            name=self._materialize_constant(value, expr.type),
            type=expr.type,
        )

    def _tensor_shape_binding_name(self, tensor_name: str, axis: int) -> str:
        return f"__shape_{tensor_name}_{axis}"

    def _tensor_stride_binding_name(self, tensor_name: str, axis: int) -> str:
        return f"__stride_{tensor_name}_{axis}"

    def _materialize_tile_memref(
        self,
        value: _RenderedValue,
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        existing = self._tile_memref_cache.get(value.name)
        if existing is not None:
            return existing
        if not isinstance(value.type, SemanticTileType):
            return value
        memref_type = _RenderedTextualType(
            self._render_memref_type(
                element_dtype=self._render_dtype_name(value.type.element_dtype),
                shape=value.type.shape if value.type.shape is not None else ("?",) * value.type.rank,
                memory_space=value.type.memory_space or "ub",
            )
        )
        memref_name = self._new_temp()
        into.append(
            self._indent(indent)
            + f"{memref_name} = pto.tile_buf_addr {value.name} : "
            + f"{self._render_type(value.type)} -> {self._render_type(memref_type)}"
        )
        rendered = _RenderedValue(name=memref_name, type=memref_type)
        self._tile_memref_cache[value.name] = rendered
        return rendered

    def _materialize_tile_valid_dim(
        self,
        binding: object,
        axis: int,
        *,
        indent: int,
        into: list[str],
        desired_name: str | None = None,
    ) -> _RenderedValue:
        cache_key = (binding.name, axis)
        existing = self._tile_valid_dim_cache.get(cache_key)
        if existing is not None:
            return existing
        source = _RenderedValue(name=binding.ssa_name, type=binding.type)
        op_name = "pto.tile_valid_rows" if axis == 0 else "pto.tile_valid_cols"
        result_name = desired_name or self._new_temp()
        into.append(
            self._indent(indent)
            + f"{result_name} = {op_name} {source.name} : "
            + f"{self._render_type(source.type)} -> index"
        )
        rendered = _RenderedValue(name=result_name, type=SemanticIndexType())
        self._tile_valid_dim_cache[cache_key] = rendered
        return rendered

    def _extract_shape_subscript_value(
        self,
        expr: SemanticSubscriptAccess,
        env: dict[str, _RenderedValue],
    ) -> int | _RenderedValue:
        if not isinstance(expr.base, SemanticAttributeAccess):
            raise NotImplementedError("only shape/stride indexing is supported in TileLang DSL v1 lowering")
        if expr.base.attr not in {"shape", "valid_shape", "strides"}:
            raise NotImplementedError(
                "only `.shape[...]`, `.valid_shape[...]`, and `.strides[...]` indexing are supported in TileLang DSL v1 lowering"
            )
        if not isinstance(expr.index, SemanticLiteralExpr) or not isinstance(expr.index.value, int):
            raise NotImplementedError("shape/stride indices must be integer literals in TileLang DSL v1 lowering")
        if not isinstance(expr.base.base, SemanticBindingRef):
            raise NotImplementedError("shape/stride indexing expects a bound TensorView or Tile value")

        base_binding = expr.base.base.binding
        base_value = env.get(base_binding.name, _RenderedValue(base_binding.ssa_name, base_binding.type))
        base_type = base_value.type
        index = expr.index.value

        if isinstance(base_type, SemanticTileType):
            if expr.base.attr == "shape":
                if base_type.shape is None:
                    raise NotImplementedError("dynamic Tile shapes are not supported in TileLang DSL v1 lowering")
                return base_type.shape[index]
            if base_type.valid_shape is None:
                raise NotImplementedError("dynamic Tile shapes are not supported in TileLang DSL v1 lowering")
            valid_dim = base_type.valid_shape[index]
            if valid_dim is not None:
                return valid_dim
            return _RenderedValue(name=base_binding.ssa_name, type=base_type)

        if isinstance(base_type, (SemanticTensorViewType, SemanticPartitionTensorViewType)):
            if expr.base.attr == "strides":
                hidden_name = self._tensor_stride_binding_name(base_binding.name, index)
            else:
                hidden_name = self._tensor_shape_binding_name(base_binding.name, index)
            hidden_value = env.get(hidden_name)
            if hidden_value is None:
                raise NotImplementedError(
                    f"missing TensorView/PartitionTensorView {expr.base.attr} binding for '{base_binding.name}.{expr.base.attr}[{index}]'"
                )
            return hidden_value

        raise NotImplementedError("shape/stride indexing expects a Tile, TensorView, or PartitionTensorView operand")

    def _format_shape_tuple(self, shape: tuple[int | None, ...]) -> str:
        return "(" + ", ".join("?" if dim is None else str(dim) for dim in shape) + ")"

    def _materialize_constant(self, value: object, ty: SemanticType) -> str:
        cache_key = (self._render_type(ty), value)
        if cache_key in self._constant_cache:
            return self._constant_cache[cache_key]

        raw_type = self._signless_integer_scalar_type(ty)
        if raw_type is not None:
            raw_name = self._materialize_constant(value, raw_type)
            name = self._constant_name(value, ty)
            self._constant_cache[cache_key] = name
            self._constant_lines.append(
                self._indent(4)
                + f"{name} = builtin.unrealized_conversion_cast {raw_name} : "
                f"{self._render_type(raw_type)} to {self._render_type(ty)}"
            )
            return name

        name = self._constant_name(value, ty)
        self._constant_cache[cache_key] = name
        self._constant_lines.append(
            self._indent(4)
            + f"{name} = arith.constant {self._format_constant(value, ty)} : "
            f"{self._render_arith_constant_type(ty)}"
        )
        return name

    def _signless_integer_scalar_type(self, ty: SemanticType) -> SemanticScalarType | None:
        if not isinstance(ty, SemanticScalarType) or not is_integer_dtype(ty.dtype):
            return None
        signedness = integer_signedness(ty.dtype)
        if signedness in {None, "signless"}:
            return None
        bitwidth = integer_bitwidth(ty.dtype)
        if bitwidth not in {8, 16, 32, 64}:
            raise NotImplementedError(
                f"unsupported integer bitwidth {bitwidth!r} for signless literal lowering"
            )
        return SemanticScalarType(dtype=ScalarType(f"i{bitwidth}"))

    def _lower_literal_expr(
        self,
        value: object,
        ty: SemanticType,
        *,
        indent: int,
        desired_name: str | None = None,
        into: list[str] | None = None,
    ) -> _RenderedValue:
        raw_type = self._signless_integer_scalar_type(ty) or ty
        if desired_name is not None and into is not None and raw_type == ty:
            into.append(
                self._indent(indent)
                + f"{desired_name} = arith.constant {self._format_constant(value, ty)} : "
                f"{self._render_arith_constant_type(ty)}"
            )
            return _RenderedValue(name=desired_name, type=ty)

        if desired_name is not None and into is not None:
            raw_name = self._new_temp()
            into.append(
                self._indent(indent)
                + f"{raw_name} = arith.constant {self._format_constant(value, raw_type)} : "
                f"{self._render_arith_constant_type(raw_type)}"
            )
            into.append(
                self._indent(indent)
                + f"{desired_name} = builtin.unrealized_conversion_cast {raw_name} : "
                f"{self._render_type(raw_type)} to {self._render_type(ty)}"
            )
            return _RenderedValue(name=desired_name, type=ty)

        return _RenderedValue(
            name=self._materialize_constant(value, ty),
            type=ty,
        )

    def _constant_name(self, value: object, ty: SemanticType) -> str:
        if isinstance(ty, SemanticIndexType):
            stem = f"c{value}"
        elif isinstance(ty, SemanticScalarType):
            if ty.dtype.name == "i1" and isinstance(value, bool):
                stem = "true" if value else "false"
            else:
                stem = f"c{value}_{ty.dtype.name}"
        else:
            stem = "cst"
        # Keep generated SSA names MLIR-safe for constants whose textual value
        # contains punctuation such as decimal points or scientific-notation
        # exponents (for example f32 max -> `3.4028235e+38`).
        stem = re.sub(r"[^0-9A-Za-z_]", "_", stem)
        stem = re.sub(r"_+", "_", stem).strip("_") or "cst"
        if stem[0].isdigit():
            stem = f"c_{stem}"

        name = f"%{stem}"
        existing = {line.split(" = ", 1)[0].strip() for line in self._constant_lines}
        if name not in existing:
            return name
        suffix = 0
        while f"{name}_{suffix}" in existing:
            suffix += 1
        return f"{name}_{suffix}"

    def _format_constant(self, value: object, ty: SemanticType) -> str:
        if isinstance(ty, SemanticIndexType):
            return str(value)
        if isinstance(ty, SemanticScalarType):
            if ty.dtype.name in {"f16", "bf16", "f32"} and isinstance(
                value, (bool, int, float)
            ):
                return self._format_float_constant(float(value), ty.dtype.name)
            if ty.dtype.name == "i1" and isinstance(value, bool):
                return "1" if value else "0"
            return str(value)
        raise NotImplementedError(f"unsupported constant type {ty!r}")

    def _render_arith_constant_type(self, ty: SemanticType) -> str:
        if isinstance(ty, SemanticScalarType) and is_integer_dtype(ty.dtype):
            width = integer_bitwidth(ty.dtype)
            if width is None:
                raise NotImplementedError(
                    f"unsupported integer dtype {ty.dtype.name!r} for arith.constant emission"
                )
            return f"i{width}"
        return self._render_type(ty)

    def _format_float_constant(self, value: float, dtype_name: str) -> str:
        # Emit stable bit-pattern literals for values that are parse-sensitive
        # (`inf`/`nan`) or sign-sensitive (`-0.0`).
        if math.isnan(value):
            return self._float_nan_bit_pattern(dtype_name)
        if math.isinf(value):
            sign_bit = value < 0.0
            return self._float_inf_bit_pattern(dtype_name, sign_bit=sign_bit)
        if value == 0.0 and math.copysign(1.0, value) < 0.0:
            return self._float_to_bit_pattern_literal(value, dtype_name)
        return str(value)

    def _float_nan_bit_pattern(self, dtype_name: str) -> str:
        if dtype_name == "f16":
            return "0x7E00"
        if dtype_name == "bf16":
            return "0x7FC0"
        if dtype_name == "f32":
            return "0x7FC00000"
        raise NotImplementedError(
            f"unsupported float dtype {dtype_name!r} for NaN constant emission"
        )

    def _float_inf_bit_pattern(self, dtype_name: str, *, sign_bit: bool) -> str:
        if dtype_name == "f16":
            return "0xFC00" if sign_bit else "0x7C00"
        if dtype_name == "bf16":
            return "0xFF80" if sign_bit else "0x7F80"
        if dtype_name == "f32":
            return "0xFF800000" if sign_bit else "0x7F800000"
        raise NotImplementedError(
            f"unsupported float dtype {dtype_name!r} for inf constant emission"
        )

    def _float_to_bit_pattern_literal(self, value: float, dtype_name: str) -> str:
        if dtype_name == "f16":
            bits = struct.unpack(">H", struct.pack(">e", value))[0]
            return f"0x{bits:04X}"
        if dtype_name == "bf16":
            bits = struct.unpack(">I", struct.pack(">f", value))[0] >> 16
            return f"0x{bits:04X}"
        if dtype_name == "f32":
            bits = struct.unpack(">I", struct.pack(">f", value))[0]
            return f"0x{bits:08X}"
        raise NotImplementedError(
            f"unsupported float dtype {dtype_name!r} for bit-pattern emission"
        )

    def _render_binary_op(self, op: str, ty: SemanticType) -> str:
        if isinstance(ty, SemanticIndexType):
            if op == "add":
                return "arith.addi"
            if op == "sub":
                return "arith.subi"
            if op == "mul":
                return "arith.muli"
            if op == "mod":
                if isinstance(ty, SemanticIndexType):
                    return "arith.remui"
            if op == "floordiv":
                return "arith.divui"
        if isinstance(ty, SemanticScalarType):
            dtype = ty.dtype
            if is_float_dtype(dtype):
                if op == "add":
                    return "arith.addf"
                if op == "sub":
                    return "arith.subf"
                if op == "mul":
                    return "arith.mulf"
            if is_integer_dtype(dtype):
                if op == "add":
                    return "arith.addi"
                if op == "sub":
                    return "arith.subi"
                if op == "mul":
                    return "arith.muli"
                if op == "mod":
                    sign = integer_signedness(dtype)
                    return "arith.remui" if sign == "unsigned" else "arith.remsi"
                if op == "floordiv":
                    sign = integer_signedness(dtype)
                    return "arith.divui" if sign == "unsigned" else "arith.floordivsi"
                if op == "bitand":
                    return "arith.andi"
                if op == "bitor":
                    return "arith.ori"
                if op == "bitxor":
                    return "arith.xori"
                if op == "lshift":
                    return "arith.shli"
                if op == "rshift":
                    sign = integer_signedness(dtype)
                    return "arith.shrui" if sign == "unsigned" else "arith.shrsi"
        raise NotImplementedError(f"unsupported binary op '{op}' for type {ty!r}")

    def _render_type(self, ty: SemanticType) -> str:
        if isinstance(ty, _RenderedTextualType):
            return ty.text
        if isinstance(ty, SemanticIndexType):
            return "index"
        if isinstance(ty, SemanticScalarType):
            return self._render_dtype_name(ty.dtype)
        if isinstance(ty, SemanticPtrType):
            return f"!pto.ptr<{self._render_dtype_name(ty.element_dtype)}, {self._render_address_space_name(ty.memory_space)}>"
        if isinstance(ty, SemanticTensorViewType):
            return self._render_tensor_view_type(
                element_dtype=self._render_dtype_name(ty.element_dtype),
                shape=("?",) * ty.rank,
            )
        if isinstance(ty, SemanticPartitionTensorViewType):
            return self._render_partition_tensor_view_type(
                element_dtype=self._render_dtype_name(ty.element_dtype),
                shape=("?",) * ty.rank,
            )
        if isinstance(ty, SemanticTileType):
            return self._render_tile_buf_type(ty)
        if isinstance(ty, SemanticAlignType):
            return "!pto.align"
        if isinstance(ty, SemanticMaskType):
            return f"!pto.mask<{ty.granularity}>"
        if isinstance(ty, SemanticVRegType):
            return f"!pto.vreg<{ty.lanes}x{self._render_dtype_name(ty.element_dtype)}>"
        if isinstance(ty, SemanticVectorType):
            dims = "x".join(str(dim) for dim in ty.shape)
            return f"vector<{dims}x{self._render_dtype_name(ty.element_dtype)}>"
        raise NotImplementedError(f"unsupported semantic type {ty!r}")

    def _render_dtype_name(self, dtype: ScalarType) -> str:
        low_precision = {
            "hif8": "!pto.hif8",
            "f8e4m3": "f8E4M3FN",
            "f8e5m2": "f8E5M2",
            "f4e1m2x2": "!pto.f4E1M2x2",
            "f4e2m1x2": "!pto.f4E2M1x2",
        }
        return low_precision.get(dtype.name, dtype.name)

    def _is_memref_like_type(self, ty: SemanticType) -> bool:
        return isinstance(ty, (SemanticTensorViewType, SemanticPartitionTensorViewType, SemanticTileType)) or (
            isinstance(ty, _RenderedTextualType) and ty.text.startswith("memref<")
        )

    def _render_copy_buffer_type(self, ty: SemanticType) -> str:
        if isinstance(ty, SemanticPtrType):
            return self._render_type(ty)
        if isinstance(ty, (SemanticTensorViewType, SemanticPartitionTensorViewType)):
            return f"!pto.ptr<{self._render_dtype_name(ty.element_dtype)}, gm>"
        if isinstance(ty, SemanticTileType):
            memory_space = ty.memory_space or "ub"
            return f"!pto.ptr<{self._render_dtype_name(ty.element_dtype)}, {self._render_address_space_name(memory_space)}>"
        return self._render_type(ty)

    def _render_memref_type(
        self,
        *,
        element_dtype: str,
        shape: tuple[int | str, ...],
        memory_space: str,
    ) -> str:
        dims = "x".join(str(dim) for dim in shape)
        return f"memref<{dims}x{element_dtype}, {self._render_memref_memory_space(memory_space)}>"

    def _render_tensor_view_type(
        self,
        *,
        element_dtype: str,
        shape: tuple[int | str, ...],
    ) -> str:
        dims = "x".join(str(dim) for dim in shape)
        return f"!pto.tensor_view<{dims}x{element_dtype}>"

    def _render_partition_tensor_view_type(
        self,
        *,
        element_dtype: str,
        shape: tuple[int | str, ...],
    ) -> str:
        dims = "x".join(str(dim) for dim in shape)
        return f"!pto.partition_tensor_view<{dims}x{element_dtype}>"

    def _render_memref_memory_space(self, memory_space: str) -> str:
        if memory_space == "gm":
            return "#pto.address_space<gm>"
        if memory_space == "ub":
            return "#pto.address_space<vec>"
        if memory_space in {"mat", "left", "right", "acc", "bias", "scaling"}:
            return f"#pto.address_space<{self._render_address_space_name(memory_space)}>"
        raise NotImplementedError(f"unsupported memref memory space '{memory_space}' in TileLang DSL v1 lowering")

    def _render_tile_buf_type(self, ty: SemanticTileType) -> str:
        if ty.shape is None:
            raise NotImplementedError("tile_buf lowering requires statically specialized Tile shape")
        if ty.rank not in (1, 2):
            raise NotImplementedError("tile_buf lowering only supports rank-1 or rank-2 Tile values")
        rows = ty.shape[0]
        cols = 1 if ty.rank == 1 else ty.shape[1]
        valid_shape = ty.valid_shape or ty.shape
        v_row = valid_shape[0]
        v_col = 1 if ty.rank == 1 else valid_shape[1]
        config = ty.config or TileConfig()
        compact_suffix = ""
        if config.compact_mode.value != "null":
            compact_suffix = f", compact={self._render_tile_buf_compact_mode(config.compact_mode)}"
        return (
            f"!pto.tile_buf<loc={self._render_tile_buf_loc(ty.memory_space or 'ub')}, "
            f"dtype={self._render_dtype_name(ty.element_dtype)}, rows={rows}, cols={cols}, "
            f"v_row={self._render_tile_buf_dim(v_row)}, v_col={self._render_tile_buf_dim(v_col)}, "
            f"blayout={config.b_layout.value}, slayout={config.s_layout.value}, "
            f"fractal={config.s_fractal_size}, pad={self._render_tile_buf_pad_value(config.pad_value)}{compact_suffix}>"
        )

    def _render_tile_buf_loc(self, memory_space: str) -> str:
        if memory_space == "ub":
            return "vec"
        if memory_space == "gm":
            return "gm"
        if memory_space in {"mat", "left", "right", "acc", "bias", "scaling"}:
            return memory_space
        raise NotImplementedError(f"unsupported tile_buf memory space '{memory_space}'")

    def _render_address_space_name(self, memory_space: str) -> str:
        mapping = {
            "gm": "gm",
            "mat": "l1",
            "left": "l0a",
            "right": "l0b",
            "acc": "l0c",
            "bias": "bt",
            "scaling": "fb",
            "ub": "ub",
        }
        try:
            return mapping[memory_space]
        except KeyError as exc:
            raise NotImplementedError(f"unsupported pointer memory space '{memory_space}'") from exc

    def _render_tile_buf_dim(self, dim: int | None) -> str:
        return "?" if dim is None else str(dim)

    def _render_tile_buf_pad_value(self, pad_value: PadValue) -> str:
        if pad_value.is_custom:
            raise NotImplementedError(
                "custom TileConfig.pad_value MLIR type rendering requires PTO tile_buf parser support for custom pad encodings"
            )
        return str(pad_value.encoded)

    def _render_tile_buf_compact_mode(self, compact_mode: CompactMode) -> str:
        if compact_mode == CompactMode.NULL:
            return "0"
        if compact_mode == CompactMode.NORMAL:
            return "1"
        if compact_mode == CompactMode.ROW_PLUS_ONE:
            return "2"
        raise NotImplementedError(f"unsupported TileConfig.compact_mode '{compact_mode}'")

    def _dtype_byte_width(self, dtype: ScalarType) -> int:
        try:
            return bytewidth(dtype)
        except TypeError as exc:
            raise NotImplementedError(f"unsupported DMA dtype '{dtype.name}' in TileLang DSL v1 lowering") from exc

    def _indent(self, indent: int) -> str:
        return " " * indent

    def _new_temp(self) -> str:
        name = f"%tmp_{self._temp_counter}"
        self._temp_counter += 1
        return name


def lower_semantic_kernel(kernel: SemanticKernel) -> AuthoringModule:
    """Lower the semantic model to the current authoring-form VPTO builder."""

    return AuthoringModule(kernel=kernel)


def lower_semantic_kernel_with_backend(
    kernel: SemanticKernel,
    backend: str = "text",
) -> "LoweringResult":
    """Lower semantic kernel with selectable backend (Issue #237).

    Args:
        kernel: The semantic kernel to lower.
        backend: Backend identifier ("text" or "pybind").

    Returns:
        LoweringResult from the specified backend.
    """
    from .lowering_backend import lower_with_backend
    return lower_with_backend(kernel, backend)


__all__ = ["AuthoringModule", "lower_semantic_kernel", "lower_semantic_kernel_with_backend"]
