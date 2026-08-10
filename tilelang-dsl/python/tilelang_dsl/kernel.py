# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Kernel descriptor surface for TileLang DSL v1."""

from __future__ import annotations

import os
import inspect
import ast
import importlib.util
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .types import (
    AnyMask,
    AnyType,
    MaskType,
    MemorySpace,
    PartitionTensorView,
    PointerType,
    ScalarType,
    TensorView,
    Tile,
    TileConfig,
    TileSpecialization,
    ViewConfig,
    TypeVariable,
    VectorType,
    WildcardType,
    is_integer_dtype,
)
from .frontend_ast import _DMA_CALL_KEYWORDS, build_frontend_kernel_node
from .lowering import lower_semantic_kernel
from .semantic import analyze_frontend_kernel
from .support_matrix import (
    ADVANCED_EXPR_PTO_CALLS,
    ADVANCED_TOPLEVEL_PTO_CALLS,
    ADVANCED_VECSCOPE_PTO_CALLS,
    CUBE_ONLY_PTO_CALLS,
    DEFERRED_PTO_SURFACES,
    SUPPORTED_TOPLEVEL_PTO_CALLS,
    SUPPORTED_VECSCOPE_PTO_CALLS,
    advanced_mode_message,
    deferred_surface_message,
)


_UNSET = object()
_INTERNAL_SOFT_MATH_MODULE_NAME = "tilelang_dsl._internal_soft_math"
_SUPPORTED_TEMPLATE_PTO_CALLS = frozenset(
    SUPPORTED_TOPLEVEL_PTO_CALLS
    | SUPPORTED_VECSCOPE_PTO_CALLS
    | ADVANCED_VECSCOPE_PTO_CALLS
    | ADVANCED_EXPR_PTO_CALLS
    | ADVANCED_TOPLEVEL_PTO_CALLS
    | CUBE_ONLY_PTO_CALLS
)

_DSL_DTYPE_NAMES = frozenset(
    {
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
        "hif8",
        "f8e4m3",
        "f8e5m2",
        "f4e1m2x2",
        "f4e2m1x2",
    }
)


_INLINE_PROC_REGISTRY: dict[tuple[str, str], "InlineProcDescriptor"] = {}
_INTERNAL_INLINE_PROC_CACHE: tuple[tuple[str, "InlineProcDescriptor"], ...] | None = None


@dataclass(frozen=True)
class InlineProcDescriptor:
    """Descriptor returned by @tilelang_dsl.inline_proc."""

    name: str
    py_fn: Callable[..., Any] = field(repr=False)
    signature: inspect.Signature = field(repr=False)
    source_info: "_FunctionSourceInfo | None" = field(repr=False, default=None)


class _InlineProcValidator(ast.NodeVisitor):
    def __init__(self, source_info: "_FunctionSourceInfo"):
        self.source_info = source_info

    def validate(self) -> None:
        fn = self.source_info.function_def
        args = fn.args
        if args.posonlyargs:
            raise self.source_info.error(args.posonlyargs[0], "inline_proc does not support positional-only parameters in TileLang DSL v1")
        if args.vararg is not None:
            raise self.source_info.error(args.vararg, "inline_proc does not support *args in TileLang DSL v1")
        if args.kwarg is not None:
            raise self.source_info.error(args.kwarg, "inline_proc does not support **kwargs in TileLang DSL v1")
        if args.kwonlyargs:
            raise self.source_info.error(args.kwonlyargs[0], "inline_proc does not support keyword-only parameters in TileLang DSL v1")
        tail_return: ast.Return | None = fn.body[-1] if fn.body and isinstance(fn.body[-1], ast.Return) else None
        for node in ast.walk(fn):
            if not isinstance(node, ast.Return):
                continue
            if node is tail_return:
                continue
            raise self.source_info.error(
                node,
                "inline_proc only supports an optional trailing `return` in TileLang DSL v1",
            )

        for stmt in fn.body:
            self.visit(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node is self.source_info.function_def:
            for stmt in node.body:
                self.visit(stmt)
            return
        raise self.source_info.error(node, "nested function definitions are not supported inside inline_proc in TileLang DSL v1")


def _inline_proc_registry_key(fn: Callable[..., Any]) -> tuple[str, str]:
    return (fn.__module__, fn.__name__)


def _find_inline_proc(name: str, *, module_name: str | None) -> InlineProcDescriptor | None:
    if module_name is None:
        return None
    descriptor = _INLINE_PROC_REGISTRY.get((module_name, name))
    if descriptor is not None:
        return descriptor
    module = sys.modules.get(module_name)
    if module is None:
        return None
    value = getattr(module, name, None)
    if isinstance(value, InlineProcDescriptor):
        return value
    return None


def _validate_inline_proc_call_surface(
    source_info: _FunctionSourceInfo,
    node: ast.Call,
    inline_proc: InlineProcDescriptor,
) -> None:
    if any(keyword.arg is None for keyword in node.keywords):
        keyword = next(keyword for keyword in node.keywords if keyword.arg is None)
        raise source_info.error(
            keyword.value,
            "keyword unpacking via `**` is not supported in TileLang DSL v1",
        )
    seen_keywords: set[str] = set()
    for keyword in node.keywords:
        assert keyword.arg is not None
        if keyword.arg in seen_keywords:
            raise source_info.error(
                keyword.value,
                f"duplicate keyword `{keyword.arg}` for inline_proc `{inline_proc.name}` in TileLang DSL v1",
            )
        seen_keywords.add(keyword.arg)
    positional_placeholders = [object() for _ in node.args]
    keyword_placeholders = {keyword.arg: object() for keyword in node.keywords if keyword.arg is not None}
    try:
        inline_proc.signature.bind(*positional_placeholders, **keyword_placeholders)
    except TypeError as exc:
        raise source_info.error(
            node,
            f"invalid inline_proc call `{inline_proc.name}` in TileLang DSL v1: {exc}",
        ) from exc


def _same_inline_proc_descriptor(
    lhs: InlineProcDescriptor,
    rhs: InlineProcDescriptor,
) -> bool:
    return lhs is rhs or lhs.py_fn is rhs.py_fn


def _format_inline_proc_origin(descriptor: InlineProcDescriptor) -> str:
    return f"{descriptor.py_fn.__module__}.{descriptor.py_fn.__name__}"


def _add_collected_inline_proc(
    collected: dict[str, InlineProcDescriptor],
    symbol: str,
    descriptor: InlineProcDescriptor,
) -> None:
    existing = collected.get(symbol)
    if existing is None:
        collected[symbol] = descriptor
        return
    if _same_inline_proc_descriptor(existing, descriptor):
        return
    raise ValueError(
        "ambiguous inline_proc name "
        f"`{symbol}` in TileLang DSL module: "
        f"{_format_inline_proc_origin(existing)} conflicts with "
        f"{_format_inline_proc_origin(descriptor)}"
    )


def _collect_inline_procs(module_name: str) -> tuple[tuple[str, InlineProcDescriptor], ...]:
    collected: dict[str, InlineProcDescriptor] = {}
    for (registered_module, symbol), descriptor in _INLINE_PROC_REGISTRY.items():
        if registered_module == module_name:
            _add_collected_inline_proc(collected, symbol, descriptor)

    module = sys.modules.get(module_name)
    if module is not None:
        for symbol, value in vars(module).items():
            if not isinstance(value, InlineProcDescriptor):
                continue
            _add_collected_inline_proc(collected, symbol, value)
            origin_module = value.py_fn.__module__
            for (registered_module, helper_name), helper in _INLINE_PROC_REGISTRY.items():
                if registered_module == origin_module:
                    _add_collected_inline_proc(collected, helper_name, helper)

    return tuple(sorted(collected.items(), key=lambda item: item[0]))


def _load_module_from_path(module_name: str, path: Path) -> Any:
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load module {module_name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _find_internal_soft_math_path() -> Path | None:
    module_path = Path(__file__).resolve()
    candidate_suffixes = (
        ("lib", "TileOps", "math.py"),
        ("share", "ptoas", "TileOps", "math.py"),
    )
    for root in (module_path.parent, *module_path.parents):
        for suffix in candidate_suffixes:
            candidate = root.joinpath(*suffix)
            if candidate.exists():
                return candidate
    return None


def _collect_internal_inline_procs() -> tuple[tuple[str, InlineProcDescriptor], ...]:
    global _INTERNAL_INLINE_PROC_CACHE
    if _INTERNAL_INLINE_PROC_CACHE is not None:
        return _INTERNAL_INLINE_PROC_CACHE

    soft_math_path = _find_internal_soft_math_path()
    if soft_math_path is None:
        _INTERNAL_INLINE_PROC_CACHE = ()
        return _INTERNAL_INLINE_PROC_CACHE

    try:
        module = _load_module_from_path(_INTERNAL_SOFT_MATH_MODULE_NAME, soft_math_path)
    except Exception:
        _INTERNAL_INLINE_PROC_CACHE = ()
        return _INTERNAL_INLINE_PROC_CACHE

    collected: dict[str, InlineProcDescriptor] = {}
    for symbol, value in vars(module).items():
        if isinstance(value, InlineProcDescriptor):
            collected.setdefault(symbol, value)

    _INTERNAL_INLINE_PROC_CACHE = tuple(sorted(collected.items(), key=lambda item: item[0]))
    return _INTERNAL_INLINE_PROC_CACHE


def _register_inline_proc(descriptor: InlineProcDescriptor) -> InlineProcDescriptor:
    _INLINE_PROC_REGISTRY[_inline_proc_registry_key(descriptor.py_fn)] = descriptor
    return descriptor


def inline_proc(
    py_fn: Callable[..., Any] | None = None,
) -> InlineProcDescriptor | Callable[[Callable[..., Any]], InlineProcDescriptor]:
    """Register a top-level compile-time inline procedure for TileLang DSL kernels."""

    def wrap(fn: Callable[..., Any]) -> InlineProcDescriptor:
        if not callable(fn):
            raise TypeError("@inline_proc can only decorate callables")
        source_info = _load_function_source_info(fn)
        if source_info is None:
            raise TypeError("@inline_proc requires source-visible Python functions")
        _InlineProcValidator(source_info).validate()
        return _register_inline_proc(
            InlineProcDescriptor(
                name=fn.__name__,
                py_fn=fn,
                source_info=source_info,
                signature=inspect.signature(fn),
            )
        )

    if py_fn is None:
        return wrap
    return wrap(py_fn)


def _validate_dtype_pattern(dtype: Any) -> ScalarType | MaskType | WildcardType | TypeVariable:
    if isinstance(dtype, (ScalarType, MaskType, WildcardType, TypeVariable)):
        return dtype
    raise TypeError(f"unsupported dtype pattern {dtype!r}")


class TileLangFrontendError(ValueError):
    """Source-located frontend diagnostic for TileLang DSL."""

    def __init__(self, path: str, line: int, column: int, message: str):
        self.path = path
        self.line = line
        self.column = column
        self.message = message
        super().__init__(f"{path}:{line}:{column}: {message}")


@dataclass(frozen=True)
class _FunctionSourceInfo:
    path: str
    start_line: int
    function_def: ast.FunctionDef

    def location(self, node: ast.AST) -> tuple[int, int]:
        line = self.start_line + getattr(node, "lineno", 1) - 1
        column = getattr(node, "col_offset", 0) + 1
        return line, column

    def error(self, node: ast.AST, message: str) -> TileLangFrontendError:
        line, column = self.location(node)
        return TileLangFrontendError(self.path, line, column, message)

    def parameter_node(self, param_name: str) -> ast.AST | None:
        for arg in self.function_def.args.args:
            if arg.arg == param_name:
                return arg.annotation or arg
        return None


class _KernelBodyValidator(ast.NodeVisitor):
    def __init__(
        self,
        source_info: _FunctionSourceInfo,
        *,
        advanced_enabled: bool,
        module_name: str | None,
        kernel_family: str,
    ):
        self.source_info = source_info
        self.advanced_enabled = advanced_enabled
        self.module_name = module_name
        self.kernel_family = kernel_family
        self._vecscope_depth = 0
        self._static_dtype_bindings: set[str] = set()

    def validate(self) -> None:
        for stmt in self.source_info.function_def.body:
            self.visit(stmt)

    def visit_While(self, node: ast.While) -> None:
        raise self.source_info.error(node, "unsupported Python syntax `while` in TileLang DSL v1")

    def visit_ListComp(self, node: ast.ListComp) -> None:
        raise self.source_info.error(
            node, "unsupported Python syntax `list comprehension` in TileLang DSL v1"
        )

    def visit_DictComp(self, node: ast.DictComp) -> None:
        raise self.source_info.error(
            node, "unsupported Python syntax `dict comprehension` in TileLang DSL v1"
        )

    def visit_SetComp(self, node: ast.SetComp) -> None:
        raise self.source_info.error(
            node, "unsupported Python syntax `set comprehension` in TileLang DSL v1"
        )

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        raise self.source_info.error(
            node, "unsupported Python syntax `generator expression` in TileLang DSL v1"
        )

    def visit_For(self, node: ast.For) -> None:
        if not isinstance(node.target, ast.Name):
            raise self.source_info.error(node.target, "for target must be a single name")
        if not isinstance(node.iter, ast.Call) or not isinstance(node.iter.func, ast.Name):
            raise self.source_info.error(node.iter, "only Python range(lb, ub, step) loops are supported")
        if node.iter.func.id != "range":
            raise self.source_info.error(node.iter, "only Python range(lb, ub, step) loops are supported")
        if node.iter.keywords:
            raise self.source_info.error(
                node.iter,
                "range() does not support keyword arguments in TileLang DSL v1",
            )
        if len(node.iter.args) != 3:
            raise self.source_info.error(node.iter, "range() expects exactly 3 arguments in TileLang DSL v1")
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_If(self, node: ast.If) -> None:
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        is_static_dtype = self._expr_is_static_dtype_expr(node.value)
        for target in node.targets:
            self._update_static_dtype_bindings(target, is_static_dtype=is_static_dtype)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
        is_static_dtype = node.value is not None and self._expr_is_static_dtype_expr(node.value)
        self._update_static_dtype_bindings(node.target, is_static_dtype=is_static_dtype)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        self._update_static_dtype_bindings(node.target, is_static_dtype=False)

    def visit_With(self, node: ast.With) -> None:
        if len(node.items) != 1:
            raise self.source_info.error(node, "only single with item is supported in TileLang DSL v1")
        item = node.items[0]
        if not isinstance(item.context_expr, ast.Call):
            raise self.source_info.error(item.context_expr, "with context must be a call in TileLang DSL v1")
        if not (
            isinstance(item.context_expr.func, ast.Attribute)
            and isinstance(item.context_expr.func.value, ast.Name)
            and item.context_expr.func.value.id == "pto"
        ):
            raise self.source_info.error(
                item.context_expr,
                "only pto.vecscope/pto.strict_vecscope are supported as with-contexts in TileLang DSL v1",
            )
        with_name = item.context_expr.func.attr
        if self.kernel_family == "cube" and with_name in {"vecscope", "strict_vecscope"}:
            raise self.source_info.error(
                item.context_expr,
                "@pto.ckernel does not support pto.vecscope()/pto.strict_vecscope(); "
                "cube kernels must use linear control flow without vecscope",
            )
        if with_name == "vecscope":
            if item.context_expr.args or item.context_expr.keywords:
                raise self.source_info.error(
                    item.context_expr,
                    "pto.vecscope() does not accept positional or keyword arguments in TileLang DSL v1",
                )
            if item.optional_vars is not None:
                raise self.source_info.error(
                    item,
                    "pto.vecscope() does not support `as` bindings in TileLang DSL v1",
                )
        elif with_name == "strict_vecscope":
            if not self.advanced_enabled:
                raise self.source_info.error(
                    item.context_expr,
                    advanced_mode_message("strict_vecscope"),
                )
            if not isinstance(item.optional_vars, ast.Tuple):
                raise self.source_info.error(item, "pto.strict_vecscope requires tuple binding in 'as'")
            for elt in item.optional_vars.elts:
                if not isinstance(elt, ast.Name):
                    raise self.source_info.error(elt, "pto.strict_vecscope bindings must be names")
        else:
            raise self.source_info.error(
                item.context_expr,
                "only pto.vecscope/pto.strict_vecscope are supported as with-contexts in TileLang DSL v1",
            )
        self._vecscope_depth += 1
        try:
            for stmt in node.body:
                self.visit(stmt)
        finally:
            self._vecscope_depth -= 1

    def _validate_no_keyword_unpacking(self, node: ast.Call) -> None:
        for keyword in node.keywords:
            if keyword.arg is None:
                raise self.source_info.error(
                    keyword.value,
                    "keyword unpacking via `**` is not supported in TileLang DSL v1",
                )

    def _reject_cube_vector_surface(self, node: ast.Call, surface_name: str) -> None:
        raise self.source_info.error(
            node,
            f"vector-only surface `{surface_name}` is not part of the @pto.ckernel contract",
        )

    def _reject_vector_cube_surface(self, node: ast.Call, surface_name: str) -> None:
        raise self.source_info.error(
            node,
            f"cube-only surface `{surface_name}` is not part of the @pto.vkernel contract",
        )

    def _validate_call_keywords(self, node: ast.Call) -> None:
        if not node.keywords:
            return
        self._validate_no_keyword_unpacking(node)

        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            namespace = node.func.value.id
            name = node.func.attr
        elif isinstance(node.func, ast.Name):
            namespace = None
            name = node.func.id
        else:
            raise self.source_info.error(
                node,
                "unsupported call surface in TileLang DSL v1",
            )

        allowed_keywords = _DMA_CALL_KEYWORDS.get(name) if namespace == "pto" else None
        if allowed_keywords is None:
            call_name = f"{namespace + '.' if namespace else ''}{name}"
            raise self.source_info.error(
                node,
                f"`{call_name}` does not support keyword arguments in TileLang DSL v1; "
                "keyword arguments are only supported on selected public call surfaces",
            )

        seen: set[str] = set()
        for keyword in node.keywords:
            assert keyword.arg is not None
            if keyword.arg in seen:
                raise self.source_info.error(
                    keyword.value,
                    f"duplicate keyword `{keyword.arg}` for `pto.{name}` in TileLang DSL v1",
                )
            if keyword.arg not in allowed_keywords:
                raise self.source_info.error(
                    keyword.value,
                    f"unsupported keyword `{keyword.arg}` for `pto.{name}` in TileLang DSL v1",
                )
            seen.add(keyword.arg)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "eval":
            if node.keywords:
                raise self.source_info.error(
                    node,
                    "`eval` does not support keyword arguments in TileLang DSL v1",
                )
            if len(node.args) > 1:
                raise self.source_info.error(
                    node,
                    "`eval()` accepts at most one positional dtype argument in TileLang DSL v1",
                )
            return

        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.attr == "as_ptr":
                if node.keywords:
                    raise self.source_info.error(
                        node,
                        "`as_ptr` does not support keyword arguments in TileLang DSL v1",
                    )
                if node.args:
                    raise self.source_info.error(
                        node,
                        "`as_ptr()` does not accept positional arguments in TileLang DSL v1",
                    )
                if self.kernel_family == "cube":
                    return
                if self.advanced_enabled:
                    return
                raise self.source_info.error(
                    node,
                    "surface `as_ptr` requires advanced=True in TileLang DSL v1",
                )
            if node.func.attr == "astype":
                if node.keywords:
                    raise self.source_info.error(
                        node,
                        "`astype` does not support keyword arguments in TileLang DSL v1",
                    )
                if len(node.args) != 1:
                    raise self.source_info.error(
                        node,
                        "`astype()` expects exactly 1 positional argument (target dtype) in TileLang DSL v1",
                    )
                # Type checking will be done during semantic analysis
                return
            if node.func.value.id == "pto" and node.func.attr == "tpl":
                if self.kernel_family == "cube":
                    self._validate_no_keyword_unpacking(node)
                else:
                    self._validate_call_keywords(node)
                return
            if node.func.value.id == "pto" and node.func.attr == "Tile":
                self._validate_no_keyword_unpacking(node)
                return
            if node.func.value.id == "pto" and self.kernel_family == "cube":
                self._validate_no_keyword_unpacking(node)
                if node.func.attr in SUPPORTED_VECSCOPE_PTO_CALLS:
                    self._reject_cube_vector_surface(node, f"pto.{node.func.attr}")
                if node.func.attr in ADVANCED_VECSCOPE_PTO_CALLS:
                    self._reject_cube_vector_surface(node, f"pto.{node.func.attr}")
                if node.func.attr in ADVANCED_TOPLEVEL_PTO_CALLS:
                    self._reject_cube_vector_surface(node, f"pto.{node.func.attr}")
                if node.func.attr in DEFERRED_PTO_SURFACES:
                    raise self.source_info.error(
                        node,
                        deferred_surface_message(node.func.attr),
                    )
                return
            if node.func.value.id == "pto" and self.kernel_family != "cube":
                if node.func.attr in CUBE_ONLY_PTO_CALLS:
                    self._reject_vector_cube_surface(node, f"pto.{node.func.attr}")
            if node.func.value.id == "pto" and node.func.attr in SUPPORTED_TOPLEVEL_PTO_CALLS:
                self._validate_call_keywords(node)
                return
            if node.func.value.id == "pto" and node.func.attr in SUPPORTED_VECSCOPE_PTO_CALLS:
                self._validate_call_keywords(node)
                return
            if node.func.value.id == "pto" and node.func.attr in ADVANCED_VECSCOPE_PTO_CALLS:
                if self.advanced_enabled:
                    self._validate_call_keywords(node)
                    return
                raise self.source_info.error(
                    node,
                    advanced_mode_message(node.func.attr),
                )
            if node.func.value.id == "pto" and (
                node.func.attr in ADVANCED_EXPR_PTO_CALLS
                or node.func.attr in ADVANCED_TOPLEVEL_PTO_CALLS
            ):
                if self.advanced_enabled:
                    self._validate_call_keywords(node)
                    return
                raise self.source_info.error(
                    node,
                    advanced_mode_message(node.func.attr),
                )
            if node.func.value.id == "pto" and node.func.attr in DEFERRED_PTO_SURFACES:
                raise self.source_info.error(
                    node,
                    deferred_surface_message(node.func.attr),
                )
            if node.func.value.id == "pto":
                raise self.source_info.error(
                    node,
                    f"unsupported op surface `pto.{node.func.attr}` in TileLang DSL v1",
                )
            raise self.source_info.error(
                node,
                f"arbitrary external call `{node.func.value.id}.{node.func.attr}` is not supported "
                "in TileLang DSL v1",
            )

        if isinstance(node.func, ast.Name):
            if node.func.id == "range":
                self._validate_call_keywords(node)
                return
            if node.func.id in self._static_dtype_bindings:
                self._validate_call_keywords(node)
                return
            inline_proc = _find_inline_proc(node.func.id, module_name=self.module_name)
            if inline_proc is not None:
                _validate_inline_proc_call_surface(self.source_info, node, inline_proc)
                return
            raise self.source_info.error(
                node,
                f"arbitrary external call `{node.func.id}` is not supported in TileLang DSL v1",
            )

        raise self.source_info.error(
            node,
            "unsupported call surface in TileLang DSL v1",
        )

    def _expr_is_static_dtype_expr(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self._static_dtype_bindings
        if isinstance(node, ast.Attribute):
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "pto"
                and node.attr in _DSL_DTYPE_NAMES
            ):
                return True
            if node.attr == "element_type":
                return True
        return False

    def _update_static_dtype_bindings(self, target: ast.expr, *, is_static_dtype: bool) -> None:
        if isinstance(target, ast.Name):
            if is_static_dtype:
                self._static_dtype_bindings.add(target.id)
            else:
                self._static_dtype_bindings.discard(target.id)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._update_static_dtype_bindings(element, is_static_dtype=False)


def _load_function_source_info(py_fn: Callable[..., Any]) -> _FunctionSourceInfo | None:
    try:
        source_lines, start_line = inspect.getsourcelines(py_fn)
        path = inspect.getsourcefile(py_fn) or inspect.getfile(py_fn)
    except (OSError, IOError, TypeError):
        return None

    source = textwrap.dedent("".join(source_lines))
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == py_fn.__name__:
            return _FunctionSourceInfo(path=path, start_line=start_line, function_def=node)
    return None


def _validate_function_body(
    source_info: _FunctionSourceInfo | None,
    *,
    advanced_enabled: bool,
    module_name: str | None,
    kernel_family: str,
) -> None:
    if source_info is None:
        return
    _KernelBodyValidator(
        source_info,
        advanced_enabled=advanced_enabled,
        module_name=module_name,
        kernel_family=kernel_family,
    ).validate()


def _raise_tile_param_error(
    source_info: _FunctionSourceInfo | None,
    param_name: str,
    message: str,
    fallback_exception: type[Exception] = ValueError,
) -> None:
    if source_info is not None:
        node = source_info.parameter_node(param_name)
        if node is not None:
            raise source_info.error(node, message)
    raise fallback_exception(message)


def _freeze_dtypes(dtypes: Any) -> tuple[tuple[Any, ...], ...]:
    if not isinstance(dtypes, (list, tuple)):
        raise TypeError("dtypes must be a sequence of signature tuples")

    frozen_signatures = []
    for signature in dtypes:
        if not isinstance(signature, (list, tuple)):
            raise TypeError("each dtypes entry must be a signature tuple")
        frozen_signature = tuple(signature)
        for dtype in frozen_signature:
            _validate_dtype_pattern(dtype)
        frozen_signatures.append(frozen_signature)

    if not frozen_signatures:
        raise ValueError("dtypes must contain at least one signature tuple")

    return tuple(frozen_signatures)


@dataclass(frozen=True)
class BoundKernelParameter:
    """One parameter after v1 monomorphic dtype binding."""

    name: str
    kind: str
    annotation: Any
    dtype: Any

    @property
    def element_dtype(self) -> ScalarType | None:
        if self.kind in ("tensorview", "partition_tensor_view", "tile", "ptr", "vector"):
            return self.dtype
        return None


@dataclass(frozen=True)
class KernelParameterSpec:
    """One validated Python function parameter before dtype selection."""

    name: str
    kind: str
    annotation: Any


@dataclass(frozen=True)
class _ConstraintValue:
    value: Any | None

    def _coerce_other(self, other: Any) -> Any | None:
        if isinstance(other, _ConstraintValue):
            return other.value
        return other

    def _arith(self, other: Any, fn: Callable[[Any, Any], Any]) -> "_ConstraintValue":
        other_value = self._coerce_other(other)
        if self.value is None or other_value is None:
            return _ConstraintValue(None)
        return _ConstraintValue(fn(self.value, other_value))

    def _compare(self, other: Any, fn: Callable[[Any, Any], bool]) -> bool:
        other_value = self._coerce_other(other)
        if self.value is None or other_value is None:
            return True
        return fn(self.value, other_value)

    def __add__(self, other: Any) -> "_ConstraintValue":
        return self._arith(other, lambda lhs, rhs: lhs + rhs)

    def __radd__(self, other: Any) -> "_ConstraintValue":
        return _ConstraintValue(self._coerce_other(other)).__add__(self)

    def __sub__(self, other: Any) -> "_ConstraintValue":
        return self._arith(other, lambda lhs, rhs: lhs - rhs)

    def __rsub__(self, other: Any) -> "_ConstraintValue":
        return _ConstraintValue(self._coerce_other(other)).__sub__(self)

    def __mul__(self, other: Any) -> "_ConstraintValue":
        return self._arith(other, lambda lhs, rhs: lhs * rhs)

    def __rmul__(self, other: Any) -> "_ConstraintValue":
        return _ConstraintValue(self._coerce_other(other)).__mul__(self)

    def __floordiv__(self, other: Any) -> "_ConstraintValue":
        return self._arith(other, lambda lhs, rhs: lhs // rhs)

    def __rfloordiv__(self, other: Any) -> "_ConstraintValue":
        return _ConstraintValue(self._coerce_other(other)).__floordiv__(self)

    def __mod__(self, other: Any) -> "_ConstraintValue":
        return self._arith(other, lambda lhs, rhs: lhs % rhs)

    def __rmod__(self, other: Any) -> "_ConstraintValue":
        return _ConstraintValue(self._coerce_other(other)).__mod__(self)

    def __eq__(self, other: Any) -> bool:  # type: ignore[override]
        return self._compare(other, lambda lhs, rhs: lhs == rhs)

    def __ne__(self, other: Any) -> bool:  # type: ignore[override]
        return self._compare(other, lambda lhs, rhs: lhs != rhs)

    def __le__(self, other: Any) -> bool:
        return self._compare(other, lambda lhs, rhs: lhs <= rhs)

    def __lt__(self, other: Any) -> bool:
        return self._compare(other, lambda lhs, rhs: lhs < rhs)

    def __ge__(self, other: Any) -> bool:
        return self._compare(other, lambda lhs, rhs: lhs >= rhs)

    def __gt__(self, other: Any) -> bool:
        return self._compare(other, lambda lhs, rhs: lhs > rhs)

    def __bool__(self) -> bool:
        if self.value is None:
            return True
        return bool(self.value)

    def __repr__(self) -> str:
        return "?" if self.value is None else repr(self.value)


class _ConstraintSequenceView:
    def __init__(self, values: tuple[Any | None, ...]):
        self._values = tuple(_ConstraintValue(value) for value in values)

    def __getitem__(self, index: int) -> _ConstraintValue:
        if -len(self._values) <= index < len(self._values):
            return self._values[index]
        return _ConstraintValue(None)

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def __repr__(self) -> str:
        return repr(tuple(self._values))


class _ConstraintParamView:
    def __init__(self, name: str, attrs: Mapping[str, Any]):
        self._name = name
        self._attrs = dict(attrs)

    def _sequence_attr(self, attr_name: str) -> _ConstraintSequenceView:
        values = self._attrs.get(attr_name)
        if values is None:
            rank = self._attrs.get("rank")
            if isinstance(rank, int) and rank > 0:
                values = (None,) * rank
            else:
                values = ()
        return _ConstraintSequenceView(tuple(values))

    @property
    def shape(self) -> _ConstraintSequenceView:
        return self._sequence_attr("shape")

    @property
    def valid_shape(self) -> _ConstraintSequenceView:
        return self._sequence_attr("valid_shape")

    @property
    def strides(self) -> _ConstraintSequenceView:
        return self._sequence_attr("strides")

    @property
    def rank(self) -> _ConstraintValue:
        rank = self._attrs.get("rank")
        if rank is None:
            shape = self._attrs.get("shape")
            if shape is not None:
                rank = len(shape)
        return _ConstraintValue(rank)

    @property
    def dtype(self) -> Any:
        return self._attrs.get("dtype")

    @property
    def value(self) -> _ConstraintValue:
        return _ConstraintValue(self._attrs.get("value"))

    @property
    def memory_space(self) -> Any:
        memory_space = self._attrs.get("memory_space")
        if memory_space is None and self._attrs.get("kind") == "tile":
            return MemorySpace.UB
        if memory_space is None:
            return None
        if isinstance(memory_space, MemorySpace):
            return memory_space
        return MemorySpace(memory_space)

    @property
    def config(self) -> TileConfig | ViewConfig | None:
        config = self._attrs.get("config")
        if config is None:
            if self._attrs.get("kind") == "tile":
                return TileConfig()
            if self._attrs.get("kind") == "view":
                return ViewConfig()
            return None
        if isinstance(config, TileConfig):
            return config
        if isinstance(config, ViewConfig):
            return config
        if isinstance(config, Mapping):
            if self._attrs.get("kind") == "view":
                return ViewConfig.from_mapping(config)
            return TileConfig.from_mapping(config)
        raise TypeError(f"unsupported config payload {config!r} in constraint view")

    def __repr__(self) -> str:
        return f"{self._name}<{self._attrs!r}>"


@dataclass(frozen=True)
class VKernelDescriptor:
    """Descriptor returned by `@tilelang_dsl.vkernel`."""

    target: str
    match_ops: tuple[str, ...]
    dtypes: tuple[tuple[Any, ...], ...]
    name: str
    verify_enabled: bool
    advanced_enabled: bool
    _parameter_specs: tuple[KernelParameterSpec, ...]
    _py_fn: Callable[..., Any] = field(repr=False)
    kernel_family: str = "vector"
    _source_info: _FunctionSourceInfo | None = field(repr=False, compare=False, default=None)
    specializations: tuple[tuple[str, TileSpecialization], ...] = ()
    constraints: tuple[Callable[[Mapping[str, Any]], Any], ...] = field(default=(), repr=False)
    priority: int = 0
    _templates: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = field(default=(), repr=False)
    _inline_procs: tuple[tuple[str, InlineProcDescriptor], ...] = field(default=(), repr=False)
    _internal_inline_procs: tuple[tuple[str, InlineProcDescriptor], ...] = field(default=(), repr=False)
    _selected_op: str | None = None
    _selected_dtype_signature: tuple[ScalarType | MaskType, ...] | None = None
    _parameters: tuple[BoundKernelParameter, ...] | None = field(default=None, repr=False)
    _constraint_context_attrs: tuple[tuple[str, Any], ...] = field(default=(), repr=False)

    @property
    def py_fn(self) -> Callable[..., Any]:
        return self._py_fn

    @property
    def op(self) -> str:
        if self._selected_op is None:
            raise ValueError(
                "descriptor requires pto.select_kernel(...) to bind a concrete op "
                "before reading descriptor.op"
            )
        return self._selected_op

    @property
    def selected_op(self) -> str | None:
        return self._selected_op

    @property
    def templates(self) -> dict[str, dict[str, str]]:
        return {
            slot: dict(op_bindings)
            for slot, op_bindings in self._templates
        }

    @property
    def inline_procs(self) -> dict[str, InlineProcDescriptor]:
        return {name: descriptor for name, descriptor in self._inline_procs}

    @property
    def internal_inline_procs(self) -> dict[str, InlineProcDescriptor]:
        return {name: descriptor for name, descriptor in self._internal_inline_procs}

    @property
    def dtype_signature(self) -> tuple[ScalarType | MaskType, ...]:
        if self._selected_dtype_signature is None:
            raise ValueError(
                "descriptor requires pto.select_kernel(...) to choose a concrete dtype signature "
                "before materialization"
            )
        return self._selected_dtype_signature

    @property
    def parameters(self) -> tuple[BoundKernelParameter, ...]:
        if self._parameters is None:
            raise ValueError(
                "descriptor requires pto.select_kernel(...) to bind concrete parameter dtypes "
                "before materialization"
            )
        return self._parameters

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "op": self._selected_op,
            "match_ops": self.match_ops,
            "selected_op": self._selected_op,
            "dtypes": self.dtypes,
            "name": self.name,
            "verify": self.verify_enabled,
            "advanced": self.advanced_enabled,
            "constraints": self.constraints,
            "priority": self.priority,
            "templates": self.templates,
            "inline_procs": tuple(sorted(self.inline_procs.keys())),
        }

    @property
    def tile_parameters(self) -> tuple[BoundKernelParameter, ...]:
        return tuple(param for param in self.parameters if param.kind == "tile")

    @property
    def specializations_by_name(self) -> dict[str, TileSpecialization]:
        return dict(self.specializations)

    @property
    def constraint_context_attrs(self) -> dict[str, Any]:
        return dict(self._constraint_context_attrs)

    def _tile_parameter_names(self) -> tuple[str, ...]:
        return tuple(param.name for param in self._parameter_specs if param.kind == "tile")

    def _bind_constraint_context_attrs(
        self,
        context_attrs: Mapping[str, Any],
    ) -> "VKernelDescriptor":
        frozen_context_attrs = tuple(
            sorted(dict(context_attrs).items(), key=lambda item: item[0])
        )
        if self._constraint_context_attrs == frozen_context_attrs:
            return self
        return VKernelDescriptor(
            target=self.target,
            match_ops=self.match_ops,
            dtypes=self.dtypes,
            name=self.name,
            verify_enabled=self.verify_enabled,
            advanced_enabled=self.advanced_enabled,
            kernel_family=self.kernel_family,
            _parameter_specs=self._parameter_specs,
            _py_fn=self._py_fn,
            _source_info=self._source_info,
            specializations=self.specializations,
            constraints=self.constraints,
            priority=self.priority,
            _templates=self._templates,
            _inline_procs=self._inline_procs,
            _internal_inline_procs=self._internal_inline_procs,
            _selected_op=self._selected_op,
            _selected_dtype_signature=self._selected_dtype_signature,
            _parameters=self._parameters,
            _constraint_context_attrs=frozen_context_attrs,
        )

    def _bind_selected_dtype_signature(
        self,
        dtype_signature: tuple[ScalarType | MaskType, ...],
    ) -> "VKernelDescriptor":
        bound_parameters = _bind_parameters(self._parameter_specs, dtype_signature)
        return VKernelDescriptor(
            target=self.target,
            match_ops=self.match_ops,
            dtypes=self.dtypes,
            name=self.name,
            verify_enabled=self.verify_enabled,
            advanced_enabled=self.advanced_enabled,
            kernel_family=self.kernel_family,
            _parameter_specs=self._parameter_specs,
            _py_fn=self._py_fn,
            _source_info=self._source_info,
            specializations=self.specializations,
            constraints=self.constraints,
            priority=self.priority,
            _templates=self._templates,
            _inline_procs=self._inline_procs,
            _internal_inline_procs=self._internal_inline_procs,
            _selected_op=self._selected_op,
            _selected_dtype_signature=dtype_signature,
            _parameters=bound_parameters,
            _constraint_context_attrs=self._constraint_context_attrs,
        )

    def _bind_selected_op(self, op: str) -> "VKernelDescriptor":
        normalized_op = _validate_op(op)
        if normalized_op not in self.match_ops:
            raise ValueError(
                f"selected op {normalized_op!r} is not in descriptor matcher set {self.match_ops!r}"
            )
        if self._selected_op == normalized_op:
            return self
        return VKernelDescriptor(
            target=self.target,
            match_ops=self.match_ops,
            dtypes=self.dtypes,
            name=self.name,
            verify_enabled=self.verify_enabled,
            advanced_enabled=self.advanced_enabled,
            kernel_family=self.kernel_family,
            _parameter_specs=self._parameter_specs,
            _py_fn=self._py_fn,
            _source_info=self._source_info,
            specializations=self.specializations,
            constraints=self.constraints,
            priority=self.priority,
            _templates=self._templates,
            _inline_procs=self._inline_procs,
            _internal_inline_procs=self._internal_inline_procs,
            _selected_op=normalized_op,
            _selected_dtype_signature=self._selected_dtype_signature,
            _parameters=self._parameters,
            _constraint_context_attrs=self._constraint_context_attrs,
        )

    def specialize(self, **bindings: Any) -> "VKernelDescriptor":
        tile_param_names = set(self._tile_parameter_names())
        if not tile_param_names:
            if bindings:
                unknown = ", ".join(sorted(bindings))
                raise TypeError(
                    f"specialize() received bindings for non-Tile parameters: {unknown}"
                )
            return self

        unknown = sorted(set(bindings) - tile_param_names)
        if unknown:
            unknown_names = ", ".join(unknown)
            raise TypeError(
                f"specialize() only accepts bare Tile parameters; got: {unknown_names}"
            )

        updated = self.specializations_by_name
        for name, binding in bindings.items():
            updated[name] = _coerce_tile_specialization(
                name,
                binding,
                self._source_info,
                kernel_family=self.kernel_family,
            )

        return VKernelDescriptor(
            target=self.target,
            match_ops=self.match_ops,
            dtypes=self.dtypes,
            name=self.name,
            verify_enabled=self.verify_enabled,
            advanced_enabled=self.advanced_enabled,
            kernel_family=self.kernel_family,
            _parameter_specs=self._parameter_specs,
            _source_info=self._source_info,
            specializations=tuple(sorted(updated.items())),
            constraints=self.constraints,
            priority=self.priority,
            _templates=self._templates,
            _inline_procs=self._inline_procs,
            _internal_inline_procs=self._internal_inline_procs,
            _selected_op=self._selected_op,
            _selected_dtype_signature=self._selected_dtype_signature,
            _parameters=self._parameters,
            _py_fn=self._py_fn,
            _constraint_context_attrs=self._constraint_context_attrs,
        )

    def _require_specialized_tiles(self, api_name: str) -> None:
        tile_names = list(self._tile_parameter_names())
        if not tile_names:
            return

        specialized = self.specializations_by_name
        missing = [name for name in tile_names if name not in specialized]
        if missing:
            missing_names = ", ".join(missing)
            _raise_tile_param_error(
                self._source_info,
                missing[0],
                f"{api_name}() requires specialize() bindings for bare Tile parameters: "
                f"{missing_names}",
            )

    def _require_materialization_binding(self, api_name: str) -> None:
        self.parameters
        if len(self.match_ops) > 1 and self._selected_op is None:
            raise ValueError(
                f"{api_name}() requires pto.select_kernel(...) to bind a concrete op "
                "before materialization"
            )

    def _constraint_context_for_evaluation(
        self,
        extra_context_attrs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        attrs = dict(self._constraint_context_attrs)
        if extra_context_attrs is not None:
            attrs.update(extra_context_attrs)
        attrs.setdefault("target", self.target)
        if self._selected_op is not None:
            attrs.setdefault("op", self._selected_op)
            attrs.setdefault("selected_op", self._selected_op)

        for index, spec in enumerate(self._parameter_specs):
            existing = attrs.get(spec.name)
            param_attrs = {} if not isinstance(existing, dict) else dict(existing)
            positional_prefix = f"arg{index}"
            param_attrs.setdefault("kind", spec.kind)
            attrs.setdefault(f"{spec.name}_kind", spec.kind)

            def set_sequence_attr(attr_name: str) -> None:
                named_key = f"{spec.name}_{attr_name}"
                positional_key = f"{positional_prefix}_{attr_name}"
                if named_key in attrs:
                    value = tuple(attrs[named_key])
                elif positional_key in attrs:
                    value = tuple(attrs[positional_key])
                    attrs.setdefault(named_key, value)
                else:
                    return
                param_attrs.setdefault(attr_name, value)

            def set_scalar_attr(attr_name: str) -> None:
                named_key = f"{spec.name}_{attr_name}"
                positional_key = f"{positional_prefix}_{attr_name}"
                if named_key in attrs:
                    value = attrs[named_key]
                elif positional_key in attrs:
                    value = attrs[positional_key]
                    attrs.setdefault(named_key, value)
                else:
                    return
                param_attrs.setdefault(attr_name, value)

            set_sequence_attr("shape")
            set_sequence_attr("valid_shape")
            set_sequence_attr("strides")
            set_scalar_attr("dtype")
            set_scalar_attr("rank")
            set_scalar_attr("value")
            set_scalar_attr("memory_space")
            set_scalar_attr("config")

            if spec.kind in ("tensorview", "partition_tensor_view"):
                # TensorView authoring form is normalized to 5D in the current DSL spec.
                param_attrs.setdefault("rank", 5)
                param_attrs.setdefault("memory_space", "gm")
                param_attrs.setdefault("config", ViewConfig())
                attrs.setdefault(f"{spec.name}_rank", 5)
                attrs.setdefault(f"{spec.name}_memory_space", "gm")
                attrs.setdefault(f"{spec.name}_config", param_attrs["config"])
            attrs[spec.name] = param_attrs

        if self._parameters is not None:
            for param in self._parameters:
                param_attrs = attrs.get(param.name)
                if not isinstance(param_attrs, dict):
                    param_attrs = {"kind": param.kind}
                param_attrs.setdefault("dtype", param.dtype)
                attrs[param.name] = param_attrs
                attrs.setdefault(f"{param.name}_dtype", param.dtype)

        for name, spec in self.specializations_by_name.items():
            effective_valid_shape = spec.shape if spec.valid_shape is None else spec.valid_shape
            param_attrs = attrs.get(name)
            if not isinstance(param_attrs, dict):
                param_attrs = {"kind": "tile"}
            param_attrs.update(
                {
                    "shape": spec.shape,
                    "rank": len(spec.shape),
                    "memory_space": spec.memory_space.value,
                    "valid_shape": effective_valid_shape,
                    "config": spec.config,
                }
            )
            attrs[name] = param_attrs
            attrs[f"{name}_shape"] = spec.shape
            attrs[f"{name}_rank"] = len(spec.shape)
            attrs[f"{name}_memory_space"] = spec.memory_space.value
            attrs[f"{name}_valid_shape"] = effective_valid_shape
            if len(spec.shape) == 1:
                attrs[f"{name}_extent"] = spec.shape[0]
                attrs[f"{name}_valid_extent"] = effective_valid_shape[0]
            elif len(spec.shape) == 2:
                attrs[f"{name}_rows"] = spec.shape[0]
                attrs[f"{name}_cols"] = spec.shape[1]
                attrs[f"{name}_valid_rows"] = effective_valid_shape[0]
                attrs[f"{name}_valid_cols"] = effective_valid_shape[1]
        return attrs

    def _validate_materialization_constraints(self, api_name: str) -> None:
        if not self.constraints:
            return
        context_attrs = self._constraint_context_for_evaluation()
        evaluation = _evaluate_constraints(self, context_attrs)
        _raise_constraint_evaluation_error(evaluation)
        if evaluation.passed:
            return
        raise LookupError(
            f"{api_name}() constraint evaluation rejected kernel {self.name!r} "
            "for the current specialization/context attributes"
        )

    def _build_authoring_module(self):
        self.parameters
        frontend_kernel = build_frontend_kernel_node(self)
        semantic_kernel = analyze_frontend_kernel(frontend_kernel)
        return lower_semantic_kernel(semantic_kernel)

    def mlir_text(self) -> str:
        self._require_materialization_binding("mlir_text")
        self._require_specialized_tiles("mlir_text")
        self._validate_materialization_constraints("mlir_text")
        return self._build_authoring_module().render()

    def mlir_module(self) -> "MaterializedMLIRModule":
        self._require_materialization_binding("mlir_module")
        self._require_specialized_tiles("mlir_module")
        return MaterializedMLIRModule(text=self.mlir_text(), target=self.target)

    def emit(self, path: str | Path) -> None:
        self._require_materialization_binding("emit")
        self._require_specialized_tiles("emit")
        self._validate_materialization_constraints("emit")
        output_path = Path(path)
        output_path.write_text(self.mlir_text(), encoding="utf-8")


@dataclass(frozen=True)
class KernelSelectionCandidateMetadata:
    """Structured selection diagnostics for one target/op-matched kernel candidate."""

    descriptor: VKernelDescriptor
    status: str
    selected_op: str | None = None
    matched_dtype_signature: tuple[ScalarType | MaskType, ...] | None = None
    reason: str | None = None
    failed_constraint_index: int | None = None
    failed_constraint_name: str | None = None
    failed_constraint_location: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    mlir_text: str | None = None
    mlir_error: str | None = None

    @property
    def name(self) -> str:
        return self.descriptor.name

    @property
    def priority(self) -> int:
        return self.descriptor.priority

    @property
    def match_ops(self) -> tuple[str, ...]:
        return self.descriptor.match_ops

    @property
    def dtype_signatures(self) -> tuple[tuple[Any, ...], ...]:
        return self.descriptor.dtypes


@dataclass(frozen=True)
class KernelSelectionReport:
    """Structured selector result returned by the opt-in metadata path."""

    target: str
    op: str
    operand_types: tuple[ScalarType | MaskType, ...]
    selected: VKernelDescriptor | None
    candidates: tuple[KernelSelectionCandidateMetadata, ...] = ()
    final_status: str = "no_candidate"
    final_error: str | None = None
    _context_attrs: tuple[tuple[str, Any], ...] = field(default=(), repr=False)

    @property
    def context_attrs(self) -> dict[str, Any]:
        return dict(self._context_attrs)

    @property
    def ok(self) -> bool:
        return self.final_status == "selected" and self.selected is not None


@dataclass(frozen=True)
class _TargetOpSelectionCandidate:
    descriptor: VKernelDescriptor


@dataclass(frozen=True)
class _DtypeSelectionCandidate:
    descriptor: VKernelDescriptor
    matched_descriptor: VKernelDescriptor | None = None
    matched_dtype_signature: tuple[ScalarType | MaskType, ...] | None = None

    @property
    def matched(self) -> bool:
        return self.matched_descriptor is not None


@dataclass(frozen=True)
class _ConstraintSelectionCandidate:
    descriptor: VKernelDescriptor
    passed: bool
    evaluation: "_ConstraintEvaluationResult"
    bound_descriptor: VKernelDescriptor | None = None


@dataclass(frozen=True)
class _PrioritySelectionResult:
    candidates: tuple[VKernelDescriptor, ...]
    highest_priority: int | None
    winners: tuple[VKernelDescriptor, ...]

    @property
    def has_tie(self) -> bool:
        return len(self.winners) > 1

    @property
    def winner(self) -> VKernelDescriptor | None:
        if len(self.winners) != 1:
            return None
        return self.winners[0]


@dataclass(frozen=True)
class _MaterializationSelectionCandidate:
    descriptor: VKernelDescriptor
    mlir_text: str | None = None
    mlir_error: str | None = None


@dataclass(frozen=True)
class _ConstraintEvaluationResult:
    passed: bool
    failed_constraint_index: int | None = None
    failed_constraint_name: str | None = None
    failed_constraint_location: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    @property
    def raised_error(self) -> bool:
        return self.error_type is not None


class KernelRegistry:
    """Explicit registry for TileLang kernel descriptors."""

    def __init__(self, descriptors: tuple[VKernelDescriptor, ...] = ()):
        self._descriptors: list[VKernelDescriptor] = []
        for descriptor in descriptors:
            self.register(descriptor)

    def register(self, descriptor: VKernelDescriptor) -> VKernelDescriptor:
        if not isinstance(descriptor, VKernelDescriptor):
            raise TypeError("KernelRegistry.register() expects a VKernelDescriptor")
        self._descriptors.append(descriptor)
        return descriptor

    @property
    def descriptors(self) -> tuple[VKernelDescriptor, ...]:
        return tuple(self._descriptors)

    def __iter__(self):
        return iter(self._descriptors)

    def __len__(self) -> int:
        return len(self._descriptors)


_DEFAULT_KERNEL_REGISTRY = KernelRegistry()


@dataclass(frozen=True)
class MaterializedMLIRModule:
    text: str
    target: str = "a5"

    def __str__(self) -> str:
        return self.text


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _validate_target(target: str) -> str:
    if not isinstance(target, str):
        raise TypeError("target must be a string")
    if target != "a5":
        raise ValueError("TileLang DSL v1 currently only supports target='a5'")
    return target


def _validate_op(op: Any) -> str:
    if not isinstance(op, str) or not op:
        raise TypeError("op must be a non-empty string")
    return op


def _is_schema_form_match_op(op: str) -> bool:
    normalized = " ".join(op.split())
    return " ins(" in normalized and "->" in normalized and " outs(" in normalized


def _freeze_match_ops(*, op: Any, ops: Any) -> tuple[str, ...]:
    if op is not None and ops is not None:
        raise ValueError("vkernel() accepts either op= or ops=, but not both")
    if op is None and ops is None:
        raise ValueError("vkernel() requires exactly one of op= or ops=")
    if op is not None:
        return (_validate_op(op),)
    if not isinstance(ops, (list, tuple)):
        raise TypeError("ops must be a sequence of non-empty strings")
    if not ops:
        raise ValueError("ops must contain at least one op")
    normalized_ops = tuple(_validate_op(candidate) for candidate in ops)
    if len(set(normalized_ops)) != len(normalized_ops):
        raise ValueError("ops must not contain duplicates")
    return normalized_ops


def _validate_template_slot_name(slot: Any) -> str:
    if not isinstance(slot, str) or not slot:
        raise TypeError("template slot names must be non-empty strings")
    return slot


def _validate_template_value(slot: str, op_name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(
            f"templates[{slot!r}][{op_name!r}] must be a non-empty pto op name string"
        )
    if value not in _SUPPORTED_TEMPLATE_PTO_CALLS:
        raise ValueError(
            f"templates[{slot!r}][{op_name!r}] maps to unsupported pto op {value!r}"
        )
    return value


def _freeze_templates(
    templates: Any,
    *,
    match_ops: tuple[str, ...],
) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    if templates in (_UNSET, None):
        return ()
    if not isinstance(templates, Mapping):
        raise TypeError("templates must be a mapping of slot names to per-op mappings")

    frozen_templates = []
    for slot, op_bindings in templates.items():
        normalized_slot = _validate_template_slot_name(slot)
        if not isinstance(op_bindings, Mapping):
            raise TypeError(
                f"templates[{normalized_slot!r}] must be a mapping of concrete ops to pto op names"
            )
        if not op_bindings:
            raise ValueError(
                f"templates[{normalized_slot!r}] must contain at least one concrete-op mapping"
            )

        frozen_bindings = []
        for concrete_op, real_op in op_bindings.items():
            normalized_concrete_op = _validate_op(concrete_op)
            if normalized_concrete_op not in match_ops:
                raise ValueError(
                    f"templates[{normalized_slot!r}] references op {normalized_concrete_op!r} "
                    f"outside descriptor matcher set {match_ops!r}"
                )
            frozen_bindings.append(
                (
                    normalized_concrete_op,
                    _validate_template_value(normalized_slot, normalized_concrete_op, real_op),
                )
            )
        frozen_templates.append((normalized_slot, tuple(frozen_bindings)))

    return tuple(frozen_templates)


def _validate_name(py_fn: Callable[..., Any], name: Any) -> str:
    if name is None:
        return py_fn.__name__
    if not isinstance(name, str) or not name:
        raise TypeError("name must be a non-empty string")
    return name


def _validate_verify(verify: Any) -> bool:
    if not isinstance(verify, bool):
        raise TypeError("verify must be a bool")
    return verify


def _validate_advanced(advanced: Any) -> bool:
    if not isinstance(advanced, bool):
        raise TypeError("advanced must be a bool")
    return advanced


def _validate_constraints(constraints: Any) -> tuple[Callable[[Mapping[str, Any]], Any], ...]:
    if constraints is _UNSET:
        return ()
    if not isinstance(constraints, (list, tuple)):
        raise TypeError("constraints must be a sequence of predicate callables")

    frozen_constraints = []
    for index, constraint in enumerate(constraints):
        if not callable(constraint):
            raise TypeError(f"constraints[{index}] must be callable")
        frozen_constraints.append(constraint)
    return tuple(frozen_constraints)


def _validate_priority(priority: Any) -> int:
    if priority is _UNSET:
        return 0
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise TypeError("priority must be an int")
    return priority


def _coerce_memory_space(value: Any, param_name: str) -> MemorySpace:
    if isinstance(value, MemorySpace):
        return value
    if isinstance(value, str):
        normalized = value.strip().upper()
        try:
            return MemorySpace[normalized]
        except KeyError as exc:
            raise ValueError(
                f"specialization for '{param_name}' uses unsupported memory_space {value!r}"
            ) from exc
    raise TypeError(
        f"specialization for '{param_name}' must provide MemorySpace or string memory_space"
    )


def _coerce_tile_config(value: Any, param_name: str) -> TileConfig | None:
    if value is None:
        return None
    if isinstance(value, TileConfig):
        return value
    if isinstance(value, dict):
        return TileConfig.from_mapping(value)
    raise TypeError(
        f"specialization for '{param_name}' must provide TileConfig, dict, or None for config"
    )


def _coerce_tile_valid_shape(
    shape: tuple[int, ...],
    value: Any,
    param_name: str,
    source_info: _FunctionSourceInfo | None,
) -> tuple[int | None, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        _raise_tile_param_error(
            source_info,
            param_name,
            f"specialization for '{param_name}' must provide valid_shape as a tuple/list",
            TypeError,
        )
    valid_shape = tuple(value)
    if len(valid_shape) != len(shape):
        _raise_tile_param_error(
            source_info,
            param_name,
            f"illegal Tile profile for '{param_name}': valid_shape rank must match shape rank",
        )

    normalized: list[int | None] = []
    for axis, (valid_dim, shape_dim) in enumerate(zip(valid_shape, shape)):
        if isinstance(valid_dim, bool):
            _raise_tile_param_error(
                source_info,
                param_name,
                f"illegal Tile profile for '{param_name}': valid_shape axis {axis} must not be bool",
                TypeError,
            )
        if isinstance(valid_dim, int):
            if valid_dim <= 0:
                _raise_tile_param_error(
                    source_info,
                    param_name,
                    f"illegal Tile profile for '{param_name}': valid_shape axis {axis} must be positive",
                )
            if valid_dim > shape_dim:
                _raise_tile_param_error(
                    source_info,
                    param_name,
                    f"illegal Tile profile for '{param_name}': valid_shape axis {axis}={valid_dim} "
                    f"must be <= shape axis {axis}={shape_dim}",
                )
            normalized.append(valid_dim)
            continue
        if valid_dim is None or isinstance(valid_dim, str):
            normalized.append(None)
            continue
        _raise_tile_param_error(
            source_info,
            param_name,
            f"illegal Tile profile for '{param_name}': valid_shape axis {axis} must be "
            "a positive int, string symbol, or None",
            TypeError,
        )
    return tuple(normalized)


def _coerce_tile_specialization(
    param_name: str,
    binding: Any,
    source_info: _FunctionSourceInfo | None,
    *,
    kernel_family: str = "vector",
) -> TileSpecialization:
    if isinstance(binding, TileSpecialization):
        spec = binding
    elif isinstance(binding, dict):
        if "shape" not in binding:
            _raise_tile_param_error(
                source_info,
                param_name,
                f"specialization for '{param_name}' must provide a static physical Tile shape",
                TypeError,
            )
        if "memory_space" not in binding:
            _raise_tile_param_error(
                source_info,
                param_name,
                f"specialization for '{param_name}' must provide memory_space",
                TypeError,
            )
        spec = TileSpecialization(
            shape=tuple(binding["shape"]),
            memory_space=_coerce_memory_space(binding["memory_space"], param_name),
            config=_coerce_tile_config(binding.get("config"), param_name),
            valid_shape=binding.get("valid_shape"),
        )
    else:
        _raise_tile_param_error(
            source_info,
            param_name,
            f"specialization for '{param_name}' must be a TileSpecialization or dict",
            TypeError,
        )

    if not spec.shape:
        _raise_tile_param_error(
            source_info,
            param_name,
            f"illegal Tile profile for '{param_name}': shape must be non-empty",
        )
    for dim in spec.shape:
        if not isinstance(dim, int) or isinstance(dim, bool):
            _raise_tile_param_error(
                source_info,
                param_name,
                f"dynamic physical Tile shape is not supported for '{param_name}'",
                TypeError,
            )
        if dim <= 0:
            _raise_tile_param_error(
                source_info,
                param_name,
                f"illegal Tile profile for '{param_name}': dimensions must be positive",
            )
    if len(spec.shape) not in (1, 2):
        _raise_tile_param_error(
            source_info,
            param_name,
            f"illegal Tile profile for '{param_name}': v1 only supports rank-1 or rank-2 Tile shapes",
        )
    allowed_memory_spaces = {
        MemorySpace.MAT,
        MemorySpace.LEFT,
        MemorySpace.RIGHT,
        MemorySpace.ACC,
        MemorySpace.BIAS,
        MemorySpace.SCALING,
        MemorySpace.UB,
    }
    if spec.memory_space not in allowed_memory_spaces:
        allowed_text = "MemorySpace.MAT/LEFT/RIGHT/ACC/BIAS/SCALING/UB"
        _raise_tile_param_error(
            source_info,
            param_name,
            f"illegal Tile profile for '{param_name}': {kernel_family} v1 only supports {allowed_text}",
        )
    valid_shape = _coerce_tile_valid_shape(spec.shape, spec.valid_shape, param_name, source_info)
    return TileSpecialization(
        shape=spec.shape,
        memory_space=spec.memory_space,
        config=spec.config,
        valid_shape=valid_shape,
    )


def _validate_leaf_dtype(dtype: Any, param_name: str) -> ScalarType | MaskType:
    if not isinstance(dtype, (ScalarType, MaskType)):
        raise TypeError(
            f"dtypes entry for parameter '{param_name}' must be a TileLang scalar or mask dtype"
        )
    return dtype


def _freeze_operand_types(operand_types: Any) -> tuple[ScalarType | MaskType, ...]:
    if not isinstance(operand_types, (list, tuple)):
        raise TypeError("operand_types must be a sequence of TileLang scalar or mask dtypes")
    return tuple(_validate_leaf_dtype(dtype, f"operand_types[{index}]") for index, dtype in enumerate(operand_types))


def _matches_wildcard(pattern: WildcardType, actual: ScalarType | MaskType) -> bool:
    if pattern.name == "AnyType":
        return isinstance(actual, ScalarType)
    if pattern.name == "AnyFloat":
        return isinstance(actual, ScalarType) and actual.name in {"f16", "bf16", "f32"}
    if pattern.name == "AnyInt":
        return isinstance(actual, ScalarType) and is_integer_dtype(actual)
    if pattern.name == "AnyMask":
        return isinstance(actual, MaskType)
    raise TypeError(f"unsupported wildcard matcher {pattern.name!r}")


def _matches_scalar_annotation(
    annotation: ScalarType | MaskType | WildcardType | TypeVariable,
    actual: ScalarType | MaskType,
) -> bool:
    if isinstance(annotation, (ScalarType, MaskType)):
        return annotation == actual
    if isinstance(annotation, WildcardType):
        return _matches_wildcard(annotation, actual)
    if isinstance(annotation, TypeVariable):
        return True
    raise TypeError(f"unsupported scalar annotation {annotation!r}")


def _match_dtype_signature(
    dtype_signature: tuple[Any, ...],
    operand_types: tuple[ScalarType | MaskType, ...],
) -> tuple[ScalarType | MaskType, ...] | None:
    if len(dtype_signature) != len(operand_types):
        return None

    typevar_bindings: dict[str, ScalarType | MaskType] = {}
    for pattern, actual in zip(dtype_signature, operand_types):
        if isinstance(pattern, (ScalarType, MaskType)):
            if pattern != actual:
                return None
            continue
        if isinstance(pattern, WildcardType):
            if not _matches_wildcard(pattern, actual):
                return None
            continue
        if isinstance(pattern, TypeVariable):
            bound = typevar_bindings.get(pattern.name)
            if bound is None:
                typevar_bindings[pattern.name] = actual
                continue
            if bound != actual:
                return None
            continue
        raise TypeError(f"unsupported dtype pattern {pattern!r}")
    return operand_types


def _match_descriptor_dtype_signature(
    descriptor: VKernelDescriptor,
    operand_types: tuple[ScalarType | MaskType, ...],
) -> tuple[ScalarType | MaskType, ...] | None:
    for dtype_signature in descriptor.dtypes:
        matched = _match_dtype_signature(dtype_signature, operand_types)
        if matched is not None:
            return matched
    return None


def _validate_parameter_spec(param: inspect.Parameter) -> KernelParameterSpec:
    if param.kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise TypeError(
            f"parameter '{param.name}' uses unsupported parameter kind for TileLang DSL v1"
        )
    if param.default is not inspect._empty:
        raise TypeError(
            f"parameter '{param.name}' must not declare a default value in TileLang DSL v1"
        )
    if param.annotation is inspect._empty:
        raise TypeError(
            f"parameter '{param.name}' must declare a TileLang DSL type annotation"
        )

    annotation = param.annotation
    if annotation is TensorView:
        return KernelParameterSpec(
            name=param.name,
            kind="tensorview",
            annotation=annotation,
        )
    if annotation is PartitionTensorView:
        return KernelParameterSpec(
            name=param.name,
            kind="partition_tensor_view",
            annotation=annotation,
        )
    if annotation is Tile:
        return KernelParameterSpec(
            name=param.name,
            kind="tile",
            annotation=annotation,
        )
    if isinstance(annotation, VectorType):
        return KernelParameterSpec(
            name=param.name,
            kind="vector",
            annotation=annotation,
        )
    if isinstance(annotation, PointerType):
        return KernelParameterSpec(
            name=param.name,
            kind="ptr",
            annotation=annotation,
        )
    if isinstance(annotation, MaskType):
        return KernelParameterSpec(
            name=param.name,
            kind="mask",
            annotation=annotation,
        )
    if isinstance(annotation, WildcardType) and annotation.name == "AnyMask":
        return KernelParameterSpec(
            name=param.name,
            kind="mask",
            annotation=annotation,
        )
    if isinstance(annotation, (ScalarType, WildcardType, TypeVariable)):
        return KernelParameterSpec(
            name=param.name,
            kind="scalar",
            annotation=annotation,
        )

    raise TypeError(
        f"parameter '{param.name}' uses unsupported annotation {annotation!r}"
    )


def _collect_parameter_specs(py_fn: Callable[..., Any]) -> tuple[KernelParameterSpec, ...]:
    signature = inspect.signature(py_fn)
    return tuple(_validate_parameter_spec(param) for param in signature.parameters.values())


def _default_dtype_signature(
    parameter_specs: tuple[KernelParameterSpec, ...],
) -> tuple[Any, ...]:
    defaults: list[Any] = []
    for param_spec in parameter_specs:
        if param_spec.kind in {"tensorview", "partition_tensor_view", "tile"}:
            defaults.append(AnyType)
            continue
        if param_spec.kind == "vector":
            defaults.append(param_spec.annotation.element_dtype)
            continue
        if param_spec.kind == "ptr":
            defaults.append(param_spec.annotation.element_dtype)
            continue
        if param_spec.kind == "mask":
            defaults.append(param_spec.annotation if isinstance(param_spec.annotation, MaskType) else AnyMask)
            continue
        if isinstance(param_spec.annotation, (WildcardType, TypeVariable)):
            defaults.append(AnyType)
            continue
        defaults.append(param_spec.annotation)
    return tuple(defaults)


def _validate_dtype_arity(
    parameter_specs: tuple[KernelParameterSpec, ...],
    dtypes: tuple[tuple[Any, ...], ...],
    *,
    kernel_family: str = "vector",
) -> None:
    if kernel_family == "cube":
        # Cube v1 reuses the shared descriptor object, but its `dtypes` surface
        # follows the cube op contract rather than Python parameter arity.
        # Keep descriptor construction permissive here and defer concrete
        # semantic checks to cube frontend/semantic analysis.
        return
    expected_arity = len(parameter_specs)
    for dtype_signature in dtypes:
        if len(dtype_signature) != expected_arity:
            raise ValueError(
                "each dtypes signature must match the decorated function parameter count"
            )


def _bind_parameter(
    param_spec: KernelParameterSpec,
    dtype: Any,
) -> BoundKernelParameter:
    bound_dtype = _validate_leaf_dtype(dtype, param_spec.name)
    if param_spec.kind in {"tensorview", "partition_tensor_view"}:
        return BoundKernelParameter(
            name=param_spec.name,
            kind=param_spec.kind,
            annotation=param_spec.annotation,
            dtype=bound_dtype,
        )
    if param_spec.kind == "tile":
        return BoundKernelParameter(
            name=param_spec.name,
            kind=param_spec.kind,
            annotation=param_spec.annotation,
            dtype=bound_dtype,
        )
    if param_spec.kind == "vector":
        if param_spec.annotation.element_dtype != bound_dtype:
            raise TypeError(
                f"vector parameter '{param_spec.name}' annotation {param_spec.annotation!r} "
                f"does not match selected dtype {bound_dtype!r}"
            )
        return BoundKernelParameter(
            name=param_spec.name,
            kind=param_spec.kind,
            annotation=param_spec.annotation,
            dtype=bound_dtype,
        )
    if param_spec.kind == "ptr":
        if param_spec.annotation.element_dtype != bound_dtype:
            raise TypeError(
                f"pointer parameter '{param_spec.name}' annotation {param_spec.annotation!r} "
                f"does not match selected dtype {bound_dtype!r}"
            )
        return BoundKernelParameter(
            name=param_spec.name,
            kind=param_spec.kind,
            annotation=param_spec.annotation,
            dtype=bound_dtype,
        )
    if param_spec.kind == "mask":
        if not isinstance(bound_dtype, MaskType):
            raise TypeError(
                f"mask parameter '{param_spec.name}' annotation {param_spec.annotation!r} "
                f"does not match selected dtype {bound_dtype!r}"
            )
        if isinstance(param_spec.annotation, MaskType) and param_spec.annotation != bound_dtype:
            raise TypeError(
                f"mask parameter '{param_spec.name}' annotation {param_spec.annotation!r} "
                f"does not match selected dtype {bound_dtype!r}"
            )
        if isinstance(param_spec.annotation, WildcardType) and not _matches_wildcard(param_spec.annotation, bound_dtype):
            raise TypeError(
                f"mask parameter '{param_spec.name}' annotation {param_spec.annotation!r} "
                f"does not match selected dtype {bound_dtype!r}"
            )
        return BoundKernelParameter(
            name=param_spec.name,
            kind=param_spec.kind,
            annotation=param_spec.annotation,
            dtype=bound_dtype,
        )
    if not _matches_scalar_annotation(param_spec.annotation, bound_dtype):
        raise TypeError(
            f"scalar parameter '{param_spec.name}' annotation {param_spec.annotation!r} "
            f"does not match selected dtype {bound_dtype!r}"
        )
    return BoundKernelParameter(
        name=param_spec.name,
        kind=param_spec.kind,
        annotation=param_spec.annotation,
        dtype=bound_dtype,
    )


def _bind_parameters(
    parameter_specs: tuple[KernelParameterSpec, ...],
    dtype_signature: tuple[ScalarType | MaskType, ...],
) -> tuple[BoundKernelParameter, ...]:
    if len(dtype_signature) != len(parameter_specs):
        raise ValueError(
            "selected dtype signature must match the decorated function parameter count"
        )
    return tuple(
        _bind_parameter(param_spec, dtype)
        for param_spec, dtype in zip(parameter_specs, dtype_signature)
    )


def _build_descriptor(
    py_fn: Callable[..., Any],
    *,
    target: str,
    op: Any,
    ops: Any,
    templates: Any,
    dtypes: Any,
    name: Any,
    verify: Any,
    advanced: Any,
    constraints: Any,
    priority: Any,
    kernel_family: str = "vector",
) -> VKernelDescriptor:
    if not callable(py_fn):
        raise TypeError("@vkernel can only decorate callables")

    source_info = _load_function_source_info(py_fn)
    advanced_enabled = _validate_advanced(advanced)
    inline_procs = _collect_inline_procs(py_fn.__module__)
    internal_inline_procs = _collect_internal_inline_procs()
    _validate_function_body(
        source_info,
        advanced_enabled=advanced_enabled,
        module_name=py_fn.__module__,
        kernel_family=kernel_family,
    )
    match_ops = _freeze_match_ops(op=op, ops=ops)
    if kernel_family == "cube":
        for match_op in match_ops:
            if _is_schema_form_match_op(match_op):
                raise ValueError(
                    "@pto.ckernel does not support schema-form op matching; "
                    "use concrete op strings such as 'pto.mad' or ops=[...]"
                )
    frozen_templates = _freeze_templates(templates, match_ops=match_ops)
    parameter_specs = _collect_parameter_specs(py_fn)
    if dtypes is None:
        dtypes = (_default_dtype_signature(parameter_specs),)
    frozen_dtypes = _freeze_dtypes(dtypes)
    _validate_dtype_arity(parameter_specs, frozen_dtypes, kernel_family=kernel_family)

    selected_op: str | None = None
    selected_dtype_signature: tuple[ScalarType | MaskType, ...] | None = None
    bound_parameters: tuple[BoundKernelParameter, ...] | None = None
    if len(match_ops) == 1:
        selected_op = match_ops[0]
    if (
        len(frozen_dtypes) == 1
        and all(isinstance(dtype, (ScalarType, MaskType)) for dtype in frozen_dtypes[0])
        and (kernel_family != "cube" or len(frozen_dtypes[0]) == len(parameter_specs))
    ):
        selected_dtype_signature = tuple(frozen_dtypes[0])
        bound_parameters = _bind_parameters(parameter_specs, selected_dtype_signature)

    return VKernelDescriptor(
        target=_validate_target(target),
        match_ops=match_ops,
        dtypes=frozen_dtypes,
        name=_validate_name(py_fn, name),
        verify_enabled=_validate_verify(verify),
        advanced_enabled=advanced_enabled,
        kernel_family=kernel_family,
        _parameter_specs=parameter_specs,
        _py_fn=py_fn,
        _source_info=source_info,
        constraints=_validate_constraints(constraints),
        priority=_validate_priority(priority),
        _templates=frozen_templates,
        _inline_procs=inline_procs,
        _internal_inline_procs=internal_inline_procs,
        _selected_op=selected_op,
        _selected_dtype_signature=selected_dtype_signature,
        _parameters=bound_parameters,
        _constraint_context_attrs=(),
    )


def _evaluate_constraints(
    descriptor: VKernelDescriptor,
    context_attrs: Mapping[str, Any],
) -> _ConstraintEvaluationResult:
    named_context: dict[str, Any] = {
        "target": context_attrs.get("target"),
        "op": context_attrs.get("op"),
        "selected_op": context_attrs.get("selected_op"),
    }
    for spec in descriptor._parameter_specs:
        param_attrs = context_attrs.get(spec.name)
        if not isinstance(param_attrs, Mapping):
            param_attrs = {}
        named_context[spec.name] = _ConstraintParamView(spec.name, param_attrs)

    for index, constraint in enumerate(descriptor.constraints):
        constraint_name = _constraint_callable_name(constraint)
        constraint_location = _constraint_callable_location(constraint)
        try:
            signature = inspect.signature(constraint)
            parameters = list(signature.parameters.values())
            kwargs: dict[str, Any] = {}
            for parameter in parameters:
                if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                    raise TypeError("constraint callables with *args are not supported")
                if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                    for key, value in named_context.items():
                        kwargs.setdefault(key, value)
                    for key, value in context_attrs.items():
                        kwargs.setdefault(key, value)
                    continue
                if parameter.name in named_context:
                    kwargs[parameter.name] = named_context[parameter.name]
                    continue
                if parameter.name in context_attrs:
                    kwargs[parameter.name] = context_attrs[parameter.name]
                    continue
                if parameter.default is not inspect._empty:
                    continue
                raise TypeError(
                    f"constraint {index} for kernel {descriptor.name!r} requires unsupported parameter "
                    f"{parameter.name!r}"
                )
            result = constraint(**kwargs)
        except Exception as exc:
            return _ConstraintEvaluationResult(
                passed=False,
                failed_constraint_index=index,
                failed_constraint_name=constraint_name,
                failed_constraint_location=constraint_location,
                error_type=type(exc).__name__,
                error_message=(
                    f"constraint {index} for kernel {descriptor.name!r} "
                    f"raised {type(exc).__name__}: {exc}"
                    f"{_format_constraint_location_suffix(constraint_location)}"
                ),
            )
        if not result:
            return _ConstraintEvaluationResult(
                passed=False,
                failed_constraint_index=index,
                failed_constraint_name=constraint_name,
                failed_constraint_location=constraint_location,
                error_message=(
                    f"constraint {index} for kernel {descriptor.name!r} returned False"
                    f"{_format_constraint_location_suffix(constraint_location)}"
                ),
            )
    return _ConstraintEvaluationResult(passed=True)


def _constraint_callable_name(constraint: Callable[..., Any]) -> str | None:
    qualname = getattr(constraint, "__qualname__", None)
    if isinstance(qualname, str) and qualname:
        return qualname
    name = getattr(constraint, "__name__", None)
    if isinstance(name, str) and name:
        return name
    return None


def _constraint_callable_location(constraint: Callable[..., Any]) -> str | None:
    code = getattr(constraint, "__code__", None)
    filename = getattr(code, "co_filename", None)
    firstlineno = getattr(code, "co_firstlineno", None)
    if isinstance(filename, str) and filename and isinstance(firstlineno, int) and firstlineno > 0:
        return f"{filename}:{firstlineno}"
    return None


def _format_constraint_location_suffix(location: str | None) -> str:
    if location is None:
        return ""
    return f" at {location}"


def _raise_constraint_evaluation_error(result: _ConstraintEvaluationResult) -> None:
    if not result.raised_error or result.error_message is None:
        return
    raise TypeError(result.error_message)


def _format_descriptor_identity(descriptor: VKernelDescriptor) -> str:
    dtype_signature = descriptor._selected_dtype_signature
    if dtype_signature is None:
        dtype_signature = tuple("?" for _ in descriptor.dtypes[0]) if descriptor.dtypes else ()
    return f"{descriptor.name}(priority={descriptor.priority}, dtypes={dtype_signature!r})"


def _bind_descriptor_for_target_op(
    descriptor: VKernelDescriptor,
    *,
    target: str,
    op: str,
) -> VKernelDescriptor | None:
    if descriptor.target != target:
        return None
    if op not in descriptor.match_ops:
        return None
    return descriptor._bind_selected_op(op)


def _collect_target_op_candidates(
    registry: KernelRegistry,
    *,
    target: str,
    op: str,
) -> tuple[_TargetOpSelectionCandidate, ...]:
    candidates: list[_TargetOpSelectionCandidate] = []
    for descriptor in registry:
        op_bound_descriptor = _bind_descriptor_for_target_op(
            descriptor,
            target=target,
            op=op,
        )
        if op_bound_descriptor is None:
            continue
        candidates.append(_TargetOpSelectionCandidate(descriptor=op_bound_descriptor))
    return tuple(candidates)


def _evaluate_dtype_candidate(
    candidate: _TargetOpSelectionCandidate,
    *,
    operand_types: tuple[ScalarType | MaskType, ...],
) -> _DtypeSelectionCandidate:
    matched_signature = _match_descriptor_dtype_signature(candidate.descriptor, operand_types)
    if matched_signature is None:
        return _DtypeSelectionCandidate(descriptor=candidate.descriptor)
    if candidate.descriptor._selected_dtype_signature == matched_signature:
        return _DtypeSelectionCandidate(
            descriptor=candidate.descriptor,
            matched_descriptor=candidate.descriptor,
            matched_dtype_signature=matched_signature,
        )
    return _DtypeSelectionCandidate(
        descriptor=candidate.descriptor,
        matched_descriptor=candidate.descriptor._bind_selected_dtype_signature(matched_signature),
        matched_dtype_signature=matched_signature,
    )


def _evaluate_dtype_candidates(
    candidates: tuple[_TargetOpSelectionCandidate, ...],
    *,
    operand_types: tuple[ScalarType | MaskType, ...],
) -> tuple[_DtypeSelectionCandidate, ...]:
    return tuple(
        _evaluate_dtype_candidate(
            candidate,
            operand_types=operand_types,
        )
        for candidate in candidates
    )


def _match_descriptor_query(
    descriptor: VKernelDescriptor,
    *,
    target: str,
    op: str,
    operand_types: tuple[ScalarType | MaskType, ...],
) -> VKernelDescriptor | None:
    op_bound_descriptor = _bind_descriptor_for_target_op(
        descriptor,
        target=target,
        op=op,
    )
    if op_bound_descriptor is None:
        return None
    dtype_result = _evaluate_dtype_candidate(
        _TargetOpSelectionCandidate(descriptor=op_bound_descriptor),
        operand_types=operand_types,
    )
    return dtype_result.matched_descriptor


def _evaluate_constraint_candidate(
    descriptor: VKernelDescriptor,
    *,
    context_attrs: Mapping[str, Any],
) -> _ConstraintSelectionCandidate:
    evaluation = _evaluate_constraints(
        descriptor,
        descriptor._constraint_context_for_evaluation(context_attrs),
    )
    if not evaluation.passed:
        return _ConstraintSelectionCandidate(
            descriptor=descriptor,
            passed=False,
            evaluation=evaluation,
        )
    return _ConstraintSelectionCandidate(
        descriptor=descriptor,
        passed=True,
        evaluation=evaluation,
        bound_descriptor=descriptor._bind_constraint_context_attrs(context_attrs),
    )


def _evaluate_constraint_candidates(
    descriptors: tuple[VKernelDescriptor, ...],
    *,
    context_attrs: Mapping[str, Any],
) -> tuple[_ConstraintSelectionCandidate, ...]:
    return tuple(
        _evaluate_constraint_candidate(
            descriptor,
            context_attrs=context_attrs,
        )
        for descriptor in descriptors
    )


def _resolve_priority_candidates(
    descriptors: tuple[VKernelDescriptor, ...],
) -> _PrioritySelectionResult:
    if not descriptors:
        return _PrioritySelectionResult(
            candidates=(),
            highest_priority=None,
            winners=(),
        )
    highest_priority = max(descriptor.priority for descriptor in descriptors)
    winners = tuple(
        descriptor
        for descriptor in descriptors
        if descriptor.priority == highest_priority
    )
    return _PrioritySelectionResult(
        candidates=descriptors,
        highest_priority=highest_priority,
        winners=winners,
    )


def _materialize_selection_candidate(
    descriptor: VKernelDescriptor,
) -> _MaterializationSelectionCandidate:
    try:
        return _MaterializationSelectionCandidate(
            descriptor=descriptor,
            mlir_text=descriptor.mlir_text(),
        )
    except Exception as exc:
        return _MaterializationSelectionCandidate(
            descriptor=descriptor,
            mlir_error=str(exc),
        )


def _collect_materialization_candidates(
    descriptors: tuple[VKernelDescriptor, ...],
) -> tuple[_MaterializationSelectionCandidate, ...]:
    return tuple(
        _materialize_selection_candidate(descriptor)
        for descriptor in descriptors
    )


def _select_kernel_no_candidate_error(
    *,
    target: str,
    op: str,
    operand_types: tuple[ScalarType | MaskType, ...],
) -> str:
    return (
        "select_kernel() found no registered kernel for "
        f"target={target!r}, op={op!r}, operand_types={operand_types!r}"
    )


def _select_kernel_constraint_error(
    *,
    target: str,
    op: str,
    operand_types: tuple[ScalarType | MaskType, ...],
) -> str:
    return (
        "select_kernel() found no registered kernel after constraint evaluation for "
        f"target={target!r}, op={op!r}, operand_types={operand_types!r}"
    )


def _select_kernel_priority_tie_error(
    *,
    target: str,
    op: str,
    operand_types: tuple[ScalarType | MaskType, ...],
    winners: tuple[VKernelDescriptor, ...],
) -> str:
    winner_set = ", ".join(sorted(_format_descriptor_identity(descriptor) for descriptor in winners))
    return (
        "select_kernel() found multiple highest-priority kernels for "
        f"target={target!r}, op={op!r}, operand_types={operand_types!r}: "
        f"{winner_set}"
    )


def _build_selection_report(
    *,
    target: str,
    op: str,
    operand_types: tuple[ScalarType | MaskType, ...],
    context_attrs: Mapping[str, Any],
    dtype_results: tuple[_DtypeSelectionCandidate, ...],
    constraint_results: tuple[_ConstraintSelectionCandidate, ...],
    materialization_results: tuple[_MaterializationSelectionCandidate, ...],
    priority_result: _PrioritySelectionResult,
    final_status: str,
    final_error: str | None,
) -> KernelSelectionReport:
    constraint_by_descriptor_id = {
        id(result.descriptor): result
        for result in constraint_results
    }
    materialization_by_descriptor_id = {
        id(result.descriptor): result
        for result in materialization_results
    }
    winner_ids = {id(descriptor) for descriptor in priority_result.winners}
    highest_priority = priority_result.highest_priority
    candidates: list[KernelSelectionCandidateMetadata] = []

    for dtype_result in dtype_results:
        if dtype_result.matched_descriptor is None:
            candidates.append(
                KernelSelectionCandidateMetadata(
                    descriptor=dtype_result.descriptor,
                    status="dtype_mismatch",
                    selected_op=dtype_result.descriptor.selected_op,
                    reason=(
                        "no dtype signature matched "
                        f"operand_types={operand_types!r}"
                    ),
                )
            )
            continue

        constraint_result = constraint_by_descriptor_id.get(id(dtype_result.matched_descriptor))
        if constraint_result is None:
            continue
        evaluation = constraint_result.evaluation
        candidate_descriptor = constraint_result.bound_descriptor or dtype_result.matched_descriptor
        materialization_result = materialization_by_descriptor_id.get(id(candidate_descriptor))
        base_kwargs = {
            "descriptor": candidate_descriptor,
            "selected_op": candidate_descriptor.selected_op,
            "matched_dtype_signature": dtype_result.matched_dtype_signature,
            "failed_constraint_index": evaluation.failed_constraint_index,
            "failed_constraint_name": evaluation.failed_constraint_name,
            "failed_constraint_location": evaluation.failed_constraint_location,
            "error_type": evaluation.error_type,
            "error_message": evaluation.error_message,
            "mlir_text": None if materialization_result is None else materialization_result.mlir_text,
            "mlir_error": None if materialization_result is None else materialization_result.mlir_error,
        }

        if evaluation.raised_error:
            candidates.append(
                KernelSelectionCandidateMetadata(
                    status="constraint_error",
                    reason=evaluation.error_message,
                    **base_kwargs,
                )
            )
            continue
        if not evaluation.passed:
            candidates.append(
                KernelSelectionCandidateMetadata(
                    status="constraint_failed",
                    reason=evaluation.error_message,
                    **base_kwargs,
                )
            )
            continue
        if id(candidate_descriptor) in winner_ids:
            status = "selected" if final_status == "selected" else "priority_tie"
            reason = None if status == "selected" else final_error
        else:
            status = "priority_shadowed"
            if highest_priority is None:
                reason = "not selected"
            else:
                reason = f"shadowed by higher-priority candidate priority={highest_priority}"
        candidates.append(
            KernelSelectionCandidateMetadata(
                status=status,
                reason=reason,
                **base_kwargs,
            )
        )

    frozen_context_attrs = tuple(
        sorted(dict(context_attrs).items(), key=lambda item: item[0])
    )
    return KernelSelectionReport(
        target=target,
        op=op,
        operand_types=operand_types,
        selected=priority_result.winner if final_status == "selected" else None,
        candidates=tuple(candidates),
        final_status=final_status,
        final_error=final_error,
        _context_attrs=frozen_context_attrs,
    )


def select_kernel(
    target: str,
    op: str,
    operand_types: Any,
    context_attrs: Mapping[str, Any] | None = None,
    registry: KernelRegistry | None = None,
    *,
    return_metadata: bool = False,
    include_mlir: bool = True,
) -> VKernelDescriptor | KernelSelectionReport:
    """Select one registered kernel descriptor for the given query."""

    normalized_target = _validate_target(target)
    normalized_op = _validate_op(op)
    normalized_operand_types = _freeze_operand_types(operand_types)

    if context_attrs is None:
        normalized_context_attrs: dict[str, Any] = {}
    elif isinstance(context_attrs, Mapping):
        normalized_context_attrs = dict(context_attrs)
    else:
        raise TypeError("context_attrs must be a mapping or None")

    active_registry = _DEFAULT_KERNEL_REGISTRY if registry is None else registry
    if not isinstance(active_registry, KernelRegistry):
        raise TypeError("registry must be a KernelRegistry or None")
    if not isinstance(return_metadata, bool):
        raise TypeError("return_metadata must be a bool")
    if not isinstance(include_mlir, bool):
        raise TypeError("include_mlir must be a bool")

    target_op_candidates = _collect_target_op_candidates(
        active_registry,
        target=normalized_target,
        op=normalized_op,
    )
    dtype_results = _evaluate_dtype_candidates(
        target_op_candidates,
        operand_types=normalized_operand_types,
    )
    type_matched_candidates = tuple(
        result.matched_descriptor
        for result in dtype_results
        if result.matched_descriptor is not None
    )

    if not type_matched_candidates:
        no_candidate_error = _select_kernel_no_candidate_error(
            target=normalized_target,
            op=normalized_op,
            operand_types=normalized_operand_types,
        )
        if return_metadata:
            return _build_selection_report(
                target=normalized_target,
                op=normalized_op,
                operand_types=normalized_operand_types,
                context_attrs=normalized_context_attrs,
                dtype_results=dtype_results,
                constraint_results=(),
                materialization_results=(),
                priority_result=_PrioritySelectionResult(candidates=(), highest_priority=None, winners=()),
                final_status="no_candidate",
                final_error=no_candidate_error,
            )
        raise LookupError(no_candidate_error)

    constraint_results = _evaluate_constraint_candidates(
        type_matched_candidates,
        context_attrs=normalized_context_attrs,
    )
    constrained_candidates = tuple(
        result.bound_descriptor
        for result in constraint_results
        if result.bound_descriptor is not None
    )
    if return_metadata:
        priority_result = _resolve_priority_candidates(constrained_candidates)
        materialization_results = (
            _collect_materialization_candidates(constrained_candidates)
            if include_mlir
            else ()
        )
        final_status = "selected"
        final_error: str | None = None
        if not constrained_candidates:
            final_status = "no_candidate"
            error_messages = [
                result.evaluation.error_message
                for result in constraint_results
                if result.evaluation.error_message is not None
            ]
            final_error = error_messages[0] if error_messages else _select_kernel_constraint_error(
                target=normalized_target,
                op=normalized_op,
                operand_types=normalized_operand_types,
            )
        elif priority_result.has_tie:
            final_status = "priority_tie"
            final_error = _select_kernel_priority_tie_error(
                target=normalized_target,
                op=normalized_op,
                operand_types=normalized_operand_types,
                winners=priority_result.winners,
            )
        return _build_selection_report(
            target=normalized_target,
            op=normalized_op,
            operand_types=normalized_operand_types,
            context_attrs=normalized_context_attrs,
            dtype_results=dtype_results,
            constraint_results=constraint_results,
            materialization_results=materialization_results,
            priority_result=priority_result,
            final_status=final_status,
            final_error=final_error,
        )
    for result in constraint_results:
        _raise_constraint_evaluation_error(result.evaluation)
    if not constrained_candidates:
        raise LookupError(
            _select_kernel_constraint_error(
                target=normalized_target,
                op=normalized_op,
                operand_types=normalized_operand_types,
            )
        )

    priority_result = _resolve_priority_candidates(constrained_candidates)
    if priority_result.has_tie:
        raise LookupError(
            _select_kernel_priority_tie_error(
                target=normalized_target,
                op=normalized_op,
                operand_types=normalized_operand_types,
                winners=priority_result.winners,
            )
        )
    assert priority_result.winner is not None
    return priority_result.winner


def vkernel(
    py_fn: Callable[..., Any] | None = None,
    *,
    target: str = "a5",
    op: str | None = None,
    ops: tuple[str, ...] | list[str] | None = None,
    templates: Any = _UNSET,
    dtypes: Any = None,
    name: str | None = None,
    verify: bool = True,
    advanced: bool = False,
    constraints: Any = _UNSET,
    priority: Any = _UNSET,
) -> VKernelDescriptor | Callable[[Callable[..., Any]], VKernelDescriptor]:
    """Create a TileLang DSL v1 kernel descriptor.

    v1 keeps only the minimal descriptor metadata surface:
    `target`, `op`/`ops`, `templates`, `dtypes`, `constraints`, `priority`, `name`,
    `verify`, and opt-in `advanced`.
    """

    def wrap(fn: Callable[..., Any]) -> VKernelDescriptor:
        descriptor = _build_descriptor(
            fn,
            target=target,
            op=op,
            ops=ops,
            templates=templates,
            dtypes=dtypes,
            name=name,
            verify=verify,
            advanced=advanced,
            constraints=constraints,
            priority=priority,
            kernel_family="vector",
        )
        return _DEFAULT_KERNEL_REGISTRY.register(descriptor)

    if py_fn is None:
        return wrap
    return wrap(py_fn)


def ckernel(
    py_fn: Callable[..., Any] | None = None,
    *,
    target: str = "a5",
    op: str | None = None,
    ops: tuple[str, ...] | list[str] | None = None,
    templates: Any = _UNSET,
    dtypes: Any = None,
    name: str | None = None,
    constraints: Any = _UNSET,
    priority: Any = _UNSET,
) -> VKernelDescriptor | Callable[[Callable[..., Any]], VKernelDescriptor]:
    """Create a TileLang DSL cube-kernel descriptor.

    This public entrypoint intentionally reuses the existing descriptor and
    registry path first; cube-specific semantic and lowering behavior is added
    incrementally in follow-up tasks.
    """

    def wrap(fn: Callable[..., Any]) -> VKernelDescriptor:
        descriptor = _build_descriptor(
            fn,
            target=target,
            op=op,
            ops=ops,
            templates=templates,
            dtypes=dtypes,
            name=name,
            verify=True,
            advanced=False,
            constraints=constraints,
            priority=priority,
            kernel_family="cube",
        )
        return _DEFAULT_KERNEL_REGISTRY.register(descriptor)

    if py_fn is None:
        return wrap
    return wrap(py_fn)


__all__ = [
    "BoundKernelParameter",
    "InlineProcDescriptor",
    "KernelRegistry",
    "KernelSelectionCandidateMetadata",
    "KernelSelectionReport",
    "MaterializedMLIRModule",
    "TileLangFrontendError",
    "VKernelDescriptor",
    "ckernel",
    "inline_proc",
    "select_kernel",
    "vkernel",
]
