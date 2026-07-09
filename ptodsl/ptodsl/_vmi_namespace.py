# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Public PTODSL namespace for formal VMI APIs."""

from __future__ import annotations

from collections.abc import Sequence

from mlir.dialects import pto as _pto
from mlir.ir import BF16Type, F16Type, F32Type, IntegerType, MemRefType

from ._scalar_coercion import coerce_scalar_to_type
from ._surface_values import _coerce_index_value, unwrap_surface_value, wrap_surface_value
from ._types import _ensure_tensor_storage_dtype, _resolve, vmi_mask_type, vmi_vreg_type


def _missing_vmi_support_error(op_name: str) -> NotImplementedError:
    return NotImplementedError(
        f"{op_name} is not available in the current PTO Python bindings or "
        "backend support. Rebuild PTO Python bindings and update VMI support "
        "for this operation before using the PTODSL pto.vmi surface."
    )


def _unsupported_vmi_feature_error(op_name: str, feature: str) -> NotImplementedError:
    return NotImplementedError(
        f"{op_name} {feature} is not available in the current generated VMI "
        "binding/backend support; update the PTO bindings or use an explicit "
        "supported VMI form."
    )


def _generated(op_name: str):
    fn = getattr(_pto, f"vmi_{op_name}", None)
    if fn is None:
        raise _missing_vmi_support_error(f"pto.vmi.{op_name}")
    return fn


def _raw(value):
    return unwrap_surface_value(value)


def _raw_sequence(values):
    if _is_sequence(values):
        return [_raw(value) for value in values]
    return [_raw(values)]


def _is_sequence(value) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _wrap_result(result):
    if hasattr(result, "type"):
        return wrap_surface_value(result)
    try:
        count = len(result)
    except TypeError:
        count = None
    if count is not None:
        return tuple(wrap_surface_value(result[index]) for index in range(count))
    if _is_sequence(result) or hasattr(result, "__iter__"):
        return tuple(wrap_surface_value(value) for value in result)
    return wrap_surface_value(result)


def _type_of(value):
    return _raw(value).type


def _result_type_or(value, result_type, *, context: str):
    if result_type is None:
        return _type_of(value)
    return _resolve(result_type)


def _require_result_type(result_type, *, context: str):
    if result_type is None:
        raise TypeError(f"{context} requires explicit result_type")
    return _resolve(result_type)


def _result_type_list(result_type, *, context: str, duplicate_for_dintlv: bool = False):
    if result_type is None:
        raise TypeError(f"{context} requires explicit result_type")
    if _is_sequence(result_type):
        return [_resolve(item) for item in result_type]
    resolved = _resolve(result_type)
    if duplicate_for_dintlv:
        return [resolved, resolved]
    return [resolved]


def _two_result_types(result_types, lhs, rhs):
    if result_types is None:
        return _type_of(lhs), _type_of(rhs)
    if not _is_sequence(result_types) or len(result_types) != 2:
        raise TypeError("result_types must be a two-item sequence")
    return _resolve(result_types[0]), _resolve(result_types[1])


def _as_vmi_vreg_type(type_obj, *, context: str):
    vreg_type_cls = getattr(_pto, "VMIVRegType", None)
    if vreg_type_cls is None:
        raise _missing_vmi_support_error("!pto.vmi.vreg")
    try:
        return vreg_type_cls(type_obj)
    except Exception as exc:
        raise TypeError(f"{context} expects a !pto.vmi.vreg value, got {type_obj}") from exc


def _vmi_element_type(type_obj, *, context: str):
    return _as_vmi_vreg_type(type_obj, context=context).element_type


def _pointer_element_type(type_obj, *, context: str):
    ptr_type_cls = getattr(_pto, "PtrType", None)
    if ptr_type_cls is not None:
        try:
            return ptr_type_cls(type_obj).element_type
        except Exception:
            pass
    try:
        return MemRefType(type_obj).element_type
    except Exception as exc:
        raise TypeError(f"{context} expects a pointer or memref source, got {type_obj}") from exc


def _type_bit_width(type_obj, *, context: str):
    if IntegerType.isinstance(type_obj):
        return IntegerType(type_obj).width
    if F16Type.isinstance(type_obj) or BF16Type.isinstance(type_obj):
        return 16
    if F32Type.isinstance(type_obj):
        return 32
    raise TypeError(f"{context} does not support element type {type_obj}")


def _derive_vcvt_result_type(source, to_dtype, *, context: str):
    if to_dtype is None:
        raise TypeError(f"{context} requires to_dtype when result_type is omitted")
    source_type = _as_vmi_vreg_type(_type_of(source), context=context)
    elem_type = _ensure_tensor_storage_dtype(to_dtype, context=context)
    return _pto.VMIVRegType.get(
        source_type.element_count,
        elem_type,
        layout=source_type.layout,
    )


def _coerce_scalar_like_vmi_element(vector_value, scalar_value, *, context: str):
    elem_type = _vmi_element_type(_type_of(vector_value), context=context)
    return coerce_scalar_to_type(scalar_value, elem_type, context=context)


def _optional_mask(mask):
    if mask is None:
        return []
    return [_raw(mask)]


def _required_mask(mask, *, context: str):
    if mask is None:
        raise TypeError(f"{context} requires a mask operand")
    return _raw(mask)


def _i16_value(value, *, context: str):
    if value is None:
        return None
    return coerce_scalar_to_type(value, IntegerType.get_signless(16), context=context)


def _resolve_vmi_mask_result_type(size, result_type, *, context: str):
    if result_type is not None:
        return _resolve(result_type)
    if size is None:
        raise TypeError(f"{context} requires size when result_type is omitted")
    return _resolve(vmi_mask_type(size))


def _resolve_vmi_unpack_result_type(source, size, to_dtype, *, context: str):
    if to_dtype is None:
        raise TypeError(
            f'{context} requires to_dtype when dist_mode="unpack" and result_type is omitted'
        )
    source_type = _pointer_element_type(_type_of(source), context=context)
    result_type = _ensure_tensor_storage_dtype(to_dtype, context=context)
    source_bits = _type_bit_width(source_type, context=context)
    result_bits = _type_bit_width(result_type, context=context)
    if source_bits * 2 != result_bits:
        raise TypeError(
            f"{context} requires unpack to widen by exactly one step; got "
            f"{source_type} -> {result_type}"
        )
    return _pto.VMIVRegType.get(size, result_type)


def _resolve_vmi_vload_result_types(source, size, result_type, *, dist_mode, to_dtype, context: str):
    if to_dtype is not None and dist_mode != "unpack":
        raise TypeError(f'{context} accepts to_dtype only when dist_mode="unpack"')
    if result_type is not None:
        resolved_types = _result_type_list(
            result_type,
            context=context,
            duplicate_for_dintlv=(dist_mode == "dintlv"),
        )
        if dist_mode == "unpack" and to_dtype is not None:
            unpack_type = _as_vmi_vreg_type(resolved_types[0], context=context)
            expected_elem_type = _ensure_tensor_storage_dtype(to_dtype, context=context)
            if unpack_type.element_type != expected_elem_type:
                raise TypeError(
                    f"{context} result_type element type {unpack_type.element_type} "
                    f"does not match to_dtype {expected_elem_type}"
                )
            if size is not None and unpack_type.element_count != size:
                raise TypeError(
                    f"{context} result_type lane count {unpack_type.element_count} "
                    f"does not match size {size}"
                )
        return resolved_types
    if size is None:
        raise TypeError(f"{context} requires size when result_type is omitted")
    if dist_mode == "unpack":
        return [_resolve_vmi_unpack_result_type(source, size, to_dtype, context=context)]
    element_type = _pointer_element_type(_type_of(source), context=context)
    resolved = _pto.VMIVRegType.get(size, element_type)
    if dist_mode == "dintlv":
        return [resolved, resolved]
    return [resolved]


def _call_value(op_name: str, *args, **kwargs):
    return _wrap_result(_generated(op_name)(*args, **kwargs))


def _emit_binary(op_name: str, lhs, rhs, mask=None, *, result_type=None, pmode=None, loc=None, ip=None):
    return _call_value(
        op_name,
        _result_type_or(lhs, result_type, context=f"pto.vmi.{op_name}(...)"),
        _raw(lhs),
        _raw(rhs),
        _optional_mask(mask),
        pmode=pmode,
        loc=loc,
        ip=ip,
    )


def _emit_unary(op_name: str, source, mask=None, *, result_type=None, pmode=None, loc=None, ip=None):
    return _call_value(
        op_name,
        _result_type_or(source, result_type, context=f"pto.vmi.{op_name}(...)"),
        _raw(source),
        _optional_mask(mask),
        pmode=pmode,
        loc=loc,
        ip=ip,
    )


def _emit_vec_scalar(op_name: str, source, scalar, mask, *, result_type=None, pmode=None, loc=None, ip=None):
    context = f"pto.vmi.{op_name}(...)"
    return _call_value(
        op_name,
        _result_type_or(source, result_type, context=context),
        _raw(source),
        _coerce_scalar_like_vmi_element(source, scalar, context=context),
        _required_mask(mask, context=context),
        pmode=pmode,
        loc=loc,
        ip=ip,
    )


def _emit_reduce(op_name: str, source, mask, *, result_type, group=None, pmode=None, loc=None, ip=None, reassoc=None):
    kwargs = {"group": group, "pmode": pmode, "loc": loc, "ip": ip}
    if reassoc is not None:
        kwargs["reassoc"] = reassoc
    return _call_value(
        op_name,
        _require_result_type(result_type, context=f"pto.vmi.{op_name}(...)"),
        _raw(source),
        _required_mask(mask, context=f"pto.vmi.{op_name}(...)"),
        **kwargs,
    )


class _VMINamespace:
    vreg = staticmethod(vmi_vreg_type)
    mask = staticmethod(vmi_mask_type)

    @staticmethod
    def vload(
        source,
        offset,
        *,
        size=None,
        to_dtype=None,
        result_type=None,
        stride=None,
        block_stride=None,
        repeat_stride=None,
        dist_mode=None,
        group=None,
        pmode=None,
        loc=None,
        ip=None,
    ):
        result_types = _resolve_vmi_vload_result_types(
            source,
            size,
            result_type,
            dist_mode=dist_mode,
            to_dtype=to_dtype,
            context="pto.vmi.vload(...)",
        )
        return _call_value(
            "vload",
            result_types,
            _raw(source),
            _coerce_index_value(offset),
            stride=None if stride is None else _coerce_index_value(stride),
            block_stride=_i16_value(block_stride, context="pto.vmi.vload(block_stride)"),
            repeat_stride=_i16_value(repeat_stride, context="pto.vmi.vload(repeat_stride)"),
            dist_mode=dist_mode,
            group=group,
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vstore(
        values,
        destination,
        offset,
        mask=None,
        *,
        stride=None,
        block_stride=None,
        repeat_stride=None,
        dist_mode=None,
        group=None,
        pmode=None,
        loc=None,
        ip=None,
    ):
        return _generated("vstore")(
            _raw_sequence(values),
            _raw(destination),
            _coerce_index_value(offset),
            _optional_mask(mask),
            stride=None if stride is None else _coerce_index_value(stride),
            block_stride=_i16_value(block_stride, context="pto.vmi.vstore(block_stride)"),
            repeat_stride=_i16_value(repeat_stride, context="pto.vmi.vstore(repeat_stride)"),
            dist_mode=dist_mode,
            group=group,
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vci(base, *, result_type=None, order=None, loc=None, ip=None):
        result_type = _require_result_type(result_type, context="pto.vmi.vci(...)")
        base = coerce_scalar_to_type(
            base,
            _vmi_element_type(result_type, context="pto.vmi.vci(...)"),
            context="pto.vmi.vci(base)",
        )
        return _call_value("vci", result_type, base, order=order, loc=loc, ip=ip)

    vadd = staticmethod(lambda lhs, rhs, mask=None, **kw: _emit_binary("vadd", lhs, rhs, mask, **kw))
    vsub = staticmethod(lambda lhs, rhs, mask=None, **kw: _emit_binary("vsub", lhs, rhs, mask, **kw))
    vmul = staticmethod(lambda lhs, rhs, mask=None, **kw: _emit_binary("vmul", lhs, rhs, mask, **kw))
    vdiv = staticmethod(lambda lhs, rhs, mask=None, **kw: _emit_binary("vdiv", lhs, rhs, mask, **kw))
    vmax = staticmethod(lambda lhs, rhs, mask=None, **kw: _emit_binary("vmax", lhs, rhs, mask, **kw))
    vmin = staticmethod(lambda lhs, rhs, mask=None, **kw: _emit_binary("vmin", lhs, rhs, mask, **kw))
    vand = staticmethod(lambda lhs, rhs, mask=None, **kw: _emit_binary("vand", lhs, rhs, mask, **kw))
    vor = staticmethod(lambda lhs, rhs, mask=None, **kw: _emit_binary("vor", lhs, rhs, mask, **kw))
    vxor = staticmethod(lambda lhs, rhs, mask=None, **kw: _emit_binary("vxor", lhs, rhs, mask, **kw))
    vshl = staticmethod(lambda lhs, rhs, mask=None, **kw: _emit_binary("vshl", lhs, rhs, mask, **kw))
    vshr = staticmethod(lambda lhs, rhs, mask=None, **kw: _emit_binary("vshr", lhs, rhs, mask, **kw))

    vabs = staticmethod(lambda source, mask=None, **kw: _emit_unary("vabs", source, mask, **kw))
    vneg = staticmethod(lambda source, mask=None, **kw: _emit_unary("vneg", source, mask, **kw))
    vrelu = staticmethod(lambda source, mask=None, **kw: _emit_unary("vrelu", source, mask, **kw))
    vexp = staticmethod(lambda source, mask=None, **kw: _emit_unary("vexp", source, mask, **kw))
    vln = staticmethod(lambda source, mask=None, **kw: _emit_unary("vln", source, mask, **kw))
    vsqrt = staticmethod(lambda source, mask=None, **kw: _emit_unary("vsqrt", source, mask, **kw))
    vnot = staticmethod(lambda source, mask=None, **kw: _emit_unary("vnot", source, mask, **kw))

    vadds = staticmethod(lambda source, scalar, mask, **kw: _emit_vec_scalar("vadds", source, scalar, mask, **kw))
    vmuls = staticmethod(lambda source, scalar, mask, **kw: _emit_vec_scalar("vmuls", source, scalar, mask, **kw))
    vmaxs = staticmethod(lambda source, scalar, mask, **kw: _emit_vec_scalar("vmaxs", source, scalar, mask, **kw))
    vmins = staticmethod(lambda source, scalar, mask, **kw: _emit_vec_scalar("vmins", source, scalar, mask, **kw))
    vshls = staticmethod(lambda source, scalar, mask, **kw: _emit_vec_scalar("vshls", source, scalar, mask, **kw))
    vshrs = staticmethod(lambda source, scalar, mask, **kw: _emit_vec_scalar("vshrs", source, scalar, mask, **kw))

    @staticmethod
    def vcmp(lhs, rhs, seed, cmp, *, result_type=None, pmode=None, loc=None, ip=None):
        return _call_value(
            "vcmp",
            _result_type_or(seed, result_type, context="pto.vmi.vcmp(...)"),
            _raw(lhs),
            _raw(rhs),
            _raw(seed),
            cmp,
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vcmps(source, scalar, seed, cmp, *, result_type=None, pmode=None, loc=None, ip=None):
        context = "pto.vmi.vcmps(...)"
        return _call_value(
            "vcmps",
            _result_type_or(seed, result_type, context=context),
            _raw(source),
            _coerce_scalar_like_vmi_element(source, scalar, context=context),
            _raw(seed),
            cmp,
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vsel(mask, true_value, false_value, *, result_type=None, pmode=None, loc=None, ip=None):
        return _call_value(
            "vsel",
            _result_type_or(true_value, result_type, context="pto.vmi.vsel(...)"),
            _raw(mask),
            _raw(true_value),
            _raw(false_value),
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vselr(source, index, *, result_type=None, loc=None, ip=None):
        return _call_value(
            "vselr",
            _require_result_type(result_type, context="pto.vmi.vselr(...)"),
            _raw(source),
            _raw(index),
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vbrc(value, *, result_type=None, group=None, loc=None, ip=None):
        context = "pto.vmi.vbrc(...)"
        result_type = _require_result_type(result_type, context=context)
        raw_value = _raw(value)
        if not hasattr(raw_value, "type") or not _is_vmi_vreg_type(raw_value.type):
            raw_value = coerce_scalar_to_type(
                value,
                _vmi_element_type(result_type, context=context),
                context="pto.vmi.vbrc(value)",
            )
        return _call_value("vbrc", result_type, raw_value, group=group, loc=loc, ip=ip)

    vcadd = staticmethod(lambda source, mask, *, result_type=None, group=None, pmode=None, reassoc=None, loc=None, ip=None: _emit_reduce("vcadd", source, mask, result_type=result_type, group=group, pmode=pmode, reassoc=reassoc, loc=loc, ip=ip))
    vcmax = staticmethod(lambda source, mask, *, result_type=None, group=None, pmode=None, loc=None, ip=None: _emit_reduce("vcmax", source, mask, result_type=result_type, group=group, pmode=pmode, loc=loc, ip=ip))
    vcmin = staticmethod(lambda source, mask, *, result_type=None, group=None, pmode=None, loc=None, ip=None: _emit_reduce("vcmin", source, mask, result_type=result_type, group=group, pmode=pmode, loc=loc, ip=ip))

    @staticmethod
    def vcvt(
        source,
        to_dtype=None,
        mask=None,
        *,
        result_type=None,
        rounding=None,
        saturate=None,
        sign=None,
        pmode=None,
        loc=None,
        ip=None,
    ):
        if mask is not None:
            raise _unsupported_vmi_feature_error("pto.vmi.vcvt", "masked form")
        if result_type is None:
            result_type = _derive_vcvt_result_type(source, to_dtype, context="pto.vmi.vcvt(...)")
        else:
            result_type = _resolve(result_type)
        return _call_value(
            "vcvt",
            result_type,
            _raw(source),
            rounding=rounding,
            saturate=saturate,
            sign=sign,
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vinterpret_cast(source, *, result_type=None, loc=None, ip=None):
        return _call_value(
            "vinterpret_cast",
            _require_result_type(result_type, context="pto.vmi.vinterpret_cast(...)"),
            _raw(source),
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vexpdif(x, max_value, mask, *, result_type=None, pmode=None, loc=None, ip=None):
        return _call_value(
            "vexpdif",
            _result_type_or(max_value, result_type, context="pto.vmi.vexpdif(...)"),
            _raw(x),
            _raw(max_value),
            _required_mask(mask, context="pto.vmi.vexpdif(...)"),
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vaxpy(x, acc, alpha, mask, *, result_type=None, pmode=None, loc=None, ip=None):
        context = "pto.vmi.vaxpy(...)"
        return _call_value(
            "vaxpy",
            _result_type_or(acc, result_type, context=context),
            _raw(x),
            _raw(acc),
            _coerce_scalar_like_vmi_element(x, alpha, context=context),
            _required_mask(mask, context=context),
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vlrelu(x, slope, mask, *, result_type=None, pmode=None, loc=None, ip=None):
        return _call_value(
            "vlrelu",
            _result_type_or(x, result_type, context="pto.vmi.vlrelu(...)"),
            _raw(x),
            _coerce_scalar_like_vmi_element(x, slope, context="pto.vmi.vlrelu(...)"),
            _required_mask(mask, context="pto.vmi.vlrelu(...)"),
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vprelu(x, alpha, mask, *, result_type=None, pmode=None, loc=None, ip=None):
        return _call_value(
            "vprelu",
            _result_type_or(x, result_type, context="pto.vmi.vprelu(...)"),
            _raw(x),
            _raw(alpha),
            _required_mask(mask, context="pto.vmi.vprelu(...)"),
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vmull(a, b, mask, *, result_type=None, pmode=None, loc=None, ip=None):
        return _call_value(
            "vmull",
            _require_result_type(result_type, context="pto.vmi.vmull(...)"),
            _raw(a),
            _raw(b),
            _required_mask(mask, context="pto.vmi.vmull(...)"),
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vmula(acc, lhs, rhs, mask, *, result_type=None, pmode=None, loc=None, ip=None):
        return _call_value(
            "vmula",
            _result_type_or(acc, result_type, context="pto.vmi.vmula(...)"),
            _raw(acc),
            _raw(lhs),
            _raw(rhs),
            _required_mask(mask, context="pto.vmi.vmula(...)"),
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vhist(bin_idx, mask, *, result_type=None, pmode=None, loc=None, ip=None):
        return _call_value(
            "vhist",
            _require_result_type(result_type, context="pto.vmi.vhist(...)"),
            _raw(bin_idx),
            _required_mask(mask, context="pto.vmi.vhist(...)"),
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vgather(source, offsets, mask, *, result_type=None, pmode=None, loc=None, ip=None):
        return _call_value(
            "vgather",
            _require_result_type(result_type, context="pto.vmi.vgather(...)"),
            _raw(source),
            _raw(offsets),
            _required_mask(mask, context="pto.vmi.vgather(...)"),
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vgatherb(source, offsets, mask, *, result_type=None, pmode=None, loc=None, ip=None):
        return _call_value(
            "vgatherb",
            _require_result_type(result_type, context="pto.vmi.vgatherb(...)"),
            _raw(source),
            _raw(offsets),
            _required_mask(mask, context="pto.vmi.vgatherb(...)"),
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vscatter(value, destination, offsets, mask, *, pmode=None, loc=None, ip=None):
        return _generated("vscatter")(
            _raw(value),
            _raw(destination),
            _raw(offsets),
            _required_mask(mask, context="pto.vmi.vscatter(...)"),
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def create_mask(active_lanes, *, size=None, result_type=None, loc=None, ip=None):
        return _call_value(
            "create_mask",
            _resolve_vmi_mask_result_type(size, result_type, context="pto.vmi.create_mask(...)"),
            _coerce_index_value(active_lanes),
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def create_group_mask(
        active_elems_per_group,
        *,
        size=None,
        result_type=None,
        num_groups,
        group_size,
        loc=None,
        ip=None,
    ):
        return _call_value(
            "create_group_mask",
            _resolve_vmi_mask_result_type(size, result_type, context="pto.vmi.create_group_mask(...)"),
            _coerce_index_value(active_elems_per_group),
            num_groups,
            group_size,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vintlv(lhs, rhs, mask, *, result_types=None, pmode=None, loc=None, ip=None):
        low, high = _two_result_types(result_types, lhs, rhs)
        return _call_value(
            "vintlv",
            low,
            high,
            _raw(lhs),
            _raw(rhs),
            _required_mask(mask, context="pto.vmi.vintlv(...)"),
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vdintlv(lhs, rhs, mask, *, result_types=None, pmode=None, loc=None, ip=None):
        low, high = _two_result_types(result_types, lhs, rhs)
        return _call_value(
            "vdintlv",
            low,
            high,
            _raw(lhs),
            _raw(rhs),
            _required_mask(mask, context="pto.vmi.vdintlv(...)"),
            pmode=pmode,
            loc=loc,
            ip=ip,
        )


def _is_vmi_vreg_type(type_obj) -> bool:
    vreg_type_cls = getattr(_pto, "VMIVRegType", None)
    if vreg_type_cls is None:
        return False
    try:
        return vreg_type_cls.isinstance(type_obj)
    except Exception:
        return False


vmi = _VMINamespace()

__all__ = ["vmi"]
