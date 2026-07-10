# PTODSL VMI DSL Quick Reference

This is a compact guide derived from
`ptodsl/docs/user_guide/14-vmi-virtual-instruction-set.md`. Use the user guide
as the source of truth when an operation is missing here.

## Core Types

- `pto.vmi.vreg(lanes, dtype, *, layout=None)` creates a logical vector type.
  `lanes` must be a multiple of 64. Common full-register choices:
  - `pto.f32` / `pto.i32`: 64 lanes per 256B physical vreg.
  - `pto.f16` / `pto.bf16` / `pto.i16`: 128 lanes per physical vreg.
  - `pto.i8` / `pto.ui8` / fp8: 256 lanes per physical vreg.
- `pto.vmi.mask(lanes, *, layout=None)` creates a logical per-lane mask type.
  Its lane count must match the gated vector.

Default to omitting `layout`; VMI layout assignment is normally a PTOAS lowering
concern.

`dtype` may come from a `pto.const_expr` parameter. Prefer compile-time dtype
selection inside one PTODSL function over many small wrappers that only vary the
VMI element type:

```python
vec_ty = pto.vmi.vreg(lanes, OUT_DTYPE)
vec = pto.vmi.vcvt(src, to_dtype=OUT_DTYPE)
```

## Load And Store

```python
vec = pto.vmi.vload(ub_src, offset, size=128)
vec = pto.vmi.vload(ub_src, offset, result_type=pto.vmi.vreg(128, pto.f16))
pto.vmi.vstore(vec, ub_dst, offset, mask)
```

Use VMI load/store only for UB-resident compute operands/results. Use MTE or
tile operations for GM movement.

Useful options:

- `dist_mode="dintlv"` for deinterleaved load/store forms.
- `dist_mode="unpack", to_dtype=<dtype>` for one-step widening load.
- `stride=...`, `block_stride=...`, `repeat_stride=...`, `group=...` for
  strided or grouped access when the source algorithm truly uses it.

Backend note: prefer putting dynamic tail masks on compute/store. Do not rely on
masked loads unless the current backend explicitly supports the form.

## Masks

```python
mask = pto.vmi.create_mask(active_lanes, result_type=pto.vmi.mask(lanes))
gmask = pto.vmi.create_group_mask(
    active_per_group,
    result_type=pto.vmi.mask(lanes),
    num_groups=num_groups,
    group_size=group_size,
)
```

Use `create_mask` for prefix-active dynamic tails. Use group masks for grouped
reductions or group broadcast patterns.

## Index And Broadcast

```python
idx = pto.vmi.vci(base, result_type=pto.vmi.vreg(64, pto.i32), order="ASC")
bc = pto.vmi.vbrc(value, result_type=pto.vmi.vreg(128, pto.f16))
```

Use `vci` for lane-wise index ramps and gather/scatter offsets. Use `vbrc` for
scalar-to-vector or group-to-vector broadcast.

## Elementwise And Scalar Ops

Same-shape vector ops usually infer result type:

```python
y = pto.vmi.vadd(a, b, mask)
y = pto.vmi.vsub(a, b, mask)
y = pto.vmi.vmul(a, b, mask)
y = pto.vmi.vdiv(a, b, mask)
y = pto.vmi.vmax(a, b, mask)
y = pto.vmi.vmin(a, b, mask)
```

Unary ops:

```python
y = pto.vmi.vabs(x, mask)
y = pto.vmi.vneg(x, mask)
y = pto.vmi.vrelu(x, mask)
y = pto.vmi.vexp(x, mask)
y = pto.vmi.vln(x, mask)
y = pto.vmi.vsqrt(x, mask)
```

Vector-scalar ops require a mask:

```python
y = pto.vmi.vadds(x, scalar, mask)
y = pto.vmi.vmuls(x, scalar, mask)
y = pto.vmi.vmaxs(x, scalar, mask)
y = pto.vmi.vmins(x, scalar, mask)
```

Integer/bitwise ops include `vand`, `vor`, `vxor`, `vnot`, `vshl`, `vshr`,
`vshls`, and `vshrs`.

## Compare, Select, Reductions

```python
cmp_mask = pto.vmi.vcmp(lhs, rhs, seed_mask, "lt")
cmp_mask = pto.vmi.vcmps(x, scalar, seed_mask, "ge")
out = pto.vmi.vsel(cmp_mask, true_value, false_value)
```

Reduction result types are explicit:

```python
sum1 = pto.vmi.vcadd(x, mask, result_type=pto.vmi.vreg(1, pto.f32))
max1 = pto.vmi.vcmax(x, mask, result_type=pto.vmi.vreg(1, pto.f32))
sum_g = pto.vmi.vcadd(
    x, gmask, result_type=pto.vmi.vreg(num_groups, pto.f32), group=num_groups
)
```

## Conversion And Reinterpretation

```python
wide = pto.vmi.vcvt(x_f16, to_dtype=pto.f32)
narrow = pto.vmi.vcvt(x_f32, to_dtype=pto.f16, rounding="...", saturate=...)
bits = pto.vmi.vinterpret_cast(x, result_type=pto.vmi.vreg(64, pto.i32))
```

Use `vcvt` for numeric conversion. Use `vinterpret_cast` only for bit-level
reinterpretation.

## Gather, Scatter, Rearrangement

```python
values = pto.vmi.vgather(src_ub, offsets, mask, result_type=pto.vmi.vreg(64, pto.f32))
pto.vmi.vscatter(values, dst_ub, offsets, mask)
lo, hi = pto.vmi.vintlv(a, b, mask)
even, odd = pto.vmi.vdintlv(a, b, mask)
sel = pto.vmi.vselr(source, index, result_type=...)
```

Only use explicit VMI interleave/deinterleave when it changes the logical data
ordering. If the AscendC source uses interleave only to repair physical register
layout after widening or packing, collapse it into the logical VMI value.

## Common AscendC SIMD To VMI Patterns

- `vlds` from UB -> `pto.vmi.vload`.
- `vsts` to UB -> `pto.vmi.vstore`.
- `vcvt f16/bf16 -> f32` -> `pto.vmi.vcvt(..., to_dtype=pto.f32)`.
- `vcvt f32 -> f16/fp8` -> `pto.vmi.vcvt(..., to_dtype=<dst>)`.
- `vmul`, `vadd`, `vsub`, `vmax`, `vmin` -> corresponding VMI elementwise op.
- `vcmp` + `vsel` -> VMI compare mask plus `vsel`.
- `vintlv`/`vdintlv` trees, `PART_P*`, `PART_EVEN/ODD`, packed store modes:
  usually physical-only lowering details; collapse unless they change the
  logical result order.

## Lane Selection Heuristic

Choose the largest contiguous logical chunk that matches the algorithm and VMI
constraints:

1. Keep dtype equal to the logical element type at that stage.
2. Prefer one full physical-register worth of elements when the algorithm and
   UB layout are contiguous.
3. Use smaller multiples of 64 for row tails or natural row/group widths.
4. For dynamic remainders, use `create_mask` with the same lane count and keep
   offsets symbolic.
