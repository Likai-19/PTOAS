# 14. The `pto.vmi` Namespace

This chapter introduces PTODSL's formal VMI authoring surface: `pto.vmi`.
Use this namespace when you want to write against the logical VMI instruction
set directly, with operation names and types aligned to the formal
`pto.vmi.*` specification.

The goal of `pto.vmi` is different from the existing top-level PTODSL vector
helpers. The top-level helpers remain the established PTODSL vector programming
surface. `pto.vmi` is a separate, explicit namespace for kernels that want to
author logical VMI vectors, masks, and operations directly.

## 14.1 What `pto.vmi` is for

`pto.vmi` is the public PTODSL namespace for the formal VMI surface.

It gives you:

- logical VMI vector types: `pto.vmi.vreg(...)`
- logical VMI mask types: `pto.vmi.mask(...)`
- formal VMI operation names such as `pto.vmi.vload`, `pto.vmi.vadd`,
  `pto.vmi.vcvt`, and `pto.vmi.vstore`
- formal predicate construction through `pto.vmi.create_mask` and
  `pto.vmi.create_group_mask`

This surface is useful when:

- you want PTODSL source to mirror the formal VMI spec naming closely
- you want to author logical vector IR directly instead of using the older
  top-level vector helper surface
- you need explicit VMI logical vector and mask types in authored code

## 14.2 Namespace shape

The VMI surface lives under `pto.vmi`:

```python
from ptodsl import pto

vec_ty = pto.vmi.vreg(64, pto.f32)
mask_ty = pto.vmi.mask(64)
```

Operation names follow the formal `pto.vmi` prefix:

```python
from ptodsl import pto

mask = pto.vmi.create_mask(active_lanes, size=64)
vec = pto.vmi.vload(src_ptr, offset, size=64)
out = pto.vmi.vadd(vec, other, mask)
pto.vmi.vstore(out, dst_ptr, offset, mask)
```

The namespace is intentionally explicit. PTODSL does not silently remap the
existing top-level vector helpers into `pto.vmi.*` calls.

## 14.3 VMI logical types

`pto.vmi` adds two logical type constructors.

### `pto.vmi.vreg(lanes, dtype, *, layout=None)`

Creates a logical VMI vector type descriptor.

```python
vec_f32 = pto.vmi.vreg(64, pto.f32)
vec_i32 = pto.vmi.vreg(64, pto.i32)
vec_f16 = pto.vmi.vreg(128, pto.f16)
```

Use this type when:

- an operation result cannot be inferred safely and still needs
  `result_type=...`
- you want a function-local authored value to carry an explicit VMI logical
  vector type

The `lanes` argument is the logical lane count, not a physical register count.

### `pto.vmi.mask(lanes, *, layout=None)`

Creates a logical VMI mask type descriptor.

```python
mask64 = pto.vmi.mask(64)
mask128 = pto.vmi.mask(128)
```

Use this type when:

- spelling an explicit VMI mask type in authored code
- annotating advanced logical mask layouts directly

For the public PTODSL VMI surface, predicate granularity is implicit and
always resolves to `pred`.

## 14.4 Result typing rules

`pto.vmi` only infers result types when the result is unambiguous from the
operands.

Same-shape logical compute usually infers naturally:

```python
out = pto.vmi.vadd(lhs, rhs, mask)
```

Operations that do not have a unique result shape or element type require an
explicit `result_type`. Common examples include:

- `pto.vmi.vci(...)`
- `pto.vmi.vbrc(...)`
- `pto.vmi.vselr(...)`
- `pto.vmi.vinterpret_cast(...)`
- reduction-style ops whose result lane count changes
- gather and histogram-style ops

Example:

```python
idx = pto.vmi.vci(0, result_type=pto.vmi.vreg(64, pto.i32))
vec = pto.vmi.vbrc(0.0, result_type=pto.vmi.vreg(64, pto.f32))
```

`pto.vmi.vcvt(...)` is a special case. When you provide `to_dtype`, PTODSL can
derive the result lane count from the source and the result element type from
the conversion target:

```python
wide = pto.vmi.vcvt(src_f16, pto.f32)
narrow = pto.vmi.vcvt(src_f32, pto.f16)
```

`pto.vmi.vload(...)` is also a common inferred form: when you provide
`size=...`, PTODSL builds `!pto.vmi.vreg<size x ptr_element_type>` directly
from the source pointer element type.

For `dist_mode="unpack"`, the widened element type is authored through
`to_dtype=...`:

```python
wide = pto.vmi.vload(src_ptr, offset, size=128, dist_mode="unpack", to_dtype=pto.i16)
```

## 14.5 Predicate creation

The public predicate construction APIs in the VMI namespace are:

- `pto.vmi.create_mask(active_lanes, size=...)`
- `pto.vmi.create_group_mask(active_elems_per_group, size=..., num_groups=..., group_size=...)`

These APIs are the public PTODSL entry points for VMI predicate generation.
When authoring `pto.vmi` code, prefer them over implementation-oriented
predicate helper names.

```python
mask = pto.vmi.create_mask(
    active_lanes,
    size=64,
)

group_mask = pto.vmi.create_group_mask(
    active_elems_per_group,
    size=128,
    num_groups=8,
    group_size=16,
)
```

## 14.6 Relationship to the existing PTODSL vector surface

`pto.vmi` does not replace the existing top-level PTODSL vector helper surface.
Both surfaces can exist in the same codebase, but they represent different
authoring styles.

- Use top-level PTODSL vector helpers when you are writing against the existing
  PTODSL vector programming model.
- Use `pto.vmi.*` when you want your code to follow the formal VMI naming and
  logical typing model directly.

In practice, that means:

- `pto.vmi.vreg(...)` is distinct from `pto.vreg_type(...)`
- `pto.vmi.mask(...)` is distinct from `pto.mask_type(...)`
- `pto.vmi.vadd(...)` is a formal VMI namespace call, not a synonym for the
  top-level vector helper of the same spelling

Choose one surface intentionally inside a given helper or kernel region, and
keep the authored style consistent.

## 14.7 Operation family map

The formal `pto.vmi` surface is easiest to read by operation family rather than
by one long flat API list.

| Family | Ops |
|--------|-----|
| Load / Store | `vload`, `vstore` |
| Index generation | `vci` |
| Elementwise compute | `vadd`, `vsub`, `vmul`, `vdiv`, `vmax`, `vmin`, `vabs`, `vneg`, `vrelu`, `vexp`, `vln`, `vsqrt`, `vand`, `vor`, `vxor`, `vnot`, `vshl`, `vshr` |
| Vector-scalar compute | `vadds`, `vmuls`, `vmaxs`, `vmins`, `vshls`, `vshrs` |
| Compare / Select | `vcmp`, `vcmps`, `vsel`, `vselr` |
| Broadcast | `vbrc` |
| Reduce | `vcadd`, `vcmax`, `vcmin` |
| Convert | `vcvt`, `vinterpret_cast` |
| SFU / fused / gather-scatter | `vexpdif`, `vaxpy`, `vlrelu`, `vprelu`, `vmull`, `vmula`, `vhist`, `vgather`, `vgatherb`, `vscatter` |
| Predicate construction | `create_mask`, `create_group_mask` |
| Data rearrange | `vintlv`, `vdintlv` |

All of these wrappers emit `pto.vmi.*` operations and keep the authored code in
the VMI logical type system.

## 14.8 Load and store

### `pto.vmi.vload`

`vload` reads from a UB pointer and materializes one logical VMI vector result.
It is the main entry point for bringing UB data into a `pto.vmi.vreg(...)`
value.

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"vmi.vector_pipeline","symbol":"vmi_vector_pipeline_probe","compile":{}} -->
```python
mask = pto.vmi.create_mask(active_lanes, size=64)
lhs = pto.vmi.vload(src_ptr, offset, size=64)
rhs = pto.vmi.vload(other_ptr, offset, size=64)
out = pto.vmi.vadd(lhs, rhs, mask)
pto.vmi.vstore(out, dst_ptr, offset, mask)
```

Common authored shape:

- `source`: a UB pointer such as `pto.ptr(pto.f32, "ub")`
- `offset`: an `index`-like element offset
- `size`: required for the common inferred form; PTODSL derives the element
  type from the source pointer
- `to_dtype`: required for `dist_mode="unpack"` when result typing is inferred

The wrapper also exposes the formal VMI access-pattern controls:

- `dist_mode=...` for logical access modes such as continuous, deinterleave,
  broadcast, or unpack
- `group=...` with `stride=...` for grouped strided access
- `block_stride=...` and `repeat_stride=...` for block-stride forms

If you author `dist_mode="dintlv"`, the Python wrapper returns two values:

```python
even, odd = pto.vmi.vload(
    src_ptr,
    offset,
    size=64,
    dist_mode="dintlv",
)
```

For widening access modes such as `dist_mode="unpack"`, author the widened
element type explicitly:

```python
wide = pto.vmi.vload(
    src_ptr,
    offset,
    size=128,
    dist_mode="unpack",
    to_dtype=pto.i16,
)
```

The unpack form widens by one adjacent step, so `to_dtype` must be exactly one
width larger than the source pointer element type.

### `pto.vmi.vstore`

`vstore` writes one logical vector result, or a deinterleaved pair, back to UB.

```python
pto.vmi.vstore(vec, dst_ptr, offset, mask)
```

Common authored shape:

- `values`: one VMI vector, or two vectors for `dist_mode="dintlv"`
- `destination`: a UB pointer
- `offset`: an `index`-like element offset
- `mask`: optional in the default continuous form, available in the PTODSL
  surface where the formal op allows it

The grouped and block-stride store forms follow the same authoring pattern as
`vload`: you spell the access mode explicitly in the call rather than relying
on a different helper name.

## 14.9 Index generation and broadcast

### `pto.vmi.vci`

`vci` builds a logical lane-wise ramp from a scalar base. Use it when you need
an index vector for lane addressing, gather/scatter offsets, or dynamic lane
selection.

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"vmi.index_select","symbol":"vmi_index_select_probe","compile":{}} -->
```python
idx = pto.vmi.vci(0, result_type=pto.vmi.vreg(64, pto.i32), order="ASC")
out = pto.vmi.vselr(src, idx, result_type=pto.vmi.vreg(64, pto.f32))
```

`vci` always needs an explicit `result_type`, because the lane count and result
element type cannot be inferred from the scalar base alone.

### `pto.vmi.vbrc`

`vbrc` broadcasts a scalar, or a compact group-shaped input, across a full
logical vector.

```python
bias = pto.vmi.vbrc(0.0, result_type=pto.vmi.vreg(64, pto.f32))
```

Grouped broadcast is also exposed through the same API:

```python
expanded = pto.vmi.vbrc(compact, result_type=pto.vmi.vreg(256, pto.f32), group=8)
```

Like `vci`, `vbrc` requires an explicit `result_type`.

## 14.10 Elementwise compute

The main elementwise families are:

- binary vector-vector: `vadd`, `vsub`, `vmul`, `vdiv`, `vmax`, `vmin`,
  `vand`, `vor`, `vxor`, `vshl`, `vshr`
- unary vector: `vabs`, `vneg`, `vrelu`, `vexp`, `vln`, `vsqrt`, `vnot`
- vector-scalar: `vadds`, `vmuls`, `vmaxs`, `vmins`, `vshls`, `vshrs`

Typical binary usage:

```python
out = pto.vmi.vadd(lhs, rhs, mask)
```

Typical unary usage:

```python
out = pto.vmi.vrelu(src, mask)
```

Typical vector-scalar usage:

```python
out = pto.vmi.vmuls(src, scale, mask)
```

Authoring rules for this family:

- same-shape unary and binary ops infer their result type from the first vector
  operand unless you override it with `result_type=...`
- vector-scalar ops keep the vector lane shape and element type
- the scalar operand is authored as a plain PTODSL scalar value or Python
  literal compatible with the vector element type
- `mask` is optional for unary and binary same-shape ops in the PTODSL surface
- `mask` is required for the vector-scalar forms

These helpers are the natural VMI equivalent of the familiar top-level PTODSL
vector arithmetic surface, but with VMI logical types and VMI naming.

## 14.11 Compare and select

### `pto.vmi.vcmp` and `pto.vmi.vcmps`

These operations compare vector data and produce a logical VMI mask.

```python
pred = pto.vmi.vcmp(lhs, rhs, seed_mask, "ogt")
pred2 = pto.vmi.vcmps(src, 0.0, seed_mask, "oge")
```

Use:

- `vcmp` for vector-vector comparisons
- `vcmps` for vector-scalar comparisons

Both keep the logical lane count of the compared vector and return a
`pto.vmi.mask(...)` value.

### `pto.vmi.vsel`

`vsel` chooses between two same-shaped vectors lane by lane under a VMI mask.

```python
out = pto.vmi.vsel(mask, true_value, false_value)
```

### `pto.vmi.vselr`

`vselr` performs dynamic lane selection from a source vector using an index
vector.

```python
out = pto.vmi.vselr(src, idx, result_type=pto.vmi.vreg(64, pto.f32))
```

`vselr` requires an explicit `result_type`, because PTODSL should not guess the
result shape from the index vector alone.

## 14.12 Reduction

The reduction family is:

- `vcadd`
- `vcmax`
- `vcmin`

These ops reduce a logical vector under a VMI mask and produce a smaller logical
result.

```python
total = pto.vmi.vcadd(src, mask, result_type=pto.vmi.vreg(1, pto.f32), reassoc=True)
peak = pto.vmi.vcmax(src, mask, result_type=pto.vmi.vreg(1, pto.f32))
```

Grouped reduction uses the same operations with `group=...` and a result type
whose lane count matches the number of groups:

```python
group_max = pto.vmi.vcmax(
    src,
    mask,
    result_type=pto.vmi.vreg(8, pto.f32),
    group=8,
)
```

Important authored rules:

- reduction ops require an explicit `result_type`
- `mask` is required
- `vcadd` on floating-point data may require `reassoc=True` to spell the
  intended reassociative reduction contract explicitly

## 14.13 Conversion and reinterpretation

### `pto.vmi.vcvt`

`vcvt` performs numeric conversion between logical VMI vector element types.

```python
wide = pto.vmi.vcvt(src_f16, pto.f32)
narrow = pto.vmi.vcvt(src_f32, pto.f16)
```

For the common form, PTODSL infers the result type from:

- the source lane count
- the target element type given by `to_dtype`

You can still override the result explicitly with `result_type=...` when
needed.

### `pto.vmi.vinterpret_cast`

`vinterpret_cast` reinterprets the same logical vector bits under a different
element type.

```python
as_i32 = pto.vmi.vinterpret_cast(src, result_type=pto.vmi.vreg(64, pto.i32))
```

This is a type reinterpretation API, not a numeric conversion API, so it
requires an explicit `result_type`.

## 14.14 SFU, fused, and indexed memory-style ops

This family contains:

- fused math and activation: `vexpdif`, `vaxpy`, `vlrelu`, `vprelu`
- widening and accumulate forms: `vmull`, `vmula`
- histogram and indexed UB access: `vhist`, `vgather`, `vgatherb`, `vscatter`

Typical usage patterns:

```python
exp_shifted = pto.vmi.vexpdif(x, max_vec, mask)
updated = pto.vmi.vaxpy(x, acc, alpha, mask)
mul64 = pto.vmi.vmull(a32, b32, mask, result_type=pto.vmi.vreg(64, pto.i64))
hist = pto.vmi.vhist(bin_idx, mask, result_type=pto.vmi.vreg(64, pto.ui32))
g = pto.vmi.vgather(src_ptr, offsets, mask, result_type=pto.vmi.vreg(64, pto.f32))
pto.vmi.vscatter(value, dst_ptr, offsets, mask)
```

Authoring guidance:

- `vmull`, `vhist`, `vgather`, and `vgatherb` require explicit `result_type`
- `vscatter` has no result value
- gather/scatter offsets are authored as VMI integer vectors
- these ops are still part of the VMI logical surface, even when they involve
  pointer operands or indexed memory access

This is the family to reach for when a pure same-shape elementwise op is not
enough and the formal spec provides a fused or indexed VMI primitive.

## 14.15 Predicate construction

The public VMI predicate constructors are:

- `create_mask`
- `create_group_mask`

Prefix-style mask creation:

```python
mask = pto.vmi.create_mask(active_lanes, size=64)
```

Grouped mask creation:

```python
group_mask = pto.vmi.create_group_mask(
    active_per_group,
    size=128,
    num_groups=8,
    group_size=16,
)
```

Use:

- `create_mask` when one active prefix controls the whole logical vector
- `create_group_mask` when the vector is logically partitioned into repeated
  groups and each group has the same active prefix width

Both constructors require a compile-time logical lane count through `size=...`.

## 14.16 Data rearrange

The rearrange family is:

- `vintlv`
- `vdintlv`

These ops return two logical vectors and are written naturally as Python tuple
unpacking:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"vmi.predicate_and_rearrange","symbol":"vmi_predicate_and_rearrange_probe","compile":{}} -->
```python
group_mask = pto.vmi.create_group_mask(
    active_per_group,
    size=128,
    num_groups=8,
    group_size=16,
)
total = pto.vmi.vcadd(src, mask, result_type=pto.vmi.vreg(1, pto.f32), reassoc=True)
lo, hi = pto.vmi.vintlv(src, src, mask)
even, odd = pto.vmi.vdintlv(lo, hi, mask)
_ = group_mask
_ = total
_ = even
_ = odd
```

Use:

- `vintlv` to interleave two logical vectors
- `vdintlv` to undo that interleaving and recover separated streams

If you do not pass `result_types=...`, PTODSL infers the two result types from
the input vectors. This makes the family convenient for in-register data
reorganization where the logical lane shape stays the same.

## 14.17 Choosing a family

As a quick rule of thumb:

- start with `vload` / `vstore` when you need to move UB data into or out of a
  VMI logical vector
- use the elementwise and vector-scalar families when the result shape is the
  same as the input shape
- use `vcmp` / `vcmps` when the output should be a logical VMI mask
- use `vcadd` / `vcmax` / `vcmin` when the lane count shrinks
- use `vcvt` for numeric conversion and `vinterpret_cast` for type
  reinterpretation
- use `vci`, `vselr`, `vgather`, `vgatherb`, and `vscatter` when lane indices
  are part of the program logic
- use `vintlv` / `vdintlv` when you are reorganizing values already in logical
  vector form
