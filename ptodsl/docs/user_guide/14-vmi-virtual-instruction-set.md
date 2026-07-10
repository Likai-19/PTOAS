# 14. The Virtual Micro-instruction Set (VMI)

The Virtual Micro-instruction Set (VMI) is a logical SIMD vector instruction
set exposed through the `pto.vmi` namespace. It provides a complete set of
virtual micro-instructions for writing vectorized kernels directly against a
hardware-abstracted instruction set — load, store, compute, compare, reduce,
convert, rearrange, and predicate control.

Use `pto.vmi` when you want to:

- author kernels against a stable, logical vector instruction set
- write SIMD code that mirrors the formal VMI specification directly
- carry explicit logical vector and mask types in your authored code
- bypass the top-level PTODSL vector helpers and work one level closer to the
  hardware abstraction

VMI is not a replacement for the existing top-level vector helpers
(`pto.vadd`, `pto.vlds`, etc.). The two surfaces coexist: the top-level helpers
remain the established PTODSL vector programming surface, while `pto.vmi` is
the explicit, instruction-set-oriented alternative.

## 14.1 VMI logical types

VMI introduces two logical type constructors. They describe a logical vector
register and a logical predicate mask at the PTODSL level — the physical
register mapping is handled by the backend.

### `pto.vmi.vreg(lanes, dtype, *, layout=None)`

**Description**: Creates a logical VMI vector register type descriptor.
`lanes` is the logical lane count (not a physical register count). `dtype` is
a PTODSL element type token such as `pto.f32`, `pto.f16`, `pto.i32`, etc.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `lanes` | `int` | Logical lane count. Must be a multiple of 64. See Constraints below |
| `dtype` | `DType` | Element type token (`pto.f32`, `pto.f16`, `pto.i32`, etc.) |
| `layout` | VMI layout or `None` | Physical register distribution of logical lanes. `None` (default) means contiguous. See the layout note below |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `type_descriptor` | VMI vreg type | Logical VMI vector type descriptor |

**Constraints**:

- `lanes` must be a multiple of 64.
- `lanes · bitwidth(dtype)` determines the physical register count
  `K = ⌈ lanes · bitwidth(dtype) / 2048 ⌉`. Each physical register is
  256 B (2048 bits).
- Common legal combinations:

  | dtype | bitwidth | lanes per physical reg | example `lanes` |
  |-------|----------|------------------------|-----------------|
  | `f32`, `i32`, `ui32`, `si32` | 32 | 64 | 64, 128, 256 |
  | `f16`, `bf16`, `i16`, `ui16`, `si16` | 16 | 128 | 64, 128, 256 |
  | `i8`, `ui8`, `si8`, `fp8_e4m3`, `fp8_e5m2` | 8 | 256 | 64, 128, 256 |

- Compact/partial vectors (`K < 1` in the formula above, e.g. `vreg(64, pto.f16)`
  = 128 B) still occupy one physical register; lanes outside the logical value
  are undefined and must be masked out.

**Example**:

```python
vec_f32 = pto.vmi.vreg(64, pto.f32)    # 1 physical reg (64 × 32b = 256B)
vec_i32 = pto.vmi.vreg(64, pto.i32)    # 1 physical reg
vec_f16 = pto.vmi.vreg(128, pto.f16)   # 1 physical reg (128 × 16b = 256B)
vec_f32_x2 = pto.vmi.vreg(128, pto.f32) # 2 physical regs (128 × 32b = 512B)
```

---

### `pto.vmi.mask(lanes, *, layout=None)`

**Description**: Creates a logical VMI mask type descriptor. The predicate
granularity is always per-lane: one mask bit governs one vector lane.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `lanes` | `int` | Logical lane count. Must match the gated vector's lanes. See Constraints below |
| `layout` | VMI layout or `None` | Physical register distribution. `None` (default) means contiguous. See the layout note below |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `type_descriptor` | VMI mask type | Logical VMI mask type descriptor |

**Constraints**:

- `lanes` must match the lane count of the vector being gated. A `mask(64)`
  gates a `vreg(64, ...)`; a `mask(128)` gates a `vreg(128, ...)`.
- The mask's `layout` should match the gated vector's layout. When the vector
  is deinterleaved, the mask must carry the same deinterleaved layout so that
  per-register predicate lowering stays consistent. In the common case where
  `layout` is omitted on both, they default to contiguous and match naturally.

**Example**:

```python
mask64 = pto.vmi.mask(64)    # gates a vreg(64, ...)
mask128 = pto.vmi.mask(128)  # gates a vreg(128, ...)
```

**About VMI layouts.** A VMI logical vector value may span `K` physical vector
registers (256 B each). The `layout` on a `vreg` or `mask` type describes how
logical lanes are distributed across those physical registers. Two common
layouts are:

| Layout | Description |
|--------|-------------|
| `contiguous` (default) | Stride-1 mapping: lane `i` sits at position `i mod (2048/bitwidth(T))` within physical register `⌊i / (2048/bitwidth(T))⌋`. This is the implicit layout when `layout` is omitted |
| `deinterleaved` | Parity split: EVEN lanes occupy the first `K/2` physical registers, ODD lanes occupy the second `K/2`. This is the natural output layout of a widening `vcvt` (e.g., `f16 → f32`) or a `vload` with `dist_mode="dintlv"` |

Layouts are primarily a lowering concern managed by `pto.as`. The system
propagates layouts automatically through Category A ops (most elementwise
compute) and inserts materialization at Category C boundaries. In day-to-day
authoring you almost never need to spell a layout explicitly — the common case
is to omit `layout` and let the type carry the default contiguous annotation.
You would only set `layout` explicitly when declaring a `result_type` that
must match the deinterleaved output of a prior narrowing/widening step.

VMI types are only used as type annotations and `result_type` arguments. They
are not Python callables that produce values — use `pto.vmi.vload`,
`pto.vmi.vci`, etc. to produce actual VMI vector values, and
`pto.vmi.create_mask` / `pto.vmi.create_group_mask` to produce actual VMI mask
values.

---

## 14.2 Load and store

The load/store family moves data between UB memory and VMI logical vector
registers. These are the primary entry and exit points for VMI vector data.

### `pto.vmi.vload(source, offset, *, size=None, to_dtype=None, result_type=None, dist_mode=None, stride=None, block_stride=None, repeat_stride=None, group=None, pmode=None)`

**Description**: Loads a logical VMI vector from a UB pointer. The element
type is derived from the source pointer; the lane count comes from `size` (or
`result_type`). This is the main entry point for bringing UB data into a
`pto.vmi.vreg(...)` value.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `PtrType` (ub) | UB source pointer |
| `offset` | `IndexLike` | Element offset into the source buffer |
| `size` | `int` | Logical lane count. Required when `result_type` is omitted; PTODSL derives the element type from the source pointer |
| `to_dtype` | `DType` | Target element type. Required when `dist_mode="unpack"` and `result_type` is omitted |
| `result_type` | VMI vreg type | Explicit result type. When provided, `size` and `to_dtype` are not used for type inference |
| `dist_mode` | `str` or `None` | Access pattern: `None` (continuous), `"dintlv"` (deinterleave), `"unpack"` (widen), `"brc"` (broadcast) |
| `stride` | `IndexLike` or `None` | Element stride for strided access patterns |
| `block_stride` | `int` or `None` | 16-bit block stride for block-strided access |
| `repeat_stride` | `int` or `None` | 16-bit repeat stride for block-strided access |
| `group` | `int` or `None` | Group count for grouped access patterns |
| `pmode` | `int` or `None` | Optional pipeline mode |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Loaded logical vector (single return for continuous, unpack, broadcast modes) |
| `(even, odd)` | `(VRegType, VRegType)` | Deinterleaved vector pair (when `dist_mode="dintlv"`) |

**Example** — continuous load:

```python
lhs = pto.vmi.vload(src_ptr, offset, size=64)
rhs = pto.vmi.vload(other_ptr, offset, size=64)
out = pto.vmi.vadd(lhs, rhs, mask)
pto.vmi.vstore(out, dst_ptr, offset, mask)
```

**Example** — deinterleaved load:

```python
even, odd = pto.vmi.vload(
    src_ptr,
    offset,
    size=64,
    dist_mode="dintlv",
)
```

**Example** — unpack (widening) load:

```python
wide = pto.vmi.vload(
    src_ptr,
    offset,
    size=128,
    dist_mode="unpack",
    to_dtype=pto.i16,
)
```

The unpack form widens by exactly one adjacent step: `to_dtype` must be one
width larger than the source pointer element type (e.g., `i8` → `i16`,
`f16` → `f32`).

**Constraints**:
- `result_type` and `size` cannot both be omitted for the default continuous
  form.
- `to_dtype` is only accepted when `dist_mode="unpack"`.
- The `unpack` form widens by exactly one bit-width step.

---

### `pto.vmi.vstore(values, destination, offset, mask=None, *, dist_mode=None, stride=None, block_stride=None, repeat_stride=None, group=None, pmode=None)`

**Description**: Writes one logical VMI vector (or a deinterleaved pair) back
to a UB pointer.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `values` | `VRegType` or `(VRegType, VRegType)` | One VMI vector, or a pair for `dist_mode="dintlv"` |
| `destination` | `PtrType` (ub) | UB destination pointer |
| `offset` | `IndexLike` | Element offset into the destination buffer |
| `mask` | VMI mask or `None` | Optional predicate mask gating which lanes are written |
| `dist_mode` | `str` or `None` | Access pattern: `None` (continuous) or `"dintlv"` (interleave) |
| `stride` | `IndexLike` or `None` | Element stride for strided access |
| `block_stride` | `int` or `None` | 16-bit block stride |
| `repeat_stride` | `int` or `None` | 16-bit repeat stride |
| `group` | `int` or `None` | Group count for grouped access |
| `pmode` | `int` or `None` | Optional pipeline mode |

**Returns**: None (side-effect operation).

**Example**:

```python
pto.vmi.vstore(vec, dst_ptr, offset, mask)
```

Grouped and block-stride store forms follow the same access-mode spelling as
`vload`.

---

## 14.3 Index generation and broadcast

These instructions produce a new logical vector from a scalar seed — either as
a lane-wise ramp or a uniform broadcast.

### `pto.vmi.vci(base, *, result_type, order=None)`

**Description**: Builds a logical lane-wise index ramp starting from a scalar
base value. Use it when you need an index vector for lane addressing,
gather/scatter offsets, or dynamic lane selection.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `base` | `ScalarType` | Scalar starting value for the ramp. Coerced to the result element type |
| `result_type` | VMI vreg type | **Required.** The result type, specifying both lane count and element type |
| `order` | `str` or `None` | Ramp order: `"ASC"` for ascending (default if omitted), or `"DESC"` for descending |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `idx` | `VRegType` | Lane-wise index vector |

**Example**:

```python
idx = pto.vmi.vci(0, result_type=pto.vmi.vreg(64, pto.i32), order="ASC")
out = pto.vmi.vselr(src, idx, result_type=pto.vmi.vreg(64, pto.f32))
```

**Constraints**:
- `result_type` is always required — the lane count and element type cannot be
  inferred from the scalar base alone.
- `base` is automatically coerced to the element type of `result_type`.

---

### `pto.vmi.vbrc(value, *, result_type, group=None)`

**Description**: Broadcasts a scalar value (or a compact group-shaped input)
across all lanes of a logical vector.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `value` | `ScalarType` or `VRegType` | Scalar to broadcast, or a compact VMI vector for grouped broadcast |
| `result_type` | VMI vreg type | **Required.** The result vector type |
| `group` | `int` or `None` | Group count for grouped broadcast. When provided, `value` is treated as a compact group-shaped input and expanded accordingly |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Broadcast vector |

**Example** — scalar broadcast:

```python
bias = pto.vmi.vbrc(0.0, result_type=pto.vmi.vreg(64, pto.f32))
```

**Example** — grouped broadcast:

```python
expanded = pto.vmi.vbrc(compact, result_type=pto.vmi.vreg(256, pto.f32), group=8)
```

**Constraints**:
- `result_type` is always required.
- When `value` is a plain scalar (Python number or PTODSL scalar), it is
  coerced to the element type of `result_type`.

---

## 14.4 Elementwise compute

Elementwise instructions operate lane-by-lane on one or two VMI vector
operands. They form the arithmetic core of VMI SIMD kernels.

### 14.4.1 Binary vector-vector

#### `pto.vmi.vadd(lhs, rhs, mask=None) -> VRegType`
#### `pto.vmi.vsub(lhs, rhs, mask=None) -> VRegType`
#### `pto.vmi.vmul(lhs, rhs, mask=None) -> VRegType`
#### `pto.vmi.vdiv(lhs, rhs, mask=None) -> VRegType`
#### `pto.vmi.vmax(lhs, rhs, mask=None) -> VRegType`
#### `pto.vmi.vmin(lhs, rhs, mask=None) -> VRegType`
#### `pto.vmi.vand(lhs, rhs, mask=None) -> VRegType`
#### `pto.vmi.vor(lhs, rhs, mask=None) -> VRegType`
#### `pto.vmi.vxor(lhs, rhs, mask=None) -> VRegType`
#### `pto.vmi.vshl(lhs, rhs, mask=None) -> VRegType`
#### `pto.vmi.vshr(lhs, rhs, mask=None) -> VRegType`

**Description**: Element-wise binary operation: `result[i] = lhs[i] <op> rhs[i]`
for lanes where `mask[i]` is true (or all lanes when `mask` is omitted).

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `lhs` | `VRegType` | First operand vector |
| `rhs` | `VRegType` | Second operand vector |
| `mask` | VMI mask or `None` | Optional predicate mask gating lane participation |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Result vector (same shape and element type as `lhs`) |

**Example**:

```python
out = pto.vmi.vadd(lhs, rhs, mask)
out = pto.vmi.vmul(scale, data, full_mask)
```

**Constraints**:
- `lhs` and `rhs` must have compatible shapes and element types.
- The result type is inferred from `lhs`. Override with `result_type=...` if
  needed.
- For bitwise ops (`vand`, `vor`, `vxor`, `vshl`, `vshr`), integer element
  types are expected. Floating-point usage is rejected.

---

### 14.4.2 Unary vector

#### `pto.vmi.vabs(source, mask=None) -> VRegType`
#### `pto.vmi.vneg(source, mask=None) -> VRegType`
#### `pto.vmi.vrelu(source, mask=None) -> VRegType`
#### `pto.vmi.vexp(source, mask=None) -> VRegType`
#### `pto.vmi.vln(source, mask=None) -> VRegType`
#### `pto.vmi.vsqrt(source, mask=None) -> VRegType`
#### `pto.vmi.vnot(source, mask=None) -> VRegType`

**Description**: Element-wise unary operation: `result[i] = op(source[i])` for
active lanes. `vrelu` = `max(0, x)`, `vnot` = bitwise NOT (integer types
only).

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `VRegType` | Input vector |
| `mask` | VMI mask or `None` | Optional predicate mask gating lane participation |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Result vector (same shape and element type as `source`) |

**Example**:

```python
activated = pto.vmi.vrelu(src, mask)
inverted = pto.vmi.vnot(int_vec)
```

---

### 14.4.3 Vector-scalar

#### `pto.vmi.vadds(source, scalar, mask) -> VRegType`
#### `pto.vmi.vmuls(source, scalar, mask) -> VRegType`
#### `pto.vmi.vmaxs(source, scalar, mask) -> VRegType`
#### `pto.vmi.vmins(source, scalar, mask) -> VRegType`
#### `pto.vmi.vshls(source, scalar, mask) -> VRegType`
#### `pto.vmi.vshrs(source, scalar, mask) -> VRegType`

**Description**: Element-wise operation with a uniform scalar second operand:
`result[i] = source[i] <op> scalar`. The scalar is broadcast to all active
lanes.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `VRegType` | Input vector |
| `scalar` | `ScalarType` | Scalar operand (Python number or PTODSL scalar). Automatically coerced to the vector element type |
| `mask` | VMI mask | **Required.** Predicate mask gating lane participation |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Result vector (same shape and element type as `source`) |

**Example**:

```python
shifted = pto.vmi.vsubs(scores, row_max, col_mask)
scaled = pto.vmi.vmuls(data, 0.5, full_mask)
```

**Constraints**:
- `mask` is always required for vector-scalar ops — unlike binary and unary
  ops, there is no mask-optional form.
- `scalar` is coerced to match the element type of `source`.

---

## 14.5 Compare and select

Compare instructions produce logical VMI masks from vector data. Select
instructions consume masks to pick between values lane by lane.

### `pto.vmi.vcmp(lhs, rhs, seed, cmp, *, result_type=None) -> MaskType`

**Description**: Element-wise vector-vector comparison producing a VMI mask:
`result[i] = seed[i] ? (lhs[i] <cmp> rhs[i]) : 0`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `lhs` | `VRegType` | First operand vector |
| `rhs` | `VRegType` | Second operand vector |
| `seed` | VMI mask | Seed mask gating which lanes participate |
| `cmp` | `str` | Comparison predicate: `"oeq"`, `"one"`, `"olt"`, `"ole"`, `"ogt"`, `"oge"` |
| `result_type` | VMI mask type or `None` | Optional explicit result mask type; defaults to the type inferred from `seed` |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `pred` | VMI mask | Result predicate mask (same granularity and lane count as `seed`) |

---

### `pto.vmi.vcmps(source, scalar, seed, cmp, *, result_type=None) -> MaskType`

**Description**: Vector-scalar comparison: `result[i] = seed[i] ? (source[i] <cmp> scalar) : 0`.
The scalar is broadcast to all lanes.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `VRegType` | Input vector |
| `scalar` | `ScalarType` | Scalar operand (coerced to the vector element type) |
| `seed` | VMI mask | Seed mask gating lane participation |
| `cmp` | `str` | Comparison predicate |
| `result_type` | VMI mask type or `None` | Optional explicit result mask type |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `pred` | VMI mask | Result predicate mask |

**Example**:

```python
pred = pto.vmi.vcmp(lhs, rhs, seed_mask, "ogt")
pred2 = pto.vmi.vcmps(src, 0.0, seed_mask, "oge")
```

---

### `pto.vmi.vsel(mask, true_value, false_value, *, result_type=None) -> VRegType`

**Description**: Per-lane ternary select: `result[i] = mask[i] ? true_value[i] : false_value[i]`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | VMI mask | Selection predicate |
| `true_value` | `VRegType` | Value taken when mask is true |
| `false_value` | `VRegType` | Value taken when mask is false |
| `result_type` | VMI vreg type or `None` | Optional explicit result type; defaults to the type of `true_value` |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Selected vector |

---

### `pto.vmi.vselr(source, index, *, result_type) -> VRegType`

**Description**: Dynamic per-lane selection from a source vector using an
index vector: `result[i] = source[index[i]]`. This is a gather-style select
within a single vector register.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `VRegType` | Source vector to select from |
| `index` | `VRegType` | Integer index vector (per-lane source lane indices) |
| `result_type` | VMI vreg type | **Required.** The result vector type |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Selected vector |

**Example**:

```python
out = pto.vmi.vselr(src, idx, result_type=pto.vmi.vreg(64, pto.f32))
```

**Constraints**:
- `result_type` is always required — the result shape cannot be inferred from
  the index vector alone.
- `index` must be an integer-typed VMI vector.

---

## 14.6 Reduction

Reduction instructions collapse a logical vector along its lane dimension,
producing a smaller logical result.

### `pto.vmi.vcadd(source, mask, *, result_type, group=None, reassoc=None) -> VRegType`
### `pto.vmi.vcmax(source, mask, *, result_type, group=None) -> VRegType`
### `pto.vmi.vcmin(source, mask, *, result_type, group=None) -> VRegType`

**Description**: Full-vector or grouped reduction. `vcadd` computes the sum,
`vcmax` / `vcmin` compute the maximum / minimum with their lane index. When
`group` is omitted (or `None`), the reduction is across the full vector and
the result lane count is 1. When `group` is provided, the vector is
partitioned into that many equal-sized groups and a separate reduction is
performed per group.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `VRegType` | Input vector |
| `mask` | VMI mask | **Required.** Predicate mask gating lane participation |
| `result_type` | VMI vreg type | **Required.** The result vector type. For full-vector reduction, typically `pto.vmi.vreg(1, dtype)`. For grouped reduction, `pto.vmi.vreg(num_groups, dtype)` |
| `group` | `int` or `None` | Number of groups for per-group reduction. `None` means full-vector reduction |
| `reassoc` | `bool` or `None` | For `vcadd` on floating-point data only: set `True` to declare reassociative reduction semantics |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Reduced vector |

**Example** — full-vector reduction:

```python
total = pto.vmi.vcadd(src, mask, result_type=pto.vmi.vreg(1, pto.f32), reassoc=True)
peak = pto.vmi.vcmax(src, mask, result_type=pto.vmi.vreg(1, pto.f32))
```

**Example** — grouped reduction:

```python
group_max = pto.vmi.vcmax(
    src,
    mask,
    result_type=pto.vmi.vreg(8, pto.f32),
    group=8,
)
```

**Constraints**:
- `mask` is always required.
- `result_type` is always required.
- `reassoc` is only meaningful for `vcadd` on floating-point data. It
  explicitly declares the intended reassociative sum contract.

---

## 14.7 Conversion and reinterpretation

### `pto.vmi.vcvt(source, to_dtype=None, *, result_type=None, rounding=None, saturate=None, sign=None) -> VRegType`

**Description**: Numeric type conversion between VMI vector element types.
Converts the element type of `source` to the target element type. PTODSL
infers the result lane count from the source when `result_type` is omitted.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `VRegType` | Input vector (source element type) |
| `to_dtype` | `DType` | Target element type. Required when `result_type` is omitted. PTODSL derives the result vector type from the source lane count and this dtype |
| `result_type` | VMI vreg type or `None` | Explicit result type. When provided, `to_dtype` is ignored for type inference |
| `rounding` | rounding mode or `None` | Optional rounding mode token |
| `saturate` | saturate mode or `None` | Optional saturation mode token |
| `sign` | sign mode or `None` | Optional sign-control token |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Converted vector |

**Example**:

```python
wide = pto.vmi.vcvt(src_f16, pto.f32)
narrow = pto.vmi.vcvt(src_f32, pto.f16)
```

**Constraints**:
- The masked form of `vcvt` is not currently supported on this surface.
- The source and target dtype pair must be legal for the target backend.

---

### `pto.vmi.vinterpret_cast(source, *, result_type) -> VRegType`

**Description**: Bitwise reinterpretation of a VMI vector under a different
element type. The logical bit pattern is unchanged; only the element type
annotation changes. This is a reinterpretation, not a numeric conversion.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `VRegType` | Input vector |
| `result_type` | VMI vreg type | **Required.** The result type with the target element type and matching total bit width |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Reinterpreted vector |

**Example**:

```python
as_i32 = pto.vmi.vinterpret_cast(src, result_type=pto.vmi.vreg(64, pto.i32))
```

**Constraints**:
- `result_type` is always required — PTODSL must not guess a reinterpretation
  target type.
- The source and result types must have the same total bit width.

---

## 14.8 SFU, fused, and indexed memory instructions

This family covers special-function-unit ops, fused multiply-accumulate forms,
and indexed memory access (gather, scatter, histogram). They go beyond simple
elementwise arithmetic.

### `pto.vmi.vexpdif(x, max_value, mask, *, result_type=None) -> VRegType`

**Description**: Computes `exp(x[i] - max_value[i])` for active lanes — the
stable softmax numerator.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `VRegType` | Input vector |
| `max_value` | `VRegType` | Maximum value vector to subtract before exponentiation |
| `mask` | VMI mask | **Required.** Predicate mask gating lane participation |
| `result_type` | VMI vreg type or `None` | Optional explicit result type; defaults to the type of `max_value` |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | `exp(x - max_value)` |

---

### `pto.vmi.vaxpy(x, acc, alpha, mask, *, result_type=None) -> VRegType`

**Description**: Fused multiply-add: `result[i] = alpha * x[i] + acc[i]`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `VRegType` | Input vector |
| `acc` | `VRegType` | Accumulator vector |
| `alpha` | `ScalarType` | Scalar multiplier (coerced to the element type of `x`) |
| `mask` | VMI mask | **Required.** Predicate mask |
| `result_type` | VMI vreg type or `None` | Optional explicit result type |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | `alpha * x + acc` |

---

### `pto.vmi.vlrelu(x, slope, mask, *, result_type=None) -> VRegType`

**Description**: Leaky ReLU: `result[i] = x[i] >= 0 ? x[i] : slope * x[i]`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `VRegType` | Input vector |
| `slope` | `ScalarType` | Negative-slope multiplier (coerced to the element type of `x`) |
| `mask` | VMI mask | **Required.** Predicate mask |
| `result_type` | VMI vreg type or `None` | Optional explicit result type |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Leaky ReLU result |

---

### `pto.vmi.vprelu(x, alpha, mask, *, result_type=None) -> VRegType`

**Description**: Parametric ReLU with a per-lane vector alpha:
`result[i] = x[i] >= 0 ? x[i] : alpha[i] * x[i]`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `VRegType` | Input vector |
| `alpha` | `VRegType` | Per-lane slope vector |
| `mask` | VMI mask | **Required.** Predicate mask |
| `result_type` | VMI vreg type or `None` | Optional explicit result type |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Parametric ReLU result |

---

### `pto.vmi.vmull(a, b, mask, *, result_type) -> VRegType`

**Description**: Widening multiply: computes the product of two narrower
vectors and produces a result with the widened element type. The result
element type must be exactly one width larger than the operand element type
(e.g., `i32 * i32` → `i64`, `f16 * f16` → `f32`).

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `VRegType` | First operand vector (narrow type) |
| `b` | `VRegType` | Second operand vector (narrow type) |
| `mask` | VMI mask | **Required.** Predicate mask |
| `result_type` | VMI vreg type | **Required.** The widened result type |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Widened product |

**Example**:

```python
mul64 = pto.vmi.vmull(a32, b32, mask, result_type=pto.vmi.vreg(64, pto.i64))
```

---

### `pto.vmi.vmula(acc, lhs, rhs, mask, *, result_type=None) -> VRegType`

**Description**: Widening multiply-accumulate: `result = acc + (lhs * rhs)`,
where the product is computed at the widened precision of `acc`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `acc` | `VRegType` | Accumulator vector (wider type) |
| `lhs` | `VRegType` | First operand (narrow type) |
| `rhs` | `VRegType` | Second operand (narrow type) |
| `mask` | VMI mask | **Required.** Predicate mask |
| `result_type` | VMI vreg type or `None` | Optional explicit result type; defaults to the type of `acc` |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Accumulated product |

---

### `pto.vmi.vhist(bin_idx, mask, *, result_type) -> VRegType`

**Description**: Histogram bin accumulation. Treats `bin_idx` as per-lane bin
indices and accumulates one count per bin into the result vector.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `bin_idx` | `VRegType` | Per-lane bin indices (integer vector) |
| `mask` | VMI mask | **Required.** Predicate mask |
| `result_type` | VMI vreg type | **Required.** Result vector type (lane count = number of bins) |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Histogram counts per bin |

---

### `pto.vmi.vgather(source, offsets, mask, *, result_type) -> VRegType`

**Description**: Indexed gather from a UB pointer using per-lane element
offsets. Only masked-on lanes participate; masked-off lanes produce an
unspecified value.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `PtrType` (ub) | UB source pointer |
| `offsets` | `VRegType` | Per-lane element offsets (integer VMI vector) |
| `mask` | VMI mask | **Required.** Predicate mask gating lane participation |
| `result_type` | VMI vreg type | **Required.** Result vector type (lane count and element type) |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Gathered vector |

---

### `pto.vmi.vgatherb(source, offsets, mask, *, result_type) -> VRegType`

**Description**: Block gather from a UB pointer. Each participating lane
gathers one 32-byte block using byte-level offsets.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `PtrType` (ub) | UB source pointer |
| `offsets` | `VRegType` | Per-lane byte offsets (integer VMI vector) |
| `mask` | VMI mask | **Required.** Predicate mask |
| `result_type` | VMI vreg type | **Required.** Result vector type |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Block-gathered vector |

---

### `pto.vmi.vscatter(value, destination, offsets, mask) -> None`

**Description**: Indexed scatter to a UB pointer. Writes vector lanes to
irregular memory locations using per-lane element offsets.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `value` | `VRegType` | Source vector to scatter |
| `destination` | `PtrType` (ub) | UB destination pointer |
| `offsets` | `VRegType` | Per-lane element offsets (integer VMI vector) |
| `mask` | VMI mask | **Required.** Predicate mask gating lane participation |

**Returns**: None (side-effect operation).

**Example**:

```python
g = pto.vmi.vgather(src_ptr, offsets, mask, result_type=pto.vmi.vreg(64, pto.f32))
pto.vmi.vscatter(value, dst_ptr, offsets, mask)
```

---

## 14.9 Predicate construction

VMI provides two public entry points for creating predicate masks. These are
the canonical ways to author masks in `pto.vmi` code — prefer them over
lower-level predicate manipulation functions when working in the VMI surface.

### `pto.vmi.create_mask(active_lanes, *, size=None, result_type=None) -> MaskType`

**Description**: Creates a prefix-style VMI mask where the first
`active_lanes` lanes are active and all remaining lanes are inactive. This is
the primary mask constructor for tail handling and partial-vector scenarios.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `active_lanes` | `IndexLike` | Number of active lanes in the prefix |
| `size` | `int` | Total logical lane count. Required when `result_type` is omitted |
| `result_type` | VMI mask type or `None` | Explicit result mask type. When provided, `size` is not used for type inference |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | VMI mask | Prefix mask with `active_lanes` active lanes |

**Example**:

```python
full_mask = pto.vmi.create_mask(64, size=64)
tail_mask = pto.vmi.create_mask(remained, size=64)
```

---

### `pto.vmi.create_group_mask(active_elems_per_group, *, size=None, result_type=None, num_groups, group_size) -> MaskType`

**Description**: Creates a grouped prefix mask. The logical vector is
partitioned into `num_groups` equal-sized groups of `group_size` lanes each.
Within every group, the first `active_elems_per_group` lanes are active.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `active_elems_per_group` | `IndexLike` | Number of active lanes in each group |
| `size` | `int` | Total logical lane count. Required when `result_type` is omitted |
| `result_type` | VMI mask type or `None` | Explicit result mask type |
| `num_groups` | `int` | Number of groups to partition the vector into |
| `group_size` | `int` | Number of lanes per group |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | VMI mask | Grouped prefix mask |

**Constraints**:
- `size` must equal `num_groups * group_size`.
- `active_elems_per_group` must be ≤ `group_size`.

**Example**:

```python
group_mask = pto.vmi.create_group_mask(
    active_per_group,
    size=128,
    num_groups=8,
    group_size=16,
)
```

Use `create_mask` when one active prefix controls the whole logical vector.
Use `create_group_mask` when the vector is logically partitioned into repeated
groups and each group needs the same prefix pattern.

---

## 14.10 Data rearrangement

Rearrangement instructions reorganize data between VMI vector registers
without touching memory. They are used to switch between interleaved and
deinterleaved data layouts.

### `pto.vmi.vintlv(lhs, rhs, mask, *, result_types=None) -> (VRegType, VRegType)`

**Description**: Interleave two logical vectors lane-by-lane and return the
result as a pair: `low` contains the interleaved lower half, `high` contains
the upper half.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `lhs` | `VRegType` | First source vector |
| `rhs` | `VRegType` | Second source vector |
| `mask` | VMI mask | **Required.** Predicate mask gating lane participation |
| `result_types` | `(VMI vreg type, VMI vreg type)` or `None` | Optional explicit pair of result types; defaults to the types of `lhs` and `rhs` |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `low` | `VRegType` | Interleaved lower half |
| `high` | `VRegType` | Interleaved upper half |

---

### `pto.vmi.vdintlv(lhs, rhs, mask, *, result_types=None) -> (VRegType, VRegType)`

**Description**: Deinterleave a previously interleaved vector pair. This is
the inverse of `vintlv`: it separates even-positioned and odd-positioned lanes
of the logical input stream into two output vectors.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `lhs` | `VRegType` | Lower half of the interleaved input |
| `rhs` | `VRegType` | Upper half of the interleaved input |
| `mask` | VMI mask | **Required.** Predicate mask gating lane participation |
| `result_types` | `(VMI vreg type, VMI vreg type)` or `None` | Optional explicit pair of result types; defaults to the types of `lhs` and `rhs` |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `even` | `VRegType` | Lanes from even interleaved positions |
| `odd` | `VRegType` | Lanes from odd interleaved positions |

**Example**:

```python
lo, hi = pto.vmi.vintlv(src, src, mask)
even, odd = pto.vmi.vdintlv(lo, hi, mask)
```

**Constraints**:
- `lhs` and `rhs` must have the same type.
- The two returned vectors form one logical interleaved pair. Preserve their
  order when passing them to subsequent ops.

---

## 14.11 Result typing rules

VMI infers result types when the output shape is unambiguous from the inputs.
When inference is not possible, the instruction requires an explicit
`result_type`.

**Ops that infer their result type automatically**:

- Same-shape elementwise binary and unary ops: inferred from the first vector
  operand.
- Vector-scalar ops: inferred from the source vector.
- `vcmp` / `vcmps`: inferred from the `seed` mask.
- `vsel`: inferred from `true_value`.
- `vcvt` (when `to_dtype` or `result_type` is provided): inferred from the
  source lane count and target element type.
- `vload` (when `size` is provided): inferred from the source pointer element
  type and `size`.
- `vstore`: no result — side-effect only.
- `vscatter`: no result — side-effect only.
- `vintlv` / `vdintlv`: inferred from the input vector types.

**Ops that always require explicit `result_type`**:

- `vci` — lane count and element type cannot be inferred from the scalar base.
- `vbrc` — same reason.
- `vselr` — result shape cannot be inferred from the index vector alone.
- `vcadd`, `vcmax`, `vcmin` — reduction changes the lane count.
- `vinterpret_cast` — reinterpretation target type must be explicit.
- `vmull` — widening product requires explicit widened result type.
- `vhist` — bin count must be explicit.
- `vgather`, `vgatherb` — result shape is independent of the source pointer.

**Ops that infer when given the right hint**:

- `vload` with `dist_mode="unpack"` requires `to_dtype` (to derive the widened
  element type) when `result_type` is omitted.
- `vcvt` requires `to_dtype` when `result_type` is omitted.

When in doubt, provide `result_type=...` explicitly. The frontend will reject
ambiguous forms with a clear error rather than guessing.

---

## 14.12 Relationship to the top-level vector surface

`pto.vmi` and the top-level PTODSL vector helpers (`pto.vadd`, `pto.vlds`,
`pto.vcvt`, etc.) are two distinct authoring surfaces that coexist in PTODSL.

| Aspect | Top-level helpers | `pto.vmi` |
|--------|-------------------|-----------|
| Type system | `VRegType` / `MaskType` | `pto.vmi.vreg(...)` / `pto.vmi.mask(...)` |
| Naming | `pto.vadd`, `pto.vlds` | `pto.vmi.vadd`, `pto.vmi.vload` |
| Style | PTODSL vector programming model | Formal VMI instruction set |
| Predicate creation | `pto.make_mask`, `pto.pset_b32` | `pto.vmi.create_mask`, `pto.vmi.create_group_mask` |
| Return model | Varies by op | Consistent: returns new value or `(values, ...)` tuple |

Key differences in practice:

- `pto.vmi.vreg(...)` is distinct from `pto.vreg_type(...)`.
- `pto.vmi.mask(...)` is distinct from `pto.mask_type(...)`.
- `pto.vmi.vadd(...)` is a formal VMI call, not a synonym for `pto.vadd(...)`.
- VMI operations return values rather than writing to destination buffers.

Choose one surface intentionally inside a given sub-kernel or helper region,
and keep the authored style consistent. Mixing both surfaces in the same
region is possible but makes the IR intent harder to follow.

---

## 14.13 Full example: elementwise vector pipeline

The following example shows a complete VMI pipeline: load, compute under mask,
and store back.

```python
from ptodsl import pto

@pto.jit(
    name="vmi_elementwise",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
)
def vmi_elementwise(
    src_ptr: pto.ptr(pto.f32, "ub"),
    dst_ptr: pto.ptr(pto.f32, "ub"),
    count: pto.i32,
    scale: pto.f32,
):
    full_mask = pto.vmi.create_mask(64, size=64)

    lhs = pto.vmi.vload(src_ptr, 0, size=64)
    rhs = pto.vmi.vload(src_ptr, 64, size=64)

    summed = pto.vmi.vadd(lhs, rhs, full_mask)
    scaled = pto.vmi.vmuls(summed, scale, full_mask)
    activated = pto.vmi.vrelu(scaled, full_mask)

    pto.vmi.vstore(activated, dst_ptr, 0, full_mask)
```

---

## 14.14 VMI instruction quick reference

| Category | Instructions |
|----------|-------------|
| Types | `vreg`, `mask` |
| Load / Store | `vload`, `vstore` |
| Index / Broadcast | `vci`, `vbrc` |
| Binary vector-vector | `vadd`, `vsub`, `vmul`, `vdiv`, `vmax`, `vmin`, `vand`, `vor`, `vxor`, `vshl`, `vshr` |
| Unary vector | `vabs`, `vneg`, `vrelu`, `vexp`, `vln`, `vsqrt`, `vnot` |
| Vector-scalar | `vadds`, `vmuls`, `vmaxs`, `vmins`, `vshls`, `vshrs` |
| Compare / Select | `vcmp`, `vcmps`, `vsel`, `vselr` |
| Reduction | `vcadd`, `vcmax`, `vcmin` |
| Conversion | `vcvt`, `vinterpret_cast` |
| SFU / Fused | `vexpdif`, `vaxpy`, `vlrelu`, `vprelu`, `vmull`, `vmula` |
| Indexed memory | `vhist`, `vgather`, `vgatherb`, `vscatter` |
| Predicate construction | `create_mask`, `create_group_mask` |
| Data rearrangement | `vintlv`, `vdintlv` |

All instructions listed above are members of the `pto.vmi` namespace.
