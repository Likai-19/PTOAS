# 5. Vector Arithmetic and Activation Operations

> **Category:** Base tile-local VEC arithmetic
> **Pipeline:** PIPE_V

This chapter documents the TileLib arithmetic families that keep the same output tile shape as their source tiles. These instructions operate on `!pto.tile_buf` values in `loc=vec` and cover tile-tile arithmetic, tile-scalar arithmetic, unary math, and activation ops.

Reduction, partial, bitwise, conversion, broadcast / expansion, selection, and fill / padding families are documented in Chapters 6-12.

---

## 5.1 Binary Tile-Tile Arithmetic

Tile-tile arithmetic families:

| Op | Semantics |
|----|-----------|
| `pto.tadd` | `dst[i, j] = src0[i, j] + src1[i, j]` |
| `pto.tsub` | `dst[i, j] = src0[i, j] - src1[i, j]` |
| `pto.tmul` | `dst[i, j] = src0[i, j] * src1[i, j]` |
| `pto.tdiv` | `dst[i, j] = src0[i, j] / src1[i, j]` |
| `pto.tmax` | `dst[i, j] = max(src0[i, j], src1[i, j])` |
| `pto.tmin` | `dst[i, j] = min(src0[i, j], src1[i, j])` |

### Common Syntax

```mlir
pto.<op> ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer. |
| `src1` | `pto.tile_buf` | Second source tile buffer. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Constraints:**

- `src0`, `src1`, and `dst` must be shape-compatible tile buffers on `loc=vec`.
- The valid region must match across all three tiles.
- Element type legality is target-defined; ops specialize over the tile dtype selected at expansion time.
- `pto.tdiv` uses element-wise division; **undefined behavior** on divide-by-zero.
- `pto.tdiv` additionally accepts `precisionType = #pto<div_precision default|high_precision>`.
  Omitted means `default`.
  `high_precision` is currently legal only when the tile element type is `f16` or `f32`.

**Example:**

```mlir
pto.tadd ins(%a, %b : !pto.tile_buf<vec, 16x16xf16>, !pto.tile_buf<vec, 16x16xf16>)
         outs(%c : !pto.tile_buf<vec, 16x16xf16>)
```

---

## 5.2 Tile-Scalar Arithmetic

Tile-scalar families:

| Op | Supported operand form(s) | Semantics |
|----|---------------------------|-----------|
| `pto.tadds` | `tile, scalar` | `dst[i, j] = src[i, j] + scalar` |
| `pto.tsubs` | `tile, scalar` | `dst[i, j] = src[i, j] - scalar` |
| `pto.tmuls` | `tile, scalar` | `dst[i, j] = src[i, j] * scalar` |
| `pto.tdivs` | `tile, scalar` and `scalar, tile` | `dst = src / scalar` or `dst = scalar / src` |
| `pto.tmaxs` | `tile, scalar` | `dst[i, j] = max(src[i, j], scalar)` |
| `pto.tmins` | `tile, scalar` | `dst[i, j] = min(src[i, j], scalar)` |

### Common Syntax

For `pto.tadds`, `pto.tsubs`, `pto.tmuls`, `pto.tmaxs`, and `pto.tmins`:

```mlir
pto.<op> ins(%src, %scalar : !pto.tile_buf<...>, <scalar_type>)
          outs(%dst : !pto.tile_buf<...>)
```

For `pto.tdivs`:

```mlir
pto.tdivs ins(%src, %scalar : !pto.tile_buf<...>, <scalar_type>)
          outs(%dst : !pto.tile_buf<...>)
          {precisionType = #pto<div_precision high_precision>}

pto.tdivs ins(%scalar, %src : <scalar_type>, !pto.tile_buf<...>)
          outs(%dst : !pto.tile_buf<...>)
          {precisionType = #pto<div_precision high_precision>}
```

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer. |
| `scalar` | signless integer / floating-point scalar | Scalar broadcast across the tile. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Constraints:**

- `src` and `dst` must be shape-compatible `loc=vec` tile buffers.
- The scalar element type must be compatible with the tile element type.
- `pto.tdivs` is the only scalar family with two public operand orders. **Undefined behavior** on divide-by-zero (either `scalar==0` or any `src[i,j]==0` in the `scalar/src` form).
- `pto.tdivs` additionally accepts `precisionType = #pto<div_precision default|high_precision>`.
  Omitted means `default`.
  `high_precision` is currently legal only when the tile element type is `f16` or `f32`.

**Example:**

```mlir
pto.tadds ins(%a, %s : !pto.tile_buf<vec, 32x32xf32>, f32)
          outs(%c : !pto.tile_buf<vec, 32x32xf32>)
```

```mlir
pto.tdivs ins(%s, %a : f32, !pto.tile_buf<vec, 32x32xf32>)
          outs(%c : !pto.tile_buf<vec, 32x32xf32>)
```

---

## 5.3 Unary Math

All ops below share the common form:

```mlir
pto.<op> ins(%src : !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

| Op | Semantics |
|----|-----------|
| `pto.tabs` | `dst = abs(src)` |
| `pto.tneg` | `dst = -src` |
| `pto.texp` | `dst = exp(src)` |
| `pto.tlog` | `dst = ln(src)` |
| `pto.tsqrt` | `dst = sqrt(src)` |
| `pto.trsqrt` | `dst = 1 / sqrt(src)` |
| `pto.trecip` | `dst = 1 / src` |

**Constraints:**

- `src` and `dst` must have the same valid region.
- These ops are numeric Tile Instruction ops on `loc=vec`.
- On unsigned element types `pto.tneg` is the negation modulo the element
  width, which is the only meaning available: the A5 `pto.vneg` element types
  do not include unsigned, so the template multiplies by the all-ones scalar
  instead, matching how `vneg` itself maps onto `RV_VMULS`.
- **Undefined behavior** on out-of-domain inputs: `tlog(<=0)`, `tsqrt(<0)`, `trsqrt(<=0)`, `trecip(0)`.
- Selected unary math ops additionally accept op-specific `precisionType` attrs:
  `#pto<exp_precision default|high_precision>` for `pto.texp`,
  `#pto<log_precision default|high_precision>` for `pto.tlog`,
  `#pto<sqrt_precision default|high_precision>` for `pto.tsqrt`,
  `#pto<rsqrt_precision default|high_precision>` for `pto.trsqrt`, and
  `#pto<recip_precision default|high_precision>` for `pto.trecip`.
  Omitted means `default`.
  For these ops, `high_precision` is currently legal on their supported floating-point element types.

**Precision-Type Form:**

```mlir
pto.<op> ins(%src : !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
         {precisionType = #pto<exp_precision high_precision>}
```

**Example:**

```mlir
pto.tabs ins(%a : !pto.tile_buf<vec, 16x16xf16>)
         outs(%c : !pto.tile_buf<vec, 16x16xf16>)
```

---

## 5.4 Activation Operations

Activation family:

| Op | Semantics |
|----|-----------|
| `pto.trelu` | `dst[i, j] = max(0, src[i, j])` |
| `pto.tlrelu` | `dst[i, j] = src[i, j] > 0 ? src[i, j] : slope * src[i, j]` |

### Common Forms

ReLU:

```mlir
pto.trelu ins(%src : !pto.tile_buf<...>)
          outs(%dst : !pto.tile_buf<...>)
```

Leaky ReLU:

```mlir
pto.tlrelu ins(%src, %slope : !pto.tile_buf<...>, f32)
           outs(%dst : !pto.tile_buf<...>)
```

**Constraints:**

- `src` and `dst` must have the same valid region.
- `pto.trelu` supports `f16`, `f32`, and `i32`.
- `pto.tlrelu` supports `f16` and `f32`, with the slope passed as an `f32` scalar operand.
- Both ops execute on `loc=vec` tiles via the vector pipeline.

**Example:**

```mlir
pto.trelu ins(%src : !pto.tile_buf<vec, 16x64xf32>)
          outs(%dst : !pto.tile_buf<vec, 16x64xf32>)
```
