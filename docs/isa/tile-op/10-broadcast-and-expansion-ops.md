# 10. Broadcast and Expansion Operations

> **Category:** Tile-local VEC broadcast and expansion compute
> **Pipeline:** PIPE_V

This chapter documents the TileLib broadcast, row-expansion, and column-expansion families. These ops populate destination tiles by broadcasting one logical scalar across a larger region — either from a standalone scalar operand, one source value per destination row, or one source value per destination column.

---

## 10.1 Scalar Broadcast: `pto.texpands`

- **syntax:**
```mlir
pto.texpands ins(%scalar : <scalar_type>)
             outs(%dst : !pto.tile_buf<...>)
```
- **semantics:** `dst[i, j] = scalar` for every element inside `dst`'s valid region.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `scalar` | integer / floating-point scalar | Scalar value broadcast into the destination tile. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Constraints:**

- The TileLib template is VEC-oriented and fills `dst.valid_shape`.
- The scalar type must be compatible with `dst.dtype`.
- On A5, `ui8`, `ui16`, and `ui32` scalars are accepted with matching destination element types.

**Example:**

```mlir
pto.texpands ins(%scalar : f32)
             outs(%dst : !pto.tile_buf<vec, 16x64xf32>)
```

---

## 10.2 Row-Wise Broadcast: `pto.trowexpand`

- **syntax:**
```mlir
pto.trowexpand ins(%src : !pto.tile_buf<...>)
               outs(%dst : !pto.tile_buf<...>)
```
- **semantics:** `dst[row, col] = src[row, 0]`.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.tile_buf` | Source tile carrying one logical scalar per destination row. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Constraints:**

- `src` and `dst` must have the same number of valid rows.
- `src` must encode exactly one logical source value per destination row.
- Templates target row-major VEC layouts.
- On A5, matching `ui8`, `ui16`, and `ui32` source and destination types are supported.

**Example:**

```mlir
pto.trowexpand ins(%src : !pto.tile_buf<vec, 16x1xf32>)
               outs(%dst : !pto.tile_buf<vec, 16x16xf32>)
```

---

## 10.3 Row-Wise Broadcast Arithmetic and Transform Families

The row-expansion family combines a full-sized tile with a row-broadcast operand. Either
input may be the expanded one: the full-sized operand is the one whose `valid_shape` matches
`%dst`, with `%src0` taking precedence when both match (as in `pto-isa`'s `src0eqdst`).
As in `pto-isa` (`TROWEXPANDADD`), the broadcast form is selected by the **expanded**
operand's layout:

- **Mode 1** — `%src1` is **ColMajor with exactly one column**: one scalar per row, splatted
  across the entire row.
- **Mode 2** — `%src1` is **RowMajor with `32 / sizeof(dtype)` columns**: that 32-byte block
  **repeats** across the row, once per `32 / sizeof(dtype)` elements.

Let `B = 32 / sizeof(dtype)` (8 for `f32`, 16 for `f16`/`i16`, 32 for `i8`).

| Op | Mode 1 (ColMajor, 1 column) | Mode 2 (RowMajor, `B` columns) |
|----|------------------------------|--------------------------------|
| `pto.trowexpandadd` | `dst[row, col] = src0[row, col] + src1[row, 0]` | `dst[row, col] = src0[row, col] + src1[row, col % B]` |
| `pto.trowexpandsub` | `dst[row, col] = src0[row, col] - src1[row, 0]` | `dst[row, col] = src0[row, col] - src1[row, col % B]` |
| `pto.trowexpandmul` | `dst[row, col] = src0[row, col] * src1[row, 0]` | `dst[row, col] = src0[row, col] * src1[row, col % B]` |
| `pto.trowexpanddiv` | `dst[row, col] = src0[row, col] / src1[row, 0]` | `dst[row, col] = src0[row, col] / src1[row, col % B]` |
| `pto.trowexpandmax` | `dst[row, col] = max(src0[row, col], src1[row, 0])` | `dst[row, col] = max(src0[row, col], src1[row, col % B])` |
| `pto.trowexpandmin` | `dst[row, col] = min(src0[row, col], src1[row, 0])` | `dst[row, col] = min(src0[row, col], src1[row, col % B])` |
| `pto.trowexpandexpdif` | `dst[row, col] = exp(src0[row, col] - src1[row, 0])` | `dst[row, col] = exp(src0[row, col] - src1[row, col % B])` |

> Write the row-expanded operand as `E` and the full-sized one as `F`; the table above uses
> `E = %src1`, `F = %src0` and reads `dst[row, col] = F[row, col] OP E[...]`. When `%src0` is
> the expanded one, substitute `E = %src0`, `F = %src1`. The operation always keeps the
> authored order `%src0 OP %src1`.
>
> In Mode 2 the block is indexed by `col % B`, so a `RowMajor` expanded operand with `B`
> columns contributes all `B` values to the row.

### Common Syntax

```mlir
pto.<op> ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src0` | `pto.tile_buf` | Main source tile. |
| `src1` | `pto.tile_buf` | Either the full-sized operand matching `dst`, or the row-broadcast operand: one scalar per destination row (Mode 1) or one 32-byte block per destination row (Mode 2). |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Constraints:**

- The full-sized operand — `%src0`, or `%src1` when only `%src1`'s valid shape matches `dst` —
  must be shape/valid-region compatible with `dst`.
- The expanded operand must provide one row-broadcast operand per destination row: bound
  `ColMajor`, exactly one column (`valid_shape[1] == 1`); bound `RowMajor`, exactly
  `32 / sizeof(dtype)` columns.
- `pto.trowexpandadd` also accepts an optional scratch `%tmp` operand for
  cross-architecture API compatibility.
- Templates target row-major VEC layouts.
- On A5, add, sub, mul, max, and min accept matching `bf16`, `ui8`, `ui16`, and `ui32` tiles.
- On A5, `pto.trowexpanddiv` also accepts matching `bf16`, `ui8`, `ui16`, and `ui32` tiles;
  unsigned integer division truncates toward zero and requires a nonzero divisor.
- `pto.trowexpandexpdif` is floating-point-only (`f16` or `f32`).
- `pto.trowexpanddiv` additionally accepts `precisionType = #pto<div_precision default|high_precision>`.
  Omitted means `default`.
  `high_precision` requires floating-point element type and an explicit `tmp` operand.

**Example:**

```mlir
pto.trowexpandadd ins(%src0, %src1 : !pto.tile_buf<vec, 16x128xf32>,
                                     !pto.tile_buf<vec, 16x1xf32, blayout=col_major>)
                  outs(%dst : !pto.tile_buf<vec, 16x128xf32>)
```

```mlir
pto.trowexpandadd ins(%src0, %src1, %tmp : !pto.tile_buf<vec, 16x128xf32>,
                                            !pto.tile_buf<vec, 16x1xf32, blayout=col_major>,
                                            !pto.tile_buf<vec, 16x128xf32>)
                  outs(%dst : !pto.tile_buf<vec, 16x128xf32>)
```

```mlir
pto.trowexpanddiv ins(%src0, %src1, %tmp : !pto.tile_buf<vec, 16x128xf32>,
                                             !pto.tile_buf<vec, 16x1xf32, blayout=col_major>,
                                             !pto.tile_buf<vec, 16x128xf32>)
                  outs(%dst : !pto.tile_buf<vec, 16x128xf32>)
                  {precisionType = #pto<div_precision high_precision>}
```

---

## 10.4 Column-Wise Broadcast: `pto.tcolexpand`

- **syntax:**
```mlir
pto.tcolexpand ins(%src : !pto.tile_buf<...>)
               outs(%dst : !pto.tile_buf<...>)
```
- **semantics:** `dst[row, col] = src[0, col]`.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.tile_buf` | Source tile carrying one logical scalar per destination column. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Constraints:**

- `src` and `dst` must have the same number of valid columns.
- `src` must encode exactly one logical source value per destination column.
- Templates target row-major VEC layouts.
- On A5, matching `ui8`, `ui16`, and `ui32` source and destination types are supported.

**Example:**

```mlir
pto.tcolexpand ins(%src : !pto.tile_buf<vec, 1x16xf32>)
               outs(%dst : !pto.tile_buf<vec, 16x16xf32>)
```

---

## 10.5 Column-Wise Broadcast Arithmetic and Transform Families

The column-expansion family combines a full-sized tile with a per-column scalar carrier.
Either input may be the carrier: the full-sized operand is the one whose `valid_shape`
matches `%dst`, with `%src0` taking precedence when both match (as in `pto-isa`'s
`src0eqdst`):

| Op | Semantics |
|----|-----------|
| `pto.tcolexpandadd` | `dst[row, col] = src0[row, col] + src1[0, col]` |
| `pto.tcolexpandsub` | `dst[row, col] = src0[row, col] - src1[0, col]` |
| `pto.tcolexpandmul` | `dst[row, col] = src0[row, col] * src1[0, col]` |
| `pto.tcolexpanddiv` | `dst[row, col] = src0[row, col] / src1[0, col]` |
| `pto.tcolexpandmax` | `dst[row, col] = max(src0[row, col], src1[0, col])` |
| `pto.tcolexpandmin` | `dst[row, col] = min(src0[row, col], src1[0, col])` |
| `pto.tcolexpandexpdif` | `dst[row, col] = exp(src0[row, col] - src1[0, col])` |

> Write the carrier as `E` and the full-sized operand as `F`; the table above uses
> `E = %src1`, `F = %src0` and reads `dst[row, col] = F[row, col] OP E[0, col]`. When `%src0`
> is the carrier, substitute `E = %src0`, `F = %src1`. The operation always keeps the
> authored order `%src0 OP %src1`.

### Common Syntax

```mlir
pto.<op> ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src0` | `pto.tile_buf` | Main source tile. |
| `src1` | `pto.tile_buf` | Either the full-sized operand matching `dst`, or the tile carrying one logical scalar per destination column. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Constraints:**

- The full-sized operand — `%src0`, or `%src1` when only `%src1`'s valid shape matches `dst` —
  must be shape/valid-region compatible with `dst`.
- The expanded operand must provide one logical scalar per destination column
  (`valid_shape[1] == dst valid_shape[1]`).
- Templates target row-major VEC layouts.
- On A5, add, sub, mul, max, and min accept matching `bf16`, `ui8`, `ui16`, and `ui32` tiles.
- On A5, `pto.tcolexpanddiv` also accepts matching `bf16`, `ui8`, `ui16`, `ui32`,
  `i16`, and `i32` tiles; integer division truncates toward zero and requires a
  nonzero divisor.
- `pto.tcolexpandexpdif` is floating-point-only (`f16` or `f32`).
- `pto.tcolexpanddiv` additionally accepts `precisionType = #pto<div_precision default|high_precision>`.
  Omitted means `default`.
  `high_precision` is currently legal only when the tile element type is `f16` or `f32`.

**Example:**

```mlir
pto.tcolexpandadd ins(%src0, %src1 : !pto.tile_buf<vec, 16x128xf32>,
                                     !pto.tile_buf<vec, 1x128xf32>)
                  outs(%dst : !pto.tile_buf<vec, 16x128xf32>)
```

```mlir
pto.tcolexpanddiv ins(%src0, %src1 : !pto.tile_buf<vec, 16x128xf32>,
                                     !pto.tile_buf<vec, 1x128xf32>)
                  outs(%dst : !pto.tile_buf<vec, 16x128xf32>)
                  {precisionType = #pto<div_precision high_precision>}
```
