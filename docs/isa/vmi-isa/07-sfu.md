# 7. SFU

> **Category:** A (fused arithmetic), B (`thistogram`, widening `tmul`), C (gather/scatter).
> **Mask:** `Pg` on all except sort-like ops.
>
> Special-function / domain-accelerator ops. Mixed categories: `thistogram` produces
> a `half` axis (B); gather/scatter are Category C tile/permute ops; fused
> activation/arithmetic ops are Category A `tilereg→tilereg`.

---

## 7.1 Fused Arithmetic

### `pto.vmi.vexpdif`

- **semantics:** Fused `exp(x − max)` for softmax numerical stability. Single
  hardware instruction.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? exp(x[i] - max[i]) : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %e = pto.vmi.vexpdif %x, %max, %mask : !pto.vmi.tilereg<MxNxT_x>, !pto.vmi.tilereg<MxNxf32>, !pto.vmi.tilereg<MxNxi1> -> !pto.vmi.tilereg<MxNxf32>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `x` | `!pto.vmi.tilereg<MxNxT_x>` | Input (`f16` or `f32`) |
  | `max` | `!pto.vmi.tilereg<MxNxf32>` | Subtracted max (always `f32`) |
  | `mask` | `!pto.vmi.tilereg<MxNxi1>` | Governing predicate |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.tilereg<MxNxf32>` | `exp(x − max)` (always `f32`) |

- **attributes:** `pmode` (`"zero"` / `"merge"`)
- **datatypes:** Input `x`: `f16`, `f32`; `max` and result: always `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vexpdif
  ```
  `#mi = K`, `dep = 1`. Fuses `tsub` + `texp`.

- **example:**
  ```mlir
  %e = pto.vmi.vexpdif %x, %max, %mask
      : !pto.vmi.tilereg<1x64xf32>, !pto.vmi.tilereg<1x64xf32>, !pto.vmi.tilereg<1x64xi1>
      -> !pto.vmi.tilereg<1x64xf32>
  ```

- **notes:** `vexpdif` remains a VMI-only name. `trowexpandexpdif` is only a
  legal optimization when `max` is provably one scalar per row, all values are
  `f32`, and the valid-region behavior is equivalent to the mask and `pmode`.

### `pto.vmi.taxpy`

- **semantics:** Fused `α·x + y` (scale-add). Single hardware instruction.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? (alpha * x[i] + acc[i]) : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %y = pto.vmi.taxpy %x, %acc, %alpha, %mask : !pto.vmi.tilereg<MxNxT>, !pto.vmi.tilereg<MxNxT>, T, !pto.vmi.tilereg<MxNxi1> -> !pto.vmi.tilereg<MxNxT>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `x` | `!pto.vmi.tilereg<MxNxT>` | Input vector |
  | `acc` | `!pto.vmi.tilereg<MxNxT>` | Accumulator (`y`) |
  | `alpha` | `T` (float scalar) | Scale factor |
  | `mask` | `!pto.vmi.tilereg<MxNxi1>` | Governing predicate |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.tilereg<MxNxT>` | `α·x + acc` |

- **datatypes:** `f16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vaxpy
  ```
  `#mi = K`, `dep = 1`. Fuses `tmuls` + `tadd`.

### `pto.vmi.tlrelu`

- **semantics:** Leaky ReLU: `y = x > 0 ? x : slope × x`. The slope is a
  scalar shared across all lanes.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? (src[i] > 0 ? src[i] : slope * src[i]) : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %y = pto.vmi.tlrelu %x, %slope, %mask : !pto.vmi.tilereg<MxNxT>, T, !pto.vmi.tilereg<MxNxi1> -> !pto.vmi.tilereg<MxNxT>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `x` | `!pto.vmi.tilereg<MxNxT>` | Input |
  | `slope` | `T` (float scalar) | Negative-slope multiplier |
  | `mask` | `!pto.vmi.tilereg<MxNxi1>` | Governing predicate |

- **datatypes:** `f16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vlrelu
  ```
  `#mi = K`, `dep = 1`.

### `pto.vmi.tprelu`

- **semantics:** Parametric ReLU: `y = max(x, 0) + alpha × min(x, 0)`. The
  `alpha` is a per-lane parameter vector (not a shared scalar).

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? (max(src[i], 0) + alpha[i] * min(src[i], 0)) : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %y = pto.vmi.tprelu %x, %alpha, %mask : !pto.vmi.tilereg<MxNxT>, !pto.vmi.tilereg<MxNxT>, !pto.vmi.tilereg<MxNxi1> -> !pto.vmi.tilereg<MxNxT>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `x` | `!pto.vmi.tilereg<MxNxT>` | Input |
  | `alpha` | `!pto.vmi.tilereg<MxNxT>` | Per-lane negative-slope parameter |
  | `mask` | `!pto.vmi.tilereg<MxNxi1>` | Governing predicate |

- **datatypes:** `f16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vprelu
  ```
  `#mi = K`, `dep = 1`.

### `pto.vmi.tmul`

- **semantics:** Widening 32-bit × 32-bit → 64-bit multiply. The former
  `vmull` uses the `tmul` name with an `i64`/`ui64` result dtype. The result
  occupies two physical registers (hi + lo) accessed through a virtual `width`
  axis.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? (int64_t)a[i] * (int64_t)b[i] : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %res = pto.vmi.tmul %a, %b, %mask : !pto.vmi.tilereg<MxNxi32>, !pto.vmi.tilereg<MxNxi32>, !pto.vmi.tilereg<MxNxi1> -> !pto.vmi.tilereg<MxNxi64>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `a` | `!pto.vmi.tilereg<MxNxi32>` | First operand |
  | `b` | `!pto.vmi.tilereg<MxNxi32>` | Second operand |
  | `mask` | `!pto.vmi.tilereg<MxNxi1>` | Governing predicate |

- **results:** `!pto.vmi.tilereg<MxNxi64>` (2 physical regs per logical value)
- **datatypes:** `i32` → `i64` (also `ui32` → `ui64`)
- **lowering to `pto.mi`:**
  ```
  K × pto.vmull (produces hi+lo pair per reg)
  ```
  `#mi = K`, `dep = 1`. Two result regs per input reg → Category B (`width` axis).

- **example:**
  ```mlir
  %res = pto.vmi.tmul %a, %b, %mask
      : !pto.vmi.tilereg<1x64xi32>, !pto.vmi.tilereg<1x64xi32>, !pto.vmi.tilereg<1x64xi1>
      -> !pto.vmi.tilereg<1x64xi64>
  ```

### `pto.vmi.vmula`

- **semantics:** Fused multiply-add: `acc = acc + lhs × rhs`. Single hardware
  instruction. The accumulator is both an input and output (writes back).

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? (acc[i] + lhs[i] * rhs[i]) : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %acc1 = pto.vmi.vmula %acc, %lhs, %rhs, %mask : !pto.vmi.tilereg<MxNxT>, !pto.vmi.tilereg<MxNxT>, !pto.vmi.tilereg<MxNxT>, !pto.vmi.tilereg<MxNxi1> -> !pto.vmi.tilereg<MxNxT>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `acc` | `!pto.vmi.tilereg<MxNxT>` | Accumulator (read-modify-write) |
  | `lhs` | `!pto.vmi.tilereg<MxNxT>` | First multiply operand |
  | `rhs` | `!pto.vmi.tilereg<MxNxT>` | Second multiply operand |
  | `mask` | `!pto.vmi.tilereg<MxNxi1>` | Governing predicate |

- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vmula
  ```
  `#mi = K`, `dep = 1`. Fuses `tmul` + `tadd`.

- **example:**
  ```mlir
  %acc1 = pto.vmi.vmula %acc, %a, %b, %mask
      : !pto.vmi.tilereg<1x64xf32>, !pto.vmi.tilereg<1x64xf32>, !pto.vmi.tilereg<1x64xf32>,
        !pto.vmi.tilereg<1x64xi1> -> !pto.vmi.tilereg<1x64xf32>
  ```

- **notes:** `vmula` remains a VMI-only name. Its fused tile-tile
  multiply-accumulate semantics are not the same as scalar `taxpy`, and a
  `tmul` plus `tadd` sequence is not a single equivalent fused operation.

---

## 7.2 Histogram

### `pto.vmi.thistogram`

- **semantics:** Histogram bin count with the former `vhist`/`chistv2`
  semantics. Counts per-bin occurrences over a channel-index vector,
  producing a `half`-axis (`Bin_N0`/`Bin_N1`) pair accessible through the
  result's width axis. This form uses the same-semantic Tile Op name
  `thistogram`.

  ```c
  // Hardware chistv2: two halves (Bin_N0, Bin_N1), 256 bins total
  uint16_t bins[256] = {0};
  for (int i = 0; i < L; i++)
      if (mask[i])
          bins[bin_idx[i]]++;
  // dst carries Bin_N0 (bins 0–127) and Bin_N1 (bins 128–255) on a half axis
  ```

- **syntax:**
  ```mlir
  %h = pto.vmi.thistogram %bin_idx, %mask : !pto.vmi.tilereg<MxNxi8>, !pto.vmi.tilereg<MxNxi1> -> !pto.vmi.tilereg<MxNxi16>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `bin_idx` | `!pto.vmi.tilereg<MxNxi8>` | Per-lane bin index (unsigned 8-bit) |
  | `mask` | `!pto.vmi.tilereg<MxNxi1>` | Governing predicate |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.tilereg<MxNxT_count>` | Bin counts (half axis: Bin_N0/N1 pair) |

- **attributes:** `pmode` (`"zero"` / `"merge"`).

- **datatypes:** Bin index: `i8`/`ui8`; result count type: typically `i16`/`i32`
- **lowering to `pto.mi`:**
  ```
  chistv2 Bin_N0 + Bin_N1 (two-half fanout) + widen/accumulate
  ```
  `#mi ≈ 2K`, `dep = 2–3`. INTLV merge on store.

- **example:**
  ```mlir
  // Channel histogram, half-axis Bin_N0/Bin_N1
  %h = pto.vmi.thistogram %bin_idx, %mask
      : !pto.vmi.tilereg<1x256xi8>, !pto.vmi.tilereg<1x256xi1> -> !pto.vmi.tilereg<1x256xi16>
  // → pto.as: Bin_N0 + Bin_N1 fanout → INTLV merge on vstore
  ```

### `pto.vmi.dhist` (no direct same-semantic Tile Op)

- **semantics:** Full 256-bin distribution histogram over unsigned 8-bit
  source lanes. It starts from a `1x256xui16` accumulator, increments the bin
  selected by each active source lane, and returns the updated accumulator.

  ```c
  for (int b = 0; b < 256; b++)
      dst[0][b] = acc[0][b];
  for (int i = 0; i < L; i++)
      if (mask[0][i]) dst[0][source[0][i]]++;
  ```

- **syntax:**
  ```mlir
  %d = pto.vmi.dhist %acc, %source, %mask
      : !pto.vmi.tilereg<1x256xui16>, !pto.vmi.tilereg<1xLxui8>,
        !pto.vmi.tilereg<1xLxi1> -> !pto.vmi.tilereg<1x256xui16>
  ```

- **operands/results:** `acc` and `result` are `1x256xui16`; `source` is
  `1xLxui8`; `mask` is `1xLxi1` and has the same shape as `source`.

Tile Op `thistogram` is a row-wise source/index/destination update with a
different operand-role and result contract, so it does not directly implement
`dhist`. The VMI name is therefore retained.

---

## 7.3 Gather / Scatter

> **Category C** — contiguous-required. `pto.as` materializes `.contiguous()`
> before these ops if the input layout is non-contiguous.

### `pto.vmi.tgather`

- **semantics:** Indexed gather from UB at B32 granularity. For each active
  lane `i`, load `src[offsets[i]]`.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? ub[base + offsets[i]] : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %g = pto.vmi.tgather %src, %offsets, %mask : !pto.ptr<T, ub>, !pto.vmi.tilereg<MxNxi32>, !pto.vmi.tilereg<MxNxi1> -> !pto.vmi.tilereg<MxNxT>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `src` | `!pto.ptr<T, ub>` | UB base pointer |
  | `offsets` | `!pto.vmi.tilereg<MxNxi32>` | Per-lane element offset |
  | `mask` | `!pto.vmi.tilereg<MxNxi1>` | Governing predicate |

- **results:** `!pto.vmi.tilereg<MxNxT>`
- **attributes:** `pmode`
- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vgather2
  ```
  `#mi = K`, `dep = 1`, util data-dependent.

### `pto.vmi.tgatherb`

- **semantics:** Byte-granularity indexed gather. Mask lane count equals result
  lane count (may differ from offset lane count).

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? ub_byte[base_byte + offsets[i]] : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %gb = pto.vmi.tgatherb %src, %offsets, %mask : !pto.ptr<T, ub>, !pto.vmi.tilereg<MxNxi32>, !pto.vmi.tilereg<MxNxi1> -> !pto.vmi.tilereg<MxNxT>
  ```
- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vgatherb
  ```
  `#mi = K`, `dep = 1`.

### `pto.vmi.tscatter`

- **semantics:** Indexed scatter to UB. For each active lane `i`,
  write `value[i]` to `dest[offsets[i]]`.

  ```c
  for (int i = 0; i < L; i++)
      if (mask[i])
          ub[base + offsets[i]] = value[i];
  ```

- **syntax:**
  ```mlir
  pto.vmi.tscatter %value, %dest, %offsets, %mask : !pto.vmi.tilereg<MxNxT>, !pto.ptr<T, ub>, !pto.vmi.tilereg<MxNxi32>, !pto.vmi.tilereg<MxNxi1>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `value` | `!pto.vmi.tilereg<MxNxT>` | Values to scatter |
  | `dest` | `!pto.ptr<T, ub>` | UB destination base pointer |
  | `offsets` | `!pto.vmi.tilereg<MxNxi32>` | Per-lane element offset |
  | `mask` | `!pto.vmi.tilereg<MxNxi1>` | Governing predicate |

- **results:** *(none)*
- **attributes:** `pmode`
- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vscatter
  ```
  `#mi = K`, `dep = 1`.

- **example:**
  ```mlir
  pto.vmi.tscatter %v, %dest, %offsets, %mask
      : !pto.vmi.tilereg<1x64xf32>, !pto.ptr<f32, ub>, !pto.vmi.tilereg<1x64xi32>, !pto.vmi.tilereg<1x64xi1>
  ```
