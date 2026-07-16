# 7. SFU

> **Category:** A (fused arithmetic, `vmull`), B (`vchist`, `vdhist`), C (gather/scatter).
> **Mask:** `Pg` on all except sort-like ops.
>
> Special-function / domain-accelerator ops. Mixed categories: `vchist`
> produces a `half` axis (B); `vdhist` yields a plain per-bin count (B);
> gather/scatter are Category C tile/permute ops; fused activation/arithmetic
> ops and pair-result `vmull` are Category A layout-passthrough operations.

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
  %e = pto.vmi.vexpdif %x, %max, %mask : !pto.vmi.vreg<L×T_x>, !pto.vmi.vreg<L×f32>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×f32>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `x` | `!pto.vmi.vreg<L×T_x>` | Input (`f16` or `f32`) |
  | `max` | `!pto.vmi.vreg<L×f32>` | Subtracted max (always `f32`) |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<L×f32>` | `exp(x − max)` (always `f32`) |

- **attributes:** `pmode` (`"zero"` / `"merge"`)
- **datatypes:** Input `x`: `f16`, `f32`; `max` and result: always `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vexpdif
  ```
  `#mi = K`, `dep = 1`. Fuses `vsub` + `vexp`.

- **example:**
  ```mlir
  %e = pto.vmi.vexpdif %x, %max, %mask
      : !pto.vmi.vreg<64×f32>, !pto.vmi.vreg<64×f32>, !pto.vmi.mask<64>
      -> !pto.vmi.vreg<64×f32>
  ```

### `pto.vmi.vaxpy`

- **semantics:** Fused `α·x + y` (scale-add). Single hardware instruction.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? (alpha * x[i] + acc[i]) : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %y = pto.vmi.vaxpy %x, %acc, %alpha, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, T, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `x` | `!pto.vmi.vreg<L×T>` | Input vector |
  | `acc` | `!pto.vmi.vreg<L×T>` | Accumulator (`y`) |
  | `alpha` | `T` (float scalar) | Scale factor |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<L×T>` | `α·x + acc` |

- **datatypes:** `f16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vaxpy
  ```
  `#mi = K`, `dep = 1`. Fuses `vmuls` + `vadd`.

### `pto.vmi.vlrelu`

- **semantics:** Leaky ReLU: `y = x > 0 ? x : slope × x`. The slope is a
  scalar shared across all lanes.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? (src[i] > 0 ? src[i] : slope * src[i]) : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %y = pto.vmi.vlrelu %x, %slope, %mask : !pto.vmi.vreg<L×T>, T, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `x` | `!pto.vmi.vreg<L×T>` | Input |
  | `slope` | `T` (float scalar) | Negative-slope multiplier |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate |

- **datatypes:** `f16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vlrelu
  ```
  `#mi = K`, `dep = 1`.

### `pto.vmi.vprelu`

- **semantics:** Parametric ReLU: `y = max(x, 0) + alpha × min(x, 0)`. The
  `alpha` is a per-lane parameter vector (not a shared scalar).

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? (max(src[i], 0) + alpha[i] * min(src[i], 0)) : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %y = pto.vmi.vprelu %x, %alpha, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `x` | `!pto.vmi.vreg<L×T>` | Input |
  | `alpha` | `!pto.vmi.vreg<L×T>` | Per-lane negative-slope parameter |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate |

- **datatypes:** `f16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vprelu
  ```
  `#mi = K`, `dep = 1`.

### `pto.vmi.vmull`

- **semantics:** Widening 32-bit × 32-bit multiply. The operation returns the
  low and high 32-bit halves as two logical vectors of the same type as the
  inputs. Inactive lanes in both results are zero.

  ```c
  for (int i = 0; i < L; i++) {
      uint64_t product = T == i32
          ? (uint64_t)((int64_t)(int32_t)a[i] * (int64_t)(int32_t)b[i])
          : (uint64_t)(uint32_t)a[i] * (uint64_t)(uint32_t)b[i];
      low[i] = mask[i] ? (uint32_t)product : 0;
      high[i] = mask[i] ? (uint32_t)(product >> 32) : 0;
  }
  ```

- **syntax:**
  ```mlir
  %low, %high = pto.vmi.vmull %a, %b, %mask
      : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, !pto.vmi.mask<L>
        -> !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `a` | `!pto.vmi.vreg<L×T>` | First operand |
  | `b` | `!pto.vmi.vreg<L×T>` | Second operand |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate |

- **results:** `%low` and `%high`, both `!pto.vmi.vreg<L×T>`
- **datatypes:** `T` is exactly `i32` or `ui32`; `L` is exactly 64, 128, or
  256. Omitted `pmode` means zeroing; the only explicit legal value is
  `pmode = "zero"`.
- **lowering to `pto.mi`:**
  ```
  K × pto.vmull (produces hi+lo pair per reg)
  ```
  For contiguous layout, `K = L / 64`; therefore 64, 128, and 256 lanes lower
  to 1, 2, and 4 operations respectively.

- **example:**
  ```mlir
  %low, %high = pto.vmi.vmull %a, %b, %mask
      : !pto.vmi.vreg<64×i32>, !pto.vmi.vreg<64×i32>, !pto.vmi.mask<64>
        -> !pto.vmi.vreg<64×i32>, !pto.vmi.vreg<64×i32>
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
  %acc1 = pto.vmi.vmula %acc, %lhs, %rhs, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `acc` | `!pto.vmi.vreg<L×T>` | Accumulator (read-modify-write) |
  | `lhs` | `!pto.vmi.vreg<L×T>` | First multiply operand |
  | `rhs` | `!pto.vmi.vreg<L×T>` | Second multiply operand |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate |

- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vmula
  ```
  `#mi = K`, `dep = 1`. Fuses `vmul` + `vadd`.

- **example:**
  ```mlir
  %acc1 = pto.vmi.vmula %acc, %a, %b, %mask
      : !pto.vmi.vreg<64×f32>, !pto.vmi.vreg<64×f32>, !pto.vmi.vreg<64×f32>,
        !pto.vmi.mask<64> -> !pto.vmi.vreg<64×f32>
  ```

---

## 7.2 Histogram

### `pto.vmi.vchist`

- **semantics:** **Cumulative histogram** — the existing `chistv2`
  semantics. Counts per-bin occurrences over a bin-index vector and produces
  a `half`-axis (`Bin_N0`/`Bin_N1`) pair accessible through the result's
  width axis.

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
  %h = pto.vmi.vchist %bin_idx, %mask : !pto.vmi.vreg<L×i8>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×i16>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `bin_idx` | `!pto.vmi.vreg<L×i8>` | Per-lane bin index (unsigned 8-bit) |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<L×T_count>` | Bin counts (half axis: Bin_N0/N1 pair) |

- **attributes:**

  | Attribute | Values | Default | Description |
  |---|---|---|---|
  | `pmode` | `"zero"`, `"merge"` | `"zero"` | Inactive-lane behavior |
- **datatypes:** Bin index: `i8`/`ui8`; result count type: typically `i16`/`i32`
- **lowering to `pto.mi`:**
  ```
  chistv2 Bin_N0 + Bin_N1 (two-half fanout) + widen/accumulate
  ```
  `#mi ≈ 2K`, `dep = 2–3`. INTLV merge on store.

- **example:**
  ```mlir
  // Cumulative histogram, half-axis Bin_N0/Bin_N1
  %h = pto.vmi.vchist %bin_idx, %mask
      : !pto.vmi.vreg<256×i8>, !pto.vmi.mask<256> -> !pto.vmi.vreg<256×i16>
  // → pto.as: Bin_N0 + Bin_N1 fanout → INTLV merge on vstore
  ```

### `pto.vmi.vdhist`

- **semantics:** **Distribution histogram** — count per bin over a
  value/index vector, yielding a plain per-bin count vector (no `half`
  axis).

  ```c
  // Plain per-bin distribution count
  uint16_t bins[N] = {0};
  for (int i = 0; i < L; i++)
      if (mask[i])
          bins[bin_idx[i]]++;
  ```

- **syntax:**
  ```mlir
  %d = pto.vmi.vdhist %bin_idx, %mask : !pto.vmi.vreg<L×i8>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×i16>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `bin_idx` | `!pto.vmi.vreg<L×i8>` | Per-lane bin index (unsigned 8-bit) |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<L×T_count>` | Plain per-bin count vector |

- **attributes:**

  | Attribute | Values | Default | Description |
  |---|---|---|---|
  | `pmode` | `"zero"`, `"merge"` | `"zero"` | Inactive-lane behavior |
- **datatypes:** Bin index: `i8`/`ui8`; result count type: typically `i16`/`i32`
- **lowering to `pto.mi`:**
  ```
  distribution histogram accumulate (no half-axis fanout)
  ```
  `#mi ≈ K`, `dep = 2`.

- **example:**
  ```mlir
  // Distribution histogram, plain per-bin count
  %d = pto.vmi.vdhist %bin_idx, %mask
      : !pto.vmi.vreg<256×i8>, !pto.vmi.mask<256> -> !pto.vmi.vreg<256×i16>
  ```

---

## 7.3 Gather / Scatter

> **Category C** — contiguous-required. `pto.as` materializes `.contiguous()`
> before these ops if the input layout is non-contiguous.

### `pto.vmi.vgather`

- **semantics:** Indexed gather from UB at B32 granularity. For each active
  lane `i`, load `src[offsets[i]]`.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? ub[base + offsets[i]] : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %g = pto.vmi.vgather %src, %offsets, %mask : !pto.ptr<T, ub>, !pto.vmi.vreg<L×i32>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `src` | `!pto.ptr<T, ub>` | UB base pointer |
  | `offsets` | `!pto.vmi.vreg<L×i32>` | Per-lane element offset |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate |

- **results:** `!pto.vmi.vreg<L×T>`
- **attributes:** `pmode`
- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vgather2
  ```
  `#mi = K`, `dep = 1`, util data-dependent.

### `pto.vmi.vgatherb`

- **semantics:** Byte-granularity indexed gather. Mask lane count equals result
  lane count (may differ from offset lane count).

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? ub_byte[base_byte + offsets[i]] : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %gb = pto.vmi.vgatherb %src, %offsets, %mask : !pto.ptr<T, ub>, !pto.vmi.vreg<L×i32>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vgatherb
  ```
  `#mi = K`, `dep = 1`.

### `pto.vmi.vscatter`

- **semantics:** Indexed scatter to UB. For each active lane `i`,
  write `value[i]` to `dest[offsets[i]]`.

  ```c
  for (int i = 0; i < L; i++)
      if (mask[i])
          ub[base + offsets[i]] = value[i];
  ```

- **syntax:**
  ```mlir
  pto.vmi.vscatter %value, %dest, %offsets, %mask : !pto.vmi.vreg<L×T>, !pto.ptr<T, ub>, !pto.vmi.vreg<L×i32>, !pto.vmi.mask<L>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `value` | `!pto.vmi.vreg<L×T>` | Values to scatter |
  | `dest` | `!pto.ptr<T, ub>` | UB destination base pointer |
  | `offsets` | `!pto.vmi.vreg<L×i32>` | Per-lane element offset |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate |

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
  pto.vmi.vscatter %v, %dest, %offsets, %mask
      : !pto.vmi.vreg<64×f32>, !pto.ptr<f32, ub>, !pto.vmi.vreg<64×i32>, !pto.vmi.mask<64>
  ```
