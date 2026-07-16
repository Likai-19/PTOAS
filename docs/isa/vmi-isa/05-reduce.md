# 5. Reduce

> **Category:** B (VLane-aligned), C (unaligned sub-VLane).
> **Mask:** `Pg req` (governing mask is a required operand).
>
> Reduction ops collapse lanes into compact scalars, governed by a mask.
> The source shape `MxN` controls the number and width of groups. Inactive lane behavior:
> `trowsum` treats inactive as 0; `trowmax`/`trowmin` treat inactive as `-∞`/`+∞`
> (fp) or type min/max (int).

---

## `pto.vmi.trowsum`

- **semantics:** Masked add-reduction. Each source row is reduced to one scalar.
  The `1xL -> 1x1` form is the former full reduction.

  ```c
  // Without group: full reduction to scalar
  T sum = 0;
  for (int i = 0; i < L; i++)
      if (mask[i]) sum += src[i];
  dst[0] = sum;

  // MxN: per-row/group reduction
  for (int g = 0; g < M; g++) {
      T sum = 0;
      for (int i = 0; i < N; i++)
          if (mask[g][i]) sum += src[g][i];
      dst[g][0] = sum;
  }
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.trowsum %src, %mask {reassoc} : !pto.vmi.tilereg<MxNxT>, !pto.vmi.tilereg<MxNxi1> -> !pto.vmi.tilereg<Mx1xT>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `src` | `!pto.vmi.tilereg<MxNxT>` | Source tile; each row is one group |
  | `mask` | `!pto.vmi.tilereg<MxNxi1>` | Governing predicate (required) |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.tilereg<Mx1xT>` | One scalar per source row |

- **attributes:**

  | Attribute | Values | Default | Description |
  |---|---|---|---|
  | `reassoc` | *(unit attr)* | *(absent)* | Permit reassociation (**required** for fp sources) |
  | `pmode` | `"zero"`, `"merge"` | `"zero"` | Inactive-result behavior |

- **datatypes:** `i8`–`i32`, `f16`, `f32`
- **lowering to `pto.mi`:**

  | Group / W | Category | Physical lowering | `#mi` | `dep` |
  |---|---|---|---|---|
  | No group (`C=1`), `K=1` | B | `1 × pto.vcadd` | `1` | `1` |
  | No group, `K>1` (fold) | B | `(K-1) × vadd` + `1 × vcadd` | `K` | `K` |
  | No group, `K>1` (partial) | B | `K × vcadd` + combine | `K` | `1+⌈log₂K⌉` |
  | `M=8` rows (W=32B, VLane-aligned) | B | `K × pto.vcgadd` | `K` | `1` |
  | `M=2/4` rows (W=64B/128B aligned) | B | `(k-1) × vadd` fold + `vcgadd` | `K+k-1` | `k` |

- **example:**
  ```mlir
  // Full sum reduction (to scalar)
  %sum = pto.vmi.trowsum %x, %mask {reassoc}
      : !pto.vmi.tilereg<1x64xf32>, !pto.vmi.tilereg<1x64xi1> -> !pto.vmi.tilereg<1x1xf32>

  // Grouped: 256-lane → 8 groups of 32, each VLane-aligned (W=32B)
  %sums = pto.vmi.trowsum %x, %mask
      : !pto.vmi.tilereg<8x32xf16>, !pto.vmi.tilereg<8x32xi1> -> !pto.vmi.tilereg<8x1xf16>
  ```

---

## `pto.vmi.trowmax` / `pto.vmi.trowmin`

- **semantics:** Masked max/min reduction.

  ```c
  // trowmax: inactive lanes treated as -∞
  T best = -INF;
  for (int i = 0; i < L; i++)
      if (mask[i]) best = max(best, src[i]);
  dst[0] = best;

  // trowmin: inactive lanes treated as +∞
  T best = +INF;
  for (int i = 0; i < L; i++)
      if (mask[i]) best = min(best, src[i]);
  dst[0] = best;
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.trowmax %src, %mask : !pto.vmi.tilereg<MxNxT>, !pto.vmi.tilereg<MxNxi1> -> !pto.vmi.tilereg<Mx1xT>
  ```
- **operands:** Same as `trowsum` (without `reassoc`).
- **results:** Same as `trowsum`.
- **attributes:** `pmode` (same as `trowsum`, no `reassoc`).
- **datatypes:** `i16`–`i32`, `f16`, `f32`
- **lowering to `pto.mi`:**

  | Group / W | Physical lowering |
  |---|---|
  | No group, fold | `(K-1) × vmax` + `1 × vcmax` |
  | VLane-aligned | `K × pto.vcgmax` / `K × pto.vcgmin` |

- **example:**
  ```mlir
  // Full max reduction
  %mx = pto.vmi.trowmax %x, %mask
      : !pto.vmi.tilereg<1x64xf32>, !pto.vmi.tilereg<1x64xi1> -> !pto.vmi.tilereg<1x1xf32>

  // Grouped: 8-sub-group max (MX block-scale exponent pattern)
  %maxe = pto.vmi.trowmax %exp, %mask
      : !pto.vmi.tilereg<8x32xui16>, !pto.vmi.tilereg<8x32xi1> -> !pto.vmi.tilereg<8x1xui16>
  ```
