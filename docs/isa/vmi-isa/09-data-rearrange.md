# 9. Data Rearrange

> **Category:** A (layout-transparent). **Mask:** `Pg`.
>
> In-register data movement and permutation. No UB access. `tinterleave`/`tdeinterleave`
> are per-lane, dtype-preserving ops that do not change tilereg layout — the output
> has the same `L` and `T` as the inputs. Commonly used for real+imaginary and
> value+index interleaving within a single tile register.

---

## `pto.vmi.tinterleave`

- **semantics:** Interleave two source vectors by even/odd lanes.

  ```c
  // low  = {lhs[0], rhs[0], lhs[1], rhs[1], ..., lhs[L/2-1], rhs[L/2-1]}
  // high = {lhs[L/2], rhs[L/2], lhs[L/2+1], rhs[L/2+1], ...}
  for (int i = 0; i < L/2; i++) {
      lo[2*i]     = lhs[i];
      lo[2*i + 1] = rhs[i];
      hi[2*i]     = lhs[L/2 + i];
      hi[2*i + 1] = rhs[L/2 + i];
  }
  ```

- **syntax:**
  ```mlir
  %lo, %hi = pto.vmi.tinterleave %lhs, %rhs, %mask : !pto.vmi.tilereg<1xLxT>, !pto.vmi.tilereg<1xLxT>, !pto.vmi.tilereg<1xLxi1> -> !pto.vmi.tilereg<1xLxT>, !pto.vmi.tilereg<1xLxT>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `lhs` | `!pto.vmi.tilereg<1xLxT>` | First source (provides low-half even slots) |
  | `rhs` | `!pto.vmi.tilereg<1xLxT>` | Second source (provides low-half odd slots) |
  | `mask` | `!pto.vmi.tilereg<1xLxi1>` | Governing predicate |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `low` | `!pto.vmi.tilereg<1xLxT>` | Even-odd interleaved low half |
  | `high` | `!pto.vmi.tilereg<1xLxT>` | Even-odd interleaved high half |

- **attributes:** `pmode`
- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vintlv
  ```
  `#mi = K`, `dep = 1`. Layout-transparent (Category A).

- **example:**
  ```mlir
  %lo, %hi = pto.vmi.tinterleave %a, %b, %mask
      : !pto.vmi.tilereg<1x64xf32>, !pto.vmi.tilereg<1x64xf32>, !pto.vmi.tilereg<1x64xi1>
      -> !pto.vmi.tilereg<1x64xf32>, !pto.vmi.tilereg<1x64xf32>
  ```

---

## `pto.vmi.tdeinterleave`

- **semantics:** Deinterleave a paired-source by even/odd lanes (AoS → SoA).

  ```c
  // lhs, rhs treated as pairs: (lhs[0], rhs[0]), (lhs[1], rhs[1]), ...
  // even = {lhs[0], lhs[2], lhs[4], ...} (all even-indexed slots from paired stream)
  // odd  = {lhs[1], lhs[3], lhs[5], ...} (all odd-indexed slots from paired stream)
  // More precisely:
  // low  = {lhs[0], lhs[1], lhs[2], lhs[3], ...}   ← original even slots from each pair
  // high = {rhs[0], rhs[1], rhs[2], rhs[3], ...}   ← original odd slots from each pair
  // After deinterleaving:
  // even[i] = (i % 2 == 0) ? lhs[i/2] : rhs[i/2]  — this is the tinterleave inverse
  for (int i = 0; i < L/2; i++) {
      even[i]         = lhs[2*i];      // even slots of paired input
      even[L/2 + i]   = lhs[2*i + 1];
      odd[i]          = rhs[2*i];      // odd slots of paired input
      odd[L/2 + i]    = rhs[2*i + 1];
  }
  ```

- **syntax:**
  ```mlir
  %even, %odd = pto.vmi.tdeinterleave %lhs, %rhs, %mask : !pto.vmi.tilereg<1xLxT>, !pto.vmi.tilereg<1xLxT>, !pto.vmi.tilereg<1xLxi1> -> !pto.vmi.tilereg<1xLxT>, !pto.vmi.tilereg<1xLxT>
  ```
- **operands:** Same shape as `tinterleave`.
- **results:** Same shape as `tinterleave` (two `!pto.vmi.tilereg<1xLxT>`).
- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vdintlv
  ```
  `#mi = K`, `dep = 1`.

- **example:**
  ```mlir
  %even, %odd = pto.vmi.tdeinterleave %x, %y, %mask
      : !pto.vmi.tilereg<1x64xf32>, !pto.vmi.tilereg<1x64xf32>, !pto.vmi.tilereg<1x64xi1>
      -> !pto.vmi.tilereg<1x64xf32>, !pto.vmi.tilereg<1x64xf32>
  ```

- **notes:**
  - `tinterleave` and `tdeinterleave` are inverses:
    `tdeinterleave(tinterleave(a, b))` recovers `(a, b)`.
  - Both are Category A — they do **not** change tilereg layout (parity/half/width
    axes pass through unchanged).
  - These operations use the canonical flat `1xL` form. Use `treshape` before
    and after them when the surrounding value is grouped as `MxN`.
  - Common use cases: real+imaginary interleave, value+index pair manipulation,
    complex number arithmetic.
