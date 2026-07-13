# 1. Load / Store

> **Category:** A (+B on `dintlv`/`unpack`). **Mask:** load none (A5 loads are unpredicated), store `Pg`.
>
> `vload`/`vstore` are logical memory ops. **`[dist_mode]` explicitly declares
> the access pattern**, defaulting to `continuous` (contiguous); the optional
> modes are `unpack` (widening unpack), `dintlv` (deinterleave), and `brc`
> (broadcast).

---

## `pto.vmi.vload`

- **semantics:** Load elements of type `T` from UB into a logical vector
  register starting at `%source + %offset` (element offset). The default
  (`continuous`) is a contiguous stride-1 read:

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = ub[base + offset + i];
  ```

  The access pattern is not always contiguous: depending on the attributes
  (`{dist_mode}`, `{group = C}` with a `stride` operand, or
  `%block_stride`), the load may instead read in a strided/scattered
  fashion (e.g. per-row stride for group mode, 32B-block stride for
  block-stride mode), widen/deinterleave the source, or broadcast. The exact
  pattern is determined by these mutually exclusive attributes (see
  attributes and lowering below).

- **syntax:**
  ```mlir
  %result = pto.vmi.vload %source[%offset] : !pto.ptr<T, ub> -> !pto.vmi.vreg<L×T>
  ```
- **syntax (`dintlv`):**
  ```mlir
  // fused load + deinterleave → 2 results
  %even, %odd = pto.vmi.vload %source[%offset] {dist_mode = "dintlv"}
      : !pto.ptr<T, ub> -> !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>
  ```
- **syntax (`group`):**
  ```mlir
  // strided group load: C rows of L/C elements, row g at base + g*stride
  %result = pto.vmi.vload %source[%offset], %stride {group = C}
      : !pto.ptr<T, ub>, index -> !pto.vmi.vreg<L×T>
  ```
- **syntax (block-stride):**
  ```mlir
  // block-strided load: %block_stride is a dynamic i16 operand (no mask)
  %result = pto.vmi.vload %source[%offset], %block_stride
      : !pto.ptr<T, ub>, i16 -> !pto.vmi.vreg<L×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `source` | `!pto.ptr<T, ub>` | UB base pointer |
  | `offset` | `index` | Element offset from base |
  | `stride` | `index` | Per-row stride (element units); required with `{group}`, invalid otherwise |
  | `block_stride` | `i16` | 32B-block stride between scattered blocks (block-stride mode); mutually exclusive with `{group}` and `{dist_mode}` |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<L×T>` | Loaded logical vector (1 result: `continuous`/`unpack`/`brc`) |
  | `even`, `odd` | two `!pto.vmi.vreg<L×T>` | Deinterleaved pair (2 results, `dintlv` only) |

- **attributes:**

  | Attribute | Values | Default | Description |
  |---|---|---|---|
  | `dist_mode` | `"continuous"`, `"unpack"`, `"dintlv"`, `"brc"` | `"continuous"` | Memory access pattern |
  | `group` | positive integer | *(none)* | Strided group load arity; mutually exclusive with `dist_mode`; requires `stride` |
  | `pmode` | `"zero"`, `"merge"` | `"zero"` | Inactive-lane behavior (applied at consumer, not on load) |

- **lowering to `pto.mi`:**

  | `dist_mode` | Physical lowering |
  |---|---|
  | `"continuous"` | `K × pto.vlds {dist="NORM"}` (element-width-independent `NORM` load) |
  | `"unpack"` | `K × pto.vlds {dist="UNPK_B*"}` (widening unpack; suffix from `Ptr<T>`) |
  | `"dintlv"` | `K × pto.vldsx2 {dist="DINTLV_B*"}` (dual deinterleave load); surface: 2 results `(%even, %odd)`, one per parity half |
  | `"brc"` | `1 × pto.vlds {dist="BRC_B*"}` or `BRC_BLK`; broadcast-axis (1-reg backing, replicate-read) |

  **Group mode** (`{group = C}` + `stride`) has two sub-cases, decided by the
  relation between `result.L` and `C`:
  - **Full-group load** (`result.L > C`): each group loads `L/C` elements,
    row-strided tile load: `C·(L/C) = L` elements across `C` rows,
    each row `g` at offset `base + g·stride`.
  - **Slot load** (`result.L == C`): each group loads **1 scalar** into the
    corresponding slot, producing a compact `V<C×T>`. This is the
    dual of group reduce — reduce
    folds lanes into slots, slot load reads those slots back into a vreg.
    `C ∈ {1, 2, 4, 8}`. Not combinable with `dist_mode`.

  **Block-stride mode** (`%block_stride` operand): 2D-tile block-strided load.
  Memory is read in 32B blocks with block `blk` at
  `base + blk * block_stride` (scattered access); the internal repeat stride
  defaults to 0. `%block_stride` is a dynamic `i16` operand. A5 loads are
  unpredicated, so an implicit all-active mask is applied. Not combinable
  with `dist_mode` or `group`.

  `B*` suffix is derived from `Ptr<T>` element width: `f32/i32 → B32`, `f16/bf16/i16 → B16`, `i8/fp8 → B8`.

- **examples:**

  ```mlir
  // Continuous load (default dist_mode): UB → vreg
  %v = pto.vmi.vload %ub[%offset] : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64×f32>
  // → pto.as: Ptr<f32> → B32, dist_mode=continuous → pto.mi.vlds {dist="NORM"}

  // Slot load: 1 scalar per group → compact V<8×f32> (reads back reduce output)
  %s = pto.vmi.vload %ub[%off], %stride {group = 8}
      : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<8×f32>

  // Full-group load: 8 rows × 8 elements, stride 64
  %t = pto.vmi.vload %ub[%off], %stride {group = 8}
      : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<64×f32>

  // Block-strided load: block_stride = 8 (dynamic i16 operand, no mask)
  %vb = pto.vmi.vload %ub[%off], %c8_i16
      : !pto.ptr<f32, ub>, i16 -> !pto.vmi.vreg<64×f32>

  // Broadcast load: scalar/block replicate into vreg
  %vb = pto.vmi.vload %ub[%offset] {dist_mode = "brc"} : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64×f32>

  // Widening unpack load: narrow source expanded to wide lanes
  %u = pto.vmi.vload %ub[%offset] {dist_mode = "unpack"} : !pto.ptr<bf16, ub> -> !pto.vmi.vreg<64×f32>

  // Deinterleave load: fused load + deinterleave, 2 surface results
  %even, %odd = pto.vmi.vload %ub[%offset] {dist_mode = "dintlv"}
      : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64×f32>, !pto.vmi.vreg<64×f32>
  ```

- **notes:**
  - **A5 loads are unpredicated.** A tail mask associated with a `vload` is
    never lowered as a masked load. It migrates to the consuming compute op or
    to a `vstore`.
  - `dist_mode` and layout inference are orthogonal: even with `dist_mode="continuous"`,
    `pto.as` may lower to `DINTLV_B*` to serve a downstream grouped reduce.
  - The `pmode` attribute on `vload` governs the result lane behavior at the
    *consumer*, not on the load itself.

- **attention:**
  - **Result count must match the access mode.** `dist_mode = "dintlv"` is a
    fused load + deinterleave and produces **two** results `(%even, %odd)`;
    all other `dist_mode` values, `{group}`, and `%block_stride` produce
    **one** result. If the written result count does not match the selected
    mode (e.g. a single result with `dintlv`, or two results with
    `continuous`), `pto.as` rejects the op.
  - **`{group}`, `%block_stride`, and `{dist_mode}` are mutually exclusive.**
    Specifying more than one at once is rejected by `pto.as`.
  - **`stride` operand is bound to `{group}`.** It is required with
    `{group = C}` and invalid otherwise; `block_stride` is bound to the
    block-stride mode and invalid otherwise. `vload` has no mask operand in
    any mode (A5 loads are unpredicated).

---

## `pto.vmi.vstore`

- **semantics:** Store elements from a vector register to UB starting at
  `%dest + %offset` (element offset). The default (`continuous`) is a
  contiguous stride-1 write; only lanes where `mask[i] != 0` are written
  (A5 stores are predicated):

  ```c
  for (int i = 0; i < L; i++)
      if (mask[i])
          ub[base + offset + i] = src[i];
  ```

  The access pattern is not always contiguous: depending on the attributes
  (`{dist_mode}`, `{group = C}` with a `stride` operand, or
  `%block_stride`), the store may instead write in a strided/scattered
  fashion (e.g. per-row stride for group mode, 32B-block stride for
  block-stride mode) or interleave the values. The exact pattern is
  determined by these mutually exclusive attributes (see attributes and
  lowering below).

- **syntax:**
  ```mlir
  pto.vmi.vstore %value, %dest[%offset], %mask : !pto.vmi.vreg<L×T>, !pto.ptr<T, ub>, !pto.vmi.mask<L>
  ```
- **syntax (`dintlv`):**
  ```mlir
  // fused interleave + store → 2 values
  pto.vmi.vstore %even, %odd, %dest[%offset], %mask {dist_mode = "dintlv"}
      : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, !pto.ptr<T, ub>, !pto.vmi.mask<L>
  ```
- **syntax (`group`):**
  ```mlir
  // strided group store: C rows of L/C elements, row g at base + g*stride (no mask)
  pto.vmi.vstore %value, %dest[%offset], %stride {group = C}
      : !pto.vmi.vreg<L×T>, !pto.ptr<T, ub>, index
  ```
- **syntax (block-stride):**
  ```mlir
  // block-strided store: %block_stride is a dynamic i16 operand (mask required)
  pto.vmi.vstore %value, %dest[%offset], %block_stride, %mask
      : !pto.vmi.vreg<L×T>, !pto.ptr<T, ub>, i16, !pto.vmi.mask<L>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `value` | `!pto.vmi.vreg<L×T>` | Vector value to store (1 value, `continuous`) |
  | `even`, `odd` | two `!pto.vmi.vreg<L×T>` | Interleaved pair to store (`dintlv` only) |
  | `dest` | `!pto.ptr<T, ub>` | UB destination base pointer |
  | `offset` | `index` | Element offset from base |
  | `stride` | `index` | Per-row stride (element units); required with `{group}`, invalid otherwise |
  | `block_stride` | `i16` | 32B-block stride between scattered blocks (block-stride mode); mutually exclusive with `{group}` and `{dist_mode}` |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate (variadic: 0 or 1) |

- **results:** *(none)*

- **attributes:**

  | Attribute | Values | Default | Description |
  |---|---|---|---|
  | `dist_mode` | `"continuous"`, `"dintlv"` | `"continuous"` | Memory access pattern |
  | `group` | positive integer | *(none)* | Strided group store arity; mutually exclusive with `dist_mode`; requires `stride`; forbids `mask` |
  | `pmode` | `"zero"`, `"merge"` | `"zero"` | Inactive-lane behavior: `"zero"` (default) stores 0; `"merge"` skips write on inactive lanes |

- **lowering to `pto.mi`:**

  | `dist_mode` | Physical lowering |
  |---|---|
  | `"continuous"` | `K × pto.vsts {dist="NORM_B*"}` |
  | `"dintlv"` | `K × pto.vstsx2 {dist="INTLV_B*"}`; surface consumes 2 inputs `(%even, %odd)`, interleaved at lowering |

  **Group mode** (`{group = C}` + `stride`): row-strided tile store. Not combinable with
  `dist_mode` or `mask` (group stores are unpredicated).

  **Block-stride mode** (`%block_stride` operand): 2D-tile block-strided store.
  Memory is written in 32B blocks with block `blk` at
  `base + blk * block_stride` (scattered access); the internal repeat stride
  defaults to 0. `%block_stride` is a dynamic `i16` operand. An explicit
  `mask` is applied; if absent an implicit all-active mask is used. Not
  combinable with `dist_mode` or `group`.

- **examples:**

  ```mlir
  // Continuous store (default): vreg → UB, masked
  pto.vmi.vstore %v, %ub_out[%offset], %mask : !pto.vmi.vreg<64×f32>, !pto.ptr<f32, ub>, !pto.vmi.mask<64>

  // Interleave store: fused interleave + store, 2 surface inputs
  pto.vmi.vstore %even, %odd, %ub_out[%offset], %mask {dist_mode = "dintlv"}
      : !pto.vmi.vreg<64×f32>, !pto.vmi.vreg<64×f32>, !pto.ptr<f32, ub>, !pto.vmi.mask<64>

  // Group (strided) store: 8 rows × 8 elements, stride 64 (no mask)
  pto.vmi.vstore %tile, %ub_out[%off], %stride {group = 8}
      : !pto.vmi.vreg<64×f32>, !pto.ptr<f32, ub>, index

  // Block-strided store: block_stride = 8 (dynamic i16 operand + mask)
  pto.vmi.vstore %v, %ub_out[%off], %c8_i16, %mask
      : !pto.vmi.vreg<64×f32>, !pto.ptr<f32, ub>, i16, !pto.vmi.mask<64>
  ```
