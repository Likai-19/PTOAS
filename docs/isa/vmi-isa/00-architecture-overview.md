# VMI Architecture Overview

> **Status:** draft. This document covers the architecture and foundational concepts
> of the unified `pto.vmi` instruction surface. Per-op reference docs are in the
> numbered group files that follow.

`pto.vmi` sits between high-level programming models (TileLang, pto-dsl) and
the physical `pto.mi` ISA. It exposes **logically contiguous vectors** and
**elementwise compute intent**; the physical SIMD register layout (interleave,
parity, width, part, pack, dist tokens) is held and propagated by `pto.as` and
is invisible to the user.

```
TileLang  T.parallel(N) { C[i] = cast<i32>(A[i]) + B[i] }
   │  (direct translation, elementwise semantics preserved)
   ▼
pto.vmi   %w = pto.vmi.tcvt %a; %c = pto.vmi.tadd %w, %b
   │  (pto.as: layout-assignment + lowering)
   ▼
pto.mi    vcvt EVEN/ODD + two-way vadd + vstsx2 INTLV_B32
```

- **Upper → vmi**: `T.parallel`'s logical iteration space translates directly
  to `pto.vmi` logical vector ops — elementwise → Category A op, `T.cast` →
  a `tcvt` with no explicit `part`, logical length `N` →
  `!pto.vmi.tilereg<1xNxT>`, "all active" → auto-generated tail predicate.
- **vmi → pto.mi**: `pto.as` performs layout inference + unification +
  materialization, lowering logical vectors to concrete `pto.mi` instructions
  (including `part/pack/interleave/dist`). At `K=1` this degenerates to
  zero-overhead pass-through.

---

## Logical vs Physical

A `pto.vmi` value is **logical** — a row-major `MxN` tile of type `T`, with
`L = M * N` logical lanes. A former one-dimensional vector uses the canonical
shape `1xL`; a grouped vector uses `MxN`, where each row is one group. Its
physical backing is `K` hardware vector registers (256B / 2048-bit each):

```
K = ⌈ L · bitwidth(T) / 2048 ⌉
```

At `K=1` and full-width (no partial lanes), one `pto.vmi.tilereg` maps 1:1 to
one `pto.vreg`. At `K>1`, the logical value fans out across `K` physical
registers with a layout descriptor (`#pto.vmi.layout`) tracking the mapping.

**Physical constants (A5 vector pipe):**

```
vector register file : 32 architectural vregs, 256 B (2048 bit) each
predicate file       : 8  architectural pregs, 256 bit each, 1 bit controls 1 byte
VLane                : 32 B sub-lane; 8 VLanes per vreg
E_v = 32 / sizeof(T) : lanes per VLane     (f32 → 8, f16/bf16 → 16, i8 → 32)
```

---

## Type System

### `!pto.vmi.tilereg<MxNxT>`

Logical tile register. `M` is the row/group count, `N` is the per-row lane
count, and `T` is the element type. Elements follow row-major order, so the
former lane `i` is addressed as `[i / N, i % N]`. Use `1xL` for an ordinary
one-dimensional value and `MxN` for a grouped value.

| T | bits | E_v (lanes per physical vreg) | Legal logical length |
|---|---|---|---|
| `f32` / `i32` / `ui32` / `si32` | 32 | 64 | `L = M * N > 0` |
| `f16` / `bf16` / `i16` / `ui16` / `si16` | 16 | 128 | `L = M * N > 0` |
| `i8` / `ui8` / `si8` / `fp8_e4m3` / `fp8_e5m2` | 8 | 256 | `L = M * N > 0` |

- **Full vector**: `L · bitwidth(T) == N · 2048` (integer multiple of 256B).
- **Compact/partial vector**: `L · bitwidth(T) < 2048` — still backed by one
  physical vreg (256B); only the low `L` logical slots are valid. Physical
  slots outside the logical value are `pad/undef` and must be masked out.

**Common logical ↔ physical mappings:**

| Logical type | Byte size | K | Physical vregs | Valid slots per vreg |
|---|---:|---:|---:|---|
| `tilereg<1x256xf32>` | 1024B | 4 | 4 | 64 f32 each, all valid |
| `tilereg<1x256xf16>` | 512B | 2 | 2 | 128 f16 each, all valid |
| `tilereg<1x256xi8>` | 256B | 1 | 1 | 256 i8, all valid |
| `tilereg<1x128xf32>` | 512B | 2 | 2 | 64 f32 each, all valid |
| `tilereg<1x64xf16>` | 128B | 1 | 1 | low 64 f16 valid |
| `tilereg<1x64xi8>` | 64B | 1 | 1 | low 64 i8 valid |

See the [layout assignment design](../../designs/vmi-layout-assignment-lowering-design.md) for detailed physical layout
diagrams (contiguous, parity EVEN/ODD, sub-part, stride-4 interleave) for each
logical type.

### Mask tile registers

A virtual predicate mask is represented by `!pto.vmi.tilereg<MxNxi1>` rather
than a separate mask type. Each logical mask lane corresponds to one data lane;
the mask and governed data tile must have exactly the same `MxN` shape.

### `pto.vmi.treshape`

`treshape` converts between one-dimensional and grouped views and also replaces
the former `vinterpret_cast`. It preserves the row-major bit sequence and
requires equal total bit size:

```text
Msrc * Nsrc * bitwidth(Tsrc) == Mdst * Ndst * bitwidth(Tdst)
```

Shape-only conversion keeps the dtype unchanged. A dtype-changing form is a
bit reinterpretation, not a numeric conversion; use `tcvt` for numeric casts.

---

## Category A / B / C

Every VMI op belongs to one of three lowering categories that determine how
`pto.as` handles its physical layout:

| Category | Layout relationship | `pto.as` behavior | Output layout |
|---|---|---|---|
| **A — Layout-passthrough** | Does not modify register layout | Fan-out: emit the same `pto.mi` op once per physical reg (`K × op`); mask follows per-reg (with `ppack`/`punpack` as needed) | Unchanged: preserves input parity/half/sub-part layout |
| **B — Layout-rewritable** | Modifies layout predictably | Fan-out along other axes; instantiate matching modes (`PART_EVEN/ODD`, `Bin_N0/N1`, `PK`/`UNPK`, `INTLV`/`DINTLV`) | Rewritten to the op's natural output layout |
| **C — Contiguous-required** | Requires stride-1 contiguous input (no in-place mode satisfies it) | `pto.as` inserts `.contiguous()` materialization (store+reload or explicit repack) before the op | Flattened contiguous chunk (`is_contiguous`) |

> **C-class note:** C-class ops cannot tolerate a non-contiguous physical
> layout — any parity/half/sub-part arrangement must first be materialized to
> contiguous before the op runs. `pto.as` therefore treats a C-class op as a
> **layout barrier**: upstream A/B ops may keep their compact layout right up to
> the C-class boundary, where a `.contiguous()` is forced.

---

## Mask & Predication (`pmode`)

All compute ops accept an optional governing mask operand `[pmode]`. The mask
is a `!pto.vmi.tilereg<MxNxi1>` with the same `MxN` shape as the data operand.

**`pmode` values:**

| `pmode` | Inactive lane behavior | Default? |
|---|---|---|
| `"zero"` | Inactive lanes produce 0 (hardware-native ZEROING) | ✓ (default) |
| `"merge"` | Inactive lanes preserve the destination's prior value | |

On A5, MERGE is **emulated**: the hardware predicates only in ZEROING mode, so the
compiler synthesizes merge as a predicate complement plus a `tor`/`tsel` blend
of the zeroed result with the old destination (see [Appendix C](10-appendices.md)).
On A6, some ops support native MERGE.

**A5 load restriction**: `tload` has **no** mask operand — A5 loads are
unpredicated. A logical tail mask associated with a load is never lowered as a
"masked load"; `pto.as` migrates it to the consuming compute op, the store, or
shortens the load length. `tstore` **is** predicated on A5.

---

## Group Shape

The canonical tile-register surface encodes the former `group=C` information
in a two-dimensional shape. `M=C` is the number of groups and `N=L/C` is the
per-group lane count:

- **Reduce**: consumes `tilereg<MxNxT>` and produces `tilereg<Mx1xT>`.
- **Broadcast**: consumes `tilereg<Mx1xT>` and fans each scalar back across
  `N` lanes, producing `tilereg<MxNxT>`.

The legal shapes follow the current VMI rule: `M > 0`, `N > 0`, and the former
logical length is `L = M * N`, so the group count evenly divides `L`. Use
`treshape` to convert `tilereg<1xLxT>` to `tilereg<MxNxT>` and back.

**`group → Category` decision table** (W = bytes per sub-group):

| W vs BlockLane (32B) | Category | Lowering |
|---|---|---|
| `W == 32B` (sub-group = 1 VLane) | B | `vcgadd`/`vcgmax`/`vcgmin` — one op per reg, no cross-reg combine |
| `W > 32B`, aligned | B | Fold `(k-1)× vadd/vmax/vmin` then `vcg*` |
| Unaligned | C | Materialize → contiguous → reduce |

---

## Group Index

| # | Group | Ops | Category | Mask |
|---|---|---|---|---|
| 1 | **Load / Store** | `tload`, `tstore` | A (+B on dintlv/unpack) | load: none; store: `Pg` |
| 2 | **Index-gen** | `tci` | A | none |
| 3 | **Eltwise Compute** | `tadd`, `tsub`, `tmul`, `tdiv`, `tmax`, `tmin`, `tabs`, `tneg`, `trelu`, `texp`, `tlog`, `tsqrt`, `tand`, `tor`, `txor`, `tnot`, `tshl`, `tshr`, `tadds`, `tmuls`, `tmaxs`, `tmins`, `tshls`, `tshrs`, `tcmp`, `tcmps`, `tsel`, `vselr` | A | `Pg` (except `vselr`: none) |
| 4 | **Broadcast** | `texpands`, `trowexpand` | A (scalar) / B (row) | none |
| 5 | **Reduce** | `trowsum`, `trowmax`, `trowmin` | B (VLane-aligned) / C (unaligned) | `Pg req` |
| 6 | **Convert** | `tcvt`, `treshape` | B / A | `Pg` / none |
| 7 | **SFU** | `vexpdif`, `taxpy`, `tlrelu`, `tprelu`, widening `tmul`, `vmula`, `thistogram`, `dhist`, `tgather`, `tgatherb`, `tscatter` | A (fused) / B (widening multiply, histogram) / C (gather/scatter) | `Pg` |
| 8 | **Predicate Ops** | `create_mask`, `create_group_mask` (no direct same-semantic Tile Op) | gen | gen |
| 9 | **Data Rearrange** | `tinterleave`, `tdeinterleave` | A | `Pg` |
