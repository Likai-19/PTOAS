# VMI ISA Architecture Overview

> **Status:** draft design. This chapter defines the canonical VMI surface.
> The remaining chapters specify each instruction family.

VMI expresses vector-pipeline computation with tile-register values.  Unlike a
flat virtual vector, a tile register makes row boundaries visible in the IR.
That lets the same operation vocabulary and the same row-reduction / expansion
semantics be used by VMI and Tile Op.

VMI values are register-resident and have no memory placement, valid-region, or
buffer-layout attributes. `!pto.vmi.tilereg` is therefore distinct from
`!pto.tile_buf`: the former is a value in the vector pipeline; the latter is an
on-chip buffer handle.

## Type System

### `!pto.vmi.tilereg<MxNxdtype>`

`tilereg` is the only VMI aggregate type. `M` and `N` are positive static
dimensions and `dtype` is a VMI data element type. The current VMI logical
data element types are the supported 8-bit, 16-bit, and 32-bit integer,
floating-point, index, and PTO low-precision types; individual operations may
accept a narrower subset or define a wider result.

```mlir
!pto.vmi.tilereg<1x128xf16>  // one-dimensional, 128 logical lanes
!pto.vmi.tilereg<8x16xf16>  // eight groups, 16 lanes per group
!pto.vmi.tilereg<8x1xf32>   // one reduced scalar per group
```

- A **one-dimensional tile register** has the canonical form
  `!pto.vmi.tilereg<1xLxdtype>`. It replaces the former logical vector of
  length `L`.
- A **two-dimensional tile register** has the form
  `!pto.vmi.tilereg<MxNxdtype>`. Elements use row-major logical indexing:
  `tile[m, n]` is logical lane `m * N + n` in the corresponding flattened
  vector.
- Elementwise operations preserve the complete `MxN` shape. Their data
  operands and result must have identical shape and dtype, except where the
  operation explicitly documents a scalar or a conversion result.

### Mask tile registers

A predicate is also a tile register, with `i1` as its element type:

```mlir
!pto.vmi.tilereg<1x128xi1>
!pto.vmi.tilereg<8x16xi1>
```

A governing mask must have the same `MxN` shape as the data tile it governs.
Comparison and mask-creation instructions return the same-shaped `i1` tile.
There is no separate VMI mask type and no predicate-granularity suffix.

## Legal Shapes

The current VMI logical vector-length contract is preserved as explicit tile
shapes. The existing logical vector type accepts every positive lane count
`L`; it is not restricted to one physical 256-byte register. Therefore a tile
register is type-legal when both dimensions are positive. Its flattened lane
count is `L = M * N`. The matching `i1` mask tile has the same logical shape.

| Form | Legal shape | Meaning |
|---|---|---|
| Ordinary 1-D data or mask | `1xL`, `L > 0` | The former `vreg<LxT>` or `mask<Lx...>` logical lane sequence. |
| Ordinary 2-D data or mask | `MxN`, `M > 0`, `N > 0` | A row-major view of `L = M * N` logical lanes. |
| Grouped data or mask | `MxN`, `M > 0`, `N > 0` | Replaces `num_groups = M` over a length-`L` VMI value; `M` divides `L` by construction and each row contains `N = L / M` lanes. |
| Group scalar carrier | `Mx1`, `M > 0` | One logical result lane per group, used by row reduction, group-slot load, row expansion, and group-slot store. |

For example, the currently used `vreg<128xf32>` with `group = 8` becomes
`tilereg<8x16xf32>`, while `vreg<512xf16>` with `group = 2` becomes
`tilereg<2x256xf16>`. Shapes such as `6x64xf32` are also type-legal when the
corresponding current VMI operation accepts `L = 384` and `group = 6`.
Operation-specific dtype, shape, and target restrictions still apply.

## `pto.vmi.treshape`

`treshape` is the sole view-conversion operation for tile registers. It can
change the `MxN` shape, the element type, or both, but it preserves the same
row-major bit sequence.

```mlir
%rows = pto.vmi.treshape %flat
    : !pto.vmi.tilereg<1x128xf32> -> !pto.vmi.tilereg<8x16xf32>
%flat_again = pto.vmi.treshape %rows
    : !pto.vmi.tilereg<8x16xf32> -> !pto.vmi.tilereg<1x128xf32>
%bits = pto.vmi.treshape %rows
    : !pto.vmi.tilereg<8x16xf32> -> !pto.vmi.tilereg<8x16xi32>
```

The source and result must have the same total storage size:

```text
Msrc * Nsrc * bitwidth(Tsrc) == Mdst * Ndst * bitwidth(Tdst)
```

With the same dtype, this requires the same total element count and provides
the one-dimensional/two-dimensional reshape required by VMI. With a changed
dtype, `treshape` replaces the former `vinterpret_cast`; it is a bitwise
reinterpretation rather than a numeric conversion. Use `tcvt` when element
values must be converted. A predicate tile may only reshape to another `i1`
tile with the same element count; the dtype-changing form applies to data
tiles. The name and storage-size rule align with Tile Op `treshape`.

## Group Semantics

The `group` attribute is removed from the canonical VMI surface. A two-
dimensional tile describes grouping directly:

- `M` is the number of groups (rows).
- `N` is the number of lanes in each group (columns).
- A row reduction consumes `MxN` and returns `Mx1`.
- A row expansion consumes `Mx1` and returns `MxN`.
- A grouped load, store, or mask uses the same `MxN` shape. Its stride is the
  distance between successive rows in elements.

Use `pto.vmi.treshape` before a grouped instruction and after it when a
one-dimensional consumer or producer is required.

## VMI Names Aligned with Tile Op

For functionality that exists in Tile Op, VMI uses the Tile Op spelling under
the `pto.vmi` namespace. The following table records the renames used by this
design, including the legacy VMI spellings still present in the current
surface.

| Current VMI spelling | Canonical VMI spelling | Tile Op counterpart |
|---|---|---|
| `vload`, `vstore` | `tload`, `tstore` | `tload`, `tstore` |
| `vadd`, `vsub`, `vmul`, `vdiv`, `vmax`, `vmin` | `tadd`, `tsub`, `tmul`, `tdiv`, `tmax`, `tmin` | same |
| `vmull` | `tmul` with a 64-bit result dtype | `tmul` |
| `vabs`, `vneg`, `vrelu`, `vexp`, `vln`, `vsqrt` | `tabs`, `tneg`, `trelu`, `texp`, `tlog`, `tsqrt` | same |
| `vand`, `vor`, `vxor`, `vnot`, `vshl`, `vshr` | `tand`, `tor`, `txor`, `tnot`, `tshl`, `tshr` | same |
| `vadds`, `vmuls`, `vmaxs`, `vmins`, `vshls`, `vshrs` | `tadds`, `tmuls`, `tmaxs`, `tmins`, `tshls`, `tshrs` | same |
| `vsel` | `tsel` | `tsel` |
| `vbrc` (scalar) | `texpands` | `texpands` |
| `vbrc` (grouped) | `trowexpand` | `trowexpand` |
| `vcadd`, `vcmax`, `vcmin` | `trowsum`, `trowmax`, `trowmin` | same |
| `vcvt` | `tcvt` | `tcvt` |
| `bitcast`, `vinterpret_cast`, and the shape-only view form | `treshape` | `treshape` |
| `iota`, `vci` | `tci` | `tci` |
| `cmpf`, `cmpi`, `vcmp` | `tcmp` | `tcmp` |
| `vcmps` | `tcmps` | `tcmps` |
| `vexpdif` | `vexpdif` | No direct one-op counterpart; a constrained row-splat case may use `trowexpandexpdif` |
| `vlrelu` | `tlrelu` | `tlrelu` |
| `vaxpy`, `vprelu` | `taxpy`, `tprelu` | `taxpy`, `tprelu` |
| `vhist` | `thistogram` | `thistogram` |
| `vgather`, `vgatherb`, `vscatter` | `tgather`, `tgatherb`, `tscatter` | same |
| `create_mask`, `create_group_mask` | predicate `texpands` | `texpands` |
| `vintlv`, `vdintlv` | `tinterleave`, `tdeinterleave` | same |

## VMI-only Instructions (No Tile Op Rename)

The following instructions have no Tile Op with the same functionality. Their
VMI names are retained; only their operands and results change to `tilereg`:

| Family | Retained VMI instruction |
|---|---|
| Register permutation | `vselr` |
| Lane-wise fused exponent difference | `vexpdif` |
| Fused tile-tile multiply-accumulate | `vmula` |
| Full 256-bin distribution histogram | `dhist` |

`vexpdif` is not renamed to `trowexpandexpdif`. VMI `vexpdif` accepts a
same-shaped `max` tile and computes a lane-wise exponent difference, while
Tile Op `trowexpandexpdif` consumes one scalar value per row. The two
operations overlap only when the `max` tile is known to be a row splat and the
remaining dtype and predication restrictions also match.

`dhist` is not renamed to `thistogram`. `dhist` reduces one masked
`1xLxui8` source tile into a single `1x256xui16` distribution accumulator,
whereas Tile Op `thistogram` has row-wise source, index, and destination
update semantics. The two operations therefore do not have the same semantic
contract.

## Predication

All predicated compute, reduction, store, and special instructions consume a
same-shaped `tilereg<MxNxi1>` mask. `pmode = "zero"` remains the default;
`pmode = "merge"` retains the prior destination value at inactive positions.
Loads and unpredicated rearrangement operations do not take a mask.

## Chapter Index

| Chapter | Subject |
|---|---|
| [1](01-load-store.md) | Tile-register load and store |
| [2](02-index-gen.md) | Index generation |
| [3](03-eltwise-compute.md) | Elementwise arithmetic, bitwise, compare, and select |
| [4](04-broadcast.md) | Scalar and row expansion |
| [5](05-reduce.md) | Row reductions |
| [6](06-convert.md) | Type conversion and reinterpretation |
| [7](07-sfu.md) | Fused and special computation |
| [8](08-predicate-ops.md) | `i1` tile construction and boolean operations |
| [9](09-data-rearrange.md) | Register rearrangement |
| [10](10-appendices.md) | Canonical instruction index and examples |
