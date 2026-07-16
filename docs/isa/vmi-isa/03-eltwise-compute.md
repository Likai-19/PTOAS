# 3. Elementwise Compute

This chapter uses the Tile Op spellings for VMI operations with identical
elementwise meaning. Every data-tile operand and result has the same legal
`MxN` shape. A governing mask, where present, is
`!pto.vmi.tilereg<MxNxi1>`.

## Binary arithmetic

| VMI op | Semantics |
|---|---|
| `pto.vmi.tadd` | `dst[m,n] = lhs[m,n] + rhs[m,n]` |
| `pto.vmi.tsub` | `dst[m,n] = lhs[m,n] - rhs[m,n]` |
| `pto.vmi.tmul` | `dst[m,n] = lhs[m,n] * rhs[m,n]` |
| `pto.vmi.tdiv` | `dst[m,n] = lhs[m,n] / rhs[m,n]` |
| `pto.vmi.tmax` | `dst[m,n] = max(lhs[m,n], rhs[m,n])` |
| `pto.vmi.tmin` | `dst[m,n] = min(lhs[m,n], rhs[m,n])` |

`tmul` has both a homogeneous form and the widening integer form formerly
spelled `vmull`:

```mlir
%wide = pto.vmi.tmul %lhs, %rhs, %mask
    : !pto.vmi.tilereg<MxNxi32>, !pto.vmi.tilereg<MxNxi32>,
      !pto.vmi.tilereg<MxNxi1> -> !pto.vmi.tilereg<MxNxi64>
```

The widening form preserves `MxN`, requires two identical 32-bit integer input
types, and produces the corresponding signed or unsigned 64-bit integer dtype.
The result signedness must match the inputs.

```mlir
%result = pto.vmi.tadd %lhs, %rhs, %mask {pmode = "zero"}
    : !pto.vmi.tilereg<MxNxT>, !pto.vmi.tilereg<MxNxT>,
      !pto.vmi.tilereg<MxNxi1> -> !pto.vmi.tilereg<MxNxT>
```

`tdiv` is floating-point only. Apart from the documented widening `tmul`
overload, arithmetic operations preserve dtype and accept the data types
supported by their Tile Op counterpart. With `pmode = "zero"`, an inactive
mask position returns zero; with `pmode = "merge"`, it retains the prior
destination value.

## Unary arithmetic and activation

| VMI op | Semantics |
|---|---|
| `pto.vmi.tabs` | `abs(src[m,n])` |
| `pto.vmi.tneg` | `-src[m,n]` |
| `pto.vmi.trelu` | `max(0, src[m,n])` |
| `pto.vmi.texp` | `exp(src[m,n])` |
| `pto.vmi.tlog` | `ln(src[m,n])` |
| `pto.vmi.tsqrt` | `sqrt(src[m,n])` |

```mlir
%result = pto.vmi.texp %src, %mask
    : !pto.vmi.tilereg<MxNxf32>, !pto.vmi.tilereg<MxNxi1>
      -> !pto.vmi.tilereg<MxNxf32>
```

## Bitwise and shift operations

| VMI op | Semantics |
|---|---|
| `pto.vmi.tand`, `pto.vmi.tor`, `pto.vmi.txor` | elementwise AND, OR, XOR |
| `pto.vmi.tnot` | elementwise NOT |
| `pto.vmi.tshl`, `pto.vmi.tshr` | elementwise left / unsigned-right shift |

These operations require integer data tiles. They use the same-shaped optional
`i1` governing mask as arithmetic operations.

## Tile-scalar operations

| VMI op | Semantics |
|---|---|
| `pto.vmi.tadds`, `pto.vmi.tmuls` | add / multiply a scalar at every tile position |
| `pto.vmi.tmaxs`, `pto.vmi.tmins` | maximum / minimum with a scalar |
| `pto.vmi.tshls`, `pto.vmi.tshrs` | left / unsigned-right shift by a scalar |

```mlir
%scaled = pto.vmi.tmuls %src, %scale, %mask
    : !pto.vmi.tilereg<8x16xf32>, f32, !pto.vmi.tilereg<8x16xi1>
      -> !pto.vmi.tilereg<8x16xf32>
```

## Compare, selection, and permutation

`tcmp`, `tcmps`, and `tsel` use the corresponding Tile Op names. `vselr` has
no same-function Tile Op and retains its VMI name.

```mlir
%pred = pto.vmi.tcmp %lhs, %rhs, %seed {cmp = "lt"}
    : !pto.vmi.tilereg<MxNxT>, !pto.vmi.tilereg<MxNxT>,
      !pto.vmi.tilereg<MxNxi1> -> !pto.vmi.tilereg<MxNxi1>

%scalar_pred = pto.vmi.tcmps %lhs, %rhs_scalar, %seed {cmp = "lt"}
    : !pto.vmi.tilereg<MxNxT>, T,
      !pto.vmi.tilereg<MxNxi1> -> !pto.vmi.tilereg<MxNxi1>

%selected = pto.vmi.tsel %pred, %when_true, %when_false
    : !pto.vmi.tilereg<MxNxi1>, !pto.vmi.tilereg<MxNxT>,
      !pto.vmi.tilereg<MxNxT> -> !pto.vmi.tilereg<MxNxT>

%permuted = pto.vmi.vselr %source, %index
    : !pto.vmi.tilereg<1xLxT>, !pto.vmi.tilereg<1xLxI>
      -> !pto.vmi.tilereg<1xLxT>
```

`tcmp` and `tcmps` produce `i1` tiles. `vselr` remains a one-dimensional VMI
register permutation; apply `treshape` to a two-dimensional value first if
needed.
