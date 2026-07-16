# 7. Fused and Special Computation

This chapter separates VMI instructions that now use Tile Op names from VMI
instructions that have no same-function Tile Op and therefore retain their
existing names.

## Tile Op-aligned operations

### `pto.vmi.trowexpandexpdif`

`trowexpandexpdif` replaces `vexpdif`. Its row-scalar operand is explicit:

```mlir
%result = pto.vmi.trowexpandexpdif %source, %row_max, %mask
    : !pto.vmi.tilereg<MxNxf32>, !pto.vmi.tilereg<Mx1xf32>,
      !pto.vmi.tilereg<MxNxi1> -> !pto.vmi.tilereg<MxNxf32>
```

It computes `exp(source[m,n] - row_max[m,0])` at active positions. See
[Broadcast and Row Expansion](04-broadcast.md) for the complete group-shape
contract.

### `pto.vmi.tlrelu`

`tlrelu` replaces `vlrelu` and has the Tile Op leaky-ReLU name.

```mlir
%result = pto.vmi.tlrelu %source, %slope, %mask
    : !pto.vmi.tilereg<MxNxT>, T, !pto.vmi.tilereg<MxNxi1>
      -> !pto.vmi.tilereg<MxNxT>
```

At every active position, the result is `source[m,n]` when the source is
positive, otherwise `slope * source[m,n]`. It preserves the source shape.

### Other Tile Op-aligned operations

The following VMI instructions also have same-function Tile Ops and therefore
use their Tile Op names. All vector and mask operands use `tilereg` shapes.

| Instruction | Tile-register contract |
|---|---|
| `pto.vmi.taxpy` | `MxN` input, accumulator, result, and optional `MxNxi1` mask; computes `alpha * x + acc`. |
| `pto.vmi.tprelu` | `MxN` source, slope tile, result, and optional `MxNxi1` mask. |
| `pto.vmi.thistogram` | Replaces `vhist`; its bin-index, mask, and result are one-dimensional `1xL` tiles with operation-specific element types. |
| `pto.vmi.tgather` | Replaces `vgather`; a UB source and `1xL` offset tile produce a `1xL` result governed by a `1xLxi1` mask. |
| `pto.vmi.tgatherb` | Replaces `vgatherb`; the `1xK` byte-offset tile may have a different lane count from the `1xL` result and its `1xLxi1` mask. |
| `pto.vmi.tscatter` | Replaces `vscatter`; its value, offset, and mask tiles are matching one-dimensional `1xL` shapes. |

For each instruction, operands that describe the same logical lanes have
identical shapes. `tgatherb` is the documented exception because its byte-
offset count may differ from the result lane count. The existing datatype and
target restrictions remain in force.

```mlir
%y = pto.vmi.taxpy %x, %acc, %alpha, %mask
    : !pto.vmi.tilereg<1x128xf32>, !pto.vmi.tilereg<1x128xf32>, f32,
      !pto.vmi.tilereg<1x128xi1> -> !pto.vmi.tilereg<1x128xf32>
```

## VMI-only fused and special instructions

`dhist` and `vmula` have no single same-function Tile Op, so their names are
retained. The former `vmull` is documented as the widening `tmul` overload in
[Elementwise Compute](03-eltwise-compute.md).

| Instruction | Tile-register contract |
|---|---|
| `pto.vmi.dhist` | Full 256-bin distribution histogram. The accumulator and result are `1x256xui16`; the source is `1xLxui8`; the governing mask is `1xLxi1`. |
| `pto.vmi.vmula` | Same-shaped accumulator, two tile operands, result, and mask. It is not scalar `taxpy`. |

### `pto.vmi.dhist`

```mlir
%hist = pto.vmi.dhist %acc, %source, %mask
    : !pto.vmi.tilereg<1x256xui16>, !pto.vmi.tilereg<1xLxui8>,
      !pto.vmi.tilereg<1xLxi1> -> !pto.vmi.tilereg<1x256xui16>
```

`L` is positive. The result starts with the 256 accumulator bins. Each active
source lane then increments the bin selected by its unsigned 8-bit value:

```text
result[0, b] = acc[0, b]                         for b in [0, 256)
result[0, source[0, n]] += 1                     when mask[0, n] is true
```

This is one complete distribution over all active source lanes. It is not
renamed to Tile Op `thistogram`, whose row-wise source/index/destination
update contract and element-type constraints describe a different operation.
