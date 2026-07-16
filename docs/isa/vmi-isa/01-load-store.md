# 1. Tile-Register Load and Store

`pto.vmi.tload` and `pto.vmi.tstore` are the VMI register-memory operations.
They use the Tile Op names but move values between a UB pointer and a
register-resident `tilereg`; they do not allocate or DMA a `tile_buf`.

## `pto.vmi.tload`

```mlir
%value = pto.vmi.tload %source[%offset]
    : !pto.ptr<T, ub> -> !pto.vmi.tilereg<1xLxT>
```

`tload` reads `L` contiguous elements beginning at `source[offset]` into a
one-dimensional tile register. The result must satisfy the ordinary 1-D shape
rule in the [architecture overview](00-architecture-overview.md#legal-shapes).

### Grouped load

```mlir
%value = pto.vmi.tload %source[%offset], %row_stride
    : !pto.ptr<T, ub>, index -> !pto.vmi.tilereg<MxNxT>
```

For a grouped result (`M > 1`), `tload` reads row `m` from
`source[offset + m * row_stride]`. `row_stride` is measured in elements. The
result shape must be a legal grouped `MxN` shape, and every row contains `N`
contiguous elements.

```text
for m in 0 .. M:
  for n in 0 .. N:
    result[m, n] = source[offset + m * row_stride + n]
```

`row_stride` is required for a grouped result and is invalid for an ordinary
`1xL` load.
Loads are not predicated.

## `pto.vmi.tstore`

```mlir
pto.vmi.tstore %value, %dest[%offset], %mask
    : !pto.vmi.tilereg<1xLxT>, !pto.ptr<T, ub>, !pto.vmi.tilereg<1xLxi1>
```

The one-dimensional form stores only lanes whose mask value is true. If the
mask is omitted, every lane is stored.

### Grouped store

```mlir
pto.vmi.tstore %value, %dest[%offset], %row_stride
    : !pto.vmi.tilereg<MxNxT>, !pto.ptr<T, ub>, index
```

The grouped form writes row `m` to `dest[offset + m * row_stride]`.
`row_stride` is in elements and must be at least `N`. Grouped stores are
unpredicated.

```text
for m in 0 .. M:
  for n in 0 .. N:
    dest[offset + m * row_stride + n] = value[m, n]
```

## Group scalar load and store

An `Mx1` group-scalar carrier uses the same instructions. This form is legal
only for the producer/consumer roles defined in the overview.

```mlir
%row_values = pto.vmi.tload %source[%offset], %row_stride
    : !pto.ptr<f32, ub>, index -> !pto.vmi.tilereg<8x1xf32>
pto.vmi.tstore %row_values, %dest[%offset], %row_stride
    : !pto.vmi.tilereg<8x1xf32>, !pto.ptr<f32, ub>, index
```

The load reads one scalar per row; the store writes one scalar per row. Both
forms are unmasked. Use `trowexpand` to expand the carrier and `trowsum`,
`trowmax`, or `trowmin` to create one.

## Examples

```mlir
// Flat 128-lane vector path.
%flat = pto.vmi.tload %src[%c0]
    : !pto.ptr<f16, ub> -> !pto.vmi.tilereg<1x128xf16>
%mask = pto.vmi.texpands %active
    : index -> !pto.vmi.tilereg<1x128xi1>
pto.vmi.tstore %flat, %dst[%c0], %mask
    : !pto.vmi.tilereg<1x128xf16>, !pto.ptr<f16, ub>, !pto.vmi.tilereg<1x128xi1>

// Eight independent rows of 16 fp32 values.
%rows = pto.vmi.tload %src[%c0], %stride
    : !pto.ptr<f32, ub>, index -> !pto.vmi.tilereg<8x16xf32>
```
