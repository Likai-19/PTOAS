# 4. Broadcast and Row Expansion

VMI uses the Tile Op broadcast names. Scalar broadcast and grouped broadcast
are distinct operations because the latter has visible row structure.

## `pto.vmi.texpands`

`texpands` broadcasts one scalar to every position of a tile register.

```mlir
%result = pto.vmi.texpands %scalar
    : T -> !pto.vmi.tilereg<MxNxT>
```

For an `i1` result, `texpands` also has the active-extent predicate overload
defined in [Predicate Tile Operations](08-predicate-ops.md).

```text
for m in 0 .. M:
  for n in 0 .. N:
    result[m, n] = scalar
```

The result may be either a legal `1xL` tile or a legal grouped `MxN` tile.

## `pto.vmi.trowexpand`

`trowexpand` is the grouped broadcast. It consumes one scalar per row and
expands that value across the row.

```mlir
%result = pto.vmi.trowexpand %source
    : !pto.vmi.tilereg<Mx1xT> -> !pto.vmi.tilereg<MxNxT>
```

```text
for m in 0 .. M:
  for n in 0 .. N:
    result[m, n] = source[m, 0]
```

`M` and `N` must be positive. This is the shape form of the current VMI rule
that `group = M` must be positive and evenly divide the logical lane count
`L = M * N`. The source `Mx1` is a group-scalar carrier and cannot be consumed
by an ordinary elementwise operation until it is expanded.

## Row-expansion arithmetic

The current VMI surface exposes the fused exponent-difference form as
`trowexpandexpdif`; it replaces `vexpdif`.

```mlir
%result = pto.vmi.trowexpandexpdif %source, %row_value, %mask
    : !pto.vmi.tilereg<MxNxf32>, !pto.vmi.tilereg<Mx1xf32>,
      !pto.vmi.tilereg<MxNxi1> -> !pto.vmi.tilereg<MxNxf32>
```

```text
for m in 0 .. M:
  for n in 0 .. N:
    if mask[m, n]:
      result[m, n] = exp(source[m, n] - row_value[m, 0])
```

The scalar carrier and source must have the same row count. `trowexpandexpdif`
is floating-point only. It replaces an implicit compact-vector broadcast with
an explicit `Mx1` input.

Other Tile Op row-expansion arithmetic names are reserved for future VMI
instructions; they are not introduced by this design.

## Example: row-wise softmax preparation

```mlir
%rows = pto.vmi.treshape %flat
    : !pto.vmi.tilereg<1x128xf32> -> !pto.vmi.tilereg<8x16xf32>
%mask = pto.vmi.texpands %active
    : index -> !pto.vmi.tilereg<8x16xi1>
%max = pto.vmi.trowmax %rows, %mask
    : !pto.vmi.tilereg<8x16xf32>, !pto.vmi.tilereg<8x16xi1>
      -> !pto.vmi.tilereg<8x1xf32>
%exp = pto.vmi.trowexpandexpdif %rows, %max, %mask
    : !pto.vmi.tilereg<8x16xf32>, !pto.vmi.tilereg<8x1xf32>,
      !pto.vmi.tilereg<8x16xi1> -> !pto.vmi.tilereg<8x16xf32>
```
