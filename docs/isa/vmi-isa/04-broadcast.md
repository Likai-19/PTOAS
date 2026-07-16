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

## Relation to `pto.vmi.vexpdif`

`trowexpandexpdif` is not introduced as a canonical VMI rename. Canonical VMI
`vexpdif` consumes same-shaped `x`, `max`, and result tiles and computes a
lane-wise exponent difference. The `max` operand is not required to contain
one repeated scalar per row.

Tile Op `trowexpandexpdif` instead consumes one logical scalar per row. It also
requires its data operands and result to use the same floating-point dtype,
whereas VMI `vexpdif` permits an `f16` `x` with an `f32` `max` and `f32`
result. It therefore does not have the same general operation contract.

For a row-wise softmax, use `trowexpand` to make the row broadcast explicit,
then call `vexpdif` with same-shaped inputs.

## Example: row-wise softmax preparation

```mlir
%rows = pto.vmi.treshape %flat
    : !pto.vmi.tilereg<1x128xf32> -> !pto.vmi.tilereg<8x16xf32>
%mask = pto.vmi.texpands %active
    : index -> !pto.vmi.tilereg<8x16xi1>
%max = pto.vmi.trowmax %rows, %mask
    : !pto.vmi.tilereg<8x16xf32>, !pto.vmi.tilereg<8x16xi1>
      -> !pto.vmi.tilereg<8x1xf32>
%max_full = pto.vmi.trowexpand %max
    : !pto.vmi.tilereg<8x1xf32> -> !pto.vmi.tilereg<8x16xf32>
%exp = pto.vmi.vexpdif %rows, %max_full, %mask
    : !pto.vmi.tilereg<8x16xf32>, !pto.vmi.tilereg<8x16xf32>,
      !pto.vmi.tilereg<8x16xi1> -> !pto.vmi.tilereg<8x16xf32>
```
