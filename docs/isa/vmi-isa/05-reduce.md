# 5. Row Reductions

VMI row reductions use the corresponding Tile Op names and require an explicit
two-dimensional source. A row is one logical group; the result retains the
row count and has one column.

| VMI op | Replaces | Semantics |
|---|---|---|
| `pto.vmi.trowsum` | `vcadd` | `result[m,0] = sum_n source[m,n]` |
| `pto.vmi.trowmax` | `vcmax` | `result[m,0] = max_n source[m,n]` |
| `pto.vmi.trowmin` | `vcmin` | `result[m,0] = min_n source[m,n]` |

## Common form

```mlir
%result = pto.vmi.trowsum %source, %mask {reassoc}
    : !pto.vmi.tilereg<MxNxT>, !pto.vmi.tilereg<MxNxi1>
      -> !pto.vmi.tilereg<Mx1xT>
```

```text
for m in 0 .. M:
  acc = identity
  for n in 0 .. N:
    if mask[m, n]:
      acc = reduce(acc, source[m, n])
  result[m, 0] = acc
```

`M` and `N` must be positive. This directly represents the current VMI rule
that the positive group count evenly divide the logical lane count. The input
and mask have the same `MxN` shape. The result is an `Mx1` group-scalar
carrier.

The mask is required. Inactive positions contribute the operation identity:

| Op | Inactive position contribution |
|---|---|
| `trowsum` | zero |
| `trowmax` | negative infinity for floating point; the type minimum for integer |
| `trowmin` | positive infinity for floating point; the type maximum for integer |

`trowsum` accepts `{reassoc}` and requires it for floating-point source data.
`trowmax` and `trowmin` do not accept `{reassoc}`.

## Full reduction

A full reduction is the `M = 1` row-reduction form:

```mlir
%sum = pto.vmi.trowsum %flat, %mask {reassoc}
    : !pto.vmi.tilereg<1x128xf32>, !pto.vmi.tilereg<1x128xi1>
      -> !pto.vmi.tilereg<1x1xf32>
```

No `group` attribute is used. For an existing flat tile, reshape to `1xL` (the
canonical one-dimensional form) or to a multi-row `MxN` shape before reducing.

## Example: eight independent reductions

```mlir
%rows = pto.vmi.treshape %input
    : !pto.vmi.tilereg<1x128xf32> -> !pto.vmi.tilereg<8x16xf32>
%mask = pto.vmi.texpands %active_per_row
    : index -> !pto.vmi.tilereg<8x16xi1>
%sums = pto.vmi.trowsum %rows, %mask {reassoc}
    : !pto.vmi.tilereg<8x16xf32>, !pto.vmi.tilereg<8x16xi1>
      -> !pto.vmi.tilereg<8x1xf32>
```
