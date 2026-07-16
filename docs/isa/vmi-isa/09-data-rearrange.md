# 9. Data Rearrangement

The current `vintlv` and `vdintlv` operations use the Tile Op-aligned names
`tinterleave` and `tdeinterleave`.

## `pto.vmi.tinterleave`

```mlir
%low, %high = pto.vmi.tinterleave %lhs, %rhs, %mask
    : !pto.vmi.tilereg<1xLxT>, !pto.vmi.tilereg<1xLxT>,
      !pto.vmi.tilereg<1xLxi1>
      -> !pto.vmi.tilereg<1xLxT>, !pto.vmi.tilereg<1xLxT>
```

`tinterleave` interleaves the two flat source tile registers into a low and
high result pair. The sources, results, and governing mask are all
one-dimensional `1xL` tiles with matching length; the data tiles also have
matching dtype.

## `pto.vmi.tdeinterleave`

```mlir
%even, %odd = pto.vmi.tdeinterleave %low, %high, %mask
    : !pto.vmi.tilereg<1xLxT>, !pto.vmi.tilereg<1xLxT>,
      !pto.vmi.tilereg<1xLxi1>
      -> !pto.vmi.tilereg<1xLxT>, !pto.vmi.tilereg<1xLxT>
```

`tdeinterleave` reverses the corresponding interleave. It is the
register-level AoS-to-SoA permutation form and also requires matching `1xL`
operands.

Use `treshape` before either operation if the values are currently grouped. A
permutation may then be reshaped back to the original `MxN` view, provided the
dtype and total element count are unchanged.
