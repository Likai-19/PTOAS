# 8. Predicate Tile Operations

VMI predicates are `tilereg` values with `i1` dtype. A mask therefore has the
same shape notation and `treshape` behavior as its data tile:

```mlir
!pto.vmi.tilereg<1x128xi1>
!pto.vmi.tilereg<8x16xi1>
```

There is no standalone mask register type, predicate granularity, or `pred`
element spelling in this design.

## `pto.vmi.texpands`

Mask construction uses the Tile Op-aligned `texpands` name. The predicate
overload expands one scalar active extent into an `i1` tile: positions before
the active extent are `true`, and the remaining positions are `false`.
Unlike the ordinary scalar-broadcast overload, the operand is an element count
rather than the value copied into every tile position.

### One-dimensional prefix mask

```mlir
%mask = pto.vmi.texpands %active_lanes
    : index -> !pto.vmi.tilereg<1xLxi1>
```

```text
for n in 0 .. L:
  mask[0, n] = n < active_lanes
```

`active_lanes <= 0` produces an all-false mask and `active_lanes >= L`
produces an all-true mask. Use `treshape` to turn the result into a legal
grouped `MxNxi1` tile when one flat prefix is to be viewed as rows.

### Two-dimensional per-row mask

The two-dimensional form creates the same valid/padding pattern independently
in every row. The result type supplies both the group count and group width, so
separate `num_groups` and `group_size` attributes are unnecessary.

```mlir
%mask = pto.vmi.texpands %active_per_row
    : index -> !pto.vmi.tilereg<MxNxi1>
```

```text
for m in 0 .. M:
  for n in 0 .. N:
    mask[m, n] = n < active_per_row
```

The result must be a legal grouped `MxN` shape. `active_per_row <= 0` makes
every position false, while `active_per_row >= N` makes every position true.
Both `M` and `N` are positive, matching the current VMI rule that a positive
group count evenly divides the logical lane count.

## Mask boolean operations

The Tile Op-aligned bitwise names operate on `i1` tile registers as predicate
boolean operations:

```mlir
%both = pto.vmi.tand %lhs, %rhs
    : !pto.vmi.tilereg<MxNxi1>, !pto.vmi.tilereg<MxNxi1>
      -> !pto.vmi.tilereg<MxNxi1>
%not_lhs = pto.vmi.tnot %lhs
    : !pto.vmi.tilereg<MxNxi1> -> !pto.vmi.tilereg<MxNxi1>
```

`tand`, `tor`, `txor`, and `tnot` require same-shaped `i1` operands and return
the same `i1` shape. They do not take a separate governing mask.
