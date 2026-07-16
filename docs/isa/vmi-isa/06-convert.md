# 6. Type Conversion and Reinterpretation

## `pto.vmi.tcvt`

`tcvt` replaces `vcvt` and has the same elementwise conversion meaning as
Tile Op `tcvt`.

```mlir
%result = pto.vmi.tcvt %source {rounding = "H", saturate = "SAT"}
    : !pto.vmi.tilereg<MxNxTsrc> -> !pto.vmi.tilereg<MxNxTdst>
```

The source and result have the same `MxN` shape. `tcvt` changes only the data
element type, preserving each logical tile position:

```text
for m in 0 .. M:
  for n in 0 .. N:
    result[m, n] = convert<Tdst>(source[m, n])
```

Supported conversion categories include floating-point widening and narrowing,
float/integer conversion, integer extension, and integer truncation. The
`rounding` and `saturate` retain their current VMI semantics. A grouped input
remains grouped; an `MxN` conversion does not flatten or reshape the tile.

```mlir
%wide = pto.vmi.tcvt %source
    : !pto.vmi.tilereg<8x16xf16> -> !pto.vmi.tilereg<8x16xf32>
```

## `pto.vmi.treshape`

`treshape` is also the canonical bitwise reinterpretation operation. It
replaces `bitcast` / `vinterpret_cast` and uses the corresponding Tile Op
name.

```mlir
%result = pto.vmi.treshape %source
    : !pto.vmi.tilereg<8x16xf32> -> !pto.vmi.tilereg<8x16xi32>
```

The source and result may change shape, dtype, or both, but their total
storage size must be equal:

```text
Msrc * Nsrc * bitwidth(Tsrc) == Mdst * Ndst * bitwidth(Tdst)
```

`treshape` preserves the row-major bit sequence; it does not numerically
convert individual elements. With an unchanged dtype it is the ordinary
one-dimensional/two-dimensional view conversion. Use `tcvt` whenever a
numerical value must change. Predicate tiles use only the same-dtype `i1`
shape-conversion form.
