# 2. Index Generation

`pto.vmi.tci` replaces the existing `iota` / `vci` spellings with the Tile Op
name for contiguous integer sequence generation. It creates a one-dimensional
tile-register index sequence.

## `pto.vmi.tci`

```mlir
%index = pto.vmi.tci %base {order = "asc"}
    : i32 -> !pto.vmi.tilereg<1xLxi32>
```

For a `1xL` result, `tci` produces a consecutive sequence:

```text
for n in 0 .. L:
  result[0, n] = base + n               // order = "asc"
  result[0, n] = base + (L - 1 - n)     // order = "desc"
```

`order` is `"asc"` by default; `"desc"` is also legal. The result dtype is
an integer type supported by VMI indexing.

`tci` is one-dimensional. Use `pto.vmi.treshape` after generation when a
grouped `MxN` index tile is required; the generated sequence is then partitioned
row-major without changing element order.
