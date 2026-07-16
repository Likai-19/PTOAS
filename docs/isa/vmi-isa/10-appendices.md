# Appendices

## Appendix A: Canonical VMI Instruction Index

| Family | Canonical instructions |
|---|---|
| Tile-register memory | `tload`, `tstore` |
| View / index | `treshape`, `tci` |
| Tile-aligned elementwise arithmetic | `tadd`, `tsub`, `tmul`, `tdiv`, `tmax`, `tmin`, `tabs`, `tneg`, `trelu`, `texp`, `tlog`, `tsqrt` |
| Tile-aligned bitwise and scalar | `tand`, `tor`, `txor`, `tnot`, `tshl`, `tshr`, `tadds`, `tmuls`, `tmaxs`, `tmins`, `tshls`, `tshrs` |
| Compare and select | `tcmp`, `tcmps`, `tsel`, `vselr` |
| Broadcast and reduction | `texpands`, `trowexpand`, `trowsum`, `trowmax`, `trowmin` |
| Conversion | `tcvt`; `treshape` for bitwise reinterpretation |
| Tile-aligned fused / special | `tlrelu`, `taxpy`, `tprelu`, `thistogram`, `tgather`, `tgatherb`, `tscatter` |
| VMI-only fused / special | `vexpdif`, `dhist`, `vmula` |
| Predicate construction | predicate `texpands` |
| Rearrangement | `tinterleave`, `tdeinterleave` |

## Appendix B: Flat and grouped equivalence

The following two sequences represent the same logical 128-element fp32
value. `treshape` changes only the view in this same-dtype example.

```mlir
%flat = pto.vmi.tload %src[%c0]
    : !pto.ptr<f32, ub> -> !pto.vmi.tilereg<1x128xf32>
%rows = pto.vmi.treshape %flat
    : !pto.vmi.tilereg<1x128xf32> -> !pto.vmi.tilereg<8x16xf32>
%flat_again = pto.vmi.treshape %rows
    : !pto.vmi.tilereg<8x16xf32> -> !pto.vmi.tilereg<1x128xf32>
```

## Appendix C: Grouped reduction and expansion

```mlir
%rows = pto.vmi.treshape %flat
    : !pto.vmi.tilereg<1x128xf32> -> !pto.vmi.tilereg<8x16xf32>
%mask = pto.vmi.texpands %active
    : index -> !pto.vmi.tilereg<8x16xi1>
%sum = pto.vmi.trowsum %rows, %mask {reassoc}
    : !pto.vmi.tilereg<8x16xf32>, !pto.vmi.tilereg<8x16xi1>
      -> !pto.vmi.tilereg<8x1xf32>
%broadcast_sum = pto.vmi.trowexpand %sum
    : !pto.vmi.tilereg<8x1xf32> -> !pto.vmi.tilereg<8x16xf32>
```

The shape itself expresses the former `group = 8` relationship. No group
attribute or compact VMI vector type appears in the operation syntax.
