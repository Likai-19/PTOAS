# Appendices

---

## Appendix A: Unified Ops Index

| # | Op | Group | Category | Brief |
|---|---|---|---|---|
| 1 | `pto.vmi.tload` | 1: Load/Store | A | Logical vector load from UB |
| 2 | `pto.vmi.tstore` | 1: Load/Store | A | Logical vector store to UB |
| 3 | `pto.vmi.tci` | 2: Index-gen | A | Lane-index vector generation |
| 4 | `pto.vmi.tadd` | 3: Eltwise | A | Elementwise add (fp+int unified) |
| 5 | `pto.vmi.tsub` | 3: Eltwise | A | Elementwise subtract |
| 6 | `pto.vmi.tmul` | 3: Eltwise | A | Elementwise multiply |
| 7 | `pto.vmi.tdiv` | 3: Eltwise | A | Elementwise divide (fp only) |
| 8 | `pto.vmi.tmax` | 3: Eltwise | A | Elementwise maximum |
| 9 | `pto.vmi.tmin` | 3: Eltwise | A | Elementwise minimum |
| 10 | `pto.vmi.tabs` | 3: Eltwise | A | Elementwise absolute value |
| 11 | `pto.vmi.tneg` | 3: Eltwise | A | Elementwise negate |
| 12 | `pto.vmi.trelu` | 3: Eltwise | A | Elementwise ReLU |
| 13 | `pto.vmi.texp` | 3: Eltwise | A | Elementwise exponential |
| 14 | `pto.vmi.tlog` | 3: Eltwise | A | Elementwise natural log |
| 15 | `pto.vmi.tsqrt` | 3: Eltwise | A | Elementwise square root |
| 16 | `pto.vmi.tand` | 3: Eltwise | A | Elementwise bitwise AND |
| 17 | `pto.vmi.tor` | 3: Eltwise | A | Elementwise bitwise OR |
| 18 | `pto.vmi.txor` | 3: Eltwise | A | Elementwise bitwise XOR |
| 19 | `pto.vmi.tnot` | 3: Eltwise | A | Elementwise bitwise NOT |
| 20 | `pto.vmi.tshl` | 3: Eltwise | A | Elementwise left shift |
| 21 | `pto.vmi.tshr` | 3: Eltwise | A | Elementwise unsigned right shift |
| 22 | `pto.vmi.tadds` | 3: Eltwise | A | Vector-scalar add |
| 23 | `pto.vmi.tmuls` | 3: Eltwise | A | Vector-scalar multiply |
| 24 | `pto.vmi.tmaxs` | 3: Eltwise | A | Vector-scalar maximum |
| 25 | `pto.vmi.tmins` | 3: Eltwise | A | Vector-scalar minimum |
| 26 | `pto.vmi.tshls` | 3: Eltwise | A | Vector-scalar shift left |
| 27 | `pto.vmi.tshrs` | 3: Eltwise | A | Vector-scalar shift right |
| 28 | `pto.vmi.tcmp` | 3: Eltwise | A | Elementwise compare → mask |
| 29 | `pto.vmi.tcmps` | 3: Eltwise | A | Vector-scalar compare → mask |
| 30 | `pto.vmi.tsel` | 3: Eltwise | A | Predicate select |
| 31 | `pto.vmi.vselr` | 3: Eltwise | A | Dynamic lane permute; no direct Tile Op rename |
| 32 | `pto.vmi.texpands` / `pto.vmi.trowexpand` | 4: Broadcast | A/B | Scalar / row broadcast |
| 33 | `pto.vmi.trowsum` | 5: Reduce | B | Add-reduction |
| 34 | `pto.vmi.trowmax` | 5: Reduce | B | Max-reduction |
| 35 | `pto.vmi.trowmin` | 5: Reduce | B | Min-reduction |
| 36 | `pto.vmi.tcvt` | 6: Convert | B | Unified type conversion |
| 37 | `pto.vmi.treshape` | 6: Convert | A | Bitwise reinterpret |
| 38 | `pto.vmi.vexpdif` | 7: SFU | A | Fused exp(x−max); no direct Tile Op rename |
| 39 | `pto.vmi.taxpy` | 7: SFU | A | Fused α·x+y |
| 40 | `pto.vmi.tlrelu` | 7: SFU | A | Leaky ReLU |
| 41 | `pto.vmi.tprelu` | 7: SFU | A | Parametric ReLU |
| 42 | `pto.vmi.tmul` | 7: SFU | B | Widening 32×32→64 multiply overload |
| 43 | `pto.vmi.vmula` | 7: SFU | A | Fused multiply-add; no direct Tile Op rename |
| 44 | `pto.vmi.thistogram` | 7: SFU | B | Channel histogram bin count |
| 45 | `pto.vmi.dhist` | 7: SFU | B | Distribution histogram; no direct Tile Op rename |
| 46 | `pto.vmi.tgather` | 7: SFU | C | Indexed gather (B32) |
| 47 | `pto.vmi.tgatherb` | 7: SFU | C | Byte-granularity indexed gather |
| 48 | `pto.vmi.tscatter` | 7: SFU | C | Indexed scatter |
| 49 | `pto.vmi.create_mask` | 8: Predicate | gen | Prefix / first-N tail mask; no direct Tile Op rename |
| 50 | `pto.vmi.create_group_mask` | 8: Predicate | gen | Grouped predicate mask; no direct Tile Op rename |
| 51 | `pto.vmi.tinterleave` | 9: Rearrange | A | Interleave two vectors |
| 52 | `pto.vmi.tdeinterleave` | 9: Rearrange | A | Deinterleave two vectors |

---

## Appendix C: MERGE Mode Emulation (A5)

On A5, the hardware predicates only in **ZEROING** mode (inactive lanes → 0).
MERGE mode is emulated by `pto.as`:

```mlir
// MERGE emulation on A5:  dst = Pg ? op(...) : dst_old
%npg   = pto.vmi.tnot %pg                         // complement predicate
%new_z = pto.vmi.<op> %a, %b, %pg                 // ZEROING: inactive → 0
%old_z = pto.vmi.tand %dst_old, %npg             // keep old on inactive lanes
%dst   = pto.vmi.tor %new_z, %old_z               // disjoint OR → merged
```

Alternatively, a single `tsel %pg, %new, %dst_old` can replace the `tand`+`tor`
pair.

**MERGE cost on A5:** `+1 tnot` (once per distinct `Pg`) + `+K tsel`/`tor`.
On A6, merge-capable ops take the mode natively — the `tnot`+`tor` emulation
collapses to the single predicated op.
