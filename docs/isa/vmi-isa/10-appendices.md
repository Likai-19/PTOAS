# Appendices

---

## Appendix A: Unified Ops Index

| # | Op | Group | Category | Brief |
|---|---|---|---|---|
| 1 | `pto.vmi.vload` | 1: Load/Store | A | Logical vector load from UB |
| 2 | `pto.vmi.vstore` | 1: Load/Store | A | Logical vector store to UB |
| 3 | `pto.vmi.vci` | 2: Index-gen | A | Lane-index vector generation |
| 4 | `pto.vmi.vadd` | 3: Eltwise | A | Elementwise add (fp+int unified) |
| 5 | `pto.vmi.vsub` | 3: Eltwise | A | Elementwise subtract |
| 6 | `pto.vmi.vmul` | 3: Eltwise | A | Elementwise multiply |
| 7 | `pto.vmi.vdiv` | 3: Eltwise | A | Elementwise divide (fp only) |
| 8 | `pto.vmi.vmax` | 3: Eltwise | A | Elementwise maximum |
| 9 | `pto.vmi.vmin` | 3: Eltwise | A | Elementwise minimum |
| 10 | `pto.vmi.vabs` | 3: Eltwise | A | Elementwise absolute value |
| 11 | `pto.vmi.vneg` | 3: Eltwise | A | Elementwise negate |
| 12 | `pto.vmi.vrelu` | 3: Eltwise | A | Elementwise ReLU |
| 13 | `pto.vmi.vexp` | 3: Eltwise | A | Elementwise exponential |
| 14 | `pto.vmi.vln` | 3: Eltwise | A | Elementwise natural log |
| 15 | `pto.vmi.vsqrt` | 3: Eltwise | A | Elementwise square root |
| 16 | `pto.vmi.vand` | 3: Eltwise | A | Elementwise bitwise AND |
| 17 | `pto.vmi.vor` | 3: Eltwise | A | Elementwise bitwise OR |
| 18 | `pto.vmi.vxor` | 3: Eltwise | A | Elementwise bitwise XOR |
| 19 | `pto.vmi.vnot` | 3: Eltwise | A | Elementwise bitwise NOT |
| 20 | `pto.vmi.vshl` | 3: Eltwise | A | Elementwise left shift |
| 21 | `pto.vmi.vshr` | 3: Eltwise | A | Elementwise signedness-aware right shift |
| 22 | `pto.vmi.vadds` | 3: Eltwise | A | Vector-scalar add |
| 23 | `pto.vmi.vmuls` | 3: Eltwise | A | Vector-scalar multiply |
| 24 | `pto.vmi.vmaxs` | 3: Eltwise | A | Vector-scalar maximum |
| 25 | `pto.vmi.vmins` | 3: Eltwise | A | Vector-scalar minimum |
| 26 | `pto.vmi.vshls` | 3: Eltwise | A | Vector-scalar shift left |
| 27 | `pto.vmi.vshrs` | 3: Eltwise | A | Vector-scalar shift right |
| 28 | `pto.vmi.vcmp` | 3: Eltwise | A | Elementwise compare → mask |
| 29 | `pto.vmi.vcmps` | 3: Eltwise | A | Vector-scalar compare → mask |
| 30 | `pto.vmi.vsel` | 3: Eltwise | A | Predicate select |
| 31 | `pto.vmi.vselr` | 3: Eltwise | A | Dynamic lane permute |
| 32 | `pto.vmi.vbrc` | 4: Broadcast | A/B | Broadcast scalar/group-slot |
| 33 | `pto.vmi.vcadd` | 5: Reduce | B | Add-reduction |
| 34 | `pto.vmi.vcmax` | 5: Reduce | B | Max-reduction |
| 35 | `pto.vmi.vcmin` | 5: Reduce | B | Min-reduction |
| 36 | `pto.vmi.vcvt` | 6: Convert | B | Unified type conversion |
| 37 | `pto.vmi.vinterpret_cast` | 6: Convert | A | Bitwise reinterpret |
| 38 | `pto.vmi.vexpdif` | 7: SFU | A | Fused exp(x−max) |
| 39 | `pto.vmi.vaxpy` | 7: SFU | A | Fused α·x+y |
| 40 | `pto.vmi.vlrelu` | 7: SFU | A | Leaky ReLU |
| 41 | `pto.vmi.vprelu` | 7: SFU | A | Parametric ReLU |
| 42 | `pto.vmi.vmull` | 7: SFU | A | Pair-result widening 32×32 multiply |
| 43 | `pto.vmi.vmula` | 7: SFU | A | Fused multiply-add |
| 44 | `pto.vmi.vchist` | 7: SFU | B | Cumulative histogram (half-axis) |
| 45 | `pto.vmi.vdhist` | 7: SFU | B | Distribution histogram (plain per-bin) |
| 46 | `pto.vmi.vgather` | 7: SFU | C | Indexed gather (B32) |
| 47 | `pto.vmi.vgatherb` | 7: SFU | C | Byte-granularity indexed gather |
| 48 | `pto.vmi.vscatter` | 7: SFU | C | Indexed scatter |
| 49 | `pto.vmi.create_mask` | 8: Predicate | gen | Prefix / first-N tail mask |
| 50 | `pto.vmi.create_group_mask` | 8: Predicate | gen | Grouped predicate mask |
| 51 | `pto.vmi.vintlv` | 9: Rearrange | A | Interleave two vectors |
| 52 | `pto.vmi.vdintlv` | 9: Rearrange | A | Deinterleave two vectors |

---

## Appendix C: MERGE Mode Emulation (A5)

On A5, the hardware predicates only in **ZEROING** mode (inactive lanes → 0).
MERGE mode is emulated by `pto.as`:

```mlir
// MERGE emulation on A5:  dst = Pg ? op(...) : dst_old
%npg   = pto.vmi.vnot %pg                         // complement predicate
%new_z = pto.vmi.<op> %a, %b, %pg                 // ZEROING: inactive → 0
%old_z = pto.vmi.vand %dst_old, %npg             // keep old on inactive lanes
%dst   = pto.vmi.vor %new_z, %old_z               // disjoint OR → merged
```

Alternatively, a single `vsel %pg, %new, %dst_old` can replace the `vand`+`vor`
pair.

**MERGE cost on A5:** `+1 vnot` (once per distinct `Pg`) + `+K vsel`/`vor`.
On A6, merge-capable ops take the mode natively — the `vnot`+`vor` emulation
collapses to the single predicated op.
