# VMI Layout Lowering Cases

本文是 VMI layout/lowering 的典型 case catalog，不是完整设计总文档。它只回答一个问题：
一个 VMI logical vector 在某个场景下选择某种 layout 后，`vmi-to-vpto` 必须生成什么
VPTO 结果。这里不写动机式描述；每个场景都给出 layout assignment 和 lowering result。

## 1. Layout Families

### 1.1 Dense Layout

Dense layout 的每个 logical lane 都有语义值。

```text
#pto.vmi.layout<contiguous>
```

Physical ordering:

```text
chunk c, lane l -> logical lane c * L + l
```

`L` is the physical lanes per 256B VPTO vector register for the element type.

```text
#pto.vmi.layout<deinterleaved = F, block_elems = B>
```

`block_elems` defaults to `1`. Existing spellings are shorthands:

```text
#pto.vmi.layout<deinterleaved = 2>
  == #pto.vmi.layout<deinterleaved = 2, block_elems = 1>

#pto.vmi.layout<deinterleaved = 4>
  == #pto.vmi.layout<deinterleaved = 4, block_elems = 1>
```

Logical-to-physical mapping:

```text
logical lane i
block q        = i / B
in_block lane r = i % B
part p         = q % F
part_block t   = q / F

physical part p, physical lane t * B + r
```

Required invariants:

```text
F > 0
B > 0
N % (F * B) == 0 for the direct full-chunk paths in this document
```

### 1.2 Sparse Group-Slot Layout

Sparse group-slot layout is not dense. Only `G` lanes have semantic values.

```text
#pto.vmi.layout<num_groups = G, slots = K>
```

Physical slot mapping:

```text
N = logical lane count
S = N / G                 // logical lanes per source group

slot_block(g) = g / K
slot_lane(g)  = g % K
```

Required invariants:

```text
G > 0
K > 0
G % K == 0
K must fit in the physical vreg element count
```

`K` is selected by the producer/consumer plan. It is not always 8. For
`VCGADD`-packed results, `K = 8` matches the eight 32B block results written to
the low lanes of one destination vreg. For row-local reductions where each
logical group already occupies one full 256B vreg, `K = 1` keeps each group's
scalar result in lane 0 of its own physical vreg and avoids an unsupported
cross-vreg scalar pack.

Only these lanes are semantic:

```text
physical slot block slot_block(g), lane slot_lane(g)
```

All other lanes are undefined for ordinary VMI consumers. They may only be read
by group-aware ops that define how to interpret group slots.

## 2. Plan Selection Rules

VMI cast ops must not hard-code one physical `vcvt` plan as their semantic
layout rule.

```text
dense cast:
  source/result are dense layouts.
  lowering may require deinterleaved(F, block_elems=1) around VCVT.

group-slot cast:
  source/result are both group_slots(G,K).
  lowering preserves slot_block(g) and slot_lane(g). Width-changing casts are
  legal only when a slot-preserving VPTO plan is registered, or when the cast
  can be commuted through a later group-aware consumer such as group_broadcast.
```

Illegal consumer mix:

```text
group_slots value -> ordinary dense store/add/mul
```

This must fail unless an explicit semantic op converts the sparse value:

```text
group_broadcast
group_store
future explicit group-pack op
```

## 3. Lowering Results

The following examples use symbolic VPTO names. `PAT_ALL_B*` means an all-true
predicate with the element granularity required by the instruction. `PAT_VLk`
means a prefix predicate for the first `k` lanes.

Completeness rule for this section: every numbered endpoint below must contain
VMI input, assigned layouts, VPTO lowering result, and either a memory result or
an explicit diagnostic.  Non-endpoint layout notes may appear only as setup for
the immediately following complete endpoints.

```text
3.1 f16 -> f32 -> store                                  complete
3.2 f32 -> f16 -> store                                  complete
3.3 f8 -> f32 -> compute -> f8                           complete
3.4 group_reduce S=8 -> group_store                      complete
3.5.1 group_reduce S=16 -> group_store                   complete
3.5.2 group_reduce S=16 -> broadcast -> compute -> reduce -> store
                                                            complete
3.5.3 group_reduce S=16 -> elemwise(rhs) -> group_store  complete
3.6.1 group_reduce S=32 -> group_store                   complete
3.6.2 group_reduce S=32 -> elemwise(rhs) -> group_store  complete
3.6.3 group_reduce S=32 -> broadcast -> compute -> reduce -> store
                                                            complete
3.7.1 group_reduce S=64 -> group_store                   complete
3.7.2 group_reduce S=64 -> elemwise(rhs) -> group_store  complete
3.7.3 group_reduce S=64 -> broadcast -> compute -> reduce -> store
                                                            complete
3.8 group_reduce -> truncf -> broadcast -> dense store   complete
3.9 dense store of group slots                           illegal diagnostic
3.10 non-load producer feeding S=32 group_reduce         complete
3.11 partial tail groups                                 complete/diagnostic
3.12 control-flow join before group_reduce               complete
3.13 direct group-slot f32 -> f16 cast                   illegal diagnostic
3.14 unsupported group size                              illegal diagnostic
3.15 compact S=12 written as logical S=16                complete/design
3.16 group_slot_load layout contract                     complete
3.17 group_broadcast physical arity alias                complete
3.18 one value with dense and group-reduce consumers     complete/materialization
3.19 S=16 reduce block_elems plan selection              complete/diagnostic
3.20 group_slots control-flow join                       complete
3.21 S=32 tail with full-tile-readable source            complete/design
3.22 scf.for loop-carried layout                         complete
3.23 group_broadcast with multiple dense consumers       complete
3.24 mask with elementwise/select/store                  complete
3.25 function boundary layout specialization             complete/design
```

### 3.1 `f16 -> f32 -> store`

VMI input:

```text
%x16 = pto.vmi.load %base[%off]
  : memref<128xf16> -> !pto.vmi.vreg<128xf16>
%x32 = pto.vmi.extf %x16
  : !pto.vmi.vreg<128xf16> -> !pto.vmi.vreg<128xf32>
pto.vmi.store %x32, %out[%off]
```

Assigned layouts:

```text
%x16 : !pto.vmi.vreg<128xf16, #pto.vmi.layout<contiguous>>
%x32 : !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>
```

VPTO lowering result:

```text
%x16_0 = pto.vlds %base[%off] {dist = "NORM"}
  : !pto.ptr<f16, ub> -> !pto.vreg<128xf16>

%x32_p0 = pto.vcvt %x16_0, PAT_ALL_B16 {part = "EVEN"}
  : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<64xf32>
%x32_p1 = pto.vcvt %x16_0, PAT_ALL_B16 {part = "ODD"}
  : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<64xf32>

pto.vstsx2 %x32_p0, %x32_p1, %out[%off], "INTLV_B32", PAT_ALL_B32
  : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index,
    !pto.mask<b32>
```

Alternative complete VPTO lowering result if `vstsx2 INTLV_B32` is unavailable:

```text
%x16_0 = pto.vlds %base[%off] {dist = "NORM"}
  : !pto.ptr<f16, ub> -> !pto.vreg<128xf16>

%x32_p0 = pto.vcvt %x16_0, PAT_ALL_B16 {part = "EVEN"}
  : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<64xf32>
%x32_p1 = pto.vcvt %x16_0, PAT_ALL_B16 {part = "ODD"}
  : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<64xf32>

%x32_d0, %x32_d1 = pto.vintlv %x32_p0, %x32_p1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

pto.vsts %x32_d0, %out[%off], PAT_ALL_B32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %x32_d1, %out[%off_plus_64], PAT_ALL_B32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for i = 0..127:
  out[off + i] = extf(base[off + i])
```

### 3.2 Dense `f32 -> f16 -> store`

VMI input:

```text
%x32 = pto.vmi.load %base[%off]
  : memref<128xf32> -> !pto.vmi.vreg<128xf32>
%x16 = pto.vmi.truncf %x32
  : !pto.vmi.vreg<128xf32> -> !pto.vmi.vreg<128xf16>
pto.vmi.store %x16, %out[%off]
```

Assigned layouts:

```text
%x32 : !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>
%x16 : !pto.vmi.vreg<128xf16, #pto.vmi.layout<contiguous>>
```

VPTO lowering result:

```text
%x32_p0, %x32_p1 = pto.vldsx2 %base[%off], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%part0 = pto.vcvt %x32_p0, PAT_ALL_B32
  {part = "EVEN", rnd = "R", sat = "SAT"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>

%part1 = pto.vcvt %x32_p1, PAT_ALL_B32
  {part = "ODD", rnd = "R", sat = "SAT"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>

%x16_0 = pto.vor %part0, %part1, PAT_ALL_B16
  : !pto.vreg<128xf16>

pto.vsts %x16_0, %out[%off], PAT_ALL_B16 {dist = "NORM_B16"}
  : !pto.vreg<128xf16>, !pto.ptr<f16, ub>, !pto.mask<b16>
```

Alternative complete VPTO lowering result if the source has already been loaded
as two contiguous f32 chunks and must be materialized to `deinterleaved=2` before
the conversion:

```text
%x32_d0 = pto.vlds %base[%off] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x32_d1 = pto.vlds %base[%off_plus_64] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x32_p0, %x32_p1 = pto.vdintlv %x32_d0, %x32_d1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%part0 = pto.vcvt %x32_p0, PAT_ALL_B32
  {part = "EVEN", rnd = "R", sat = "SAT"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>

%part1 = pto.vcvt %x32_p1, PAT_ALL_B32
  {part = "ODD", rnd = "R", sat = "SAT"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>

%x16_0 = pto.vor %part0, %part1, PAT_ALL_B16
  : !pto.vreg<128xf16>

pto.vsts %x16_0, %out[%off], PAT_ALL_B16 {dist = "NORM_B16"}
  : !pto.vreg<128xf16>, !pto.ptr<f16, ub>, !pto.mask<b16>
```

Memory result:

```text
for i = 0..127:
  out[off + i] = truncf(base[off + i])
```

### 3.3 Dense `f8 -> f32 -> compute -> f8`

VMI input:

```text
%x8  = pto.vmi.load %base[%off]
%x32 = pto.vmi.extf %x8
%scale = pto.vmi.broadcast %scale_s : f32 -> !pto.vmi.vreg<256xf32>
%y32 = pto.vmi.mulf %x32, %scale
%y8  = pto.vmi.truncf %y32
pto.vmi.store %y8, %out[%off]
```

Assigned layouts:

```text
%x8  : !pto.vmi.vreg<256xf8,  #pto.vmi.layout<contiguous>>
%x32 : !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>
%scale : !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>
%y32 : !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>
%y8  : !pto.vmi.vreg<256xf8,  #pto.vmi.layout<contiguous>>
```

VPTO lowering result:

```text
%x8_0 = pto.vlds %base[%off] {dist = "NORM"}
  : !pto.ptr<f8, ub> -> !pto.vreg<256xf8>

%x32_p0 = pto.vcvt %x8_0, PAT_ALL_B8 {part = "P0"}
  : !pto.vreg<256xf8>, !pto.mask<b8> -> !pto.vreg<64xf32>
%x32_p1 = pto.vcvt %x8_0, PAT_ALL_B8 {part = "P1"}
  : !pto.vreg<256xf8>, !pto.mask<b8> -> !pto.vreg<64xf32>
%x32_p2 = pto.vcvt %x8_0, PAT_ALL_B8 {part = "P2"}
  : !pto.vreg<256xf8>, !pto.mask<b8> -> !pto.vreg<64xf32>
%x32_p3 = pto.vcvt %x8_0, PAT_ALL_B8 {part = "P3"}
  : !pto.vreg<256xf8>, !pto.mask<b8> -> !pto.vreg<64xf32>

%scale_p0 = pto.vdup %scale_s, PAT_ALL_B32
  : f32, !pto.mask<b32> -> !pto.vreg<64xf32>
%scale_p1 = pto.vdup %scale_s, PAT_ALL_B32
  : f32, !pto.mask<b32> -> !pto.vreg<64xf32>
%scale_p2 = pto.vdup %scale_s, PAT_ALL_B32
  : f32, !pto.mask<b32> -> !pto.vreg<64xf32>
%scale_p3 = pto.vdup %scale_s, PAT_ALL_B32
  : f32, !pto.mask<b32> -> !pto.vreg<64xf32>

%y32_p0 = pto.vmul %x32_p0, %scale_p0, PAT_ALL_B32
%y32_p1 = pto.vmul %x32_p1, %scale_p1, PAT_ALL_B32
%y32_p2 = pto.vmul %x32_p2, %scale_p2, PAT_ALL_B32
%y32_p3 = pto.vmul %x32_p3, %scale_p3, PAT_ALL_B32

%y8_p0 = pto.vcvt %y32_p0, PAT_ALL_B32
  {part = "P0", rnd = "R", sat = "SAT"} -> !pto.vreg<256xf8>
%y8_p1 = pto.vcvt %y32_p1, PAT_ALL_B32
  {part = "P1", rnd = "R", sat = "SAT"} -> !pto.vreg<256xf8>
%y8_p2 = pto.vcvt %y32_p2, PAT_ALL_B32
  {part = "P2", rnd = "R", sat = "SAT"} -> !pto.vreg<256xf8>
%y8_p3 = pto.vcvt %y32_p3, PAT_ALL_B32
  {part = "P3", rnd = "R", sat = "SAT"} -> !pto.vreg<256xf8>

%y8_01 = pto.vor %y8_p0, %y8_p1, PAT_ALL_B8
%y8_23 = pto.vor %y8_p2, %y8_p3, PAT_ALL_B8
%y8_0  = pto.vor %y8_01, %y8_23, PAT_ALL_B8

pto.vsts %y8_0, %out[%off], PAT_ALL_B8 {dist = "NORM_B8"}
  : !pto.vreg<256xf8>, !pto.ptr<f8, ub>, !pto.mask<b8>
```

Memory result:

```text
for i = 0..255:
  out[off + i] = truncf(extf(base[off + i]) * scale_s)
```

### 3.4 `group_reduce` S=8 f32

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<64xf32> -> !pto.vmi.vreg<64xf32>
%mask = pto.vmi.create_mask %c64 : index -> !pto.vmi.mask<64xpred>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
  : !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64xpred>
 -> !pto.vmi.vreg<64xf32>
pto.vmi.group_store %sum, %sum_out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%x    : !pto.vmi.vreg<64xf32, #pto.vmi.layout<contiguous>>
%mask : !pto.vmi.mask<64xpred, #pto.vmi.layout<contiguous>>
%sum  : !pto.vmi.vreg<64xf32,
          #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering result for one full 8-row tile:

```text
%mask_chunk = pto.pge_b32 "PAT_ALL"

%x_chunk = pto.vlds %base[%tile_off] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

%sum_block = pto.vcgadd %x_chunk, %mask_chunk
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%store8 = pto.pge_b32 "PAT_VL8"
pto.vsts %sum_block, %sum_out[%group_tile_off], %store8 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Lowering result for one chunk, per the `visa.txt` VCGADD contract:

```text
%sum_block lane 0 = reduce %x lanes 0..7
%sum_block lane 1 = reduce %x lanes 8..15
...
%sum_block lane 7 = reduce %x lanes 56..63
all non-slot lanes are non-semantic
```

Layout result:

```text
G = N / 8
K = 8

slot_block(g) = g / 8
slot_lane(g)  = g % 8
```

Memory result:

```text
for r = 0..7:
  sum_out[group_tile_off + r] = reduce(row_r[0..7])
```

### 3.5 `group_reduce` S=16 f32, load-fused split

The facts used by this lowering are checked against the current repo:

```text
pto.vldsx2 supports "BDINTLV".
pto.vstsx2 supports only "INTLV_B8" / "INTLV_B16" / "INTLV_B32".
visa.txt says VCGADD writes one 32B-block result continuously to destination
LSBs; the current repository golden tests follow lanes 0..7 for f32.
```

There are three complete consumers for this layout today:

```text
load -> group_reduce -> group_store(sum)
load -> group_reduce -> elementwise compute on group-slot values
     -> group_store
load -> group_reduce -> group_broadcast -> elementwise compute
     -> group_reduce -> group_store
```

#### 3.5.1 Reduce And Store Group Sums

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<Nxf32> -> !pto.vmi.vreg<Nxf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = N / 16}
pto.vmi.group_store %sum, %sum_out[%group_off], %c1 {num_groups = N / 16}
```

Assigned layouts:

```text
%x : !pto.vmi.vreg<Nxf32,
       #pto.vmi.layout<deinterleaved = 2, block_elems = 8>>

%sum : !pto.vmi.vreg<Nxf32,
         #pto.vmi.layout<num_groups = N / 16, slots = 8>>
```

For each 8-row tile:

```text
row r = 16xf32 = row_r.lo8, row_r.hi8
```

VPTO lowering result for one full 8-row tile:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"

%lo, %hi = pto.vldsx2 %base[%tile_off], "BDINTLV"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%lo lanes 0..7   = row0.lo8
%lo lanes 8..15  = row1.lo8
...
%lo lanes 56..63 = row7.lo8

%hi lanes 0..7   = row0.hi8
%hi lanes 8..15  = row1.hi8
...
%hi lanes 56..63 = row7.hi8

%lo_sum = pto.vcgadd %lo, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%hi_sum = pto.vcgadd %hi, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%sum_block = pto.vadd %lo_sum, %hi_sum, %sum_mask
  : !pto.vreg<64xf32>

%store8 = pto.pge_b32 "PAT_VL8"
pto.vsts %sum_block, %sum_out[%group_tile_off], %store8 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

`BDINTLV` here denotes the ISA `#bdintlv` block-based interleaving load mode:
it loads `2 * VL` bytes and sends even 32B blocks to the first destination
register and odd 32B blocks to the second destination register. For f32,
one 32B block is `8xf32`, matching `block_elems = 8`.

Tail tiles use the same dataflow with `%all_b32` replaced by masks derived from
the VMI mask for the low and high 8-lane halves of each row.

Layout result:

```text
G = N / 16
K = 8

slot_block(g) = g / 8
slot_lane(g)  = g % 8

%sum_block lane 0 = reduce row0 lanes 0..15
%sum_block lane 1 = reduce row1 lanes 0..15
...
%sum_block lane 7 = reduce row7 lanes 0..15
```

No VMI value exposes `%lo_sum` or `%hi_sum`. They are internal VPTO values.

Memory result:

```text
sum_out[group_tile_off + 0] = reduce row0 lanes 0..15
sum_out[group_tile_off + 1] = reduce row1 lanes 0..15
...
sum_out[group_tile_off + 7] = reduce row7 lanes 0..15
```

This endpoint is fully specified: the only sparse value is `%sum`; `group_store`
stores the low 8 slot lanes with an ordinary prefix store.

#### 3.5.2 Reduce, Broadcast, Elementwise, Reduce, Store

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<Nxf32> -> !pto.vmi.vreg<Nxf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = N / 16}
%b = pto.vmi.group_broadcast %sum {num_groups = N / 16}
%y = pto.vmi.mulf %x, %b
%ysum = pto.vmi.group_reduce_addf %y, %mask {num_groups = N / 16}
pto.vmi.group_store %ysum, %out[%group_off], %c1 {num_groups = N / 16}
```

Assigned layouts:

```text
%x   : !pto.vmi.vreg<Nxf32,
         #pto.vmi.layout<deinterleaved = 2, block_elems = 8>>
%sum : !pto.vmi.vreg<Nxf32,
         #pto.vmi.layout<num_groups = N / 16, slots = 8>>
%b   : !pto.vmi.vreg<Nxf32,
         #pto.vmi.layout<deinterleaved = 2, block_elems = 8>>
%y   : !pto.vmi.vreg<Nxf32,
         #pto.vmi.layout<deinterleaved = 2, block_elems = 8>>
%ysum : !pto.vmi.vreg<Nxf32,
          #pto.vmi.layout<num_groups = N / 16, slots = 8>>
```

VPTO lowering result for one full 8-row tile:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"

%x_lo, %x_hi = pto.vldsx2 %base[%tile_off], "BDINTLV"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%x_lo_sum = pto.vcgadd %x_lo, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%x_hi_sum = pto.vcgadd %x_hi, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%sum_block = pto.vadd %x_lo_sum, %x_hi_sum, %sum_mask
  : !pto.vreg<64xf32>

%lane_id = pto.vci %c0_i32 : i32 -> !pto.vreg<64xi32>
%broadcast_idx = pto.vshrs %lane_id, %c3_i16, %all_b32
  : !pto.vreg<64xi32>, i16, !pto.mask<b32> -> !pto.vreg<64xi32>

// This is the materialization of pto.vmi.group_broadcast.  The group sums are
// in %sum_block lanes 0..7; vselr expands each sum to the 8 lanes of the
// corresponding row half.  The following vmul/vcgadd consume an ordinary dense
// physical vector.
%b_rows = pto.vselr %sum_block, %broadcast_idx
  : !pto.vreg<64xf32>, !pto.vreg<64xi32> -> !pto.vreg<64xf32>

%y_lo = pto.vmul %x_lo, %b_rows, %all_b32
  : !pto.vreg<64xf32>
%y_hi = pto.vmul %x_hi, %b_rows, %all_b32
  : !pto.vreg<64xf32>

%y_lo_sum = pto.vcgadd %y_lo, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%y_hi_sum = pto.vcgadd %y_hi, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

// Final per-row reduction and store.
%ysum_block = pto.vadd %y_lo_sum, %y_hi_sum, %sum_mask
  : !pto.vreg<64xf32>

%store8 = pto.pge_b32 "PAT_VL8"
pto.vsts %ysum_block, %out[%group_tile_off], %store8 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

This trace processes 8 logical rows at once. `num_groups = N / 16` means each
logical group is one `16xf32` row, and one full f32 VPTO tile covers 8 such
groups:

```text
64 f32 lanes per physical part = 8 rows * 8 f32 lanes per half-row
```

Tail tiles use the same dataflow with `%all_b32` replaced by masks derived from
the VMI mask for the low and high 8-lane halves of each row.

Physical lane result for the tile:

```text
%x_lo lanes 0..7   = row0[0..7]
%x_lo lanes 8..15  = row1[0..7]
...
%x_lo lanes 56..63 = row7[0..7]

%x_hi lanes 0..7   = row0[8..15]
%x_hi lanes 8..15  = row1[8..15]
...
%x_hi lanes 56..63 = row7[8..15]

%sum_block lanes 0..7 =
  reduce(row0[0..15]), reduce(row1[0..15]), ..., reduce(row7[0..15])

%b_rows lanes 0..7   = reduce(row0[0..15])
%b_rows lanes 8..15  = reduce(row1[0..15])
...
%b_rows lanes 56..63 = reduce(row7[0..15])

For each row `r` in this 8-row tile:

%y_lo lanes r*8 .. r*8+7 =
  row_r[0..7] * reduce(row_r[0..15])

%y_hi lanes r*8 .. r*8+7 =
  row_r[8..15] * reduce(row_r[0..15])

Concretely:
%y_lo lanes 0..7   = row0[0..7] * reduce(row0[0..15])
%y_lo lanes 8..15  = row1[0..7] * reduce(row1[0..15])
...
%y_lo lanes 56..63 = row7[0..7] * reduce(row7[0..15])

%y_hi lanes 0..7   = row0[8..15] * reduce(row0[0..15])
%y_hi lanes 8..15  = row1[8..15] * reduce(row1[0..15])
...
%y_hi lanes 56..63 = row7[8..15] * reduce(row7[0..15])

%ysum_block lanes 0..7 =
  reduce(%y row0), reduce(%y row1), ..., reduce(%y row7)
```

Memory result:

```text
out[group_tile_off + r] =
    reduce_i((row_r[i] * reduce_j(row_r[j])) for i in 0..15)
  = reduce(row_r[0..15]) * reduce(row_r[0..15])
for r = 0..7
```

If a later consumer requires row-major contiguous order, `vmi-to-vpto` must
materialize:

```text
deinterleaved=2, block_elems=8 -> contiguous
```

This materialization cannot be implemented with `vstsx2 INTLV_B32`, because
that instruction interleaves individual b32 elements, not 32B row halves. Until
a concrete block-interleave register materialization or store op is selected,
row-major store of this layout must be rejected with:

```text
VMI-LAYOUT-CONTRACT:
  pto.vmi.store requires materializing
  #pto.vmi.layout<deinterleaved = 2, block_elems = 8> to contiguous, but no
  VPTO block-interleave materialization/store plan is registered.
```

#### 3.5.3 Reduce Result, Elementwise, Store

This case computes a per-row reduction, applies an elementwise operation to the
reduced values themselves, and stores one result per group.  There is no
`group_broadcast` in this flow because the elementwise op is not applied to the
original `8x16xf32` matrix elements.

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<128xf32> -> !pto.vmi.vreg<128xf32>
%rhs = pto.vmi.group_slot_load %rhs_base[%rhs_off], %c1 {num_groups = 8}
  : !pto.ptr<f32, ub> -> !pto.vmi.vreg<128xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
%outv = pto.vmi.addf %sum, %rhs
pto.vmi.group_store %outv, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%x for reduce:
  !pto.vmi.vreg<128xf32,
    #pto.vmi.layout<deinterleaved = 2, block_elems = 8>>

%sum:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>

%rhs:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>

%outv:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

For this endpoint, the RHS is a packed per-group vector:

```text
rhs_base[rhs_off + r] = rhs(row r), for r = 0..7
```

Layout assignment must treat `group_slot_load` as a group-slot producer: one
f32 value per group is placed in the live slot lanes.  It must not use
`group_load`, which loads `group_size` data elements per group instead of one
per-group scalar.

The elementwise op runs only on the live group-slot lanes:

```text
%sum lanes 0..7 =
  reduce(row0[0..15]), reduce(row1[0..15]), ..., reduce(row7[0..15])

%rhs lanes 0..7 =
  rhs(row0), rhs(row1), ..., rhs(row7)

%outv lanes 0..7 =
  %sum lanes 0..7 + %rhs lanes 0..7

lanes 8..63 remain dead/zero and are masked off by PAT_VL8.
```

VPTO lowering result for one full 8-row tile:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"
%one_block = pto.pge_b32 "PAT_VL1"

// Reduction path: use BDINTLV to feed two VCG reductions.
%x_lo, %x_hi = pto.vldsx2 %base[%tile_off], "BDINTLV"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%x_lo_sum = pto.vcgadd %x_lo, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%x_hi_sum = pto.vcgadd %x_hi, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%sum_block = pto.vadd %x_lo_sum, %x_hi_sum, %sum_mask
  : !pto.vreg<64xf32>

// Packed RHS group-slot load.  %rhs_tile_base points to rhs_base[rhs_off].
// One 32B block contains 8 f32 RHS values and materializes lanes 0..7; all
// other lanes are dead/zero.
%rhs_block = pto.vsldb %rhs_tile_base, %c0_i16, %c0_i16, %one_block
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>

// Elementwise compute on group-slot values.  Only lanes 0..7 are live.
%outv_block = pto.vadd %sum_block, %rhs_block, %sum_mask
  : !pto.vreg<64xf32>

pto.vsts %outv_block, %out[%group_tile_off], %sum_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  s = reduce(row_r[0..15])
  out[group_tile_off + r] = s + rhs[r]
```

### 3.6 `group_reduce` S=32 f32, 4-way split

This case covers one `8x32xf32` tile.  Each logical row is 128B, so it must be
split into four 32B partial rows before `vcgadd` can reduce it efficiently.

The canonical layout for the input is:

```text
%x : !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>
```

With `deinterleaved = 4`, physical part `p` contains columns whose logical
column index is `p mod 4`:

```text
%x_p0 lanes r*8 .. r*8+7 =
  row_r[0], row_r[4], row_r[8],  ..., row_r[28]

%x_p1 lanes r*8 .. r*8+7 =
  row_r[1], row_r[5], row_r[9],  ..., row_r[29]

%x_p2 lanes r*8 .. r*8+7 =
  row_r[2], row_r[6], row_r[10], ..., row_r[30]

%x_p3 lanes r*8 .. r*8+7 =
  row_r[3], row_r[7], row_r[11], ..., row_r[31]
```

Each physical part now has exactly 8 f32 values per row, so one `vcgadd` per
part computes one partial sum per row.  The four partial sums are then added
under `PAT_VL8`.

The full contiguous-to-4-way materialization for one tile should fuse the first
deinterleave level into the load.  `vldsx2 DINTLV_B32` loads `2 * VL` bytes and
splits even/odd f32 elements into two physical vectors.  Two such loads cover
the `8x32xf32` tile, and a second register `vdintlv` level splits even columns
into `mod4 = 0/2` and odd columns into `mod4 = 1/3`.

This setup documentation is repeated inside every complete 32-wide endpoint
below.

```text
%x_even_0, %x_odd_0 = pto.vldsx2 %base[%tile_off_0], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_even_1, %x_odd_1 = pto.vldsx2 %base[%tile_off_1], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%x_p0, %x_p2 = pto.vdintlv %x_even_0, %x_even_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_p1, %x_p3 = pto.vdintlv %x_odd_0, %x_odd_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
```

Each endpoint below inlines this materialization before the first consumer of
`%x_p0..%x_p3`.

#### 3.6.1 Reduce And Store Group Sums

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<256xf32> -> !pto.vmi.vreg<256xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
pto.vmi.group_store %sum, %sum_out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%x:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>

%sum:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering result for one full 8-row tile:

```text
%x_even_0, %x_odd_0 = pto.vldsx2 %base[%tile_off_0], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_even_1, %x_odd_1 = pto.vldsx2 %base[%tile_off_1], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%x_p0, %x_p2 = pto.vdintlv %x_even_0, %x_even_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_p1, %x_p3 = pto.vdintlv %x_odd_0, %x_odd_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"

%s0 = pto.vcgadd %x_p0, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s1 = pto.vcgadd %x_p1, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s2 = pto.vcgadd %x_p2, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s3 = pto.vcgadd %x_p3, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%s01 = pto.vadd %s0, %s1, %sum_mask : !pto.vreg<64xf32>
%s23 = pto.vadd %s2, %s3, %sum_mask : !pto.vreg<64xf32>
%sum_block = pto.vadd %s01, %s23, %sum_mask : !pto.vreg<64xf32>

pto.vsts %sum_block, %sum_out[%group_tile_off], %sum_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  sum_out[group_tile_off + r] = reduce(row_r[0..31])
```

#### 3.6.2 Reduce Result, Elementwise, Store

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<256xf32> -> !pto.vmi.vreg<256xf32>
%rhs = pto.vmi.group_slot_load %rhs_base[%rhs_off], %c1 {num_groups = 8}
  : !pto.ptr<f32, ub> -> !pto.vmi.vreg<256xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
%outv = pto.vmi.addf %sum, %rhs
pto.vmi.group_store %outv, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%x:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>

%sum, %rhs, %outv:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering result for one full 8-row tile:

```text
%x_even_0, %x_odd_0 = pto.vldsx2 %base[%tile_off_0], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_even_1, %x_odd_1 = pto.vldsx2 %base[%tile_off_1], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%x_p0, %x_p2 = pto.vdintlv %x_even_0, %x_even_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_p1, %x_p3 = pto.vdintlv %x_odd_0, %x_odd_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"
%one_block = pto.pge_b32 "PAT_VL1"

%s0 = pto.vcgadd %x_p0, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s1 = pto.vcgadd %x_p1, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s2 = pto.vcgadd %x_p2, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s3 = pto.vcgadd %x_p3, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%s01 = pto.vadd %s0, %s1, %sum_mask : !pto.vreg<64xf32>
%s23 = pto.vadd %s2, %s3, %sum_mask : !pto.vreg<64xf32>
%sum_block = pto.vadd %s01, %s23, %sum_mask : !pto.vreg<64xf32>

// Packed RHS group-slot load.  %rhs_tile_base points to rhs_base[rhs_off].
%rhs_block = pto.vsldb %rhs_tile_base, %c0_i16, %c0_i16, %one_block
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>

%outv_block = pto.vadd %sum_block, %rhs_block, %sum_mask
  : !pto.vreg<64xf32>

pto.vsts %outv_block, %out[%group_tile_off], %sum_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  out[group_tile_off + r] = reduce(row_r[0..31]) + rhs[r]
```

#### 3.6.3 Reduce, Broadcast, Elementwise, Reduce, Store

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<256xf32> -> !pto.vmi.vreg<256xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
%b = pto.vmi.group_broadcast %sum {num_groups = 8}
%y = pto.vmi.mulf %x, %b
%ysum = pto.vmi.group_reduce_addf %y, %mask {num_groups = 8}
pto.vmi.group_store %ysum, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%x, %b, %y:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>

%sum, %ysum:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering result for one full 8-row tile:

```text
%x_even_0, %x_odd_0 = pto.vldsx2 %base[%tile_off_0], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_even_1, %x_odd_1 = pto.vldsx2 %base[%tile_off_1], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%x_p0, %x_p2 = pto.vdintlv %x_even_0, %x_even_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_p1, %x_p3 = pto.vdintlv %x_odd_0, %x_odd_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"

%s0 = pto.vcgadd %x_p0, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s1 = pto.vcgadd %x_p1, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s2 = pto.vcgadd %x_p2, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s3 = pto.vcgadd %x_p3, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%s01 = pto.vadd %s0, %s1, %sum_mask : !pto.vreg<64xf32>
%s23 = pto.vadd %s2, %s3, %sum_mask : !pto.vreg<64xf32>
%sum_block = pto.vadd %s01, %s23, %sum_mask : !pto.vreg<64xf32>

%lane_id = pto.vci %c0_i32 : i32 -> !pto.vreg<64xi32>
%broadcast_idx = pto.vshrs %lane_id, %c3_i16, %all_b32
  : !pto.vreg<64xi32>, i16, !pto.mask<b32> -> !pto.vreg<64xi32>

// group_broadcast materialized for each deinterleaved=4 physical part.
%b_p0 = pto.vselr %sum_block, %broadcast_idx
  : !pto.vreg<64xf32>, !pto.vreg<64xi32> -> !pto.vreg<64xf32>
%b_p1 = pto.vselr %sum_block, %broadcast_idx
  : !pto.vreg<64xf32>, !pto.vreg<64xi32> -> !pto.vreg<64xf32>
%b_p2 = pto.vselr %sum_block, %broadcast_idx
  : !pto.vreg<64xf32>, !pto.vreg<64xi32> -> !pto.vreg<64xf32>
%b_p3 = pto.vselr %sum_block, %broadcast_idx
  : !pto.vreg<64xf32>, !pto.vreg<64xi32> -> !pto.vreg<64xf32>

%y_p0 = pto.vmul %x_p0, %b_p0, %all_b32 : !pto.vreg<64xf32>
%y_p1 = pto.vmul %x_p1, %b_p1, %all_b32 : !pto.vreg<64xf32>
%y_p2 = pto.vmul %x_p2, %b_p2, %all_b32 : !pto.vreg<64xf32>
%y_p3 = pto.vmul %x_p3, %b_p3, %all_b32 : !pto.vreg<64xf32>

%ys0 = pto.vcgadd %y_p0, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%ys1 = pto.vcgadd %y_p1, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%ys2 = pto.vcgadd %y_p2, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%ys3 = pto.vcgadd %y_p3, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%ys01 = pto.vadd %ys0, %ys1, %sum_mask : !pto.vreg<64xf32>
%ys23 = pto.vadd %ys2, %ys3, %sum_mask : !pto.vreg<64xf32>
%ysum_block = pto.vadd %ys01, %ys23, %sum_mask : !pto.vreg<64xf32>

pto.vsts %ysum_block, %out[%group_tile_off], %sum_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  s = reduce(row_r[0..31])
  out[group_tile_off + r] =
      reduce_i(row_r[i] * s for i = 0..31)
    = s * s
```

### 3.7 `group_reduce` S=64 f32, row-local reduction

This case covers one `8x64xf32` tile. Each logical row is exactly 256B, so the
input does not need a deinterleaved layout:

```text
row r = 64xf32 = one !pto.vreg<64xf32>
```

The reduction is two-stage but row-local:

```text
vcgadd(row_r)       -> 8 partial sums in lanes 0..7
vcadd(PAT_VL8)     -> one row sum in lane 0
```

The result layout is therefore not `slots = 8`. It is:

```text
#pto.vmi.layout<num_groups = 8, slots = 1>
```

Physical slot mapping for this tile:

```text
slot_block(r) = r
slot_lane(r)  = 0

%sum0 lane 0 = reduce row0 lanes 0..63
%sum1 lane 0 = reduce row1 lanes 0..63
...
%sum7 lane 0 = reduce row7 lanes 0..63
```

Trying to canonicalize this result to `slots = 8` would require packing lane 0
from eight different physical vregs into lanes 0..7 of one vreg. This document
does not use that plan. `slots = 1` is the canonical layout for S=64 row-local
group reductions.

#### 3.7.1 Reduce And Store Group Sums

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<512xf32> -> !pto.vmi.vreg<512xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
pto.vmi.group_store %sum, %sum_out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%x:
  !pto.vmi.vreg<512xf32, #pto.vmi.layout<contiguous>>

%sum:
  !pto.vmi.vreg<512xf32, #pto.vmi.layout<num_groups = 8, slots = 1>>
```

VPTO lowering result for one full 8-row tile:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%block8 = pto.pge_b32 "PAT_VL8"
%one_b32 = pto.pge_b32 "PAT_VL1"

%x0 = pto.vlds %base[%row_off_0] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x1 = pto.vlds %base[%row_off_1] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x2 = pto.vlds %base[%row_off_2] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x3 = pto.vlds %base[%row_off_3] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x4 = pto.vlds %base[%row_off_4] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x5 = pto.vlds %base[%row_off_5] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x6 = pto.vlds %base[%row_off_6] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x7 = pto.vlds %base[%row_off_7] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

%p0 = pto.vcgadd %x0, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%p1 = pto.vcgadd %x1, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%p2 = pto.vcgadd %x2, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%p3 = pto.vcgadd %x3, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%p4 = pto.vcgadd %x4, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%p5 = pto.vcgadd %x5, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%p6 = pto.vcgadd %x6, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%p7 = pto.vcgadd %x7, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%sum0 = pto.vcadd %p0, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum1 = pto.vcadd %p1, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum2 = pto.vcadd %p2, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum3 = pto.vcadd %p3, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum4 = pto.vcadd %p4, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum5 = pto.vcadd %p5, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum6 = pto.vcadd %p6, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum7 = pto.vcadd %p7, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

pto.vsts %sum0, %sum_out[%group_tile_off_0], %one_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %sum1, %sum_out[%group_tile_off_1], %one_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %sum2, %sum_out[%group_tile_off_2], %one_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %sum3, %sum_out[%group_tile_off_3], %one_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %sum4, %sum_out[%group_tile_off_4], %one_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %sum5, %sum_out[%group_tile_off_5], %one_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %sum6, %sum_out[%group_tile_off_6], %one_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %sum7, %sum_out[%group_tile_off_7], %one_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  sum_out[group_tile_off + r] = reduce(row_r[0..63])
```

#### 3.7.2 Reduce Result, Elementwise, Store

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<512xf32> -> !pto.vmi.vreg<512xf32>
%rhs = pto.vmi.group_slot_load %rhs_base[%rhs_off], %c1 {num_groups = 8}
  : !pto.ptr<f32, ub> -> !pto.vmi.vreg<512xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
%outv = pto.vmi.addf %sum, %rhs
pto.vmi.group_store %outv, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%x:
  !pto.vmi.vreg<512xf32, #pto.vmi.layout<contiguous>>

%sum, %rhs, %outv:
  !pto.vmi.vreg<512xf32, #pto.vmi.layout<num_groups = 8, slots = 1>>
```

VPTO lowering result for one full 8-row tile:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%block8 = pto.pge_b32 "PAT_VL8"
%one_b32 = pto.pge_b32 "PAT_VL1"

%x0 = pto.vlds %base[%row_off_0] {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x1 = pto.vlds %base[%row_off_1] {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x2 = pto.vlds %base[%row_off_2] {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x3 = pto.vlds %base[%row_off_3] {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x4 = pto.vlds %base[%row_off_4] {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x5 = pto.vlds %base[%row_off_5] {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x6 = pto.vlds %base[%row_off_6] {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x7 = pto.vlds %base[%row_off_7] {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

%p0 = pto.vcgadd %x0, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%p1 = pto.vcgadd %x1, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%p2 = pto.vcgadd %x2, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%p3 = pto.vcgadd %x3, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%p4 = pto.vcgadd %x4, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%p5 = pto.vcgadd %x5, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%p6 = pto.vcgadd %x6, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%p7 = pto.vcgadd %x7, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%sum0 = pto.vcadd %p0, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum1 = pto.vcadd %p1, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum2 = pto.vcadd %p2, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum3 = pto.vcadd %p3, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum4 = pto.vcadd %p4, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum5 = pto.vcadd %p5, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum6 = pto.vcadd %p6, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum7 = pto.vcadd %p7, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%rhs0 = pto.vsldb %rhs_ptr_0, %c0_i16, %c0_i16, %one_b32
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>
%rhs1 = pto.vsldb %rhs_ptr_1, %c0_i16, %c0_i16, %one_b32
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>
%rhs2 = pto.vsldb %rhs_ptr_2, %c0_i16, %c0_i16, %one_b32
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>
%rhs3 = pto.vsldb %rhs_ptr_3, %c0_i16, %c0_i16, %one_b32
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>
%rhs4 = pto.vsldb %rhs_ptr_4, %c0_i16, %c0_i16, %one_b32
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>
%rhs5 = pto.vsldb %rhs_ptr_5, %c0_i16, %c0_i16, %one_b32
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>
%rhs6 = pto.vsldb %rhs_ptr_6, %c0_i16, %c0_i16, %one_b32
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>
%rhs7 = pto.vsldb %rhs_ptr_7, %c0_i16, %c0_i16, %one_b32
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>

%out0 = pto.vadd %sum0, %rhs0, %one_b32 : !pto.vreg<64xf32>
%out1 = pto.vadd %sum1, %rhs1, %one_b32 : !pto.vreg<64xf32>
%out2 = pto.vadd %sum2, %rhs2, %one_b32 : !pto.vreg<64xf32>
%out3 = pto.vadd %sum3, %rhs3, %one_b32 : !pto.vreg<64xf32>
%out4 = pto.vadd %sum4, %rhs4, %one_b32 : !pto.vreg<64xf32>
%out5 = pto.vadd %sum5, %rhs5, %one_b32 : !pto.vreg<64xf32>
%out6 = pto.vadd %sum6, %rhs6, %one_b32 : !pto.vreg<64xf32>
%out7 = pto.vadd %sum7, %rhs7, %one_b32 : !pto.vreg<64xf32>

pto.vsts %out0, %out[%group_tile_off_0], %one_b32 {dist = "NORM_B32"} : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %out1, %out[%group_tile_off_1], %one_b32 {dist = "NORM_B32"} : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %out2, %out[%group_tile_off_2], %one_b32 {dist = "NORM_B32"} : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %out3, %out[%group_tile_off_3], %one_b32 {dist = "NORM_B32"} : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %out4, %out[%group_tile_off_4], %one_b32 {dist = "NORM_B32"} : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %out5, %out[%group_tile_off_5], %one_b32 {dist = "NORM_B32"} : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %out6, %out[%group_tile_off_6], %one_b32 {dist = "NORM_B32"} : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %out7, %out[%group_tile_off_7], %one_b32 {dist = "NORM_B32"} : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  out[group_tile_off + r] = reduce(row_r[0..63]) + rhs[r]
```

#### 3.7.3 Reduce, Broadcast, Elementwise, Reduce, Store

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<512xf32> -> !pto.vmi.vreg<512xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
%b = pto.vmi.group_broadcast %sum {num_groups = 8}
%y = pto.vmi.mulf %x, %b
%ysum = pto.vmi.group_reduce_addf %y, %mask {num_groups = 8}
pto.vmi.group_store %ysum, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%x, %b, %y:
  !pto.vmi.vreg<512xf32, #pto.vmi.layout<contiguous>>

%sum, %ysum:
  !pto.vmi.vreg<512xf32, #pto.vmi.layout<num_groups = 8, slots = 1>>
```

VPTO lowering result for one full 8-row tile:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%block8 = pto.pge_b32 "PAT_VL8"
%one_b32 = pto.pge_b32 "PAT_VL1"

// The compiler emits this row-local block once for each r in 0..7.
%x_r = pto.vlds %base[%row_off_r] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

%p_r = pto.vcgadd %x_r, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum_r = pto.vcadd %p_r, %block8
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

// This vdup is the lowering of pto.vmi.group_broadcast for slots=1.
%b_r = pto.vdup %sum_r, %all_b32 {position = "LOWEST"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%y_r = pto.vmul %x_r, %b_r, %all_b32 : !pto.vreg<64xf32>

%yp_r = pto.vcgadd %y_r, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%ysum_r = pto.vcadd %yp_r, %block8
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

pto.vsts %ysum_r, %out[%group_tile_off_r], %one_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

The row-local block above is not a runtime loop requirement. It is the repeated
VPTO shape for row offsets `%row_off_0` through `%row_off_7` and store offsets
`%group_tile_off_0` through `%group_tile_off_7`.

Memory result:

```text
for r = 0..7:
  s = reduce(row_r[0..63])
  out[group_tile_off + r] =
      reduce_i(row_r[i] * s for i = 0..63)
    = s * s
```

### 3.8 `group_reduce -> truncf -> group_broadcast -> store`

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<128xf32> -> !pto.vmi.vreg<128xf32>
%sum32 = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
%sum16 = pto.vmi.truncf %sum32
%b16   = pto.vmi.group_broadcast %sum16 {num_groups = 8}
pto.vmi.store %b16, %out[%off]
```

Assigned layouts:

```text
%x     : !pto.vmi.vreg<128xf32,
           #pto.vmi.layout<deinterleaved = 2, block_elems = 8>>
%sum32 : !pto.vmi.vreg<128xf32,
           #pto.vmi.layout<num_groups = 8, slots = 8>>
%sum16 : semantic value only; not materialized as a group-slot VPTO value
%b32   : !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>
%b16   : !pto.vmi.vreg<128xf16, #pto.vmi.layout<contiguous>>
```

This case is supported by commuting `truncf` after `group_broadcast`:

```text
group_broadcast(truncf(group_reduce(x)))
  == truncf(group_broadcast(group_reduce(x)))
```

This avoids materializing a group-slot f16 value. The only cast emitted is the
existing dense `f32 deinterleaved=2 -> contiguous f16` truncation.

VPTO lowering result for one full 8-row tile:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"

%x_lo, %x_hi = pto.vldsx2 %base[%tile_off], "BDINTLV"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%x_lo_sum = pto.vcgadd %x_lo, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%x_hi_sum = pto.vcgadd %x_hi, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%sum32_block = pto.vadd %x_lo_sum, %x_hi_sum, %sum_mask
  : !pto.vreg<64xf32>

%lane_id = pto.vci %c0_i32 : i32 -> !pto.vreg<64xi32>
%broadcast_idx = pto.vshrs %lane_id, %c3_i16, %all_b32
  : !pto.vreg<64xi32>, i16, !pto.mask<b32> -> !pto.vreg<64xi32>

// This vselr is the VPTO lowering of pto.vmi.group_broadcast.  The later store
// only writes lanes as-is; it does not duplicate group-slot values.
%b32_rows = pto.vselr %sum32_block, %broadcast_idx
  : !pto.vreg<64xf32>, !pto.vreg<64xi32> -> !pto.vreg<64xf32>

// The broadcasted f32 value is dense deinterleaved=2.
// Both parity parts carry the same per-row broadcast values.
%b16_even = pto.vcvt %b32_rows, %all_b32 {part = "EVEN", rnd = "R", sat = "SAT"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>
%b16_odd = pto.vcvt %b32_rows, %all_b32 {part = "ODD", rnd = "R", sat = "SAT"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>

%all_b16 = pto.pge_b16 "PAT_ALL"
%b16 = pto.vor %b16_even, %b16_odd, %all_b16
  : !pto.vreg<128xf16>

pto.vsts %b16, %out[%off], %all_b16 {dist = "NORM_B16"}
  : !pto.vreg<128xf16>, !pto.ptr<f16, ub>, !pto.mask<b16>
```

Memory result:

```text
for r = 0..7:
  s32 = reduce(row_r[0..15])
  s16 = truncf(s32)
  out[r * 16 + 0 .. r * 16 + 15] = splat(s16)
```

### 3.9 Illegal Dense Consumer Of Group Slots

VMI input:

```text
%sum32 = pto.vmi.group_reduce_addf %x, %mask {num_groups = G}
pto.vmi.store %sum32, %out[%off]
```

Assigned layouts before the illegal consumer:

```text
%sum32 : group_slots(G,K)
```

Required diagnostic:

```text
VMI-LAYOUT-CONTRACT:
  pto.vmi.store cannot consume #pto.vmi.layout<num_groups = G, slots = K>
  as a dense vector. Use pto.vmi.group_store, pto.vmi.group_broadcast, or an
  explicit group-pack op.
```

It must not be diagnosed as:

```text
dense store materializes group slots implicitly
```

That behavior would silently reinterpret a sparse group-slot value as a dense
vector.

### 3.10 Non-Load Producer Feeding S=32 `group_reduce`

This case proves that layout assignment is consumer-driven. The producer of the
S=32 input is an elementwise op, not a load. The S=32 `group_reduce` still
requires the elementwise result to be `deinterleaved = 4`, and that requirement
must propagate backward through the elementwise op to both operands.

VMI input:

```text
%a = pto.vmi.load %base[%off]
  : memref<256xf32> -> !pto.vmi.vreg<256xf32>
%bias = pto.vmi.broadcast %bias_s
  : f32 -> !pto.vmi.vreg<256xf32>
%x = pto.vmi.addf %a, %bias
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
pto.vmi.group_store %sum, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%a, %bias, %x:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>

%sum:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering result for one full `8x32xf32` tile:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"

%a_even_0, %a_odd_0 = pto.vldsx2 %base[%tile_off_0], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%a_even_1, %a_odd_1 = pto.vldsx2 %base[%tile_off_1], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%a_p0, %a_p2 = pto.vdintlv %a_even_0, %a_even_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%a_p1, %a_p3 = pto.vdintlv %a_odd_0, %a_odd_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%bias_p0 = pto.vdup %bias_s, %all_b32 : f32, !pto.mask<b32> -> !pto.vreg<64xf32>
%bias_p1 = pto.vdup %bias_s, %all_b32 : f32, !pto.mask<b32> -> !pto.vreg<64xf32>
%bias_p2 = pto.vdup %bias_s, %all_b32 : f32, !pto.mask<b32> -> !pto.vreg<64xf32>
%bias_p3 = pto.vdup %bias_s, %all_b32 : f32, !pto.mask<b32> -> !pto.vreg<64xf32>

%x_p0 = pto.vadd %a_p0, %bias_p0, %all_b32 : !pto.vreg<64xf32>
%x_p1 = pto.vadd %a_p1, %bias_p1, %all_b32 : !pto.vreg<64xf32>
%x_p2 = pto.vadd %a_p2, %bias_p2, %all_b32 : !pto.vreg<64xf32>
%x_p3 = pto.vadd %a_p3, %bias_p3, %all_b32 : !pto.vreg<64xf32>

%s0 = pto.vcgadd %x_p0, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s1 = pto.vcgadd %x_p1, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s2 = pto.vcgadd %x_p2, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s3 = pto.vcgadd %x_p3, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%s01 = pto.vadd %s0, %s1, %sum_mask : !pto.vreg<64xf32>
%s23 = pto.vadd %s2, %s3, %sum_mask : !pto.vreg<64xf32>
%sum_block = pto.vadd %s01, %s23, %sum_mask : !pto.vreg<64xf32>

pto.vsts %sum_block, %out[%group_tile_off], %sum_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  out[group_tile_off + r] =
    reduce_i(base[row_r, i] + bias_s for i = 0..31)
```

### 3.11 Partial Tail Groups

Tail handling must be separated by the physical input layout. Row-local S=64
can avoid inactive rows entirely. Load-fused S=16/S=32 cannot safely do that
with the current `vldsx2` materialization unless the source is known to be
full-tile readable.

#### 3.11.1 S=64 Active Row Tail

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<384xf32> -> !pto.vmi.vreg<384xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 6}
pto.vmi.group_store %sum, %out[%group_off], %c1 {num_groups = 6}
```

Assigned layouts:

```text
%x:
  !pto.vmi.vreg<384xf32, #pto.vmi.layout<contiguous>>

%sum:
  !pto.vmi.vreg<384xf32, #pto.vmi.layout<num_groups = 6, slots = 1>>
```

VPTO lowering result:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%block8 = pto.pge_b32 "PAT_VL8"
%one_b32 = pto.pge_b32 "PAT_VL1"

// Emit this row-local block for r = 0..5 only. No load or store is emitted for
// rows 6 and 7.
%x_r = pto.vlds %base[%row_off_r] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%p_r = pto.vcgadd %x_r, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum_r = pto.vcadd %p_r, %block8
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
pto.vsts %sum_r, %out[%group_tile_off_r], %one_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..5:
  out[group_tile_off + r] = reduce(row_r[0..63])
```

#### 3.11.2 S=32 Tail Without Full-Tile Read Contract

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<192xf32> -> !pto.vmi.vreg<192xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 6}
pto.vmi.group_store %sum, %out[%group_off], %c1 {num_groups = 6}
```

Assigned layout requested by the consumer:

```text
%x:
  !pto.vmi.vreg<192xf32, #pto.vmi.layout<deinterleaved = 4>>

%sum:
  !pto.vmi.vreg<192xf32, #pto.vmi.layout<num_groups = 6, slots = 8>>
```

Required diagnostic when the source does not carry a full-tile-readable
contract:

```text
VMI-LAYOUT-CONTRACT:
  pto.vmi.group_reduce_addf with group size 32 and num_groups tail 6 requires
  materializing #pto.vmi.layout<deinterleaved = 4>. The registered fast plan
  uses vldsx2 DINTLV_B32 over a full 8-row tile. This source is not marked
  full-tile-readable, and the stable gather tail plan is not implemented.
```

If a future option enables the stable gather tail plan, the same VMI input may
lower by gathering only the active lanes. Until that plan is registered, the
converter must not silently issue the full-tile `vldsx2` loads.

### 3.12 Control-Flow Join Before `group_reduce`

The layout carried by a value must survive block arguments. In MLIR converter
terms, the logical VMI value lowered through control flow becomes a tuple of
physical VPTO values with one tuple type per assigned layout.

VMI input:

```text
%x = scf.if %cond -> !pto.vmi.vreg<256xf32> {
  %a = pto.vmi.load %a_base[%a_off]
    : memref<256xf32> -> !pto.vmi.vreg<256xf32>
  scf.yield %a : !pto.vmi.vreg<256xf32>
} else {
  %b = pto.vmi.load %b_base[%b_off]
    : memref<256xf32> -> !pto.vmi.vreg<256xf32>
  scf.yield %b : !pto.vmi.vreg<256xf32>
}
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
pto.vmi.group_store %sum, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%a, %b, %x:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>

%sum:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering result for the join:

```text
%x_p0, %x_p1, %x_p2, %x_p3 =
  scf.if %cond
    -> (!pto.vreg<64xf32>, !pto.vreg<64xf32>,
        !pto.vreg<64xf32>, !pto.vreg<64xf32>) {
    %a_even_0, %a_odd_0 = pto.vldsx2 %a_base[%a_tile_off_0], "DINTLV_B32"
      : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
    %a_even_1, %a_odd_1 = pto.vldsx2 %a_base[%a_tile_off_1], "DINTLV_B32"
      : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
    %a_p0, %a_p2 = pto.vdintlv %a_even_0, %a_even_1
      : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
    %a_p1, %a_p3 = pto.vdintlv %a_odd_0, %a_odd_1
      : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
    scf.yield %a_p0, %a_p1, %a_p2, %a_p3
      : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.vreg<64xf32>
  } else {
    %b_even_0, %b_odd_0 = pto.vldsx2 %b_base[%b_tile_off_0], "DINTLV_B32"
      : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
    %b_even_1, %b_odd_1 = pto.vldsx2 %b_base[%b_tile_off_1], "DINTLV_B32"
      : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
    %b_p0, %b_p2 = pto.vdintlv %b_even_0, %b_even_1
      : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
    %b_p1, %b_p3 = pto.vdintlv %b_odd_0, %b_odd_1
      : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
    scf.yield %b_p0, %b_p1, %b_p2, %b_p3
      : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.vreg<64xf32>
  }
```

The consumer after the join is the same S=32 reduction plan as section 3.6:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"

%s0 = pto.vcgadd %x_p0, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s1 = pto.vcgadd %x_p1, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s2 = pto.vcgadd %x_p2, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s3 = pto.vcgadd %x_p3, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%s01 = pto.vadd %s0, %s1, %sum_mask : !pto.vreg<64xf32>
%s23 = pto.vadd %s2, %s3, %sum_mask : !pto.vreg<64xf32>
%sum_block = pto.vadd %s01, %s23, %sum_mask : !pto.vreg<64xf32>

pto.vsts %sum_block, %out[%group_tile_off], %sum_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  selected_row = cond ? a_row_r : b_row_r
  out[group_tile_off + r] = reduce(selected_row[0..31])
```

If the two branches cannot be assigned the same layout and no materialization
plan exists before `scf.yield`, the required diagnostic is:

```text
VMI-LAYOUT-CONTRACT:
  scf.yield joins incompatible VMI layouts for !pto.vmi.vreg<256xf32>.
  Expected #pto.vmi.layout<deinterleaved = 4> on every incoming value.
```

### 3.13 Direct Group-Slot `f32 -> f16` Cast

This case is intentionally illegal for the current S=16/S=32 packed
group-slot layout. It prevents the compiler from treating a width-changing
`vcvt` as if it preserved low-lane group slots.

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<128xf32> -> !pto.vmi.vreg<128xf32>
%sum32 = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
%sum16 = pto.vmi.truncf %sum32
pto.vmi.group_store %sum16, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts before the illegal cast:

```text
%x:
  !pto.vmi.vreg<128xf32,
    #pto.vmi.layout<deinterleaved = 2, block_elems = 8>>

%sum32:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

Required diagnostic:

```text
VMI-LAYOUT-CONTRACT:
  pto.vmi.truncf cannot lower from
  #pto.vmi.layout<num_groups = 8, slots = 8> f32 to f16 because no
  slot-preserving width-changing VPTO plan is registered. f32->f16 vcvt writes
  even/odd sub-lanes, not lanes 0..7. Use group_broadcast before truncf, or
  keep the group_store element type as f32.
```

This does not contradict section 3.8. Section 3.8 is legal because the cast is
commuted after `group_broadcast`, where the value is dense again.

### 3.14 Unsupported Group Size

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<96xf32> -> !pto.vmi.vreg<96xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
pto.vmi.group_store %sum, %out[%group_off], %c1 {num_groups = 8}
```

Here `S = 96 / 8 = 12` f32 elements per group. The current VCG-based plans use
32B groups, i.e. 8 f32 elements per row fragment:

```text
S = 8   -> one VCGADD block per group
S = 16  -> two 8-lane row fragments, add partial sums
S = 32  -> four 8-lane row fragments, add partial sums
S = 64  -> one full 256B row, VCGADD then VCADD
```

Required diagnostic:

```text
VMI-LAYOUT-CONTRACT:
  pto.vmi.group_reduce_addf with f32 group size 12 has no registered VPTO
  layout plan. Supported VCG-based f32 group sizes are 8, 16, 32, and 64.
  A scalar/gather fallback or a rewrite to logical group size 16 with an
  explicit per-group mask is required.
```

### 3.15 Compact S=12 Written As Logical S=16

If the program wants to use the S=16 lowering for data with 12 semantic f32
elements per group, the IR must distinguish two sizes:

```text
logical group size used by VMI ops: 16
active elements per group:          12
```

The mask is not a prefix mask over the whole vector. It is a per-group mask:

```text
mask lane i is active iff (i % 16) < 12
```

The group load surface carries the physical source stride as an SSA operand:

```text
%x = pto.vmi.group_load %base[%off], %source_group_stride
  {num_groups = G, group_size = S}
  : !pto.ptr<T, ub>, index -> !pto.vmi.vreg<NxT>
```

`source_group_stride` is in elements, not bytes. It is an operand because it may
come from a dynamic leading dimension, a subview, or a runtime tile descriptor.
Static strides use a constant index operand and can be canonicalized later.
`group_size` remains an attribute in this design because it selects the logical
load layout. `active_elems_per_group` belongs to the mask producer, not to the
load.

Grouped masks use a paired `pto.vmi.create_group_mask` op. It is intentionally
separate from ordinary prefix `pto.vmi.create_mask` so the IR makes group
semantics explicit next to `pto.vmi.group_load` / `pto.vmi.group_reduce_*`:

```text
%mask = pto.vmi.create_group_mask %active_elems_per_group
  {num_groups = G, group_size = S}
  : index -> !pto.vmi.mask<(G*S)xpred>
```

Semantics:

```text
lane i is active iff (i % S) < active_elems_per_group
```

Ordinary `pto.vmi.create_mask %active_lanes` keeps the prefix-mask meaning:

```text
lane i is active iff i < active_lanes
```

#### 3.15.1 Existing Design Works If Source Row Stride Is 16

If memory already has a 16-f32 row stride, the user can write a logical S=16
tile and mask off the last four lanes of every group.

VMI input:

```text
%stride16 = arith.constant 16 : index
%x = pto.vmi.group_load %base[%off], %stride16
  {num_groups = 8, group_size = 16}
  : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<128xf32>
%c12 = arith.constant 12 : index
%mask = pto.vmi.create_group_mask %c12 {num_groups = 8, group_size = 16}
  : index -> !pto.vmi.mask<128xpred>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
pto.vmi.group_store %sum, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%x:
  !pto.vmi.vreg<128xf32,
    #pto.vmi.layout<deinterleaved = 2, block_elems = 8>>

%mask:
  !pto.vmi.mask<128xpred,
    #pto.vmi.layout<deinterleaved = 2, block_elems = 8>>

%sum:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering result for one `8x16xf32` tile:

```text
%lo_mask = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"

%lane = pto.vci %c0_i32 : i32 -> !pto.vreg<64xi32>
%row = pto.vshrs %lane, %c3_i16, %lo_mask
  : !pto.vreg<64xi32>, i16, !pto.mask<b32> -> !pto.vreg<64xi32>
%row8 = pto.vshls %row, %c3_i16, %lo_mask
  : !pto.vreg<64xi32>, i16, !pto.mask<b32> -> !pto.vreg<64xi32>
%col = pto.vsub %lane, %row8, %lo_mask
  : !pto.vreg<64xi32>
%hi4_mask = pto.vcmps %col, %c4_i32, %lo_mask, "lt"
  : !pto.vreg<64xi32>, i32, !pto.mask<b32> -> !pto.mask<b32>

%lo, %hi = pto.vldsx2 %base[%tile_off], "BDINTLV"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%lo lanes r*8 .. r*8+7 = row_r[0..7]
%hi lanes r*8 .. r*8+3 = row_r[8..11]
%hi lanes r*8+4 .. r*8+7 = row_r[12..15]  // inactive by mask

%lo_sum = pto.vcgadd %lo, %lo_mask
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%hi_sum = pto.vcgadd %hi, %hi4_mask
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%sum_block = pto.vadd %lo_sum, %hi_sum, %sum_mask
  : !pto.vreg<64xf32>

pto.vsts %sum_block, %out[%group_tile_off], %sum_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  out[group_tile_off + r] = reduce(row_r[0..11])
```

Design requirement added by this case: VMI mask lowering must support
group-periodic masks by generating the predicate from lane indices. It must not
rewrite this mask to `PAT_M4`: VISA defines `M4` as multiples of 4, not the
first four lanes of each 8-lane block.

```text
lane = vci(0)
row  = lane >> 3
col  = lane - (row << 3)
mask = col < 4
```

#### 3.15.2 Source Row Stride Greater Than 16

For now, support the non-compact case where each physical row has at least 16
f32 slots and the row stride is greater than 16. The fast strided-block path
requires the row stride to be a multiple of one 32B block:

```text
source_group_stride % 8 == 0
```

The example below uses `source_group_stride = 24`. Each row has 12 semantic
values, 4 masked-but-readable slots, and 8 extra skipped slots:

```text
row_r[0..11]   semantic
row_r[12..15]  readable but inactive for the S=16 logical group
row_r[16..23]  outside the logical group
```

VMI input:

```text
%stride24 = arith.constant 24 : index
%x = pto.vmi.group_load %base[%off], %stride24
  {num_groups = 8, group_size = 16}
  : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<128xf32>
%c12 = arith.constant 12 : index
%mask = pto.vmi.create_group_mask %c12 {num_groups = 8, group_size = 16}
  : index -> !pto.vmi.mask<128xpred>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
pto.vmi.group_store %sum, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts are the same as section 3.15.1:

```text
%x, %mask:
  #pto.vmi.layout<deinterleaved = 2, block_elems = 8>
%sum:
  #pto.vmi.layout<num_groups = 8, slots = 8>
```

VPTO lowering result:

```text
%lo_mask = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"

%lane = pto.vci %c0_i32 : i32 -> !pto.vreg<64xi32>
%row = pto.vshrs %lane, %c3_i16, %lo_mask
  : !pto.vreg<64xi32>, i16, !pto.mask<b32> -> !pto.vreg<64xi32>
%row8 = pto.vshls %row, %c3_i16, %lo_mask
  : !pto.vreg<64xi32>, i16, !pto.mask<b32> -> !pto.vreg<64xi32>
%col = pto.vsub %lane, %row8, %lo_mask
  : !pto.vreg<64xi32>
%hi4_mask = pto.vcmps %col, %c4_i32, %lo_mask, "lt"
  : !pto.vreg<64xi32>, i32, !pto.mask<b32> -> !pto.mask<b32>

// source_group_stride = 24 f32 = 3 * 32B blocks.
%stride_blocks = %c3_i16

%base_lo = %base + tile_off
%base_hi = %base + tile_off + 8

%lo = pto.vsldb %base_lo, %stride_blocks, %c0_i16, %lo_mask
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>
%hi = pto.vsldb %base_hi, %stride_blocks, %c0_i16, %lo_mask
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>

%lo lanes r*8 .. r*8+7 = row_r[0..7]
%hi lanes r*8 .. r*8+7 = row_r[8..15]

%lo_sum = pto.vcgadd %lo, %lo_mask
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%hi_sum = pto.vcgadd %hi, %hi4_mask
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%sum_block = pto.vadd %lo_sum, %hi_sum, %sum_mask
  : !pto.vreg<64xf32>

pto.vsts %sum_block, %out[%group_tile_off], %sum_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  out[group_tile_off + r] =
    reduce(base[tile_off + r * 24 + 0 .. tile_off + r * 24 + 11])
```

If `source_group_stride > 16` but is not a multiple of 8 f32 elements, this
strided-block path is not legal because `vsldb` block addresses are 32B based.
That case remains unsupported until a gather materialization is selected.

#### 3.15.3 Compact Source Row Stride 12

Compact storage is explicitly out of scope for the first implementation:

```text
row0[0..11], row1[0..11], row2[0..11], ...
```

Required diagnostic:

```text
VMI-LAYOUT-CONTRACT:
  logical group size 16 with active_elems_per_group 12 and
  source_group_stride 12 requires compact-row gather materialization. This
  plan is not part of the initial VMI layout lowering.
```

### 3.16 `group_slot_load` Layout Contract

`group_slot_load` is separate from `group_load`.

```text
group_load:
  loads group_size data elements per group and produces dense grouped data.

group_slot_load:
  loads one scalar value per group and produces sparse group slots.
```

Surface form:

```text
%v = pto.vmi.group_slot_load %base[%off], %source_group_stride
  {num_groups = G}
  : !pto.ptr<T, ub>, index -> !pto.vmi.vreg<NxT>
```

Semantics:

```text
semantic group slot g = base[off + g * source_group_stride]
```

The result logical lane count `N` remains the surrounding VMI value shape. Only
the `G` group slots are semantic. Layout assignment chooses the sparse physical
placement requested by the consumer:

```text
#pto.vmi.layout<num_groups = G, slots = 8>
#pto.vmi.layout<num_groups = G, slots = 1>
```

#### 3.16.1 Packed `group_slot_load`, `slots = 8`

VMI input:

```text
%rhs = pto.vmi.group_slot_load %rhs_base[%rhs_off], %c1 {num_groups = 8}
  : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<128xf32>
pto.vmi.group_store %rhs, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layout:

```text
%rhs:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering result:

```text
%slot_mask = pto.pge_b32 "PAT_VL8"
%one_block = pto.pge_b32 "PAT_VL1"

// source_group_stride = 1, so one 32B block contains all 8 scalar group slots.
%rhs_block = pto.vsldb %rhs_base[%rhs_off], %c0_i16, %c0_i16, %one_block
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>

pto.vsts %rhs_block, %out[%group_off], %slot_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for g = 0..7:
  out[group_off + g] = rhs_base[rhs_off + g]
```

If `source_group_stride != 1`, this packed `slots = 8` plan requires a
strided/gather group-slot load materializer. Until that plan is registered,
`group_slot_load` with `slots = 8` and non-unit stride must diagnose instead of
silently using full-group `group_load`.

#### 3.16.2 Row-Local `group_slot_load`, `slots = 1`

VMI input:

```text
%rhs = pto.vmi.group_slot_load %rhs_base[%rhs_off], %c1 {num_groups = 8}
  : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<512xf32>
pto.vmi.group_store %rhs, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layout:

```text
%rhs:
  !pto.vmi.vreg<512xf32, #pto.vmi.layout<num_groups = 8, slots = 1>>
```

VPTO lowering result:

```text
%one_b32 = pto.pge_b32 "PAT_VL1"

// Emit this shape for r = 0..7.  Each result value carries one semantic slot
// in lane 0, matching the S=64 row-local group_reduce result layout.
%rhs_r = pto.vsldb %rhs_base[%rhs_off_plus_r], %c0_i16, %c0_i16, %one_b32
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>

pto.vsts %rhs_r, %out[%group_off_plus_r], %one_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  out[group_off + r] = rhs_base[rhs_off + r]
```

### 3.17 `group_broadcast` Physical Arity Alias

This case fixes a lowering invariant: a layout determines physical arity.  A
`deinterleaved = 2` result has two physical bundle entries even when both
entries can reuse the same VPTO SSA value.

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<128xf32> -> !pto.vmi.vreg<128xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
%b = pto.vmi.group_broadcast %sum {num_groups = 8}
%h = pto.vmi.truncf %b
pto.vmi.store %h, %out[%off]
```

Assigned layouts:

```text
%x:
  !pto.vmi.vreg<128xf32,
    #pto.vmi.layout<deinterleaved = 2, block_elems = 8>>

%sum:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>

%b:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>

%h:
  !pto.vmi.vreg<128xf16, #pto.vmi.layout<contiguous>>
```

VPTO lowering result:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"

%x_lo, %x_hi = pto.vldsx2 %base[%tile_off], "BDINTLV"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%lo_sum = pto.vcgadd %x_lo, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%hi_sum = pto.vcgadd %x_hi, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum_block = pto.vadd %lo_sum, %hi_sum, %sum_mask
  : !pto.vreg<64xf32>

%lane_id = pto.vci %c0_i32 : i32 -> !pto.vreg<64xi32>
%broadcast_idx = pto.vshrs %lane_id, %c3_i16, %all_b32
  : !pto.vreg<64xi32>, i16, !pto.mask<b32> -> !pto.vreg<64xi32>

%b_rows = pto.vselr %sum_block, %broadcast_idx
  : !pto.vreg<64xf32>, !pto.vreg<64xi32> -> !pto.vreg<64xf32>

// Physical bundle binding for %b, not emitted VPTO ops:
//   physical entry 0 = %b_rows
//   physical entry 1 = %b_rows
// The layout still has two physical entries; they alias the same SSA value
// because every even/odd logical lane pair contains the same broadcast value.

%h_even = pto.vcvt %b_rows, %all_b32 {part = "EVEN", rnd = "R", sat = "SAT"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>
%h_odd = pto.vcvt %b_rows, %all_b32 {part = "ODD", rnd = "R", sat = "SAT"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>

%all_b16 = pto.pge_b16 "PAT_ALL"
%h0 = pto.vor %h_even, %h_odd, %all_b16
  : !pto.vreg<128xf16>

pto.vsts %h0, %out[%off], %all_b16 {dist = "NORM_B16"}
  : !pto.vreg<128xf16>, !pto.ptr<f16, ub>, !pto.mask<b16>
```

Memory result:

```text
for r = 0..7:
  s = reduce(row_r[0..15])
  out[r * 16 + 0 .. r * 16 + 15] = truncf(s)
```

### 3.18 One Value With Dense And Group-Reduce Consumers

This case forces layout assignment to handle a solvable use-site conflict.  One
consumer requires an S=32 group-reduce layout; another consumer requires dense
row-major store.  This is not semantically illegal.  It must be solved by
use-site materialization or producer rematerialization when a registered plan
exists.

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<256xf32> -> !pto.vmi.vreg<256xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
pto.vmi.group_store %sum, %sum_out[%group_off], %c1 {num_groups = 8}
pto.vmi.store %x, %copy_out[%off]
```

Assigned layouts:

```text
%x for group_reduce:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>

%sum:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>

%x for dense store:
  requires #pto.vmi.layout<contiguous>
```

If `%x` is cheap to rematerialize, layout assignment may clone the producer for
the dense store.  Otherwise, if the registry has a `deinterleaved = 4 ->
contiguous` materialization plan, layout assignment may keep `%x` in
`deinterleaved = 4` and insert `ensure_layout` before the dense store.

VPTO lowering result:

```text
%x_even_0, %x_odd_0 = pto.vldsx2 %base[%tile_off_0], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_even_1, %x_odd_1 = pto.vldsx2 %base[%tile_off_1], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%x_p0, %x_p2 = pto.vdintlv %x_even_0, %x_even_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_p1, %x_p3 = pto.vdintlv %x_odd_0, %x_odd_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"

%s0 = pto.vcgadd %x_p0, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s1 = pto.vcgadd %x_p1, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s2 = pto.vcgadd %x_p2, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s3 = pto.vcgadd %x_p3, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%s01 = pto.vadd %s0, %s1, %sum_mask : !pto.vreg<64xf32>
%s23 = pto.vadd %s2, %s3, %sum_mask : !pto.vreg<64xf32>
%sum_block = pto.vadd %s01, %s23, %sum_mask : !pto.vreg<64xf32>

pto.vsts %sum_block, %sum_out[%group_off], %sum_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>

// Dense store materialization for the second consumer.
%even0, %even1 = pto.vintlv %x_p0, %x_p2
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%odd0, %odd1 = pto.vintlv %x_p1, %x_p3
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%d0, %d1 = pto.vintlv %even0, %odd0
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%d2, %d3 = pto.vintlv %even1, %odd1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

pto.vsts %d0, %copy_out[%off_0], %all_b32 {dist = "NORM_B32"}
pto.vsts %d1, %copy_out[%off_64], %all_b32 {dist = "NORM_B32"}
pto.vsts %d2, %copy_out[%off_128], %all_b32 {dist = "NORM_B32"}
pto.vsts %d3, %copy_out[%off_192], %all_b32 {dist = "NORM_B32"}
```

Memory result:

```text
for r = 0..7:
  sum_out[group_off + r] = reduce(row_r[0..31])

for i = 0..255:
  copy_out[off + i] = base[off + i]
```

If the `deinterleaved = 4 -> contiguous` plan is not registered, the required
diagnostic is:

```text
VMI-LAYOUT-CONTRACT:
  value %x is required as #pto.vmi.layout<deinterleaved = 4> by
  pto.vmi.group_reduce_addf and as #pto.vmi.layout<contiguous> by
  pto.vmi.store, but no registered materialization plan exists at the store
  use site.
```

### 3.19 S=16 Reduce `block_elems` Plan Selection

S=16 f32 group reduction has two legal dense input layouts:

```text
#pto.vmi.layout<deinterleaved = 2, block_elems = 1>
#pto.vmi.layout<deinterleaved = 2, block_elems = 8>
```

`block_elems = 1` is the element-parity layout required by f32->f16 `truncf`.
It is also a valid S=16 reduction layout: each physical part contains eight
values per row, so `VCGADD` can reduce each part and `VADD` can combine the two
partial sums.

`block_elems = 8` is still useful when the producer is a block load plan such
as `BDINTLV` or `vsldb` over 32B row fragments.  Layout assignment must select
between these plans by producer/consumer cost.  It must not hard-code S=16
reduce to `block_elems = 8`.

#### 3.19.1 Continuous S=16 Reduce And Truncf, `block_elems = 1`

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<128xf32> -> !pto.vmi.vreg<128xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
pto.vmi.group_store %sum, %sum_out[%group_off], %c1 {num_groups = 8}
%h = pto.vmi.truncf %x
  : !pto.vmi.vreg<128xf32> -> !pto.vmi.vreg<128xf16>
pto.vmi.store %h, %out[%off]
```

Assigned layouts:

```text
%x:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>

%sum:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>

%h:
  !pto.vmi.vreg<128xf16, #pto.vmi.layout<contiguous>>
```

Physical lane map:

```text
%x_p0 lanes r*8 .. r*8+7 =
  row_r[0], row_r[2], row_r[4], ..., row_r[14]

%x_p1 lanes r*8 .. r*8+7 =
  row_r[1], row_r[3], row_r[5], ..., row_r[15]
```

VPTO lowering result:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"

%x_p0, %x_p1 = pto.vldsx2 %base[%tile_off], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%s0 = pto.vcgadd %x_p0, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s1 = pto.vcgadd %x_p1, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum_block = pto.vadd %s0, %s1, %sum_mask
  : !pto.vreg<64xf32>

pto.vsts %sum_block, %sum_out[%group_off], %sum_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>

%h_even = pto.vcvt %x_p0, %all_b32 {part = "EVEN", rnd = "R", sat = "SAT"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>
%h_odd = pto.vcvt %x_p1, %all_b32 {part = "ODD", rnd = "R", sat = "SAT"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>

%all_b16 = pto.pge_b16 "PAT_ALL"
%h0 = pto.vor %h_even, %h_odd, %all_b16
  : !pto.vreg<128xf16>
pto.vsts %h0, %out[%off], %all_b16 {dist = "NORM_B16"}
  : !pto.vreg<128xf16>, !pto.ptr<f16, ub>, !pto.mask<b16>
```

Memory result:

```text
for r = 0..7:
  sum_out[group_off + r] = reduce(row_r[0..15])

for i = 0..127:
  out[off + i] = truncf(base[off + i])
```

#### 3.19.2 Block-Load Producer Fixed To `block_elems = 8`

This is the real conflict case.  The value is fixed to `block_elems = 8`
because the producer is a registered block-load plan.  A later `truncf`
requires element-parity `block_elems = 1`.

VMI input:

```text
%stride24 = arith.constant 24 : index
%x = pto.vmi.group_load %base[%off], %stride24
  {num_groups = 8, group_size = 16}
  : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<128xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
pto.vmi.group_store %sum, %sum_out[%group_off], %c1 {num_groups = 8}
%h = pto.vmi.truncf %x
  : !pto.vmi.vreg<128xf32> -> !pto.vmi.vreg<128xf16>
pto.vmi.store %h, %out[%off]
```

Assigned layouts before the conflicting `truncf` use:

```text
%x from strided block group_load:
  !pto.vmi.vreg<128xf32,
    #pto.vmi.layout<deinterleaved = 2, block_elems = 8>>

%sum:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

The reduction path is legal and uses the same `vsldb` block-load shape as
section 3.15.2.  The `truncf` path is legal only if one of these plans exists:

```text
1. rematerialize the original memory producer as block_elems=1
2. materialize block_elems=8 -> block_elems=1 in registers
3. use an explicitly enabled scratch/reload fallback
```

If no such plan is registered, the required diagnostic is:

```text
VMI-LAYOUT-CONTRACT:
  pto.vmi.truncf requires
  #pto.vmi.layout<deinterleaved = 2, block_elems = 1>, but the source value is
  fixed to #pto.vmi.layout<deinterleaved = 2, block_elems = 8> by the selected
  strided group_load plan. Register a rematerialization or preserving
  materialization plan, or avoid consuming this block-loaded value with truncf.
```

### 3.20 `group_slots` Control-Flow Join

`group_slots` values must be allowed to cross control flow.  The join type is a
sparse physical tuple, not a dense vector.

VMI input:

```text
%sum = scf.if %cond -> !pto.vmi.vreg<128xf32> {
  %x = pto.vmi.load %base[%off]
    : memref<128xf32> -> !pto.vmi.vreg<128xf32>
  %a = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
  scf.yield %a : !pto.vmi.vreg<128xf32>
} else {
  %b = pto.vmi.group_slot_load %rhs_base[%rhs_off], %c1 {num_groups = 8}
    : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<128xf32>
  scf.yield %b : !pto.vmi.vreg<128xf32>
}
%bias = pto.vmi.group_slot_load %bias_base[%bias_off], %c1 {num_groups = 8}
  : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<128xf32>
%outv = pto.vmi.addf %sum, %bias
pto.vmi.group_store %outv, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%x:
  !pto.vmi.vreg<128xf32,
    #pto.vmi.layout<deinterleaved = 2, block_elems = 8>>

%a, %b, %sum, %bias, %outv:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering result for the join:

```text
%sum_block = scf.if %cond -> !pto.vreg<64xf32> {
  %all_b32 = pto.pge_b32 "PAT_ALL"
  %sum_mask = pto.pge_b32 "PAT_VL8"

  %x_lo, %x_hi = pto.vldsx2 %base[%tile_off], "BDINTLV"
    : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
  %lo_sum = pto.vcgadd %x_lo, %all_b32
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  %hi_sum = pto.vcgadd %x_hi, %all_b32
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  %a_block = pto.vadd %lo_sum, %hi_sum, %sum_mask
    : !pto.vreg<64xf32>
  scf.yield %a_block : !pto.vreg<64xf32>
} else {
  %one_block = pto.pge_b32 "PAT_VL1"
  %b_block = pto.vsldb %rhs_base[%rhs_off], %c0_i16, %c0_i16, %one_block
    : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>
  scf.yield %b_block : !pto.vreg<64xf32>
}

%one_block = pto.pge_b32 "PAT_VL1"
%slot_mask = pto.pge_b32 "PAT_VL8"
%bias_block = pto.vsldb %bias_base[%bias_off], %c0_i16, %c0_i16, %one_block
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>
%out_block = pto.vadd %sum_block, %bias_block, %slot_mask
  : !pto.vreg<64xf32>

pto.vsts %out_block, %out[%group_off], %slot_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  lhs = cond ? reduce(row_r[0..15]) : rhs_base[rhs_off + r]
  out[group_off + r] = lhs + bias_base[bias_off + r]
```

### 3.21 S=32 Tail With Full-Tile-Readable Source

This is the positive counterpart to section 3.11.2.  Tail participation is
still expressed by masks, but the source additionally promises that reading the
rounded-up 8-row physical tile is memory-safe.

VMI input:

```text
%x = pto.vmi.load %base[%off] {full_tile_readable}
  : memref<192xf32> -> !pto.vmi.vreg<192xf32>
%mask = pto.vmi.create_mask %c192 : index -> !pto.vmi.mask<192xpred>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 6}
pto.vmi.group_store %sum, %out[%group_off], %c1 {num_groups = 6}
```

Assigned layouts:

```text
%x:
  !pto.vmi.vreg<192xf32, #pto.vmi.layout<deinterleaved = 4>>

%mask:
  !pto.vmi.mask<192xpred, #pto.vmi.layout<deinterleaved = 4>>

%sum:
  !pto.vmi.vreg<192xf32, #pto.vmi.layout<num_groups = 6, slots = 8>>
```

VPTO lowering result:

```text
// Full-tile-readable allows the load plan to read the rounded-up 8-row tile.
// Only rows 0..5 are semantically active.
%data_mask = pto.pge_b32 "PAT_VL48"  // 6 rows * 8 lanes per physical part
%sum_mask = pto.pge_b32 "PAT_VL6"

%x_even_0, %x_odd_0 = pto.vldsx2 %base[%tile_off_0], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_even_1, %x_odd_1 = pto.vldsx2 %base[%tile_off_1], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%x_p0, %x_p2 = pto.vdintlv %x_even_0, %x_even_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_p1, %x_p3 = pto.vdintlv %x_odd_0, %x_odd_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%s0 = pto.vcgadd %x_p0, %data_mask
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s1 = pto.vcgadd %x_p1, %data_mask
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s2 = pto.vcgadd %x_p2, %data_mask
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s3 = pto.vcgadd %x_p3, %data_mask
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%s01 = pto.vadd %s0, %s1, %sum_mask : !pto.vreg<64xf32>
%s23 = pto.vadd %s2, %s3, %sum_mask : !pto.vreg<64xf32>
%sum_block = pto.vadd %s01, %s23, %sum_mask : !pto.vreg<64xf32>

pto.vsts %sum_block, %out[%group_off], %sum_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..5:
  out[group_off + r] = reduce(row_r[0..31])
```

Rows 6 and 7 may be physically loaded because of `full_tile_readable`, but
their lanes are not active in `%data_mask`, and their group slots are not stored
because `%sum_mask` is `PAT_VL6`.

### 3.22 `scf.for` Loop-Carried Layout

Loop-carried VMI values require a layout fixed point.  The iter_arg, body block
argument, yield operand, loop result, and later consumer must all agree on one
layout, or `vmi-layout-assignment` must insert a materialization at a legal
dominating use site.

VMI input:

```text
%init = pto.vmi.load %base[%off]
  : memref<256xf32> -> !pto.vmi.vreg<256xf32>
%acc = scf.for %i = %c0 to %steps step %c1
    iter_args(%arg = %init) -> !pto.vmi.vreg<256xf32> {
  %bias = pto.vmi.broadcast %bias_s
    : f32 -> !pto.vmi.vreg<256xf32>
  %next = pto.vmi.addf %arg, %bias
  scf.yield %next : !pto.vmi.vreg<256xf32>
}
%sum = pto.vmi.group_reduce_addf %acc, %mask {num_groups = 8}
pto.vmi.group_store %sum, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%init, %arg, %bias, %next, %acc:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>

%sum:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering result:

```text
%init_even_0, %init_odd_0 = pto.vldsx2 %base[%tile_off_0], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%init_even_1, %init_odd_1 = pto.vldsx2 %base[%tile_off_1], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%init_p0, %init_p2 = pto.vdintlv %init_even_0, %init_even_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%init_p1, %init_p3 = pto.vdintlv %init_odd_0, %init_odd_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%acc_p0, %acc_p1, %acc_p2, %acc_p3 =
  scf.for %i = %c0 to %steps step %c1
      iter_args(%arg_p0 = %init_p0, %arg_p1 = %init_p1,
                %arg_p2 = %init_p2, %arg_p3 = %init_p3)
      -> (!pto.vreg<64xf32>, !pto.vreg<64xf32>,
          !pto.vreg<64xf32>, !pto.vreg<64xf32>) {
    %all_b32 = pto.pge_b32 "PAT_ALL"
    %bias_p0 = pto.vdup %bias_s, %all_b32
      : f32, !pto.mask<b32> -> !pto.vreg<64xf32>
    %bias_p1 = pto.vdup %bias_s, %all_b32
      : f32, !pto.mask<b32> -> !pto.vreg<64xf32>
    %bias_p2 = pto.vdup %bias_s, %all_b32
      : f32, !pto.mask<b32> -> !pto.vreg<64xf32>
    %bias_p3 = pto.vdup %bias_s, %all_b32
      : f32, !pto.mask<b32> -> !pto.vreg<64xf32>

    %next_p0 = pto.vadd %arg_p0, %bias_p0, %all_b32 : !pto.vreg<64xf32>
    %next_p1 = pto.vadd %arg_p1, %bias_p1, %all_b32 : !pto.vreg<64xf32>
    %next_p2 = pto.vadd %arg_p2, %bias_p2, %all_b32 : !pto.vreg<64xf32>
    %next_p3 = pto.vadd %arg_p3, %bias_p3, %all_b32 : !pto.vreg<64xf32>
    scf.yield %next_p0, %next_p1, %next_p2, %next_p3
      : !pto.vreg<64xf32>, !pto.vreg<64xf32>,
        !pto.vreg<64xf32>, !pto.vreg<64xf32>
  }

%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"
%s0 = pto.vcgadd %acc_p0, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s1 = pto.vcgadd %acc_p1, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s2 = pto.vcgadd %acc_p2, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s3 = pto.vcgadd %acc_p3, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s01 = pto.vadd %s0, %s1, %sum_mask : !pto.vreg<64xf32>
%s23 = pto.vadd %s2, %s3, %sum_mask : !pto.vreg<64xf32>
%sum_block = pto.vadd %s01, %s23, %sum_mask : !pto.vreg<64xf32>
pto.vsts %sum_block, %out[%group_off], %sum_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  for c = 0..31:
    acc[row_r, c] = base[row_r, c] + steps * bias_s
  out[group_off + r] = reduce(acc[row_r, 0..31])
```

### 3.23 `group_broadcast` With Multiple Dense Consumers

One `group_slots` value may feed multiple `group_broadcast` uses with different
dense result layout requirements.  Layout assignment should rematerialize the
broadcast per use instead of forcing one result layout onto all consumers.

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<128xf32> -> !pto.vmi.vreg<128xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}

%b_for_mul = pto.vmi.group_broadcast %sum {num_groups = 8}
%y = pto.vmi.mulf %x, %b_for_mul
%ysum = pto.vmi.group_reduce_addf %y, %mask {num_groups = 8}
pto.vmi.group_store %ysum, %sum_out[%group_off], %c1 {num_groups = 8}

%b_for_cast = pto.vmi.group_broadcast %sum {num_groups = 8}
%h = pto.vmi.truncf %b_for_cast
pto.vmi.store %h, %dense_out[%off]
```

Assigned layouts:

```text
%x, %b_for_mul, %y:
  !pto.vmi.vreg<128xf32,
    #pto.vmi.layout<deinterleaved = 2, block_elems = 8>>

%sum, %ysum:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>

%b_for_cast:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>

%h:
  !pto.vmi.vreg<128xf16, #pto.vmi.layout<contiguous>>
```

VPTO lowering result:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"

%x_lo, %x_hi = pto.vldsx2 %base[%tile_off], "BDINTLV"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_lo_sum = pto.vcgadd %x_lo, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%x_hi_sum = pto.vcgadd %x_hi, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum_block = pto.vadd %x_lo_sum, %x_hi_sum, %sum_mask
  : !pto.vreg<64xf32>

%lane_id = pto.vci %c0_i32 : i32 -> !pto.vreg<64xi32>
%broadcast_idx = pto.vshrs %lane_id, %c3_i16, %all_b32
  : !pto.vreg<64xi32>, i16, !pto.mask<b32> -> !pto.vreg<64xi32>

// Use 1: broadcast for the S=16 block_elems=8 multiply path.  Both row halves
// use the same per-row broadcast vector.
%b_rows_for_mul = pto.vselr %sum_block, %broadcast_idx
  : !pto.vreg<64xf32>, !pto.vreg<64xi32> -> !pto.vreg<64xf32>
%y_lo = pto.vmul %x_lo, %b_rows_for_mul, %all_b32 : !pto.vreg<64xf32>
%y_hi = pto.vmul %x_hi, %b_rows_for_mul, %all_b32 : !pto.vreg<64xf32>
%y_lo_sum = pto.vcgadd %y_lo, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%y_hi_sum = pto.vcgadd %y_hi, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%ysum_block = pto.vadd %y_lo_sum, %y_hi_sum, %sum_mask
  : !pto.vreg<64xf32>
pto.vsts %ysum_block, %sum_out[%group_off], %sum_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>

// Use 2: rematerialize broadcast for the f32->f16 parity cast path.  The
// deinterleaved=2 physical bundle has two entries that alias this SSA value.
%b_rows_for_cast = pto.vselr %sum_block, %broadcast_idx
  : !pto.vreg<64xf32>, !pto.vreg<64xi32> -> !pto.vreg<64xf32>
%h_even = pto.vcvt %b_rows_for_cast, %all_b32
  {part = "EVEN", rnd = "R", sat = "SAT"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>
%h_odd = pto.vcvt %b_rows_for_cast, %all_b32
  {part = "ODD", rnd = "R", sat = "SAT"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>
%all_b16 = pto.pge_b16 "PAT_ALL"
%h0 = pto.vor %h_even, %h_odd, %all_b16 : !pto.vreg<128xf16>
pto.vsts %h0, %dense_out[%off], %all_b16 {dist = "NORM_B16"}
  : !pto.vreg<128xf16>, !pto.ptr<f16, ub>, !pto.mask<b16>
```

Memory result:

```text
for r = 0..7:
  s = reduce(row_r[0..15])
  sum_out[group_off + r] = reduce_i(row_r[i] * s for i = 0..15)
  dense_out[r * 16 + 0 .. r * 16 + 15] = truncf(s)
```

### 3.24 Mask With Elementwise, Select, And Store

This case separates compute masking from memory effects.  A masked elementwise
operation with passthrough semantics can be represented as ordinary compute
plus `select`; a masked store uses the mask only on the store effect.

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<64xf32> -> !pto.vmi.vreg<64xf32>
%rhs = pto.vmi.load %rhs_base[%off]
  : memref<64xf32> -> !pto.vmi.vreg<64xf32>
%mask = pto.vmi.create_mask %c48
  : index -> !pto.vmi.mask<64xpred>
%sum = pto.vmi.addf %x, %rhs
%passthrough = pto.vmi.select %mask, %sum, %x
pto.vmi.store %passthrough, %dense_out[%off]
pto.vmi.masked_store %sum, %masked_out[%off], %mask
```

Assigned layouts:

```text
%x, %rhs, %sum, %passthrough:
  !pto.vmi.vreg<64xf32, #pto.vmi.layout<contiguous>>

%mask:
  !pto.vmi.mask<64xpred, #pto.vmi.layout<contiguous>>
```

VPTO lowering result:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%m = pto.pge_b32 "PAT_VL48"

%x0 = pto.vlds %base[%off] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%rhs0 = pto.vlds %rhs_base[%off] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%sum0 = pto.vadd %x0, %rhs0, %all_b32 : !pto.vreg<64xf32>

%pass0 = pto.vsel %sum0, %x0, %m
  : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
pto.vsts %pass0, %dense_out[%off], %all_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>

pto.vsts %sum0, %masked_out[%off], %m {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for i = 0..63:
  if i < 48:
    dense_out[off + i] = base[off + i] + rhs_base[off + i]
    masked_out[off + i] = base[off + i] + rhs_base[off + i]
  else:
    dense_out[off + i] = base[off + i]
    masked_out[off + i] is unchanged
```

### 3.25 Function Boundary Layout Specialization

Function boundaries cannot rely on hidden layout side tables.  Either the
function is internal and layout-specialized by `vmi-layout-assignment`, or a
public/external VMI boundary must diagnose until a stable VMI ABI is defined.

#### 3.25.1 Internal Function Specialized To Consumer Layout

VMI input:

```text
func.func private @producer(%base: !pto.ptr<f32, ub>, %off: index)
    -> !pto.vmi.vreg<256xf32> {
  %x = pto.vmi.load %base[%off]
    : memref<256xf32> -> !pto.vmi.vreg<256xf32>
  return %x : !pto.vmi.vreg<256xf32>
}

func.func @caller(%base: !pto.ptr<f32, ub>, %off: index, %out: !pto.ptr<f32, ub>) {
  %x = call @producer(%base, %off)
    : (!pto.ptr<f32, ub>, index) -> !pto.vmi.vreg<256xf32>
  %sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
  pto.vmi.group_store %sum, %out[%off], %c1 {num_groups = 8}
  return
}
```

Assigned layouts:

```text
@producer result:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>

%x in @caller:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>

%sum:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering result for the function boundary:

```text
func.func private @producer(...)
    -> (!pto.vreg<64xf32>, !pto.vreg<64xf32>,
        !pto.vreg<64xf32>, !pto.vreg<64xf32>) {
  %x_even_0, %x_odd_0 = pto.vldsx2 %base[%tile_off_0], "DINTLV_B32"
    : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
  %x_even_1, %x_odd_1 = pto.vldsx2 %base[%tile_off_1], "DINTLV_B32"
    : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
  %x_p0, %x_p2 = pto.vdintlv %x_even_0, %x_even_1
    : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
  %x_p1, %x_p3 = pto.vdintlv %x_odd_0, %x_odd_1
    : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
  return %x_p0, %x_p1, %x_p2, %x_p3
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>,
      !pto.vreg<64xf32>, !pto.vreg<64xf32>
}

func.func @caller(...) {
  %x_p0, %x_p1, %x_p2, %x_p3 = call @producer(...)
    : (...) -> (!pto.vreg<64xf32>, !pto.vreg<64xf32>,
                !pto.vreg<64xf32>, !pto.vreg<64xf32>)

  %all_b32 = pto.pge_b32 "PAT_ALL"
  %sum_mask = pto.pge_b32 "PAT_VL8"
  %s0 = pto.vcgadd %x_p0, %all_b32
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  %s1 = pto.vcgadd %x_p1, %all_b32
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  %s2 = pto.vcgadd %x_p2, %all_b32
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  %s3 = pto.vcgadd %x_p3, %all_b32
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  %s01 = pto.vadd %s0, %s1, %sum_mask : !pto.vreg<64xf32>
  %s23 = pto.vadd %s2, %s3, %sum_mask : !pto.vreg<64xf32>
  %sum_block = pto.vadd %s01, %s23, %sum_mask : !pto.vreg<64xf32>
  pto.vsts %sum_block, %out[%group_off], %sum_mask {dist = "NORM_B32"}
    : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
}
```

Memory result:

```text
for r = 0..7:
  out[off + r] = reduce(row_r[0..31])
```

#### 3.25.2 Public Or External VMI Boundary

VMI input:

```text
func.func @public_producer(%base: !pto.ptr<f32, ub>, %off: index)
    -> !pto.vmi.vreg<256xf32> attributes {public} {
  %x = pto.vmi.load %base[%off]
    : memref<256xf32> -> !pto.vmi.vreg<256xf32>
  return %x : !pto.vmi.vreg<256xf32>
}
```

Required diagnostic for the initial design:

```text
VMI-LAYOUT-CONTRACT:
  public or external function boundary returns !pto.vmi.vreg<256xf32> without a
  stable VMI layout ABI. Mark the function internal for layout specialization,
  inline it before vmi-layout-assignment, or define an explicit ABI layout.
```
