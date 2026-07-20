# VMI Group-Value Cast And Broadcast Generalization

This note describes the generalized rule behind the
`group_reduce -> truncf -> group_broadcast` case.  The goal is to avoid a
case-specific optimization such as "if f32->f16 and slots=8, do X".  The
implementation should instead treat group-value layouts, width-changing casts,
and group broadcast as independent layout facts.

This document intentionally keeps the current layout spelling:

```text
#pto.vmi.layout<num_groups = G, slots = K>
#pto.vmi.layout<num_groups = G, slots = K, lane_stride = LS>
```

Renaming the attribute syntax is separate work.

## 1. Terms

### 1.1 Group-Value Layout

A group-value VMI value contains one logical scalar value per logical group.
The VMI type element count is therefore the group count:

```text
!pto.vmi.vreg<GxT, #pto.vmi.layout<num_groups = G, slots = K, lane_stride = LS>>
```

The layout fields mean:

```text
G:
  number of logical groups.  This is redundant with the VMI vreg element count
  and with the relevant group op attr, but the current IR spelling stores it.

K:
  number of logical group values placed in one physical chunk.

LS:
  physical lane distance, measured in element-sized lanes of T, between
  adjacent logical group values inside a physical chunk.
```

For logical group `g`:

```text
chunk = g / K
slot  = g % K
lane  = slot * LS
```

Example:

```text
8xf32 group_values, K=8, LS=1:
  f32 physical lanes 0,1,2,3,4,5,6,7

8xf16 group_values, K=8, LS=2:
  f16 physical lanes 0,2,4,6,8,10,12,14

8xui8 group_values, K=8, LS=4:
  ui8 physical lanes 0,4,8,12,16,20,24,28
```

This is distinct from dense layouts.  Dense values contain ordinary element
streams.  Group-value layouts contain one scalar per group and cannot be read
by ordinary dense consumers without an explicit operation such as
`group_broadcast` or `group_store`.

### 1.2 Group Size

Do not confuse these two quantities:

```text
G = num_groups = number of group scalar values
S = group_size = dense lanes per group = dense_element_count / G
K = slots = group scalar values per physical chunk
```

`group_reduce` consumes a dense value with group size `S` and produces a
group-value result with `G` logical lanes.  The result layout's `K` describes
how those `G` scalars are physically placed.

## 2. General Layout Rules

### 2.1 Valid Group-Value Layout

A concrete group-value layout is valid for a physical lowering path when:

```text
G == vreg element count
K > 0
LS > 0
physical arity = ceil(G / K)
for every chunk:
  active_slots = min(K, G - chunk * K)
  if active_slots > 0:
    (active_slots - 1) * LS < physical_lanes_per_chunk(T)
```

The final inequality is important.  `K=8, LS=2` is valid for f16 because the
last active lane is 14 and an f16 physical vector chunk has 128 lanes.  A
layout whose last active lane does not fit in one chunk must be rejected or
materialized through a different explicit layout conversion.

### 2.2 Physical Lane Helper

All group-value consumers should use one helper instead of open-coding slot
math:

```text
getGroupValuePhysicalLane(layout, group):
  require layout is group-value
  K = layout.slots
  LS = layout.lane_stride
  chunk = group / K
  slot = group % K
  lane = slot * LS
  return (chunk, lane)
```

For a chunk-local helper:

```text
getGroupValueSlotPhysicalLane(layout, slot):
  return slot * layout.lane_stride
```

This helper is the common contract for:

```text
group_broadcast VSELR index generation
group_store packed or point stores
group-value casts
debug/validation lane maps
```

## 3. Cast Generalization

Width-changing casts on group-value layouts preserve the group structure.  They
do not change `G` or `K`.  They only change the element type and, for
phase-zero casts, the lane stride.

This is not the default layout decision for every cast with matching element
widths.  A cast enters the group-value relation only when one connected source
or result value already carries a group-value layout fact from an independent
producer or consumer requirement:

```text
producer fact:
  group_reduce/group_slot_load/loop-carried group value gives the source a
  group-value layout, and the cast derives the result layout.

consumer fact:
  group_broadcast/group_store or another group-value consumer requests a
  group-value source value layout through the consumer operand overload, and
  the cast derives the inverse source or result relation through the layout
  propagator.
```

If neither side has a group-value fact, the cast must use the ordinary
type-based dense cast relation, such as the existing deinterleaved-source /
contiguous-result rule.  The cast must not invent a group-value layout from
element types alone.

The old assignment pass could reliably use only source facts that were already
known from a defining producer.  The layout propagator removes that ordering
assumption by treating cast layout equations as bidirectional transfer
relations.  Consumer-driven inverse derivation is therefore implemented by the
propagator rather than by a special `group_reduce -> truncf -> group_broadcast`
pattern.

### 3.1 Narrowing

For a narrowing cast:

```text
source_bits = R * result_bits
```

the phase-zero layout relation is:

```text
source: group_values(G, K, LS)
result: group_values(G, K, LS * R)
```

Examples:

```text
f32 -> f16, R=2:
  group_values(G, K=8, LS=1)
    -> group_values(G, K=8, LS=2)

i32 -> ui16, R=2:
  group_values(G, K=8, LS=1)
    -> group_values(G, K=8, LS=2)

i32 -> ui8, R=4:
  group_values(G, K=8, LS=1)
    -> group_values(G, K=8, LS=4)
```

The corresponding VPTO conversion part for the phase-zero result is:

```text
R=2: EVEN
R=4: P0
```

This document only covers phase-zero layouts.  Supporting odd or non-zero phase
would require an explicit lane offset/phase field, not another special case.

### 3.2 Widening

For a widening cast:

```text
result_bits = R * source_bits
```

the inverse phase-zero relation is:

```text
source: group_values(G, K, LS * R)
result: group_values(G, K, LS)
```

Examples:

```text
f16 -> f32, R=2:
  group_values(G, K=8, LS=2)
    -> group_values(G, K=8, LS=1)

ui8 -> ui32, R=4:
  group_values(G, K=8, LS=4)
    -> group_values(G, K=8, LS=1)
```

If the source layout does not have `lane_stride` divisible by `R`, the
phase-zero widening relation does not hold.  The compiler should either choose
another legal layout relation or insert an explicit layout materialization if a
supported one exists.

### 3.3 Support Query Shape

The support query should not mention a particular op use-site such as
`group_broadcast`.  It should validate the group-value cast relation:

```text
source and result are both group-value layouts
source.G == result.G
source.K == result.K
source/result element widths define R
narrow: result.LS == source.LS * R
widen:  source.LS == result.LS * R
source/result physical arity are computable and compatible with the lowering
active slots fit in each physical chunk
target has the required phase-zero conversion part
```

For the current main case this query succeeds because:

```text
source: 8xf32 group_values(G=8, K=8, LS=1)
result: 8xf16 group_values(G=8, K=8, LS=2)
R = 2
result.LS == source.LS * R
```

## 4. Group Broadcast Generalization

`group_broadcast` consumes a group-value source and produces a dense result.
It should not inspect whether the source was produced by `group_reduce`,
`group_slot_load`, `truncf`, `trunci`, or an elementwise op.  Its only layout
obligation is to select the physical source lane for each output group.

For result logical lane `i`:

```text
group = i / group_size
slot  = group % K
index = slot * source_lane_stride
source_chunk = group / K
```

The VSELR index vector therefore depends on the source group-value layout:

```text
source K=8, LS=1:
  index lanes are 0,1,2,3,4,5,6,7 repeated by group_size

source K=8, LS=2:
  index lanes are 0,2,4,6,8,10,12,14 repeated by group_size

source K=8, LS=4:
  index lanes are 0,4,8,12,16,20,24,28 repeated by group_size
```

This rule is the same for contiguous and supported deinterleaved dense results.
The only difference is how result physical lanes map back to result logical
lanes before computing `group`.

## 5. Assignment Flow

Assignment should stabilize layouts before `vmi-to-vpto`.  The generalized
flow is:

```text
group_reduce/group_slot_load:
  choose a group-value layout for the result:
    unit packed plan      -> K=8, LS=1
    row-local plan        -> K=1, LS=1
    existing constrained layout wins if legal

group-value narrow cast:
  if the source already has a group-value layout:
    set result layout with same G/K and LS multiplied by R
    request the source layout
  else:
    use the ordinary type-based dense cast layout

group-value widen cast:
  if the source already has a group-value layout:
    derive the result layout through the widening relation
  else:
    use the ordinary type-based dense cast layout

group_broadcast:
  request a concrete group-value source layout
  set a dense result layout chosen by the consumer or by the broadcast support
  policy

group_store:
  request a concrete group-value source layout compatible with row stride and
  store dist support
```

No step should commute:

```text
group_broadcast(truncf(x)) <-> truncf(group_broadcast(x))
```

as a required legality mechanism.  A separate cost optimization may still
choose such a rematerialization later, but the basic assigned layout must be
legal without changing the semantic op order.

## 6. Lowering Flow

### 6.1 Group-Value Narrow Cast

For a phase-zero group-value narrow cast:

```text
source: group_values(G,K,LS)
result: group_values(G,K,LS*R)
```

lower each physical source chunk independently:

```text
active_slots = min(K, G - chunk * K)
mask = prefix mask active_slots in source element granularity
part = EVEN for R=2, P0 for R=4
result_chunk = vcvt source_chunk, mask, part
```

Main case:

```text
%sum32 : !pto.vreg<64xf32> with active lanes 0..7
%mask  = pto.pge_b32 "PAT_VL8"
%sum16 = pto.vcvt %sum32, %mask {part = "EVEN", rnd = "R", sat = "SAT"}
       : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>
```

The resulting f16 group values are in lanes:

```text
0,2,4,6,8,10,12,14
```

### 6.2 Group Broadcast

After the cast, broadcast uses the source layout only:

```text
source layout: group_values(G=8,K=8,LS=2)
group_size: 16
index vector:
  [0 repeated 16,
   2 repeated 16,
   4 repeated 16,
   6 repeated 16,
   8 repeated 16,
   10 repeated 16,
   12 repeated 16,
   14 repeated 16]

%b16 = pto.vselr %sum16, %index
```

This is a group-broadcast lowering rule, not a truncf-specific rule.

### 6.3 Store

The broadcast result is dense contiguous f16 in the main case:

```text
pto.vsts %b16, %out[%off] {dist = "NORM_B16"}
```

If a later case stores group-value data directly, `group_store` should use the
same `getGroupValuePhysicalLane` helper or a target-specific store dist that is
proven equivalent to that lane map.

## 7. Scenarios

### 7.1 Main Floating-Point Broadcast Case

VMI:

```text
%sum32 = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
%sum16 = pto.vmi.truncf %sum32
%b16   = pto.vmi.group_broadcast %sum16 {num_groups = 8}
pto.vmi.store %b16, %out[%off]
```

Assigned:

```text
%sum32 : 8xf32 group_values(G=8,K=8,LS=1)
%sum16 : 8xf16 group_values(G=8,K=8,LS=2)
%b16   : 128xf16 contiguous
```

Lowering skeleton:

```text
vcgadd/vadd -> vcvt EVEN PAT_VL8 -> vselr index stride 2 -> vsts NORM_B16
```

### 7.2 Integer Narrow Direct Store

VMI:

```text
%narrow = pto.vmi.trunci %wide
pto.vmi.group_store %narrow, %out[%off], %c1 {num_groups = 8}
```

Assigned:

```text
%wide   : 8xi32 group_values(G=8,K=8,LS=1)
%narrow : 8xui8 group_values(G=8,K=8,LS=4)
```

A direct `group_store` may lower this through a target-specific packed store
such as `PK4_B32` when that store is equivalent to reading lanes
`0,4,8,...,28` in ui8 lane units.  This is a store lowering strategy; it must
not redefine the group-value layout relation.

### 7.3 Group Slot Load Then Broadcast

VMI:

```text
%slots = pto.vmi.group_slot_load %base[%off], %c1 {num_groups = 8}
%dense = pto.vmi.group_broadcast %slots {num_groups = 8}
```

Assigned:

```text
%slots : 8xT group_values(G=8,K=8,LS=1)
%dense : dense result chosen by consumer/support
```

Broadcast uses VSELR indices `0..7` repeated by group size, or a memory-source
optimization may replace the whole load+broadcast chain with
`group_broadcast_load` when that transformation is legal.

### 7.4 Widening A Lane-Strided Group Value

VMI:

```text
%wide = pto.vmi.extf %narrow
```

Legal phase-zero relation:

```text
%narrow : Gxf16 group_values(G,K,LS=2)
%wide   : Gxf32 group_values(G,K,LS=1)
```

If `%narrow` has `LS=1`, this particular phase-zero relation cannot produce a
packed f32 group-value result without either selecting a different layout or
materializing a layout conversion first.

### 7.5 Unsupported Overflowing Layout

For element type T with `L` physical lanes per chunk:

```text
group_values(G,K,LS)
```

is unsupported when:

```text
(active_slots - 1) * LS >= L
```

Example:

```text
K=128, LS=2 for f16
last lane = 254
f16 lanes per chunk = 128
```

This cannot be represented in one physical chunk with the stated `K`; assignment
must choose a smaller `K`, insert an explicit materialization, or reject the
layout if no legal route exists.

## 8. Implementation Plan

### 8.1 Shared Helpers

Add or consolidate helpers with these responsibilities:

```text
isConcreteGroupValueLayout(type/layout)
getGroupValueSlots(layout)
getGroupValueLaneStride(layout)
getGroupValuePhysicalLane(layout, group)
getGroupValueSlotPhysicalLane(layout, slot)
checkGroupValueLaneSpan(type/layout)
deriveGroupValueNarrowLayout(sourceLayout, factor)
deriveGroupValueWidenLayout(sourceLayout, factor)
```

These helpers should be used by support checks and lowering.  Open-coded
`group % slots` lane selection should disappear from broadcast/store/cast
lowering unless the code is explicitly computing a logical slot number before
calling the physical-lane helper.

### 8.2 Support Layer

Update support checks so the same relation is used by floating-point and
integer casts:

```text
truncf/trunci group-value narrow:
  source/result group-value layouts
  same G and K
  result LS = source LS * narrow factor

extf/extsi/extui group-value widen:
  source/result group-value layouts
  same G and K
  source LS = result LS * widen factor

group_broadcast:
  source group-value layout with concrete K/LS
  result dense layout with registered VSELR lowering support
```

The first implementation must explicitly enumerate the element-width pairs with
known VPTO conversion parts in the support helper.  A pair not listed there is
unsupported and must fail with a diagnostic that says which relation failed, not
which high-level pattern was expected.

### 8.3 Assignment

Assignment should call the support-layer layout derivation instead of duplicating
cast-specific cases:

```text
if source layout is group-value and cast is narrowing:
  result layout = deriveGroupValueNarrowLayout(source layout, factor)
  request source layout

if source layout is group-value and cast is widening:
  result layout = deriveGroupValueWidenLayout(source layout, factor)
  request source layout
```

When the cast source has no known layout yet, normal producer facts still apply.
For the main case, `group_reduce` naturally produces `K=8, LS=1`; `truncf`
then derives `K=8, LS=2`.

Backward derivation from a result consumer request is handled by the generic
request propagator.  For example, if a consumer requests a group-value cast
result, the cast transfer must request the matching source layout when the
relation is unique and supported.  Do not extend the old single walk with
order-dependent checks.

### 8.4 Layout Request Propagation

The generic request propagation design is described separately in
[vmi-layout-request-propagation.md](vmi-layout-request-propagation.md).  This
document only needs the case-specific consequence: if consumer-driven
group-value casts become required, do not extend the current single walk with
more order-dependent checks.  Add value layout requests to the propagator, using
the operand overload when the request comes from a specific operand, and let
them propagate through the cast relation.

The old assignment implementation could avoid that framework change because
the main case was producer-driven:

```text
group_reduce sets source group-value layout before truncf is visited
truncf derives its result layout immediately
```

When the propagator is used, the cast width relation remains generic:

```text
if the source value is requested/propagated as group-value:
  derive result group-value layout
  request the result value layout

if the result value is requested/propagated as group-value:
  derive inverse source value layout
  request the source value layout through the source operand overload
```

If the derived source layout conflicts with the source value's existing
`assignment.layout`, the operand overload records a use-side conflict in the
source value's `assignment.conflicts`.  Apply materializes it with
`ensure_layout` if the layouts still differ.

### 8.5 VMI To VPTO

Lowering work:

```text
group-value truncf/trunci:
  use active slot prefix mask
  use conversion part from the width factor
  preserve per-chunk arity

group-value extf/extsi/extui:
  use active slot prefix mask
  use conversion part from the width factor
  preserve per-chunk arity

group_broadcast:
  build VSELR index vectors through getGroupValuePhysicalLane
```

The physical type policy must not make `lane_stride` mean only "integer carrier
packing".  Floating-point group-value lane stride must be representable as a
normal floating-point VPTO vector with sparse active lanes, because the main
case needs:

```text
vcvt f32 -> f16 EVEN
then vselr.f16 from the even f16 lanes
```

Existing target-specific integer store paths may keep using packed store dists
when they are proven equivalent to the same group-value lane map.

## 9. Regression Tests

Add or update focused lit tests:

```text
test/lit/vmi_new/vmi_layout_assignment_group_reduce_s16_truncf_broadcast_store.pto
  CHECK assignment:
    group_reduce result: slots=8
    truncf result: slots=8, lane_stride=2
    group_broadcast consumes the truncf result
    no f32 group_broadcast + ensure_layout + truncf shape

  CHECK lowering:
    vcgadd/vadd before vcvt
    vcvt before vselr
    vselr before vsts
    no remaining pto.vmi ops

test/lit/vmi_new/vmi_to_vpto_group_broadcast_lane_stride_source.pto
  Direct assigned-IR test where the source is
  group_values(G=8,K=8,LS=2); CHECK vselr is generated.

test/lit/vmi_new/vmi_layout_gate_group_value_cast_invalid.pto
  Invalid relation, for example result LS not equal source LS * factor.
```

Keep existing integer lane-stride tests, but adjust their checks only if the
generalized physical type policy changes their printed VPTO shape.  The semantic
expectation remains the same: the direct store must be equivalent to the
group-value lane map.

## 10. First-Phase Required Support

The first implementation must support the current phase-zero group-value cases.
The non-goals below are only for layouts and rewrites beyond the current
phase-zero relation.

Required support:

```text
phase-zero group-value layout:
  lane = slot * LS
  no non-zero lane offset
  no odd-phase layout

group-value producer facts:
  group_reduce and group_slot_load seed their current natural group-value
  layouts.

group-value cast transfer:
  narrowing and widening use the shared group-value cast equations for every
  element-width pair explicitly listed by the VPTO support helper.
  source-driven and result-consumer-driven propagation must both use the same
  facts.

group_broadcast:
  consumes a concrete group-value source layout and produces a dense result
  layout accepted by the broadcast support helper.

group_broadcast_load / e2b optimization:
  remains an optimization over the same logical group-value broadcast relation;
  it must not be required to make the assigned layout legal.
```

## 11. Non-Goals

This design does not include:

```text
attr syntax rename
non-zero lane offset or odd-phase group-value layouts
cost search between broadcast-before-cast and cast-before-broadcast
automatic rematerialization as a legality mechanism
arbitrary LS values without VPTO conversion/store support
generic register compaction/expansion for every group-value layout
```

Those can be added later as separate design items once the phase-zero
group-value relation is implemented and covered by tests.
