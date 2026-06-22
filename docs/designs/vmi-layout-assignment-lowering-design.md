# VMI Layout Assignment And Lowering Design

本文是新的 VMI layout assignment / lowering 设计文档。它只以
`docs/designs/vmi-layout-lowering-cases.md` 为 source of truth，不继承旧
`vmi-dialect-design.md` 的 layout 设计，以避免旧上下文污染。

目标：

```text
VMI surface IR
  -> pto-validate-vmi-ir
  -> vmi-layout-assignment          // hard legalization baseline
  -> canonicalize/cse
  -> vmi-layout-fold-consumers      // optional optimization
  -> canonicalize/cse
  -> vmi-layout-rematerialize       // optional optimization
  -> canonicalize/cse
  -> vmi-layout-sink-materialization // optional optimization
  -> canonicalize/cse
  -> optional later layout optimization passes
  -> canonicalize/cse
  -> vmi-legalize-arith-select
  -> pto-validate-vmi-layout-ir
  -> layout-assigned and optimized VMI IR
  -> vmi-to-vpto
  -> VPTO IR
```

核心验收约束：

```text
vmi-to-vpto 不允许通过上下文猜 lowering。

任何需要 producer/consumer/control-flow/memory/mask 上下文才能决定的事，
必须在 vmi-layout-assignment 或后续 VMI layout optimization 阶段变成显式 IR：

1. vmi.vreg/vmi.mask 的 layout
2. current-op attrs/operands that make the local recipe deterministic
3. use-site ensure_layout / ensure_mask_layout / ensure_mask_granularity
4. rematerialized or cloned producer
5. target capability diagnostic
```

## 0. Hard Legalization And Optimization Boundary

Layout assignment is a stage, not necessarily one monolithic pass.  The design
separates correctness from optimization:

```text
hard legalization:
  produces legal layout-assigned VMI IR for all supported semantics
  inserts conservative ensure_* helpers at incompatible uses
  may choose a simple canonical layout even when a fused consumer recipe exists
  must diagnose unsupported semantics before vmi-to-vpto has to guess

layout optimization:
  rewrites already legal VMI IR into cheaper but equivalent VMI IR
  may fold ensure_layout into a layout-aware consumer
  may clone/rematerialize cheap producers for different use-site layouts
  may sink or hoist layout materialization through pure elementwise chains
  may specialize private VMI function signatures
```

The driver currently runs MLIR's normal `canonicalize` and `cse` between these
VMI-specific passes.  They are allowed to clean up trivially unused helpers,
merge identical rematerialized producers, and expose simpler use-def shapes.
They are not a source of hidden lowering information; after every optimization,
the IR must still carry enough local information for `vmi-to-vpto`.

The baseline hard pass may emit:

```text
%x_c = pto.vmi.ensure_layout %x : deinterleaved=2 -> contiguous
pto.vmi.store %x_c
```

A later optimization may replace that use with:

```text
pto.vmi.store %x : deinterleaved=2
```

only if the store op itself has a local deterministic recipe for preserving the
same row-major memory effect, such as a layout-aware `vstsx2 INTLV` lowering.
Both forms are semantically complete.  The second form is an optimization, not
a hard requirement for correctness.

## 1. Source Case Coverage

设计必须覆盖 case catalog 中的端到端场景：

```text
dense cast:
  f16 -> f32 -> store
  f32 -> f16 -> store
  f8  -> f32 -> compute -> f8
  f16 -> f32 shared by dense store and S=16 reduce
  f32 shared by f8 store and S=32 reduce

group reduce:
  S=8, S=16, S=32, S=64
  reduce -> group_store
  reduce -> group_slot_load/elemwise -> group_store
  reduce -> group_broadcast -> elemwise -> reduce -> store
  one group_slots result fanning out to group_store and group_broadcast
  grouped tail -> broadcast -> reduce -> store

layout conflict:
  one value with dense and group-reduce consumers
  one value with S=16 and S=32 group-reduce consumers
  one scalar broadcast rematerialized for dense and grouped users
  one non-rematerializable value materialized with use-site ensure_layout
  one scalar group-slot source rematerialized as slots=8 and slots=1
  S=16 block_elems=1/8 recipe selection
  dense consumer of group_slots diagnostic
  packed group-slot width-changing cast diagnostic
  S=64 slots=1 group-slot width-changing cast

control flow:
  scf.if before group_reduce
  group_slots across scf.if
  scf.for loop-carried layout fixed point
  group_slots as scf.for loop-carried accumulator
  internal function boundary specialization
  internal function argument boundary materialization
  public/external VMI ABI diagnostic

mask and tail:
  prefix mask
  group-periodic mask
  dynamic group-periodic mask
  masked_load tail with explicit passthrough instead of padding
  masked_load grouped tail feeding group_reduce
  masked select/store
  one semantic mask used by multiple predicate granularities
  S=32 tail with and without full_tile_readable
  compact S=12 diagnostic

strided memory:
  group_load source stride greater than logical group size
  strided group_load feeding broadcast and a second group_reduce
  group_slot_load slots=1 with non-unit source stride
  group_store slots=1 with non-unit output stride
```

### 1.1 Case-Set Sufficiency

The current case set is sufficient to define the first implementation of layout
assignment and lowering.  It covers every decision axis that has changed the
design so far:

```text
physical dense layout:
  contiguous, deinterleaved=2/4, block_elems=1/8

sparse result layout:
  group_slots(G, slots=8) for packed VCG results
  group_slots(G, slots=1) for row-local S=64 results

producer-driven layout:
  load, group_load, group_slot_load, broadcast, create_mask,
  create_group_mask

consumer-driven pressure:
  dense store, group_reduce, group_store, group_broadcast, truncf,
  elementwise/select, masked_load/masked_store

conflict resolution:
  cheap rematerialization, explicit ensure_layout, explicit diagnostics

control-flow propagation:
  scf.if, scf.for iter_args/results, internal/private function boundaries,
  public ABI rejection

memory legality:
  full_tile_readable proof, grouped masks, predicate granularity, aligned
  strided group memory, stable gather diagnostic
```

No extra layout kind should be added unless a new case proves that the existing
layouts and recipes cannot express the logical behavior.  The remaining open
items are not missing layout semantics:

```text
dynamic active_elems_per_group runtime source:
  create_group_mask layout lowering is defined and has both lit and SIM
  coverage. The supported runtime source is a kernel scalar argument cast to
  index inside vecscope; vmi-to-vpto does not recover this value from GM/UB
  scalar loads or surrounding context.

private vector function runtime:
  private/internal single-block helpers are runtime-covered by ptoas inlining
  private physical VMI helpers after vmi-to-vpto and before VPTO vecscope/backend
  emission. This is a post-physicalization backend hygiene step; vmi-to-vpto
  still lowers only from assigned layouts and helper ops.

diagnostic-only cases:
  compact S=12 gather fallback, packed slots=8 width-changing cast, public VMI
  ABI, unsafe masked_load tail, and unaligned/dynamic group memory remain
  explicit capability boundaries.
```

## 2. Layout Domain

Layout is a property of a layout-assigned VMI value, not a property inferred by
the final lowering pattern.

### 2.1 Dense Layouts

```text
#pto.vmi.layout<contiguous>
#pto.vmi.layout<deinterleaved = F, block_elems = B>
```

`block_elems` defaults to `1`:

```text
#pto.vmi.layout<deinterleaved = 2>
  == #pto.vmi.layout<deinterleaved = 2, block_elems = 1>
```

Dense layouts preserve one semantic value for every logical lane.

Lane map for `deinterleaved = F, block_elems = B`:

```text
logical lane i
block q          = i / B
in-block lane r  = i % B
part p           = q % F
part block t     = q / F

physical part p, physical lane t * B + r
```

Important consequence:

```text
deinterleaved=2, block_elems=1
deinterleaved=2, block_elems=8
```

are different layouts.  They cannot be treated as compatible because `F` is the
same.

### 2.2 Sparse Group-Slot Layouts

```text
#pto.vmi.layout<num_groups = G, slots = K>
```

Only `G` lanes have semantic values:

```text
slot_block(g) = g / K
slot_lane(g)  = g % K
```

All non-slot lanes are undefined and may only be read by group-aware operations.
Ordinary dense `add/mul/store/truncf` cannot consume `group_slots`.

`K` is selected by the producer/consumer local recipe:

```text
S=8/16/32 packed VCG result -> slots=8
S=64 row-local result       -> slots=1
```

## 3. Lowering Context Must Become Explicit IR Output

`vmi-to-vpto` may inspect only:

```text
1. op name and explicit op attrs
2. converted operand/result types with layout
3. helper/materialization ops written by layout assignment
4. inserted helper ops
5. target capability registry
```

It must not:

```text
1. walk to defining op to infer layout
2. inspect all users to choose a lowering path
3. infer memory legality from a later mask
4. decide S=16 block_elems=1 vs block_elems=8 locally
5. decide whether group_broadcast should be materialized for one or many users
6. specialize function signatures during vmi-to-vpto
```

Any of those decisions belongs to the layout stage before `vmi-to-vpto`.

## 4. Explicit Assignment Products

After `vmi-layout-assignment`, every VMI data and mask value must be in one of
these states:

```text
layout-assigned type:
  !pto.vmi.vreg<NxT, #pto.vmi.layout<...>>
  !pto.vmi.mask<Nxpred, #pto.vmi.layout<...>>

or explicit helper:
  pto.vmi.ensure_layout
  pto.vmi.ensure_mask_layout
  pto.vmi.ensure_mask_granularity
```

`vmi-to-vpto` is allowed to choose a deterministic recipe from local
information on the current op:

```text
current op name
current op attrs
operand/result types and layouts
current op operand values such as stride and offset
target capability and pass options
```

This is not context inference.  What remains forbidden is walking to producers,
users, sibling users, branch/loop bodies, callees/callers, or nearby memory/MTE
ops to recover a lowering decision or a memory-safety proof.

If a decision cannot be made from that local information, layout assignment
must rewrite the IR until the decision is explicit in attrs, operand/result
layouts, helper ops, cloned producers, or diagnostics.  `vmi-to-vpto` must not
consume a separate string recipe attr.

### 4.1 Local Recipe Contract

The lowering recipe is derived from op + assigned operand/result layouts +
explicit attrs/operands.  If two legal recipes cannot be distinguished from
that local information, the IR is missing a semantic carrier and must be
extended before the recipe is implemented.

Locally deterministic decisions in the current implementation:

```text
group_load:
  result layout, num_groups, row_stride, source type, and target capability
  decide contiguous chunks versus S=16/S=32 block8 vsldb lowering.  Unit-stride
  vldsx2/BDINTLV can be a local peephole for the same block8 layout.

group_slot_load:
  result group_slots layout and source_group_stride decide packed slots=8
  versus row-local slots=1 vsldb lowering.  A single source op may still be
  rematerialized into two ops when different users require different result
  layouts; each clone is then locally deterministic.

group_reduce_addf:
  source/mask layout, result group_slots layout, num_groups, element type, and
  reassoc decide S=8 contiguous vcgadd, S=16/S=32 deinterleaved vcgadd trees,
  and S=64 row-local vcadd/vsel lowering.

group_broadcast:
  source group_slots layout, result dense layout, num_groups, and element type
  decide vdup/vselr materialization.

truncf:
  source/result group_slots layouts and element widths decide the slots=1
  f32->f16 slot-preserving vcvt path.
```

Other layout-only or attr-only decisions in the current implementation:

```text
load:
  result layout plus explicit memory attrs decide the lowering.  full_read_elems
  is the memory-safety proof; vmi-to-vpto may not recover that proof from MTE or
  caller context.

group_store:
  source group_slots layout and explicit output stride decide packed slots=8
  versus row-local slots=1 store legality.  If another legal store recipe
  needs more information, assignment must make that information explicit in the
  op or helper IR before vmi-to-vpto uses it.

masked_load:
  explicit passthrough, mask layout, full physical read, shaped safe-tail memref,
  or an explicit diagnostic decide legality.  A future stable gather fallback
  must be made explicit by assignment before vmi-to-vpto lowers it.

masked_store/select/elementwise:
  operand/result layouts and explicit mask granularity decide the lowering.
  They remain transfer ops unless a future case introduces competing recipes.

extf/truncf:
  dense width-changing paths are layout-determined today.  Any future
  commute-through-group-broadcast or alternative VCVT recipe must have an
  explicit IR carrier first.
```

Forbidden non-local recipe recovery:

```text
No pattern may synthesize a recipe or memory proof by:
  - walking from group_reduce to the load/group_load producer
  - walking from store/broadcast/truncf to the group_reduce producer
  - scanning sibling users of a group_slots value
  - inspecting branch bodies or loop bodies from a control-flow boundary
  - inspecting private callee bodies while lowering a call
```

If the current op lacks enough local information, `vmi-to-vpto` emits
`VMI-LAYOUT-CONTRACT` at the current op and prints the op name, logical type,
assigned layouts, and the missing decision class.

## 5. Local Recipe Registry

The compiler owns a target-aware local recipe registry.  Layout assignment and
layout optimization query this registry to decide which explicit IR shape to
produce.  `vmi-to-vpto` queries the same registry only to verify and lower the
current op from local information.

The registry is not serialized as a separate recipe-selection attribute.  If
two legal physical recipes cannot be distinguished by the current op's name,
attrs, operands, operand/result layouts, helper ops, and target options, the
VMI IR is missing a carrier.  Add an explicit attr, operand, helper op, or
semantic op before implementing that recipe.

### 5.1 Recipe Kinds

```text
ProducerRecipe:
  op can produce result layout L
  example: load -> deinterleaved=4 using DINTLV_B32 + vdintlv

ConsumerRecipe:
  op can consume operand layout L
  example: group_reduce S=32 consumes deinterleaved=4

TransferRecipe:
  op ties operand/result layouts
  example: addf requires same dense layout for operands/result

MaterializationRecipe:
  layout A -> layout B without changing logical value
  example: deinterleaved=4 -> contiguous by vintlv tree

RematerializationRecipe:
  cheap producer can be cloned for a use-site layout
  example: broadcast/create_mask/group_broadcast

DiagnosticRecipe:
  known unsupported semantic/capability boundary
  example: compact S=12 requires gather materialization
```

### 5.2 Dense Recipes From Cases

```text
f16 -> f32:
  source contiguous f16
  result deinterleaved=2, block_elems=1

f8 -> f32:
  source contiguous f8
  result deinterleaved=4, block_elems=1

f32 -> f16:
  source deinterleaved=2, block_elems=1
  result contiguous f16

f32 -> f8:
  source deinterleaved=4, block_elems=1
  result contiguous f8

elementwise dense:
  all dense operands/results share the same layout

broadcast scalar:
  rematerializable to any dense layout requested by the consumer

load:
  may be rematerialized per use when two consumers request incompatible dense
  layouts, such as S=16 deinterleaved=2 and S=32 deinterleaved=4
```

### 5.3 Group Recipes From Cases

```text
group_reduce f32 S=8:
  input contiguous
  result group_slots(G, slots=8)

group_reduce f32 S=16:
  legal input layout A: deinterleaved=2, block_elems=1
  legal input layout B: deinterleaved=2, block_elems=8
  result group_slots(G, slots=8)

group_reduce f32 S=32:
  legal input layout A: deinterleaved=4, block_elems=1
  legal input layout B: deinterleaved=4, block_elems=8
  result group_slots(G, slots=8)

group_reduce f32 S=64:
  input contiguous
  result group_slots(G, slots=1)

group_slot_load:
  result group_slots(G, slots=8) for packed slots
  result group_slots(G, slots=1) for row-local slots

group_broadcast:
  source group_slots(G,K)
  result is dense layout requested by each consumer
  rematerialize per use instead of forcing one result layout

group_store:
  source group_slots(G,K)

group_slot_cast f32 -> f16:
  slots=1 row-local source/result is legal with
  group_slot_cast_slots1_f32_to_f16
  slots=8 packed source is illegal unless a packed slot-preserving recipe is
  registered
```

### 5.4 Tail And Memory Safety Recipes

Mask semantics and memory legality are separate:

```text
mask:
  decides which logical lanes participate in compute/store semantics

full_tile_readable:
  decides whether a rounded-up physical load is allowed to read inactive lanes
```

The full-tile-readable proof must be explicit.  It may be carried by a
statically shaped memref source, or by `pto.vmi.load {full_read_elems = N}` for
pointer sources.  `vmi-to-vpto` consumes only this proof carrier; it does not
inspect surrounding MTE copies, producer bodies, callers, or later consumers to
decide whether inactive physical lanes are safe to read.

Example:

```text
S=32 tail num_groups=6:
  without full_tile_readable:
    fast DINTLV_B32 full-tile load is illegal

  with full_tile_readable:
    full 8-row physical tile may be loaded
    compute mask is PAT_VL48 per physical part
    group store mask is PAT_VL6

S=16 grouped tail active_elems_per_group=12:
  low 8-lane row half uses PAT_ALL
  high 8-lane row half uses lane_mod_8 < 4
  the same split applies before and after group_broadcast

one mask used by f32 and f16 consumers:
  f32 use materializes a b32 predicate
  f16 use materializes a b16 predicate
  vmi-to-vpto consumes the assigned per-use mask materialization
```

### 5.5 Case-Driven Request Matrix

The first implementation should build requests from the following finite table.
This table is deliberately case-derived; adding a new request kind requires a
new catalog case or a proof that it is equivalent to one listed here.

```text
dense store:
  requests dense contiguous source
  if source is deinterleaved, assignment must insert ensure_layout or select a
  store recipe such as vstsx2 that consumes the assigned layout explicitly

truncf f32 -> f16:
  requests source deinterleaved=2, block_elems=1
  requests result contiguous f16

truncf f32 -> f8:
  requests source deinterleaved=4, block_elems=1
  requests result contiguous f8

group_reduce S=8:
  requests source contiguous
  requests result group_slots(num_groups, slots=8)

group_reduce S=16:
  requests source deinterleaved=2, block_elems=1 or block_elems=8
  requests result group_slots(num_groups, slots=8)

group_reduce S=32:
  requests source deinterleaved=4, block_elems=1 or block_elems=8
  requests result group_slots(num_groups, slots=8)

group_reduce S=64:
  requests source contiguous
  requests result group_slots(num_groups, slots=1)

group_broadcast:
  requests source group_slots(num_groups, slots=K)
  produces one dense result layout per consumer request
  is cloned per incompatible dense consumer

group_store:
  requests source group_slots(num_groups, slots=K)
  explicit output stride attrs/operands decide store legality

dense elementwise add/mul/fma/min/max/select:
  requests all dense data operands and results use one dense layout
  mask operands request the same data layout and the consumer element
  granularity

group-slot elementwise add/mul/select:
  requests all group-slot operands and results use the same
  group_slots(num_groups, slots=K)
  rejects mixing dense and group_slots without explicit group_broadcast or
  group_store

group_slot_load:
  requests result group_slots(num_groups, slots=8) for packed unit-stride slots
  requests result group_slots(num_groups, slots=1) for row-local aligned slots

group_load:
  requests result deinterleaved=2/4, block_elems=8 for S=16/S=32 block
  fragment recipes, or contiguous for row-local full-chunk recipes

masked_load:
  requests result layout from its consumers
  requests mask layout matching the result
  requires explicit passthrough; padding is not synthesized

masked_store:
  requests dense source layout selected by the store recipe
  requests mask layout matching the source layout and store element granularity
  does not choose memory safety for an earlier load

create_mask/create_group_mask:
  produces whichever mask layout each consumer requests
  may be cloned per incompatible mask layout or granularity

scf.if/scf.for/call/return:
  requests equality across carried VMI values, yielded values, call operands,
  callee arguments, and function results
  private/internal functions may specialize or materialize at boundaries
  public/external VMI boundaries are diagnostics until an ABI is defined
```

Important negative requests:

```text
ordinary dense add/mul/store/truncf cannot request group_slots
packed group_slots(slots=8) cannot request width-changing cast unless a packed
slot-preserving cast recipe is registered
slots=1 group_store cannot request unit-stride row-major output until a pack or
unaligned-store recipe exists
```

### 5.6 Conflict Resolution Matrix

When one value receives incompatible requests, assignment resolves it using the
first legal row below.  `vmi-to-vpto` never repeats this decision.

```text
cheap producer with multiple requested layouts:
  clone the producer and assign each clone independently
  examples: load, broadcast, create_mask, create_group_mask, group_broadcast
  memory-read producers require the same explicit no-alias and safe-read proof
  at each clone site

non-cheap value with registered materialization:
  keep one chosen layout on the value and insert ensure_layout at the use site
  examples: deinterleaved=4 -> contiguous before dense store

layout-transparent chain:
  assign the whole equivalence class to the non-contiguous consumer request when
  that avoids materialization
  examples: broadcast -> addf -> S=32 group_reduce

control-flow join:
  all incoming values must be materialized to one layout before yield/branch
  examples: scf.if yielding group_slots, scf.for loop-carried group_slots

private function boundary:
  specialize or materialize at call/callee-entry before vmi-to-vpto

no clone/materialization/specialization recipe:
  emit a diagnostic naming the requesting op and both layouts
```

The cost model may choose between legal rows only when the observable contract
is identical.  For example, S=16 `block_elems=1` and `block_elems=8` are both
valid reduce inputs, but `block_elems=8` is selected only when a producer recipe
such as strided `group_load` naturally creates 32B row fragments or when cost
proves it cheaper without breaking another consumer such as `truncf`.

## 6. Layout Assignment Algorithm

`vmi-layout-assignment` is module-level.  It must see function/call/control-flow
connections before choosing layouts.

### 6.1 Variables

Create a layout variable for:

```text
1. every VMI OpResult
2. every VMI BlockArgument
3. every function argument/result that is allowed to carry VMI
4. every VMI mask value
```

Create a use-site request for:

```text
1. every operand use that requires a specific layout
2. every control-flow yield/branch/call/return edge
3. every memory operation that requires a memory legality recipe
```

### 6.2 Constraints

Hard constraints:

```text
group_slots cannot feed ordinary dense consumers
direct group-slot width-changing cast requires a slot-preserving recipe
public/external VMI function boundary requires a stable ABI or diagnostic
S=32 fast tail load requires full_tile_readable or gather fallback
```

`slots = 1` row-local cast may satisfy the slot-preserving recipe requirement.
Packed `slots = 8` f32->f16 remains a diagnostic unless a separate packed cast
or unpack/materialization recipe is registered.

Equivalence constraints:

```text
dense add/mul/select:
  operands/results use same dense layout unless an explicit materialization is
  inserted at a use site

scf.if/scf.for:
  region yield operands and block arguments must have the same assigned layout
  as the region result/iter_arg
```

Candidate constraints:

```text
S=16 group_reduce:
  choose block_elems=1 or block_elems=8 by cost and explicit assignment constraints

one dense value feeding S=16 and S=32 group_reduce:
  rematerialize a cheap producer per consumer layout, or insert an explicit
  materialization recipe; the final lowering pass must not pick one layout after
  seeing both users

load/group_load:
  choose memory recipe and result layout together

group_broadcast:
  rematerialize per dense consumer layout
```

### 6.3 Solving

Recommended solving order:

```text
1. Build function/control-flow SCCs.
2. Collect candidate recipes for every op.
3. Propagate hard required layouts from consumers.
4. Propagate producer natural layouts where they are unique.
5. Resolve multi-recipe ops by cost.
6. Insert use-site materialization where a value has multiple incompatible uses.
7. Rematerialize cheap producers instead of materializing when cheaper.
8. Specialize internal function signatures.
9. Emit diagnostics for unsatisfied hard constraints.
10. Rewrite VMI types and insert explicit helper/rematerialized ops.
```

Tie-breaking must be deterministic.  Suggested priority:

```text
1. Avoid unsupported recipes.
2. Prefer rematerializing cheap producers over register materialization.
3. Prefer layouts accepted by all consumers without conversion.
4. Prefer memory-fused layout recipes over load + register rearrange.
5. Prefer fewer VPTO instructions.
6. Prefer contiguous only when cost ties and no consumer requests a special layout.
```

## 7. Control Flow And Functions

### 7.1 `scf.if`

All branch yields for one result must agree on one assigned layout.  If they do
not, assignment inserts materialization before `scf.yield` where possible.
The `scf.if` result type after assignment carries that layout, so
`vmi-to-vpto` does not need to inspect either branch body.

### 7.2 `scf.for`

Loop-carried VMI values are fixed-point variables:

```text
initial iter_arg layout
body block argument layout
yield operand layout
loop result layout
```

must converge to one layout.  If a body consumer needs another layout, it is a
use-site request inside the loop body.
The loop body block argument has no defining op.  Its layout is therefore part
of the block argument type after assignment, not information reconstructed from
the initial value or previous iteration during lowering.

### 7.3 Calls

Internal/private VMI function boundaries must make layout choices explicit in
the assigned IR.  The baseline implementation keeps function arguments in a
contiguous VMI ABI and inserts callee-entry `ensure_layout` helpers when the
callee body needs another layout.  Private helpers are then physicalized by
`vmi-to-vpto` and inlined before VPTO vecscope/backend emission so physical
`!pto.vreg`/`!pto.mask` values do not become a backend function ABI.  A later
private-function optimization may specialize signatures directly:

```text
func @producer() -> !vmi.vreg<256xf32, deinterleaved=4>
```

then physicalized by `vmi-to-vpto` into multiple VPTO function results.

Public/external VMI function boundaries are rejected until a stable VMI ABI is
defined.

## 8. vmi-to-vpto Contract

`vmi-to-vpto` receives layout-assigned VMI.  It performs no global reasoning.

For each op, the pattern:

```text
1. reads operand/result layouts
2. reads current op attrs and operand values
3. asks TypeConverter for ordered physical values
4. emits the locally implied VPTO recipe
5. fails if target capability or required local proof is absent
```

The pattern must not:

```text
1. inspect all users to decide result layout
2. inspect defining ops to decide source layout
3. choose between S=16 block_elems=1 and block_elems=8
4. decide whether a load is full_tile_readable
5. decide function signature specialization
```

Allowed local reads are deliberately narrower:

```text
arith.constant defining op:
  allowed only to materialize an operand of the current op, such as
  create_mask active_lanes or a constant memory offset

current VMI op body/attrs:
  allowed for op-local semantics, such as create_group_mask
  active_elems_per_group when lowering the create_group_mask op itself

helper materialization chain:
  allowed only to strip ensure_mask_layout / ensure_mask_granularity for
  static predicate analysis that does not choose a different layout or recipe

diagnostic embellishment:
  allowed only to improve an already-failed capability message, such as naming
  memref.subview after identity lane-to-address planning has failed
```

Anything else is a layout-assignment responsibility.  In particular, an
unsupported producer/consumer combination must be rejected before assignment
emits layout-assigned IR.  Section 3.44 is the model for supported partial S=32
grouped masks: assignment emits explicit contiguous and deinterleaved mask
values, and `vmi-to-vpto` lowers the deinterleaved mask op itself through
contiguous grouped-mask materialization followed by predicate deinterleave.  It
does not walk from `group_reduce_addf` to the mask producer to choose or reject
the recipe.  Dynamic `active_elems_per_group` follows the same rule: the
`create_group_mask` op lowers its own SSA scalar with vci/vshrs/vshls/vsub/vcmps
for contiguous chunks before any predicate deinterleave.

## 9. Physical Value Ordering

The OneToN lowering order is fixed.

```text
contiguous:
  chunk0, chunk1, ...

deinterleaved=F:
  part0_chunk0, part0_chunk1, ...,
  part1_chunk0, part1_chunk1, ...,
  ...
  part(F-1)_chunk0, ...

group_slots(G,K):
  slot_block0, slot_block1, ...
```

Two physical bundle entries may alias the same VPTO SSA value when the local
recipe proves they have the same contents, such as group_broadcast feeding both
parts of a `deinterleaved=2` broadcast result.  Arity still follows the layout;
aliasing is not a different layout.

## 10. Diagnostics

Diagnostics are part of the design.  They must name:

```text
1. the VMI op
2. source logical type
3. assigned source layout
4. requested layout
5. missing local proof or disabled fallback
6. suggested rewrite when available
```

Examples:

```text
dense store of group_slots:
  use group_store, group_broadcast, or explicit group-pack

packed group-slot f32->f16:
  group_broadcast before truncf, or keep group_store as f32

S=32 tail without full_tile_readable:
  mark source full_tile_readable or enable stable gather fallback

S=32 group_load with unaligned source_group_stride:
  choose a stride divisible by 8 f32 elements or enable stable gather fallback

public VMI function boundary:
  make function internal, inline before assignment, or define ABI layout
```

## 11. Design Completion Criteria

The design is complete only when:

```text
1. every case in vmi-layout-lowering-cases.md maps to a local recipe
2. every local recipe can be emitted without looking at producer/user context
3. every unsupported case has a precise capability diagnostic
4. every control-flow/function boundary either specializes layout or diagnoses
5. every mask has explicit data layout and predicate granularity
6. every positive case has end-to-end lit coverage
7. every simulator-supported positive case has simulator validation
```
