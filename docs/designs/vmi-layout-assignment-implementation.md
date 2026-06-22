# VMI Layout Assignment Implementation Plan

本文是 `vmi-layout-assignment` 和 `vmi-to-vpto` 的实现计划。它配套
`vmi-layout-assignment-lowering-design.md`，并以
`vmi-layout-lowering-cases.md` 为测试和验收来源。

不使用旧 `vmi-dialect-design.md` 作为设计输入。

## 1. Pipeline

Recommended pass pipeline:

```text
pto-validate-vmi-ir
  -> vmi-layout-assignment                  // hard legalization baseline
  -> canonicalize/cse
  -> vmi-layout-fold-consumers              // optional optimization
  -> canonicalize/cse
  -> vmi-layout-rematerialize               // optional optimization
  -> canonicalize/cse
  -> vmi-layout-sink-materialization        // optional optimization
  -> canonicalize/cse
  -> vmi-legalize-arith-select
  -> pto-validate-vmi-layout-ir
  -> vmi-to-vpto
  -> canonicalize/cse
  -> existing VPTO lowering/codegen
```

Only `vmi-layout-assignment` is required for the first legal implementation.
The optimization passes may be introduced one by one.  Their contract is that
they consume legal layout-assigned VMI IR and produce legal layout-assigned VMI
IR; they never move a hidden decision into `vmi-to-vpto`.

Pass responsibilities:

```text
pto-validate-vmi-ir:
  verify surface VMI has no physical VPTO layout dependency
  reject public/external VMI ABI unless explicitly enabled

vmi-layout-assignment:
  solve hard value layout constraints
  choose explicit layouts and local recipe carriers visible in IR
  insert ensure/rematerialization helpers
  make internal function boundary layouts explicit
  rewrite VMI types with layout attrs

canonicalize/cse:
  remove dead helpers and merge identical cloned producers where MLIR legality
  permits

vmi-layout-fold-consumers:
  fold use-site materialization into consumers that can directly consume the
  source layout while preserving the same logical effect
  example: ensure_layout(deinterleaved=2 -> contiguous) feeding store may become
  a store of deinterleaved=2 when the store has a local vstsx2 INTLV recipe
  current implementation: pto.vmi.store, pto.vmi.tile_write, and the value
  operand of pto.vmi.masked_store when the existing mask arity matches, fed by
  ensure_layout from deinterleaved=2/4, block_elems=1 to contiguous.  factor=2
  uses the store's vstsx2 INTLV recipe; factor=4 is still store-local, but it
  materializes through physical interleave before vsts.

vmi-layout-rematerialize:
  replace explicit ensure_* helpers with cloned cheap layout-polymorphic
  producers when the clone directly creates the requested result type
  current implementation: splat pto.vmi.constant, pto.vmi.broadcast,
  pto.vmi.iota, pto.vmi.create_mask, pto.vmi.create_group_mask, and
  pto.vmi.constant_mask
  not included in the first implementation: load, group_load, masked_load,
  group_slot_load, and group_broadcast; those require separate memory,
  execution-count, or source-layout proof before they can be rematerialized

vmi-layout-sink-materialization:
  move ensure_layout across pure layout-transparent elementwise chains when the
  rewritten IR reduces materialization cost and keeps every op locally legal
  current implementation: sink two identical operand ensure_layout helpers
  across binary add/sub/mul/div/min/max/and/or/xor/shl/shru VMI ops, or one
  source ensure_layout across unary neg/abs/sqrt/exp/ln/relu/not VMI ops,
  producing one result ensure_layout.  It also sinks matching
  ensure_mask_layout or ensure_mask_granularity helpers across
  mask_and/mask_or/mask_xor/mask_not, producing one result mask helper.  It
  does not sink through select, fma, cast, load, store, reduce,
  group_broadcast, or control-flow ops

vmi-legalize-arith-select:
  restore scalar-condition arith.select with VMI result type back to scf.if
  after canonicalize; canonicalize may fold simple scf.if into arith.select,
  but VMI values must not cross non-VMI semantic ops before vmi-to-vpto

pto-validate-vmi-layout-ir:
  verify every VMI data/mask value has layout
  verify every VMI value has an assigned layout and every non-local lowering
  choice has been serialized explicitly
  verify helper ops have registered materialization recipes.  Current
  implementation checks `ensure_layout`, `ensure_mask_layout`, and
  `ensure_mask_granularity` at the layout gate, so unsupported helper recipes
  fail before `vmi-to-vpto`.  It also checks the first semantic local-recipe
  families, non-contiguous `pto.vmi.store`/`pto.vmi.tile_write`, block8
  `pto.vmi.group_load`, `pto.vmi.group_slot_load`, group_slots
  `pto.vmi.group_store`, group_slots `pto.vmi.group_reduce_addf`,
  explicit-slots `pto.vmi.group_broadcast`, `pto.vmi.truncf`,
  `pto.vmi.extf`, and `pto.vmi.bitcast`, at the layout gate.

vmi-to-vpto:
  use OneToN type conversion
  lower only from current-op attrs/operands, operand/result layouts, and helper
  ops
  emit VPTO or precise unsupported diagnostic
```

### 1.1 Hard Constraints Versus Optimizations

Hard legalization answers "can this program be lowered correctly?"  It is
allowed to be conservative:

```text
%w = pto.vmi.extf %a                 // natural layout deinterleaved=2
%t1 = pto.vmi.mulf %w, %k1           // layout-transparent, stays deinterleaved=2
%t1_c = pto.vmi.ensure_layout %t1    // hard store contract wants contiguous
pto.vmi.store %t1_c, %OUT1
%w_c = pto.vmi.ensure_layout %w
pto.vmi.store %w_c, %OUT2
```

This is a correct legal shape.  The contiguous action is explicit at each store
use, and `vmi-to-vpto` lowers the helper with register materialization such as
`vintlv` before ordinary `vsts`.

Optimization answers "can the same external effect be cheaper?"  A fold pass
may rewrite the two store uses to consume the deinterleaved values directly:

```text
pto.vmi.store %t1, %OUT1   // value type still says deinterleaved=2
pto.vmi.store %w,  %OUT2
```

This optimized shape is legal only because `pto.vmi.store` has enough local
information to lower a `deinterleaved=2` f32 value to row-major memory, for
example with `vstsx2 INTLV_B32`.  The optimization does not require
`vmi-to-vpto` to inspect `%w`'s producer or the sibling store.

The split gives later passes room to improve layout choices:

```text
hard pass:
  guarantee legality with explicit ensure_* helpers

optimization passes:
  remove, fold, clone, or sink helpers when the optimized IR is still locally
  deterministic

vmi-to-vpto:
  physicalize exactly the IR it sees, with no global planning
```

## 2. Files To Add Or Update

Expected implementation files:

```text
include/PTO/IR/VMITypes.td
include/PTO/IR/VMIOps.td
include/PTO/IR/VMIAttrs.td
lib/PTO/IR/VMI.cpp

include/PTO/Transforms/Passes.td
lib/PTO/Transforms/PTOValidateVMIIR.cpp
lib/PTO/Transforms/VMILayoutAssignment.cpp
lib/PTO/Transforms/VMIToVPTO.cpp
lib/PTO/Transforms/VMILocalRecipeRegistry.cpp

test/lit/vmi/vmi_layout_assignment_*.pto
test/lit/vmi/vmi_to_vpto_*.pto
test/vpto/cases/vmi/*/
```

Exact names may follow project conventions, but the layering should remain:

```text
IR definitions
  -> validation
  -> assignment
  -> OneToN lowering
  -> lit and sim tests
```

## 3. IR Types And Attributes

### 3.1 Layout Attribute

Represent layout as a closed attribute family:

```text
#pto.vmi.layout<contiguous>
#pto.vmi.layout<deinterleaved = F, block_elems = B>
#pto.vmi.layout<num_groups = G, slots = K>
```

C++ form:

```c++
enum class VMILayoutKind {
  Contiguous,
  Deinterleaved,
  GroupSlots,
};

struct VMILayoutKey {
  VMILayoutKind kind;
  int64_t deinterleaveFactor = 1;
  int64_t blockElems = 1;
  int64_t numGroups = 0;
  int64_t slots = 0;
};
```

Verifier rules:

```text
contiguous:
  no extra parameters

deinterleaved:
  F > 1
  B > 0
  direct full-chunk recipes require N % (F * B) == 0

group_slots:
  G > 0
  K > 0
  G % K == 0
  K fits in one physical vreg for element type
```

Parser compatibility during migration:

```text
#pto.vmi.layout<num_groups = G>
```

is accepted as a legacy spelling for the pre-design implicit group layout. New
`vmi-layout-assignment` output must not rely on that implicit form. It must
print one of:

```text
#pto.vmi.layout<num_groups = G, slots = 8>
#pto.vmi.layout<num_groups = G, slots = 1>
```

so `vmi-to-vpto` can lower from the assigned type without reconstructing group
slot placement from producer or consumer context.

### 3.2 VMI Types

Surface:

```text
!pto.vmi.vreg<NxT>
!pto.vmi.mask<Nxpred>
```

Layout-assigned:

```text
!pto.vmi.vreg<NxT, #pto.vmi.layout<...>>
!pto.vmi.mask<Nxpred, #pto.vmi.layout<...>>
```

Surface VMI types are legal before assignment.  Layout-assigned VMI types are
required after assignment.

### 3.3 Explicit Recipe Carriers

Lowering decisions are carried by the current op and its types, not by a
separate recipe string.  The allowed carriers are:

```text
op attrs and operands
operand/result VMI layouts
mask granularity and mask layouts
helper ops such as ensure_layout / ensure_mask_layout
cloned or rematerialized producers
diagnostics for unsupported shapes
```

If assignment made a non-local choice by inspecting producers, users, sibling
users, control flow, callees, or memory context, it must rewrite the IR so that
the final choice is visible through those carriers before `vmi-to-vpto`.

Local-decision table for the current implementation:

```text
op                         local decision inputs
group_load                 result layout, num_groups, row_stride, source type
group_slot_load            result group_slots layout and source_group_stride
group_reduce_addf          source/mask/result layouts, num_groups, reassoc
group_broadcast            source/result layouts and num_groups
truncf                     source/result layouts and element widths
ensure_layout              always carries source/result layouts instead of recipe
ensure_mask_layout         always carries source/result layouts instead of recipe
ensure_mask_granularity    always carries source/result granularities instead of recipe
```

Layout/attr-only decisions today:

```text
load                       result layout plus full_read_elems/full chunk proof
group_store                source group_slots layout plus explicit output stride
masked_load                explicit passthrough, mask layout, and memory proof
masked_store/select        operand/result layouts plus mask granularity
dense extf/truncf          source/result layouts and element widths
```

Implementation rule:

```text
validate-assigned-vmi validates assigned layouts, mask granularity, boundaries,
and helper placement.
vmi-to-vpto emits VMI-LAYOUT-CONTRACT for missing local proof.
If a layout/attr-only op later gains a second legal recipe that cannot be
distinguished from current-op information, that recipe must be represented by a
new attr, helper op, or rematerialized op before vmi-to-vpto can emit it.
Unsupported shapes that have no registered recipe still diagnose through their
specific capability check rather than failing with a generic missing-recipe
error.
```

Examples of forbidden recovery in `vmi-to-vpto`:

```text
group_reduce_addf cannot walk to a load/group_load producer to choose S=16
  parity versus block8.
group_store cannot inspect the group_reduce producer; it consumes only the
  assigned source layout and explicit stride.
group_broadcast cannot inspect sibling users to decide whether to rematerialize.
masked_load cannot inspect the mask producer to prove memory safety.
func.call cannot inspect the callee body to decide physical function layout.
```

## 4. VMI Surface Ops Required By Cases

Initial op set from the case catalog:

```text
load
group_load
group_slot_load
store
masked_store

create_mask
create_group_mask

extf
truncf
addf
mulf
select
broadcast

group_reduce_addf
group_broadcast
group_store

ensure_layout                 // internal
ensure_mask_layout            // internal
ensure_mask_granularity       // internal
```

Important semantic split:

```text
load:
  optional full_read_elems=N is a memory-safety contract for pointer sources.
  It states that source[offset : offset + N) may be physically read even if the
  VMI logical result has fewer active lanes.

group_load:
  loads group_size data elements per group

group_slot_load:
  loads one scalar per group and produces group_slots
```

## 5. Local Recipe Registry

Create one target-aware local recipe registry shared by assignment and lowering.
It is not serialized as a separate recipe-selection attribute.  It answers local legality
questions from op kind, explicit attrs/operands, layouts, and target capability.

```c++
class VMILocalRecipeRegistry {
public:
  SmallVector<ProducerRecipe> getProducerRecipes(Operation *op);
  SmallVector<ConsumerRecipe> getConsumerRecipes(OpOperand &use);
  SmallVector<TransferRecipe> getTransferRecipes(Operation *op);
  FailureOr<MaterializationRecipe>
  getMaterializationRecipe(Type valueType, VMILayoutKey from,
                           VMILayoutKey to);
  bool isCheaplyRematerializable(Operation *op);
  bool hasTargetCapability(RecipeID recipe) const;
};
```

Recipe record:

```c++
struct VMILayoutRecipe {
  RecipeID id;
  SmallVector<VMILayoutKey> operandLayouts;
  SmallVector<VMILayoutKey> resultLayouts;
  int64_t cost;
  bool requiresFullTileReadable;
  bool mayReadInactivePhysicalLanes;
  DiagnosticBuilder (*explainFailure)(...);
};
```

The registry must be target-aware but deterministic.  It should not read global
mutable state.  Pass options configure fallback availability:

```text
enableScratchFallback
enableGatherFallback
enablePublicVMIABI
diagnosticVerbosity
```

Assignment and optimization passes may query the registry to decide which IR
shape to produce.  `vmi-to-vpto` may query the same registry to verify the
current op is locally lowerable.  If the same op, attrs, operands, and
operand/result layouts could map to two different physical recipes with
different observable preconditions, the IR is under-specified; add an explicit
attr, operand, helper op, or distinct VMI semantic op before implementing that
recipe.

Current implementation status: `VMILocalRecipeRegistry` exists and currently
owns nine local recipe families:

```text
contiguous store/tile_write consumer recipes:
  contiguous vsts
  deinterleaved=2 vstsx2 INTLV
  deinterleaved=4 materialize-then-vsts

helper materialization recipes:
  data/mask layout identity
  data/mask contiguous <-> deinterleaved=2/4 when source/result physical
  arity matches and the physical part shape can be materialized
  mask granularity identity or b8/b16/b32 predicate cast

group_slot_load semantic recipes:
  slots=8 unit-stride vsldb
  slots=1 aligned lane-0 vsldb per group

block8 group_load semantic recipes:
  S=16 deinterleaved=2, block_elems=8 vsldb per row fragment
  S=32 deinterleaved=4, block_elems=8 vsldb per row fragment

group_slots group_store semantic recipes:
  slots=8 unit-stride vsts
  slots=1 aligned lane-0 vsts per group

group_slots group_reduce_addf semantic recipes:
  S=8 vcgadd
  S=16 deinterleaved=2 vcgadd+vadd
  S=32 deinterleaved=4 vcgadd+vadd tree
  S=64 contiguous slots=1 vcadd/vadd/vsel row-local reduction

explicit-slots group_broadcast semantic recipes:
  slots=8/slots=1 vselr materialization to contiguous or supported
  deinterleaved result layouts

extf/truncf semantic recipes:
  contiguous f16/bf16 -> deinterleaved=2 f32
  contiguous f8-like -> deinterleaved=4 f32
  deinterleaved=2 f32 -> contiguous f16
  deinterleaved=4 f32 -> contiguous f8-like
  group_slots(G, slots=1) f32 -> f16

bitcast semantic recipes:
  per-part vbitcast for contiguous/deinterleaved layouts when source/result
  layouts match, physical arity matches, and every physical chunk carries the
  same logical bit footprint; this does not require each deinterleaved part to
  contain the same number of chunks. group_slots bitcast is unsupported until a
  slot-wise bitcast contract is defined.
```

`vmi-layout-fold-consumers`, `pto-validate-vmi-layout-ir`, and `vmi-to-vpto`
query this registry for the decisions implemented above.

## 6. Layout Assignment Data Model

### 6.1 Solver State

```c++
struct ValueLayoutState {
  Value value;
  Type logicalType;
  SmallVector<VMILayoutKey> candidates;
  std::optional<VMILayoutKey> chosen;
  SmallVector<UseRequest> useRequests;
};

struct UseRequest {
  OpOperand *operand;
  VMILayoutKey requestedLayout;
  RecipeID requestingRecipe;
  bool hard;
};

struct OpRecipeState {
  Operation *op;
  SmallVector<VMILayoutRecipe> candidates;
  std::optional<RecipeID> chosen;
};
```

### 6.2 Collection Phase

Walk the module and collect:

```text
1. every VMI value
2. every VMI block argument
3. every VMI function argument/result
4. every VMI op with candidate local recipes
5. every branch/yield/call/return edge carrying VMI
```

Build SCCs over:

```text
dataflow uses
region yields
loop iter_args
function call graph for private/internal functions
```

Public/external VMI function boundaries are rejected unless
`enablePublicVMIABI` is explicitly supported.

Block arguments are first-class layout variables.  Assignment must write the
chosen layout into the block argument type or specialized function signature.
`vmi-to-vpto` must never recover a block argument layout by walking to an
incoming branch, yield, or call operand.

### 6.3 Constraint Generation

Examples:

```text
truncf f32->f16:
  source request deinterleaved=2, block_elems=1
  result contiguous

group_reduce S=16:
  source candidate deinterleaved=2, block_elems=1
  source candidate deinterleaved=2, block_elems=8
  result group_slots(G, slots=8)

group_reduce S=32:
  source candidate deinterleaved=4, block_elems=1
  source candidate deinterleaved=4, block_elems=8
  result group_slots(G, slots=8)

group_reduce S=64:
  source request contiguous
  result group_slots(G, slots=1)

group_broadcast:
  source request group_slots(G,K)
  result candidate comes from each dense consumer request
  op is rematerializable per use

ordinary dense add/mul/select:
  operands/results same dense layout

group-slot add/mul:
  operands/results same group_slots(G,K)

ordinary store:
  dense source required
  group_slots source is illegal

group_store:
  source request group_slots(G,K)
```

Consumer-driven adoption is limited to producers that are layout-transparent or
can produce the requested memory layout directly:

```text
direct layout producer:
  load, tile_read

layout-transparent producer:
  broadcast, constant, iota
  add/sub/mul/fma/div/min/max/neg/abs/sqrt/exp/ln/relu
  integer bitwise/shift/not
  select, bitcast
```

For a non-load layout-transparent producer, only non-contiguous consumer
requests may be adopted by the producer equivalence class.  Contiguous requests
from ordinary stores are handled by use-site `ensure_layout` or
rematerialization instead.  This prevents a dense store from overwriting a
natural `deinterleaved` cast layout while still allowing:

```text
load -> broadcast -> addf -> S=32 group_reduce
```

to assign the whole producer chain as
`deinterleaved = 4, block_elems = 8` before `vmi-to-vpto`.

Memory legality constraints:

```text
S=32 tail fast load:
  requires full_tile_readable
  otherwise require gather fallback or diagnose

compact S=12 logical S=16:
  requires compact-row gather materialization
  diagnose if gather fallback is disabled/missing
```

### 6.3.1 Request Builders

Implement request generation as small per-op builders.  The builders produce
candidate recipes and use-site requests; they do not rewrite IR.

```text
buildStoreRequests:
  ordinary store -> dense contiguous request unless a layout-aware store recipe is
  selected
  group_store -> group_slots(G,K) request plus stride/alignment capability
  checks

buildCastRequests:
  extf f16->f32 -> source contiguous, result deinterleaved=2
  extf f8->f32  -> source contiguous, result deinterleaved=4
  truncf f32->f16 -> source deinterleaved=2/block_elems=1, result contiguous
  truncf f32->f8  -> source deinterleaved=4/block_elems=1, result contiguous
  group_slots slots=1 f32->f16 -> slot-preserving recipe
  group_slots slots=8 width-changing cast -> diagnostic unless a packed recipe
  exists

buildGroupReduceRequests:
  derive S = logical_lanes / num_groups
  S=8  -> contiguous source, group_slots(G,8) result
  S=16 -> deinterleaved=2/block_elems=1 or block_elems=8 source,
          group_slots(G,8) result
  S=32 -> deinterleaved=4/block_elems=1 or block_elems=8 source,
          group_slots(G,8) result
  S=64 -> contiguous source, group_slots(G,1) result
  other S -> diagnostic unless an explicit fallback recipe is enabled

buildGroupMemoryRequests:
  group_load S=16/S=32 with aligned constant stride -> block_elems=8 recipe
  group_load row-local full chunks -> contiguous recipe
  group_slot_load unit stride -> group_slots(G,8)
  group_slot_load aligned row-local stride -> group_slots(G,1)
  unsupported dynamic/unaligned grouped memory -> diagnostic

buildElementwiseRequests:
  dense add/mul/fma/min/max/select -> all dense operands/results share one
  dense layout
  group-slot add/mul/select -> all operands/results share one group_slots(G,K)
  dense/group_slots mixing -> diagnostic unless an explicit group_broadcast or
  group_store boundary exists

buildMaskRequests:
  mask layout follows each consuming data layout
  predicate granularity follows each consuming element type
  create_mask/create_group_mask may be cloned for incompatible mask layout or
  granularity requests
  masked_store requests source layout, mask layout, and store predicate
  granularity explicitly

buildControlFlowRequests:
  region yields, branch operands, loop iter_args, call operands, and returns
  create equality requests on the carried VMI layout variable

buildFunctionBoundaryRequests:
  private/internal function argument/result layouts are specialized or
  materialized with callee-entry/return-site helpers
  public/external VMI arguments/results diagnose unless enablePublicVMIABI has
  a real ABI recipe
```

Request builders must record the requesting op.  Diagnostics and inserted
helpers are use-site operations, so the user can see which consumer forced a
layout.

### 6.3.2 Producer Classes

The solver uses producer classes to decide whether a conflict can be solved by
cloning, equivalence propagation, or materialization.

```text
cheap rematerializable producers:
  load when address operands dominate the clone site, no intervening may-alias
  write exists, and any full_read_elems proof is preserved
  broadcast
  create_mask
  create_group_mask
  group_broadcast
  group_slot_load when the same address/no-alias/proof conditions as load hold
  and the memory recipe is legal at the clone site

layout-transparent producers:
  add/sub/mul/fma/min/max/neg/abs
  select
  bitcast
  integer bitwise and shift ops

fixed-layout producers:
  extf/truncf physical conversion recipes
  group_load block-fragment recipes
  group_reduce result group_slots
  masked_load when the physical memory-safety proof fixes a full-read recipe
```

Conflict policy:

```text
cheap producer:
  clone for each incompatible request when cloning does not duplicate a
  side-effect, cross an aliasing write, or duplicate an illegal memory read

layout-transparent producer:
  merge into the consumer-requested equivalence class; insert materialization
  only at incompatible uses

fixed-layout producer:
  use registered materialization only; otherwise diagnose
```

This is the rule that keeps case 3.32 legal: a plain `load` can be assigned to
`deinterleaved=4, block_elems=1` for both `truncf f32->f8` and S=32
`group_reduce`.  It also keeps case 3.19.2 diagnostic: a strided `group_load`
that selected `block_elems=8` is fixed unless a block8-to-parity
materialization or rematerialized memory recipe is registered.

### 6.4 Solving And Rewriting

Algorithm:

```text
1. Pick candidate recipe sets for every op.
2. Propagate hard constraints through SCCs.
3. Resolve transfer-equivalent dense values.
4. Choose multi-recipe ops by cost:
   - S=16 parity vs block8
   - load memory-fused vs load+materialize
   - group_slot_load slots=8 vs slots=1
5. For conflicting uses:
   - rematerialize cheap producer where legal
   - otherwise insert ensure_layout at use
   - otherwise diagnose
6. Rewrite VMI result/block/function types with chosen layouts.
7. Insert helper ops with source/result layout attrs.
```

Rewrite invariants:

```text
No VMI data/mask value after assignment has a null layout.
Any non-local choice is represented by op attrs, operand/result layouts, a
helper op, a clone, or an explicit diagnostic.
Every ensure_* helper has a registered materialization recipe.
Every function/call signature carrying VMI is specialized or diagnosed.
```

### 6.5 Rewrite Artifacts

Assignment rewrites the IR so that later lowering has no hidden choices.

```text
type rewrite:
  every VMI data/mask result and block argument receives a layout attr

clone rewrite:
  cheap producers are cloned before their divergent use sites
  each clone receives its own layout and attrs

ensure rewrite:
  non-cheap values use pto.vmi.ensure_layout or ensure_mask_layout at the use
  site, with source and target layouts visible in the types

granularity rewrite:
  one semantic mask used by f32 and f16 consumers gets
  ensure_mask_granularity or cloned mask producers

control-flow rewrite:
  scf.if/scf.for yields and block arguments are rewritten to one agreed layout;
  materialization is inserted before yield when branches differ

function rewrite:
  private VMI functions are specialized or get callee-entry ensure_layout
  public/external VMI functions are diagnosed
```

Canonical assigned IR shape for a conflicting load:

```text
%x = pto.vmi.load ...
  : ... -> !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>

%x_dense = pto.vmi.ensure_layout %x
  : !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>
 -> !pto.vmi.vreg<256xf32, #pto.vmi.layout<contiguous>>

pto.vmi.store %x_dense, ...
```

Canonical assigned IR shape for a cloned cheap producer:

```text
%x_s16 = pto.vmi.load ...
  : ... -> !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 2>>

%x_s32 = pto.vmi.load ...
  : ... -> !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>
```

Canonical assigned IR shape for `group_broadcast` multi-use:

```text
%b0 = pto.vmi.group_broadcast %slots
  : !pto.vmi.vreg<256xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
 -> !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>

%b1 = pto.vmi.group_broadcast %slots
  : !pto.vmi.vreg<256xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
 -> !pto.vmi.vreg<256xf32, #pto.vmi.layout<contiguous>>
```

If the assigned IR does not have one of these explicit shapes, `vmi-to-vpto`
must reject it instead of attempting to recover the missing decision.

### 6.6 Case-To-Implementation Closure Matrix

The current case catalog is sufficient for the first implementation.  No new
layout kind is justified by the supported endpoints.  The implementation work
should instead close the following finite matrix.  Each row names the request
builder that owns the decision, the assignment artifact that must appear in IR,
and the `vmi-to-vpto` contract.

```text
case family                     builder / owner             assignment artifact
3.1, 3.2, 3.3 dense casts       buildCastRequests           dense layout on each cast result
3.29 mask width split           buildMaskRequests           per-use mask granularity helper
3.31, 3.32 dense fanout         conflict resolver           cloned load or ensure_layout

vmi-to-vpto contract:
  consume only the assigned dense layouts.  It may emit VCVT and dense
  materialization, but it must not choose deinterleaved=2/4 by inspecting a
  later truncf, store, or group_reduce user.
```

```text
case family                     builder / owner             assignment artifact
3.4 S=8 reduce                  buildGroupReduceRequests    s8_reduce_contiguous recipe
3.5 S=16 reduce                 buildGroupReduceRequests    s16_reduce_parity/block8 recipe
3.6 S=32 reduce                 buildGroupReduceRequests    s32_reduce_dintlv4/block8 recipe
3.7 S=64 reduce                 buildGroupReduceRequests    s64_reduce_row_local recipe
3.11.1 S=64 active-row tail     buildMaskRequests           active-row store/reduce masks
3.19.1 S=16 block_elems choice  buildGroupReduceRequests    explicit block_elems layout
3.38 multi-tile S=32 reduce     buildGroupReduceRequests    multiple group_slots chunks
3.26 grouped tail               buildMaskRequests           split grouped masks
3.44, 3.45 grouped S=32 masks   buildMaskRequests           explicit deinterleaved mask values

vmi-to-vpto contract:
  lower each reduce from the current op's attrs, source/mask layout, result
  group_slots layout.  It must not walk to the load/group_load producer to
  decide parity versus block8, row-local versus packed slots, or static versus
  dynamic mask generation.
```

```text
case family                     builder / owner             assignment artifact
3.15.1 S=16 row stride 16       buildGroupMemoryRequests    block_elems=8 group_load recipe
3.15.2 S=16 row stride > 16     buildGroupMemoryRequests    strided block_elems=8 plan
3.16.1 group_slot_load slots=8  buildGroupMemoryRequests    unit-stride packed slots plan
3.16.2 group_slot_load slots=1  buildGroupMemoryRequests    row-local aligned slots plan
3.27 strided group_load         buildGroupMemoryRequests    positive block_elems=8 plan
3.28 slots=1 non-unit load      buildGroupMemoryRequests    row-local group_slot_load recipe
3.37 slots=1 strided store      buildStoreRequests          group_store stride/alignment proof
3.39 strided load fanout        conflict resolver           preserving layout or materialization

vmi-to-vpto contract:
  consume only explicit memory stride/alignment attrs, current op operands,
  and layouts.  It must not infer safe read/write placement from neighboring
  compute ops.  Unsupported dynamic, unaligned, or compact-row gather shapes
  stay diagnostics until a gather recipe is explicit in the current op.
```

```text
case family                     builder / owner             assignment artifact
3.8 reduce->truncf->broadcast   conflict resolver           slot cast plus dense materialization
3.10 non-load S=32 producer     buildElementwiseRequests    transparent deinterleaved chain
3.17 broadcast deint consumer   conflict resolver           use-site group_broadcast layout
3.18 dense + reduce users       conflict resolver           clone/rematerialize/ensure_layout
3.23 broadcast multi-user       conflict resolver           cloned group_broadcast
3.33 S=16 + S=32 users          conflict resolver           cloned load or materialization
3.34 S=64 slots=1 cast          buildCastRequests           group_slot_cast layout
3.35 slots fanout               buildElementwiseRequests    same group_slots layout on users
3.36 scalar slots=8/slots=1     conflict resolver           cloned group_slot_load/broadcast
3.40 scalar dense + grouped     conflict resolver           cloned broadcast
3.41 incompatible fixed value   conflict resolver           diagnostic or ensure_layout

vmi-to-vpto contract:
  each op instance is already single-plan.  The lowering pass never scans
  sibling users to decide whether to clone, pack, broadcast, or materialize.
```

```text
case family                     builder / owner             assignment artifact
3.21 S=32 safe full-read tail    buildMaskRequests           full_read_elems memory proof
3.24 mask/select/store          buildMaskRequests           explicit mask layout/granularity
3.12 scf.if before reduce       buildControlFlowRequests    common yielded layout
3.20 group_slots scf.if         buildControlFlowRequests    common group_slots layout
3.22 scf.for carried value      buildControlFlowRequests    fixed-point iter_arg layout
3.25 function boundary          buildFunctionBoundary       specialized/internal boundary
3.42 loop accumulator           buildControlFlowRequests    loop-carried group_slots layout
3.43 call argument materialize  buildFunctionBoundary       callee-entry/return helper

vmi-to-vpto contract:
  block argument, region result, call operand, and function result layouts are
  visible in types or helper ops.  It must not inspect branch bodies, loop
  bodies, callers, or callees to discover a layout.
```

```text
diagnostic family               builder / owner             required failure
3.7.4 slots=1 unit-stride store buildStoreRequests          no aligned row-local store recipe
3.9 dense store of group slots  buildStoreRequests          use group_store/group_broadcast
3.11.2 S=32 unsafe tail         buildMaskRequests           missing full_tile_readable/gather
3.13 slots=8 width cast         buildCastRequests           no packed slot cast recipe
3.14 unsupported group size     buildGroupReduceRequests    no registered reduce recipe
3.15.3 compact S=12            buildGroupMemoryRequests    no compact gather plan
3.16.1 slots=8 non-unit load    buildGroupMemoryRequests    no packed strided slot load recipe
3.16.2 slots=1 bad stride       buildGroupMemoryRequests    no dynamic/unaligned row-local plan
3.19.2 invalid block_elems use  conflict resolver           no preserving materialization
3.25.2 public/external ABI      buildFunctionBoundary       no stable public VMI ABI
3.27 unaligned group_load       buildGroupMemoryRequests    no gather/block fallback recipe
3.30 masked_load unsafe tail    buildMaskRequests           no padding/gather fallback

vmi-to-vpto contract:
  these cases must fail before or at the layout contract boundary with the
  requesting op named.  They must not be accepted by falling back to a generic
  dense load, dense store, or producer/user inspection.
```

Additional cases are needed only when the scope changes:

```text
stable gather fallback enabled:
  add compact S=12 positive lowering and masked_load unsafe-tail positive
  lowering before accepting either path.

pack-to-slots=8 or unaligned row-local stores enabled:
  add positive S=64 unit-stride group_store and reduce->pack->dense store cases.

public VMI ABI enabled:
  add public call/return ABI cases before removing the public-boundary
  diagnostic.

packed group-slot width cast enabled:
  add slots=8 f32->f16 cast and downstream group_store/broadcast cases.
```

## 7. OneToN Type Conversion

`vmi-to-vpto` should use OneToN conversion for VMI values.

Conversion rules:

```text
contiguous:
  ceil(N / lanesPerVReg(T)) physical vregs

deinterleaved=F:
  F * ceil((N / F) / lanesPerVReg(T)) physical vregs
  ordering: part-major, then chunk

group_slots(G,K):
  ceil(G / K) physical vregs
  each vreg has logical slot lanes 0..K-1 live
```

Mask conversion:

```text
mask layout follows data layout
mask granularity is selected from consumer element width:
  f32/i32 -> b32
  f16/i16 -> b16
  f8/i8   -> b8
```

If one logical mask is used by multiple widths, assignment inserts
`ensure_mask_granularity` or rematerializes the mask producer.

## 8. VMI-to-VPTO Pattern Rules

Each pattern uses:

```text
op
op attrs and operand values
operand/result layouts
adaptor physical values
```

Each pattern rejects:

```text
missing current-op proof for an otherwise unsafe memory recipe
missing target capability
unexpected group_slots dense consumer
```

Target local recipe matrix:

```text
load, recipe=dense_load_norm:
  result layout contiguous
  emits pto.vlds / pto.vsts NORM paths
  covers dense store users and S=64 row-local reduce input

load, recipe=load_dintlv2:
  result layout deinterleaved=2, block_elems=1
  emits vldsx2 DINTLV_B32 or normal load + vdintlv materialization
  covers f32->f16, S=16 parity reduce, f16->f32 widened values

load, recipe=load_dintlv4:
  result layout deinterleaved=4, block_elems=1
  emits two vldsx2 DINTLV_B32 plus vdintlv
  covers f32->f8, S=32 dintlv4 reduce

group_load, recipe=s16_group_load_block8_unit_stride:
  result layout deinterleaved=2, block_elems=8
  emits vldsx2/BDINTLV for 8 rows of 16xf32
  covers compact logical S=16 when source_group_stride == 16

group_load, recipe=s16_group_load_block8_stride:
  result layout deinterleaved=2, block_elems=8
  emits two vsldb strided 32B block loads
  requires source_group_stride % 8 == 0

group_load, recipe=s32_group_load_block8_stride:
  result layout deinterleaved=4, block_elems=8
  emits four vsldb strided 32B block loads
  requires source_group_stride % 8 == 0

group_load, recipe=group_load_contiguous_chunks:
  result layout contiguous
  emits one vlds per physical group chunk using row_stride address arithmetic
  covers the currently implemented full-chunk row-local group_load path

group_reduce_addf, recipe=s8_reduce_contiguous:
  consumes contiguous f32 with group size 8
  produces group_slots(G, slots=8)
  emits one vcgadd

group_reduce_addf, recipe=s16_reduce_parity:
  consumes deinterleaved=2, block_elems=1
  produces group_slots(G, slots=8)
  emits two vcgadd operations and one vadd

group_reduce_addf, recipe=s16_reduce_block8:
  consumes deinterleaved=2, block_elems=8
  produces group_slots(G, slots=8)
  emits two vcgadd operations and one vadd

group_reduce_addf, recipe=s32_reduce_dintlv4:
  consumes deinterleaved=4, block_elems=1
  produces group_slots(G, slots=8)
  emits four vcgadd operations and a vadd tree

group_reduce_addf, recipe=s32_reduce_block8_stride:
  consumes deinterleaved=4, block_elems=8
  produces group_slots(G, slots=8)
  emits four vcgadd operations and a vadd tree

group_reduce_addf, recipe=s64_reduce_row_local:
  consumes contiguous f32 with group size 64
  produces group_slots(G, slots=1)
  target lowering emits per-row vcgadd plus vcadd; the current prototype uses
  the existing row-local VCADD/VADD/VSEL sequence while preserving the same
  group_slots(G, slots=1) value contract

group_slot_load, recipe=group_slot_load_slots8_unit_stride:
  result group_slots(G, slots=8)
  requires source_group_stride == 1
  emits one packed vsldb load

group_slot_load, recipe=group_slot_load_slots1_row_local:
  result group_slots(G, slots=1)
  supports aligned non-unit source_group_stride
  requires constant positive source_group_stride divisible by 256 / elementBits
  emits one lane-0 vsldb per group

group_broadcast, recipe=group_broadcast_slots8_vselr:
  source group_slots(G, slots=8)
  result dense layout selected per use
  emits vselr using assigned result layout

group_broadcast, recipe=group_broadcast_slots1_vselr:
  source group_slots(G, slots=1)
  result dense layout selected per use
  emits vdup/vselr row-local materialization

truncf, recipe=group_slot_cast_slots1_f32_to_f16:
  source/result group_slots(G, slots=1)
  emits one lane-0 vcvt per group slot block
  rejects packed slots=8 unless another plan is registered
```

The target matrix is the implementation contract.  The staged status below
records how much of that contract the current prototype has already enforced.

Current staged implementation status:

```text
group_slot_load:
  vmi-to-vpto lowers from #pto.vmi.layout<num_groups = G, slots = 8/1>
  and source_group_stride.

group_reduce_addf:
  explicit slots=8 VCGADD lowering is selected from contiguous source/mask
  layout, slots=8 result layout, num_groups, and reassoc.
  S=16 block8 assignment emits source/mask
  #pto.vmi.layout<deinterleaved = 2, block_elems = 8>, result
  #pto.vmi.layout<num_groups = G, slots = 8>; vmi-to-vpto lowers through two
  VCGADDs plus a PAT_VL8 VADD per packed result block.
  S=32 block8 assignment emits source/mask
  #pto.vmi.layout<deinterleaved = 4, block_elems = 8>, result
  #pto.vmi.layout<num_groups = G, slots = 8>; vmi-to-vpto lowers through four
  VCGADDs plus a PAT_VL8 VADD tree per packed result block.
  S=64 row-local assignment uses #pto.vmi.layout<num_groups = G, slots = 1>
  and has focused layout-assignment/vmi-to-vpto lit coverage; the explicit
  slots=1 generic VCADD row-local path is registered and selected locally.

group_broadcast:
  explicit slots=8/1 source layouts select
  packed or row-local VSELR recipes locally. Deinterleaved block-fragment
  results use the result layout block_elems as the local vselr selection group,
  so
  `deinterleaved = 4, block_elems = 8` broadcasts one group slot across each
  32B row fragment. VSELR index vectors are materialized per physical result
  chunk.  For small-group results, layout assignment has already fixed the
  result layout, and vmi-to-vpto computes:
  `firstGroup = first logical group covered by this result chunk`,
  `sourceChunk = firstGroup / slots`, and
  `baseGroupSlot = firstGroup % slots`.  The generated index vector selects
  `baseGroupSlot .. baseGroupSlot + groupsPerResultChunk - 1`; it must not be
  reused across result chunks.

group_load:
  contiguous full-chunk path is selected from a contiguous result layout.
  S=16/S=32 block-aligned strided loads are selected from
  #pto.vmi.layout<deinterleaved = 2/4, block_elems = 8>, and lower to one
  vsldb per 32B row fragment and physical chunk.  The explicit block8 recipe
  is registered and checked by pto-validate-vmi-layout-ir before vmi-to-vpto.
  The dedicated S=16 unit-stride vldsx2/BDINTLV recipe remains a local
  peephole target.
  S=16/S=32 group_load with a non-constant, non-positive, or non-8-f32-aligned
  row_stride is rejected by vmi-layout-assignment because the stable gather
  fallback is not implemented.

truncf group-slot cast:
  layout assignment and vmi-to-vpto support group_slots(G, slots=1)
  f32 -> f16 from source/result layouts and element widths. The reduce->truncf
  -> group_store slots=1 flow has focused lit coverage and no longer relies on
  vmi-to-vpto inspecting the truncf producer.

group_store:
  row-local group_slots(G, slots=1) lowering is implemented as one lane-0
  vsts per group and is covered by the reduce->truncf->group_store lit case.
  The current plan is accepted only when row_stride is a constant positive
  multiple of the 32B store alignment in destination elements: 8 for f32,
  16 for f16, and 32 for f8. Unit-stride f32 output is rejected because only
  the first row-local store is 32B-aligned; later `group_off + r` stores are
  4B apart. A future pack-to-slots=8 or unaligned-store recipe is required before
  contiguous `%c1` slots=1 group_store can be accepted.
  Packed group_slots(G, slots=8) group_store is implemented only when
  num_groups is a multiple of 8 and row_stride is constant 1; it emits one
  PAT_VL8 store per packed slot block. Non-unit packed group stores remain a
  design target unless a strided packed-lane store recipe is made explicit.
```

Examples:

```text
group_reduce_addf, recipe=s16_reduce_parity:
  consume deinterleaved=2, block_elems=1
  emit two VCGADDs and one VADD

group_reduce_addf, recipe=s16_reduce_block8:
  consume deinterleaved=2, block_elems=8
  emit two VCGADDs and one VADD

group_reduce_addf, recipe=s32_reduce_dintlv4:
  consume deinterleaved=4
  emit four VCGADDs and reduction tree

group_broadcast:
  consume group_slots
  emit VSELR or VDUP depending slots and target dense layout

group_slot_load slots=8:
  emit one packed block load for unit stride

group_slot_load slots=1:
  emit row-local lane-0 loads for constant positive 32B-aligned strides
```

## 9. Validation Passes

### 9.1 Surface Validation

Before assignment:

```text
VMI types may omit layout.
VPTO physical op must not consume VMI values.
Public/external VMI function ABI rejected unless enabled.
Unsupported vector-to-scalar extract rejected.
```

### 9.2 Layout Validation

After assignment:

```text
Every VMI value has layout.
Every VMI mask has layout and granularity plan.
Every lowering choice is locally deterministic or explicit in attrs/layouts.
Every ensure_* helper has a materialization recipe.
Every control-flow edge has matching VMI layouts.
```

### 9.3 `vmi-to-vpto` Context Read Audit

`vmi-to-vpto` may still read defining ops in narrowly scoped cases that do not
select a layout or plan:

```text
allowed:
  arith.constant for the current op's scalar operands
  create_mask/create_group_mask internals when lowering that mask op itself
  ensure_mask_layout / ensure_mask_granularity stripping for static mask facts
  memref.subview only to improve an already-failed non-identity memref
  diagnostic

not allowed:
  walking from a consumer to a producer to decide a recipe
  walking from a consumer to a mask producer to decide whether a recipe is legal
  inspecting users to choose a result layout or materialization
  recovering full_tile_readable from surrounding MTE/caller context
```

Current audit result:

```text
3.44 partial S=32 create_group_mask:
  assignment writes explicit contiguous and deinterleaved mask values.  When
  lowering the deinterleaved create_group_mask itself, vmi-to-vpto first
  materializes contiguous grouped predicate chunks and then applies predicate
  pdintlv in the same tree shape as the data vdintlv.  It still does not walk
  from group_reduce_addf to the mask defining op to choose or reject the plan.
  The dynamic active_elems_per_group form is also op-local: vmi-to-vpto lowers
  contiguous chunks with vci/vshrs/vshls/vsub/vcmps, then uses the same
  predicate pdintlv tree for S=32 deinterleaved masks.

masked_load:
  direct lowering is load + vsel.  It does not inspect the mask producer to
  choose a different load form; memory safety is provided by full physical
  chunks, shaped memref proof, or load full_read_elems.

memref.subview:
  mentioned only after identity lane-to-address planning fails.  It is not used
  to recover a hidden base/stride lowering.
```

## 10. Diagnostics

Implement diagnostics with stable prefixes:

```text
VMI-LAYOUT-CONTRACT
VMI-UNSUPPORTED-PLAN
VMI-MISSING-CAPABILITY
VMI-PUBLIC-ABI
VMI-MASK-GRANULARITY
VMI-CONTROL-FLOW-LAYOUT
```

Minimum diagnostic payload:

```text
op name
logical type
actual layout
requested layout
selected/missing plan
recommended rewrite or option
```

Example:

```text
VMI-LAYOUT-CONTRACT:
  pto.vmi.truncf requires
  #pto.vmi.layout<deinterleaved = 2, block_elems = 1>, but the source value is
  fixed to #pto.vmi.layout<deinterleaved = 2, block_elems = 8> by the selected
  strided group_load recipe. Register a rematerialization or preserving
  materialization recipe, or avoid consuming this block-loaded value with truncf.
```

## 11. Test And Simulator Acceptance

Each numbered endpoint in `vmi-layout-lowering-cases.md` should become:

```text
1. a layout-assignment lit test
2. a vmi-to-vpto lit test
3. a simulator case when the VPTO sequence is supported by the current backend
4. a diagnostic lit test when the case is explicitly unsupported
```

Repository locations:

```text
test/lit/vmi/
test/vpto/cases/vmi/
```

The current repository uses descriptive flat lit names rather than
case-numbered subdirectories.  New tests should follow the existing prefixes:

```text
vmi_layout_assignment_<case>.pto
vmi_to_vpto_<case>.pto
<runtime-case-name>/kernel.pto
```

The case number should still be recoverable from the coverage table in this
document and from the corresponding section in `vmi-layout-lowering-cases.md`.

### 11.1 Layout Assignment Checks

Each positive layout-assignment test must check:

```text
assigned data layouts
assigned mask layouts
assigned op attrs
direct vmi-to-vpto local lowering
inserted ensure_layout/rematerialized producers
control-flow/function signature specialization
```

Negative tests check diagnostic text.

### 11.2 VMI-to-VPTO Checks

Each positive vmi-to-vpto test must check:

```text
no pto.vmi ops remain
VPTO op sequence matches the case lowering
physical value arity and ordering are correct
mask granularity is correct
stores preserve observable logical memory order
```

### 11.3 Simulator Checks

Simulator cases should compare final memory against the memory result written in
the case catalog.

Current broad runtime sweep:

```text
WORK_SPACE=$PWD/.tmp/vmi-runtime-batch-layout-gate CASE_PREFIX='vmi/' JOBS=4 \
  test/vpto/scripts/run_host_vpto_validation_parallel.sh

PASS=43 FAIL=0
summary: .tmp/vmi-runtime-batch-layout-gate/parallel-summary.tsv
log scan: rg -n "RV_|alignment|\[ERROR\]|\[error\]|ERROR" \
  .tmp/vmi-runtime-batch-layout-gate.log
result: no matches
```

The `find: Permission denied` messages printed while discovering CANN simulator
paths are environment noise and are not treated as simulator failures.

Required groups:

```text
dense conversion:
  3.1, 3.2, 3.3, 3.31, 3.32

group reduce:
  3.4, 3.5.1, 3.5.2, 3.5.3
  3.6.1, 3.6.2, 3.6.3
  3.7.1, 3.7.2, 3.7.3
  3.7.4 diagnostic

layout/rematerialization:
  3.8, 3.10, 3.17, 3.18, 3.19.1, 3.22, 3.23, 3.31,
  3.32, 3.33, 3.34, 3.35, 3.36, 3.38, 3.40, 3.41

mask/tail:
  3.11.1, 3.15.1, 3.15.2, 3.21, 3.24, 3.26, 3.29,
  3.30, 3.44, 3.45

strided/group-slot memory:
  3.27, 3.28, 3.37, 3.39

function/control-flow:
  3.12, 3.20, 3.22, 3.25.1, 3.42, 3.43
```

Aggregate catalog headings are covered through their endpoint subcases:

```text
3.11 partial tail groups:
  3.11.1 positive S=64 active-row tail
  3.11.2 diagnostic S=32 tail without full_tile_readable

3.15 compact S=12 written as logical S=16:
  3.15.1 positive source row stride 16
  3.15.2 positive source row stride greater than 16
  3.15.3 diagnostic compact source row stride 12

3.16 group_slot_load layout contract:
  3.16.1 packed slots=8 positive and non-unit-stride diagnostic
  3.16.2 row-local slots=1 positive plus dynamic/unaligned diagnostics

3.25 function boundary layout specialization:
  3.25.1 private/internal boundary lit and runtime coverage
  3.25.2 public/external boundary diagnostics
```

Current coverage audit result:

```text
SIM-backed positive endpoints:
  3.1, 3.2, 3.3, 3.4, 3.5.1, 3.5.2, 3.5.3,
  3.6.1, 3.6.2, 3.6.3, 3.7.1, 3.7.2, 3.7.3,
  3.8, 3.10, 3.11.1, 3.12, 3.15.1, 3.15.2,
  3.16.1 positive, 3.16.2 positive, 3.17, 3.18,
  3.19.1, 3.20, 3.21, 3.22, 3.23, 3.24, 3.25.1, 3.26,
  3.27 positive, 3.28 positive, 3.29, 3.31, 3.32,
  3.33, 3.34, 3.35, 3.36, 3.37, 3.38, 3.39,
  3.40, 3.41, 3.42, 3.43, 3.44, 3.45

diagnostic endpoints:
  3.7.4, 3.9, 3.11.2, 3.13, 3.14, 3.15.3,
  3.16.1 non-unit slots=8 source stride,
  3.16.2 dynamic/unaligned slots=1 source stride,
  3.19.2, 3.25.2, 3.27 unaligned source_group_stride,
  3.30 unsafe masked_load tail

repository evidence:
  all concrete lit/runtime paths listed below exist
  all 43 runtime case directories contain kernel.pto, launch.cpp, main.cpp,
  golden.py, and compare.py
  latest broad VMI runtime sweep passed: PASS=43 FAIL=0
  latest full VMI lit sweep passed: 340/340
```

Current checked-in coverage for 3.3 dense f8->f32->compute->f8:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_f8_compute_f8.pto

runtime SIM:
  test/vpto/cases/vmi/f8-compute-f8
```

Current checked-in coverage for 3.1/3.2 dense f16/f32 conversion stores:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_dense_f16_f32_store.pto

runtime SIM:
  test/vpto/cases/vmi/widen-f16-to-f32-store-reduce
  test/vpto/cases/vmi/quant-f32-to-f16-tail
```

Current checked-in coverage for basic packed group_reduce -> group_store paths
for 3.4, 3.5.1, and 3.6.1:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_group_reduce_slots8_store.pto
  test/lit/vmi/vmi_layout_assignment_group_reduce_s16_store.pto
  test/lit/vmi/vmi_layout_assignment_group_reduce_s32_store.pto

runtime SIM:
  test/vpto/cases/vmi/group-reduce-basic-store
```

Current checked-in coverage for S=16 group broadcast continuation:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_group_slots_fanout.pto

runtime SIM:
  test/vpto/cases/vmi/group-reduce-s16-broadcast-reduce-store
```

Current checked-in coverage for 3.35 group_slots fanout to direct group_store
and group_broadcast:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_group_slots_fanout.pto

runtime SIM:
  test/vpto/cases/vmi/group-slots-fanout-store-broadcast
```

Current checked-in coverage for 3.8 `group_reduce -> truncf ->
group_broadcast -> dense store` and 3.17 `group_broadcast` feeding a
deinterleaved consumer:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_group_reduce_s16_truncf_broadcast_store.pto

runtime SIM:
  test/vpto/cases/vmi/group-reduce-s16-truncf-broadcast-store
```

Current checked-in coverage for 3.18 one dense value with dense and
group-reduce consumers:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_dense_group_reduce_multi_consumer.pto

runtime SIM:
  test/vpto/cases/vmi/dense-group-reduce-multi-consumer
```

Current checked-in coverage for 3.10 non-load producer feeding S=32
`group_reduce`:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_non_load_s32_reduce.pto

runtime SIM:
  test/vpto/cases/vmi/group-reduce-s32-add-bias-store
```

Current checked-in coverage for 3.23 group_broadcast with multiple dense
consumers:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_group_broadcast_multi_consumer.pto

runtime SIM:
  test/vpto/cases/vmi/group-broadcast-multi-consumer
```

Current checked-in coverage for S=32 contiguous group broadcast continuation:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_group_reduce_s32_broadcast_reduce.pto

runtime SIM:
  test/vpto/cases/vmi/group-reduce-s32-broadcast-reduce-store
```

Current checked-in coverage for 3.21 S=32 tail with a statically safe
full-read source:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_group_reduce_s32_tail_full_tile.pto

runtime SIM:
  test/vpto/cases/vmi/group-reduce-s32-tail-full-tile-store
  This case has `ptoas.flags` with `--enable-vmi`, because the partial pointer
  load must run through layout assignment before VPTO/LLVM emission.
```

Current checked-in coverage for 3.44 masked_load grouped tail feeding S=32
reduce:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_masked_load_group_tail_s32.pto

runtime SIM:
  test/vpto/cases/vmi/masked-load-group-tail-s32-reduce-store
```

Current checked-in coverage for 3.45 dynamic S=32 `create_group_mask`:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_create_group_mask_s32_dynamic.pto

runtime SIM:
  test/vpto/cases/vmi/dynamic-create-group-mask-s32-reduce-store

runtime scalar source:
  active_cols is passed as a kernel i32 scalar argument and cast to index inside
  vecscope before pto.vmi.create_group_mask. This is an explicit scalar ABI,
  not a value recovered by vmi-to-vpto from producer/consumer context.
```

Current checked-in runtime coverage for 3.12 control-flow join before S=32
`group_reduce`:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_cf_branch.pto
  test/lit/vmi/vmi_to_vpto_cf_branch.pto

runtime SIM:
  test/vpto/cases/vmi/group-reduce-s32-cf-join-store
```

Current checked-in runtime coverage for 3.20 `group_slots` control-flow join:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_group_slots_cf_join.pto

runtime SIM:
  test/vpto/cases/vmi/group-slots-cf-join-store
```

Current checked-in runtime coverage for 3.22 `scf.for` loop-carried VMI layout:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_scf_for.pto
  test/lit/vmi/vmi_to_vpto_scf_for.pto

runtime SIM:
  test/vpto/cases/vmi/scf-for-loop-carried-store
```

Current checked-in runtime coverage for 3.42 `group_slots` `scf.for`
loop-carried accumulator:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_group_slots_scf_for.pto

runtime SIM:
  test/vpto/cases/vmi/group-slots-scf-for-store
```

Current checked-in coverage for 3.25.1 private function result boundary:

```text
lit:
  test/lit/vmi/vmi_ptoas_private_call_inline.pto

runtime SIM:
  test/vpto/cases/vmi/private-call-inline-store

implementation note:
  after vmi-to-vpto physicalizes the private helper, ptoas inlines private
  single-block helpers whose signatures contain !pto.vreg or !pto.mask. This
  happens before VPTO vecscope/backend emission, so physical vector values do
  not escape through a function return.
```

Current checked-in coverage for 3.43 internal function argument boundary
materialization:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_call_argument_boundary.pto
  test/lit/vmi/vmi_ptoas_call_boundary_vecscope.pto

runtime SIM:
  test/vpto/cases/vmi/private-call-argument-boundary-store

implementation note:
  private physical helper inlining also covers void helper calls with physical
  VMI arguments, so the backend no longer sees a physical VPTO vector function
  ABI for this internal boundary.
```

Current checked-in coverage for packed group-slot RHS elementwise continuations
for 3.5.3 and 3.6.2:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_group_slot_load_dual_layout.pto

runtime SIM:
  test/vpto/cases/vmi/group-reduce-slot-add-store
```

Current checked-in coverage for S=64 row-local group broadcast continuation
with aligned row_stride:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_group_reduce_s64_broadcast_reduce.pto

runtime SIM:
  test/vpto/cases/vmi/group-reduce-s64-broadcast-reduce-store
```

Current checked-in coverage for S=64 active-row tail with aligned row_stride:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_group_reduce_s64_tail_store.pto

runtime SIM:
  test/vpto/cases/vmi/group-reduce-s64-tail-store
```

The companion negative lit case for contiguous `%c1` slots=1 group_store is:

```text
test/lit/vmi/vmi_layout_assignment_group_store_slots1_unit_stride_invalid.pto
```

Current checked-in coverage for S=64 row-local group-slot RHS elementwise
continuation with aligned source_group_stride and aligned output row_stride:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_group_slot_load_dual_layout.pto

runtime SIM:
  test/vpto/cases/vmi/group-reduce-s64-slot-add-store
```

Current checked-in coverage for 3.34 S=64 `slots = 1` group-slot f32->f16 cast:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_group_reduce_s64_truncf.pto

runtime SIM:
  test/vpto/cases/vmi/group-reduce-s64-truncf-store
```

The companion negative lit cases for dynamic or unaligned `%c2` slots=1
group_slot_load, and non-unit `slots = 8` group_slot_load, are:

```text
test/lit/vmi/vmi_to_vpto_group_slot_load_nonunit_slots8_invalid.pto
test/lit/vmi/vmi_layout_assignment_group_slot_load_slots1_dynamic_stride_invalid.pto
test/lit/vmi/vmi_layout_assignment_group_slot_load_slots1_unaligned_stride_invalid.pto
```

Current checked-in coverage for the strided block-load cases:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_group_load_s16_stride_store.pto
  test/lit/vmi/vmi_layout_assignment_group_load_s16_unaligned_stride_invalid.pto
  test/lit/vmi/vmi_layout_assignment_group_load_s32_stride_store.pto
  test/lit/vmi/vmi_layout_assignment_group_load_s32_stride_broadcast_reduce.pto
  test/lit/vmi/vmi_layout_assignment_group_load_s32_unaligned_stride_invalid.pto

runtime SIM:
  test/vpto/cases/vmi/group-load-s16-stride-store
  test/vpto/cases/vmi/group-load-s32-stride-store
  test/vpto/cases/vmi/group-load-s32-stride-broadcast-reduce
```

Current checked-in coverage for grouped mask S=16 tail:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_create_group_mask_s16.pto
  test/lit/vmi/vmi_create_group_mask_invalid.pto

runtime SIM:
  test/vpto/cases/vmi/group-reduce-s16-group-mask-tail-store
  test/vpto/cases/vmi/group-reduce-s16-stride-group-mask-tail-store
  test/vpto/cases/vmi/group-reduce-s16-group-mask-broadcast-reduce-store
```

Current checked-in coverage for 3.24 mask/select/masked-store semantics:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_mask_select_store.pto

runtime SIM:
  test/vpto/cases/vmi/mask-select-store
```

Current checked-in coverage for 3.29 one semantic mask with f32 and f16
consumers:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_mask_granularity_f32_f16_store.pto

runtime SIM:
  test/vpto/cases/vmi/mask-granularity-f32-f16-store
```

Current checked-in coverage for 3.31 f16->f32 feeding dense store and S=16
reduce:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_widen_f16_store_reduce.pto

runtime SIM:
  test/vpto/cases/vmi/widen-f16-to-f32-store-reduce
```

Current checked-in lit coverage for the first `vmi-layout-fold-consumers`
optimization is:

```text
test/lit/vmi/vmi_layout_fold_consumers_store.pto
test/lit/vmi/vmi_layout_fold_consumers_masked_store.pto
test/lit/vmi/vmi_layout_fold_consumers_deint4.pto
```

Current checked-in lit coverage for the first `vmi-layout-rematerialize`
optimization is:

```text
test/lit/vmi/vmi_layout_rematerialize_data.pto
test/lit/vmi/vmi_layout_rematerialize_mask.pto
```

Current checked-in lit coverage for the first
`vmi-layout-sink-materialization` optimization is:

```text
test/lit/vmi/vmi_layout_sink_materialization_binary.pto
test/lit/vmi/vmi_layout_sink_materialization_mask.pto
```

Current checked-in lit coverage for canonicalized VMI control-flow restoration is:

```text
test/lit/vmi/vmi_legalize_arith_select.pto
test/lit/vmi/vmi_ptoas_cli_control_flow.pto
```

Current checked-in lit coverage for the first semantic local-recipe layout gate
is:

```text
test/lit/vmi/vmi_layout_gate_group_slot_load_recipe_invalid.pto
test/lit/vmi/vmi_layout_gate_group_load_recipe_invalid.pto
test/lit/vmi/vmi_layout_gate_group_store_recipe_invalid.pto
test/lit/vmi/vmi_layout_gate_group_slots_unsupported_slots_invalid.pto
test/lit/vmi/vmi_layout_gate_store_recipe_invalid.pto
test/lit/vmi/vmi_layout_gate_helper_materialization_shape_invalid.pto
test/lit/vmi/vmi_layout_gate_group_reduce_recipe_invalid.pto
test/lit/vmi/vmi_layout_gate_group_reduce_slots1_recipe_invalid.pto
test/lit/vmi/vmi_layout_gate_group_broadcast_recipe_invalid.pto
test/lit/vmi/vmi_layout_gate_truncf_recipe_invalid.pto
test/lit/vmi/vmi_layout_gate_extf_recipe_invalid.pto
test/lit/vmi/vmi_layout_gate_bitcast_recipe_invalid.pto
test/lit/vmi/vmi_layout_gate_bitcast_group_slots_invalid.pto
```

Current checked-in direct `vmi-to-vpto` preflight coverage for bitcast local
recipes is:

```text
test/lit/vmi/vmi_to_vpto_bitcast_footprint_invalid.pto
test/lit/vmi/vmi_to_vpto_bitcast_group_slots_invalid.pto
```

Current checked-in coverage for 3.32 f32 feeding f8 store and S=32 reduce:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_f32_f8_store_reduce.pto

runtime SIM:
  test/vpto/cases/vmi/f32-to-f8-store-reduce
```

Current checked-in coverage for multi-tile group-slot arity:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_group_reduce_s32_multitile_store.pto

runtime SIM:
  test/vpto/cases/vmi/group-reduce-s32-multitile-store
```

Current checked-in coverage for 3.40 scalar broadcast feeding dense and grouped
users:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_broadcast_dense_group_users.pto

runtime SIM:
  test/vpto/cases/vmi/broadcast-dense-group-users
```

Current checked-in coverage for 3.41 non-rematerializable `masked_load` feeding
dense and grouped users:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_masked_load_dense_group_users.pto

runtime SIM:
  test/vpto/cases/vmi/masked-load-dense-group-users
```

Diagnostic-only cases:

```text
3.9 dense store of group slots
3.11.2 S=32 tail without full_tile_readable
3.7.4 S=64 slots=1 group_store with unit output stride
3.13 packed group-slot f32 -> f16 cast
3.14 unsupported group size
3.15.3 compact source row stride 12
3.16.1 group_slot_load slots=8 non-unit stride
3.16.2 group_slot_load slots=1 dynamic or unaligned stride
3.27 S=32 source_group_stride not divisible by 8 f32 elements
3.19.2 block_elems=8 value consumed by truncf without materialization recipe
3.25.2 public/external VMI boundary
3.30 unsafe masked_load tail without stable masked/gather fallback
```

Current checked-in diagnostic coverage for 3.9/3.13/3.14:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_dense_store_group_slots_invalid.pto
  test/lit/vmi/vmi_layout_assignment_packed_group_slots_truncf_invalid.pto
  test/lit/vmi/vmi_layout_assignment_group_reduce_s12_invalid.pto
```

Current checked-in diagnostic coverage for the remaining non-SIM diagnostic
entries:

```text
lit:
  test/lit/vmi/vmi_layout_gate_helper_recipe_invalid.pto
  test/lit/vmi/vmi_layout_gate_helper_materialization_shape_invalid.pto
  test/lit/vmi/vmi_layout_assignment_group_reduce_s32_tail_no_full_tile_invalid.pto
  test/lit/vmi/vmi_layout_assignment_group_load_s16_compact_stride12_invalid.pto
  test/lit/vmi/vmi_to_vpto_group_slot_load_nonunit_slots8_invalid.pto
  test/lit/vmi/vmi_layout_assignment_group_slot_load_slots1_dynamic_stride_invalid.pto
  test/lit/vmi/vmi_layout_assignment_group_slot_load_slots1_unaligned_stride_invalid.pto
  test/lit/vmi/vmi_layout_assignment_group_load_block8_truncf_invalid.pto
  test/lit/vmi/vmi_layout_assignment_group_store_slots1_unit_stride_invalid.pto
  test/lit/vmi/vmi_layout_assignment_group_load_s16_unaligned_stride_invalid.pto
  test/lit/vmi/vmi_layout_assignment_group_load_s32_unaligned_stride_invalid.pto
  test/lit/vmi/vmi_ptoas_public_abi_invalid.pto
  test/lit/vmi/vmi_ptoas_public_result_abi_invalid.pto
  test/lit/vmi/vmi_layout_assignment_external_call_invalid.pto
  test/lit/vmi/vmi_layout_assignment_external_decl_invalid.pto
  test/lit/vmi/vmi_to_vpto_masked_load_nonfull_invalid.pto
  test/lit/vmi/vmi_to_vpto_stable_gather_masked_load_todo_invalid.pto
```

Capability boundaries and runtime evidence notes:

```text
private physical function ABI:
  3.25.1 and 3.43 runtime coverage is closed for private/internal single-block
  helpers by inlining private physical VMI helpers after vmi-to-vpto and before
  VPTO vecscope/backend emission. Public/external VMI boundaries are still
  rejected until a stable VMI ABI is defined.

memory-proof runtime coverage:
  3.21 S=32 full-tile-readable tail is covered by a runtime case that uses
  `pto.vmi.load {full_read_elems = 256}` on a UB pointer source. The attr is
  the explicit safe-read proof consumed by `vmi-to-vpto`; no surrounding MTE,
  caller/body context, or producer/user scan is inspected to justify the
  rounded-up physical reads.
```

## 12. Implementation Slices

### Slice 1: IR Skeleton And Verifiers

```text
layout attrs
vmi.vreg/vmi.mask types
surface op definitions
surface/layout validators
```

### Slice 2: Straight-Line Dense Assignment/Lowering

```text
3.1 f16->f32->store
3.2 f32->f16->store
3.3 f8->f32->compute->f8
```

### Slice 3: Group Slots And Reductions

```text
3.4 S=8
3.5 S=16 parity/block8
3.6 S=32
3.7 S=64
group_slot_load
group_broadcast
group_store
```

### Slice 4: Layout Conflicts And Materialization

```text
3.8 cast commute through group_broadcast
3.18 dense/group-reduce multi-consumer
3.19 block_elems recipe selection
3.23 group_broadcast multi-consumer
3.32 f32 feeding f8 store and S=32 reduce
3.33 S=16/S=32 reduce multi-consumer rematerialization
3.34 slots=1 group-slot f32->f16 cast
3.35 group_slots fanout to group_store and group_broadcast
3.36 group_slot_load rematerialized for slots=8/slots=1
3.38 multi-tile group_slots arity
3.40 scalar broadcast rematerialized for dense/grouped users
3.41 non-rematerializable value with ensure_layout
```

### Slice 5: Masks, Tail, And Memory Legality

```text
create_mask
create_group_mask
masked_store
safe full-read proof
compact/gather diagnostics
mask granularity per use
group_load stride greater than group size
group_slot_load slots=1 aligned non-unit stride plus dynamic/unaligned diagnostic
group_store slots=1 non-unit output stride
strided group_load feeding broadcast and a second reduce
masked_load grouped tail feeding S=32 reduce
```

### Slice 6: Control Flow And Functions

```text
scf.if
scf.for
group_slots across control flow
group_slots loop-carried accumulator
internal function specialization
internal function argument boundary materialization
public ABI diagnostic
```

## 13. Completion Checklist

Current evidence for the case-catalog objective:

```text
1. every catalog endpoint is mapped in section 6.6 to an assignment owner,
   assignment artifact, and vmi-to-vpto contract
2. every SIM-backed positive endpoint is listed in section 11.3 and has a
   checked-in runtime case directory
3. every runtime case directory contains kernel.pto, launch.cpp, main.cpp,
   golden.py, and compare.py
4. the latest broad VMI runtime sweep passed: PASS=43 FAIL=0
5. the latest full VMI lit sweep passed: 340/340
6. every unsupported endpoint listed in section 11.3 has a diagnostic lit test
7. vmi-to-vpto decisions are represented by current-op attrs/operands,
   assigned layouts, helper ops, rematerialization, or diagnostics
8. no separate recipe string attr is emitted or consumed
9. release docs remain untouched; this is still a design/implementation plan
   under docs/designs
```
