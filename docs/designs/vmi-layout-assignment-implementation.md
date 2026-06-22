# VMI Layout Assignment Implementation Plan

本文是 `vmi-layout-assignment` 和 `vmi-to-vpto` 的实现计划。它配套
`vmi-layout-assignment-lowering-design.md`，并以
`vmi-layout-lowering-cases.md` 为测试和验收来源。

不使用旧 `vmi-dialect-design.md` 作为设计输入。

## 1. Pipeline

Recommended pass pipeline:

```text
pto-validate-vmi-surface
  -> vmi-layout-assignment
  -> pto-validate-vmi-layout
  -> vmi-to-vpto
  -> canonicalize/cse
  -> existing VPTO lowering/codegen
```

Pass responsibilities:

```text
pto-validate-vmi-surface:
  verify surface VMI has no physical VPTO layout dependency
  reject public/external VMI ABI unless explicitly enabled

vmi-layout-assignment:
  solve value layouts
  choose selected lowering plans
  insert ensure/rematerialization helpers
  make internal function boundary layouts explicit
  rewrite VMI types with layout attrs

pto-validate-vmi-layout:
  verify every VMI data/mask value has layout
  verify every context-sensitive op has selected_plan
  verify helper ops have registered materialization plans

vmi-to-vpto:
  use OneToN type conversion
  lower only from explicit layout/plan information
  emit VPTO or precise unsupported diagnostic
```

## 2. Files To Add Or Update

Expected implementation files:

```text
include/PTO/IR/VMITypes.td
include/PTO/IR/VMIOps.td
include/PTO/IR/VMIAttrs.td
lib/PTO/IR/VMI.cpp

include/PTO/Transforms/Passes.td
lib/PTO/Transforms/ValidateVMI.cpp
lib/PTO/Transforms/VMILayoutAssignment.cpp
lib/PTO/Transforms/VMIToVPTO.cpp
lib/PTO/Transforms/VMILayoutPlanRegistry.cpp

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
  direct full-chunk plans require N % (F * B) == 0

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

### 3.3 Selected Plan Attribute

Every context-sensitive op gets a selected plan attr after assignment.  The
initial implementation may use a stable string attr:

```text
vmi.selected_plan = "s16_reduce_parity"
```

Once the plan registry syntax is stable, this can become a dedicated plan attr:

```text
vmi.selected_plan = #pto.vmi.plan<s16_reduce_parity>
vmi.selected_plan = #pto.vmi.plan<dense_load_norm>
vmi.selected_plan = #pto.vmi.plan<s16_reduce_block8>
vmi.selected_plan = #pto.vmi.plan<s32_reduce_dintlv4>
vmi.selected_plan = #pto.vmi.plan<s32_reduce_block8_stride>
vmi.selected_plan = #pto.vmi.plan<s64_reduce_row_local>
vmi.selected_plan = #pto.vmi.plan<load_dintlv2>
vmi.selected_plan = #pto.vmi.plan<load_dintlv4>
vmi.selected_plan = #pto.vmi.plan<group_load_contiguous_chunks>
vmi.selected_plan = #pto.vmi.plan<s16_group_load_block8_unit_stride>
vmi.selected_plan = #pto.vmi.plan<s16_group_load_block8_stride>
vmi.selected_plan = #pto.vmi.plan<s32_group_load_block8_stride>
vmi.selected_plan = #pto.vmi.plan<group_slot_load_slots8_unit_stride>
vmi.selected_plan = #pto.vmi.plan<group_slot_load_slots1_row_local>
vmi.selected_plan = #pto.vmi.plan<group_broadcast_slots8_vselr>
vmi.selected_plan = #pto.vmi.plan<group_broadcast_slots1_vselr>
vmi.selected_plan = #pto.vmi.plan<group_slot_cast_slots1_f32_to_f16>
vmi.selected_plan = #pto.vmi.plan<safe_full_read_dintlv4>
```

Ops that are uniquely determined by layout may omit this attr, but the rule
should be conservative.  If future maintainers could reasonably ask "why this
lowering?", assignment should write a plan.

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

## 5. Plan Registry

Create one registry object shared by assignment and lowering.

```c++
class VMILayoutPlanRegistry {
public:
  SmallVector<ProducerPlan> getProducerPlans(Operation *op);
  SmallVector<ConsumerPlan> getConsumerPlans(OpOperand &use);
  SmallVector<TransferPlan> getTransferPlans(Operation *op);
  FailureOr<MaterializationPlan> getMaterializationPlan(Type valueType,
                                                        VMILayoutKey from,
                                                        VMILayoutKey to);
  bool isCheaplyRematerializable(Operation *op);
  bool hasTargetCapability(PlanID plan) const;
};
```

Plan record:

```c++
struct VMILayoutPlan {
  PlanID id;
  SmallVector<VMILayoutKey> operandLayouts;
  SmallVector<VMILayoutKey> resultLayouts;
  int64_t cost;
  bool requiresSelectedPlanAttr;
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
  PlanID requestingPlan;
  bool hard;
};

struct OpPlanState {
  Operation *op;
  SmallVector<VMILayoutPlan> candidates;
  std::optional<PlanID> chosen;
};
```

### 6.2 Collection Phase

Walk the module and collect:

```text
1. every VMI value
2. every VMI block argument
3. every VMI function argument/result
4. every VMI op with candidate plans
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

### 6.4 Solving And Rewriting

Algorithm:

```text
1. Pick candidate plan sets for every op.
2. Propagate hard constraints through SCCs.
3. Resolve transfer-equivalent dense values.
4. Choose multi-plan ops by cost:
   - S=16 parity vs block8
   - load memory-fused vs load+materialize
   - group_slot_load slots=8 vs slots=1
5. For conflicting uses:
   - rematerialize cheap producer where legal
   - otherwise insert ensure_layout at use
   - otherwise diagnose
6. Rewrite VMI result/block/function types with chosen layouts.
7. Attach selected_plan attrs where required.
8. Insert helper ops with source/result layout attrs.
```

Rewrite invariants:

```text
No VMI data/mask value after assignment has a null layout.
No context-sensitive VMI op after assignment lacks selected_plan.
Every ensure_* helper has a registered materialization plan.
Every function/call signature carrying VMI is specialized or diagnosed.
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
operand/result layouts
selected_plan
adaptor physical values
```

Each pattern rejects:

```text
missing selected_plan for context-sensitive op
layout not matching selected_plan
missing target capability
unexpected group_slots dense consumer
```

Target selected-plan matrix:

```text
load, selected_plan=dense_load_norm:
  result layout contiguous
  emits pto.vlds / pto.vsts NORM paths
  covers dense store users and S=64 row-local reduce input

load, selected_plan=load_dintlv2:
  result layout deinterleaved=2, block_elems=1
  emits vldsx2 DINTLV_B32 or normal load + vdintlv materialization
  covers f32->f16, S=16 parity reduce, f16->f32 widened values

load, selected_plan=load_dintlv4:
  result layout deinterleaved=4, block_elems=1
  emits two vldsx2 DINTLV_B32 plus vdintlv
  covers f32->f8, S=32 dintlv4 reduce

group_load, selected_plan=s16_group_load_block8_unit_stride:
  result layout deinterleaved=2, block_elems=8
  emits vldsx2/BDINTLV for 8 rows of 16xf32
  covers compact logical S=16 when source_group_stride == 16

group_load, selected_plan=s16_group_load_block8_stride:
  result layout deinterleaved=2, block_elems=8
  emits two vsldb strided 32B block loads
  requires source_group_stride % 8 == 0

group_load, selected_plan=s32_group_load_block8_stride:
  result layout deinterleaved=4, block_elems=8
  emits four vsldb strided 32B block loads
  requires source_group_stride % 8 == 0

group_load, selected_plan=group_load_contiguous_chunks:
  result layout contiguous
  emits one vlds per physical group chunk using row_stride address arithmetic
  covers the currently implemented full-chunk row-local group_load path

group_reduce_addf, selected_plan=s8_reduce_contiguous:
  consumes contiguous f32 with group size 8
  produces group_slots(G, slots=8)
  emits one vcgadd

group_reduce_addf, selected_plan=s16_reduce_parity:
  consumes deinterleaved=2, block_elems=1
  produces group_slots(G, slots=8)
  emits two vcgadd operations and one vadd

group_reduce_addf, selected_plan=s16_reduce_block8:
  consumes deinterleaved=2, block_elems=8
  produces group_slots(G, slots=8)
  emits two vcgadd operations and one vadd

group_reduce_addf, selected_plan=s32_reduce_dintlv4:
  consumes deinterleaved=4, block_elems=1
  produces group_slots(G, slots=8)
  emits four vcgadd operations and a vadd tree

group_reduce_addf, selected_plan=s32_reduce_block8_stride:
  consumes deinterleaved=4, block_elems=8
  produces group_slots(G, slots=8)
  emits four vcgadd operations and a vadd tree

group_reduce_addf, selected_plan=s64_reduce_row_local:
  consumes contiguous f32 with group size 64
  produces group_slots(G, slots=1)
  target lowering emits per-row vcgadd plus vcadd; the current prototype uses
  the existing row-local VCADD/VADD/VSEL sequence while preserving the same
  group_slots(G, slots=1) value contract

group_slot_load, selected_plan=group_slot_load_slots8_unit_stride:
  result group_slots(G, slots=8)
  requires source_group_stride == 1
  emits one packed vsldb load

group_slot_load, selected_plan=group_slot_load_slots1_row_local:
  result group_slots(G, slots=1)
  supports aligned non-unit source_group_stride
  requires constant positive source_group_stride divisible by 256 / elementBits
  emits one lane-0 vsldb per group

group_broadcast, selected_plan=group_broadcast_slots8_vselr:
  source group_slots(G, slots=8)
  result dense layout selected per use
  emits vselr using assigned result layout

group_broadcast, selected_plan=group_broadcast_slots1_vselr:
  source group_slots(G, slots=1)
  result dense layout selected per use
  emits vdup/vselr row-local materialization

truncf, selected_plan=group_slot_cast_slots1_f32_to_f16:
  source/result group_slots(G, slots=1)
  emits one lane-0 vcvt per group slot block
  rejects packed slots=8 unless another plan is registered
```

The target matrix is the implementation contract.  The staged status below
records how much of that contract the current prototype has already enforced.

Current staged implementation status:

```text
group_slot_load:
  vmi-to-vpto requires vmi.selected_plan and checks it against
  #pto.vmi.layout<num_groups = G, slots = 8/1>.

group_reduce_addf:
  explicit slots=8 VCGADD lowering requires
  vmi.selected_plan = "s8_reduce_contiguous". Legacy bare num_groups and
  generic VCADD lowering still need the plan-registry migration.
  S=16 block8 assignment emits source/mask
  #pto.vmi.layout<deinterleaved = 2, block_elems = 8>, result
  #pto.vmi.layout<num_groups = G, slots = 8>, and
  vmi.selected_plan = "s16_reduce_block8"; vmi-to-vpto checks that plan and
  lowers through two VCGADDs plus a PAT_VL8 VADD per packed result block.
  S=32 block8 assignment emits source/mask
  #pto.vmi.layout<deinterleaved = 4, block_elems = 8>, result
  #pto.vmi.layout<num_groups = G, slots = 8>, and
  vmi.selected_plan = "s32_reduce_block8_stride"; vmi-to-vpto checks that
  plan and lowers through four VCGADDs plus a PAT_VL8 VADD tree per packed
  result block.
  S=64 row-local assignment now emits
  vmi.selected_plan = "s64_reduce_row_local" and has focused
  layout-assignment/vmi-to-vpto lit coverage; the explicit slots=1 generic
  VCADD row-local path also requires and checks that selected_plan. Other
  legacy bare num_groups generic VCADD paths still need the plan-registry
  migration.

group_broadcast:
  explicit slots=8/1 source layouts require
  vmi.selected_plan = "group_broadcast_slots8_vselr" or
  "group_broadcast_slots1_vselr". Deinterleaved block-fragment results use
  the result layout block_elems as the local vselr selection group, so
  `deinterleaved = 4, block_elems = 8` broadcasts one group slot across each
  32B row fragment. VSELR index vectors are materialized per physical result
  chunk.  For small-group results, layout assignment has already fixed the
  result layout, and vmi-to-vpto computes:
  `firstGroup = first logical group covered by this result chunk`,
  `sourceChunk = firstGroup / slots`, and
  `baseGroupSlot = firstGroup % slots`.  The generated index vector selects
  `baseGroupSlot .. baseGroupSlot + groupsPerResultChunk - 1`; it must not be
  reused across result chunks. Legacy bare num_groups still needs the
  plan-registry migration.

group_load:
  contiguous full-chunk path emits and checks
  vmi.selected_plan = "group_load_contiguous_chunks". S=16/S=32
  block-aligned strided loads emit and check
  vmi.selected_plan = "s16_group_load_block8_stride" or
  "s32_group_load_block8_stride", assign
  #pto.vmi.layout<deinterleaved = 2/4, block_elems = 8>, and lower to one
  vsldb per 32B row fragment and physical chunk. The dedicated S=16 unit-stride
  vldsx2/BDINTLV plan remains a design target. S=16/S=32 group_load with a
  non-constant, non-positive, or non-8-f32-aligned row_stride is rejected by
  vmi-layout-assignment because the stable gather fallback is not implemented.

truncf group-slot cast:
  layout assignment and vmi-to-vpto support and check
  vmi.selected_plan = "group_slot_cast_slots1_f32_to_f16" for
  group_slots(G, slots=1) f32 -> f16. The reduce->truncf->group_store
  slots=1 flow has focused lit coverage and no longer relies on vmi-to-vpto
  inspecting the truncf producer.

group_store:
  row-local group_slots(G, slots=1) lowering is implemented as one lane-0
  vsts per group and is covered by the reduce->truncf->group_store lit case.
  The current plan is accepted only when row_stride is a constant positive
  multiple of the 32B store alignment in destination elements: 8 for f32,
  16 for f16, and 32 for f8. Unit-stride f32 output is rejected because only
  the first row-local store is 32B-aligned; later `group_off + r` stores are
  4B apart. A future pack-to-slots=8 or unaligned-store plan is required before
  contiguous `%c1` slots=1 group_store can be accepted.
  Packed group_slots(G, slots=8) group_store is implemented only when
  num_groups is a multiple of 8 and row_stride is constant 1; it emits one
  PAT_VL8 store per packed slot block. Non-unit packed group stores remain a
  design target unless a strided packed-lane store plan is selected explicitly.
```

Examples:

```text
group_reduce_addf, selected_plan=s16_reduce_parity:
  consume deinterleaved=2, block_elems=1
  emit two VCGADDs and one VADD

group_reduce_addf, selected_plan=s16_reduce_block8:
  consume deinterleaved=2, block_elems=8
  emit two VCGADDs and one VADD

group_reduce_addf, selected_plan=s32_reduce_dintlv4:
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
Every context-sensitive op has selected_plan.
Every selected_plan matches operand/result layouts.
Every ensure_* helper has a materialization plan.
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
  walking from a consumer to a producer to decide a selected_plan
  walking from a consumer to a mask producer to decide whether a plan is legal
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
  strided group_load plan. Register a rematerialization or preserving
  materialization plan, or avoid consuming this block-loaded value with truncf.
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
selected_plan attrs
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
WORK_SPACE=$PWD/.tmp/vmi-runtime-batch-40 CASE_PREFIX='vmi/' JOBS=4 \
  test/vpto/scripts/run_host_vpto_validation_parallel.sh

PASS=40 FAIL=0
summary: .tmp/vmi-runtime-batch-40/parallel-summary.tsv
log scan: rg -n "RV_|alignment|\[ERROR\]|\[error\]|ERROR" \
  .tmp/vmi-runtime-batch-40.log
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
  3.25.1 private/internal boundary lit coverage, runtime backend gap
  3.25.2 public/external boundary diagnostics
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
  test/lit/vmi/vmi_layout_assignment_create_group_mask_s32_dynamic.pto

runtime SIM:
  test/vpto/cases/vmi/masked-load-group-tail-s32-reduce-store
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

Current checked-in lit coverage for 3.43 internal function argument boundary
materialization:

```text
lit:
  test/lit/vmi/vmi_layout_assignment_call_argument_boundary.pto

runtime SIM:
  blocked by the current private vector callee backend path; see known
  implementation gaps below
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
3.19.2 block_elems=8 value consumed by truncf without materialization plan
3.25.1 full ptoas emission for private VMI callees that return VPTO vector values
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
  test/lit/vmi/vmi_layout_assignment_group_reduce_s32_tail_no_full_tile_invalid.pto
  test/lit/vmi/vmi_layout_assignment_group_load_s16_compact_stride12_invalid.pto
  test/lit/vmi/vmi_to_vpto_group_slot_load_nonunit_slots8_invalid.pto
  test/lit/vmi/vmi_layout_assignment_group_load_block8_truncf_invalid.pto
  test/lit/vmi/vmi_layout_assignment_group_store_slots1_unit_stride_invalid.pto
  test/lit/vmi/vmi_ptoas_public_abi_invalid.pto
  test/lit/vmi/vmi_ptoas_public_result_abi_invalid.pto
  test/lit/vmi/vmi_layout_assignment_external_call_invalid.pto
  test/lit/vmi/vmi_layout_assignment_external_decl_invalid.pto
  test/lit/vmi/vmi_ptoas_call_boundary_vecscope_invalid.pto
  test/lit/vmi/vmi_to_vpto_masked_load_nonfull_invalid.pto
  test/lit/vmi/vmi_to_vpto_stable_gather_masked_load_todo_invalid.pto
```

Known implementation gaps before all catalog cases can become runtime SIM
coverage:

```text
dynamic grouped mask runtime source:
  vmi-to-vpto supports dynamic active_elems_per_group for contiguous b32
  grouped masks and S=32 deinterleaved=4/block_elems=8 masks. Full runtime SIM
  coverage still needs a supported scalar source for active_elems_per_group in
  vector kernels. Direct GM pto.ldg crashed the Bisheng vector backend in this
  test shape, and UB pto.load_scalar reached an invalid scalar LSU address in
  the SIM. Do not replace grouped masks with prefix create_mask; that would
  change the semantics.

remaining function runtime coverage:
  3.25.1 internal function boundary specialization has layout-assignment and
  vmi-to-vpto lit coverage, but full ptoas emission still fails after
  physicalization because today's inferred pto.vecscope is resultless and VPTO
  vector-scope values cannot escape through a function return. Runtime coverage
  requires either a resultful vecscope/VPTO vector ABI or an explicit inlining
  policy before vecscope inference.

  3.43 internal function argument boundary materialization has
  layout-assignment and vmi-to-vpto lit coverage. Full ptoas emission for a
  private void vector callee currently reaches the Bisheng device backend and
  fails on the physicalized callee with:

    fatal error: error in backend: Do not know how to split the result of this operator!

  Runtime coverage requires either inlining private vector callees before the
  device backend path or adding backend support for the physical VPTO vector
  function ABI. This is a runtime/backend gap, not a license for `vmi-to-vpto`
  to infer layouts from caller/callee context.

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
selected_plan attr
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
3.19 block_elems plan selection
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

The implementation is not complete until:

```text
1. every case has a layout-assignment test
2. every positive case has a vmi-to-vpto test
3. every simulator-supported case has a sim validation
4. every unsupported case has a diagnostic test
5. vmi-to-vpto contains no producer/user context inference
6. missing selected_plan on context-sensitive ops is a hard failure
7. release docs are updated only after the design stabilizes
```
