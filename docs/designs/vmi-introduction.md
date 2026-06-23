# VMI 介绍

本文介绍 VMI 的设计入口：VMI 解决什么问题，layout 有哪些，pass pipeline
如何分工，以及这些机制分别应对哪些典型场景。更完整的逐 case lowering 结果见
`docs/designs/vmi-layout-lowering-cases.md`。

示例是设计级 IR，保留关键 type、layout、helper op 和 VPTO op 形状，
省略 module wrapper、完整 operand list 和不影响讨论的 SSA 细节。

## 1. VMI 表达什么

VMI 是 VPTO 之前的逻辑向量层。它让前端先表达“我要对 `NxT` 的逻辑向量做什么”，
再由 layout assignment 决定这个逻辑向量如何拆到 256B 物理 vector register 上。

Surface VMI 类型不携带布局：

```mlir
!pto.vmi.vreg<128xf32>
!pto.vmi.mask<128xpred>
```

Layout-assigned VMI 类型携带具体布局和 mask granularity：

```mlir
!pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>
!pto.vmi.mask<128xb32, #pto.vmi.layout<deinterleaved = 2>>
```

VMI 的核心约束是：`vmi-to-vpto` 只从当前 op 的 attrs、operands、types、
layouts 和显式 helper ops 做 lowering，不读取隐藏 plan/recipe，也不通过
defining op 或 sibling user 恢复上下文。

## 2. Layout 类型

### 2.1 `contiguous`

```mlir
#pto.vmi.layout<contiguous>
```

含义：logical lane 按顺序落入物理 register list。

```text
logical lanes:  0  1  2 ... 63 | 64 65 ... 127
physical part:  p0             | p1
```

典型场景：

```text
dense load/store
普通 elementwise compute
一个 group 天然适配当前 reduce op 时的 reduction input
caller/callee 约定 dense order 时的 control-flow/function boundary
```

### 2.2 `deinterleaved = F, block_elems = B`

```mlir
#pto.vmi.layout<deinterleaved = 2>
#pto.vmi.layout<deinterleaved = 4, block_elems = 8>
```

`block_elems` 缺省为 `1`。逻辑 lane 到物理 part 的映射是：

```text
logical lane i
block q         = i / B
in-block lane r = i % B
part p          = q % F
part block t    = q / F

physical part p, physical lane t * B + r
```

`deinterleaved=2` 的直观例子：

```text
logical lanes:   0 1 2 3 4 5 ...
physical part0:  0   2   4   ...
physical part1:    1   3   5 ...
```

`deinterleaved=4, block_elems=8` 的直观例子：

```text
logical group S=32:
  lanes  0.. 7 -> part0 lanes 0..7
  lanes  8..15 -> part1 lanes 0..7
  lanes 16..23 -> part2 lanes 0..7
  lanes 24..31 -> part3 lanes 0..7
```

典型场景：

```text
f16 -> f32:
  vcvt 天然产生 even/odd 两个 f32 part，所以结果使用 deinterleaved=2。

f32 -> f16:
  vcvt 需要 f32 source 先拆成 even/odd 两个 part，所以 source 使用
  deinterleaved=2。

S=32 group_reduce f32:
  一个 group 有 32 个 f32 element。高效 reduce path 消费四个 8-lane block，
  所以 source/mask 使用 deinterleaved=4, block_elems=8。
```

`block_elems=8` 表示一种按 32B row fragment 组织的输入形态，不表示
S=32 reduce 只能接受这一种形态。如果同一个 value 还要服务 narrow cast 等
element-parity consumer，assignment 可以选择 `deinterleaved=4, block_elems=1`
作为共同 layout，再由 lowering 生成对应的物理指令序列。

`deinterleaved` 只描述最终物理 part 中有哪些 logical lane，不描述这个 layout
由哪条指令生成。不同 producer 可以用不同方式直接产生同一个 layout；如果不能
直接产生，后续 lowering 再通过显式 materialization helper 把 source layout
转换成 consumer 需要的 layout。具体 lowering 形状见 case catalog。

### 2.3 `num_groups = G, slots = K`

```mlir
#pto.vmi.layout<num_groups = 8, slots = 8>
#pto.vmi.layout<num_groups = 8, slots = 1>
```

这是 sparse group-result layout。它不表示全部 `N` 个 logical lane 都有语义值。
只有 `G` 个 group 结果 slot 有语义值。

```text
slot_block(g) = g / K
slot_lane(g)  = g % K

physical part slot_block(g) 的 lane slot_lane(g) 保存 group g 的结果
```

`num_groups=16, slots=8` 的例子：

```text
part0 lane0..7 = group result 0..7
part1 lane0..7 = group result 8..15
other lanes    = 对普通 dense consumer 来说未定义
```

为什么 group 信息也要放进 layout：

```text
group_reduce 自身有 num_groups，但它的结果可能继续跨过 truncf、
group_broadcast、group_store、scf.if、scf.for、function call 或多个 consumer。

这些后续 op 不应该回看 producer attr。value layout 因此需要记录有多少个
group result，以及这些 result 如何 packed 到 physical slot。
```

典型场景：

```text
group_reduce result
group_slot_load result
group_store input
group_broadcast input
group-slot control-flow/function boundary
部分 row-local cast 路径，通常使用 slots=1
```

## 3. Pass Pipeline

```text
pto-validate-vmi-ir
  -> vmi-layout-assignment
  -> canonicalize/cse
  -> vmi-layout-fold-consumers
  -> canonicalize/cse
  -> vmi-layout-rematerialize
  -> canonicalize/cse
  -> vmi-layout-sink-materialization
  -> canonicalize/cse
  -> vmi-legalize-arith-select
  -> pto-validate-vmi-layout-ir
  -> vmi-to-vpto
```

### 3.1 `pto-validate-vmi-ir`

检查 surface VMI 边界。

合法输入：

```mlir
%x = pto.vmi.load %src[%off]
  : !pto.ptr<f16, ub> -> !pto.vmi.vreg<128xf16>
```

非法输入：

```mlir
%x = pto.vmi.load %src[%off]
  : !pto.ptr<f16, ub>
    -> !pto.vmi.vreg<128xf16, #pto.vmi.layout<contiguous>>
```

原因：具体 layout 由 `vmi-layout-assignment` 产生，不应该由 surface frontend
提前写入。

### 3.2 `vmi-layout-assignment`

这是硬合法化 pass。它选择具体 value layout、具体 mask granularity，
并在 layout 不匹配的 use site 插入显式 helper op。

实现上它维护 data 和 mask 两套求解状态：

```text
data value:
  每个 !pto.vmi.vreg 是一个节点，节点记录最终选择的布局。

mask value:
  每个 !pto.vmi.mask 是一个节点，节点记录最终选择的布局和 predicate 粒度。
```

data value 使用 union-find 表示“这些 value 必须共用 layout”。函数参数、
call operand/result、return/yield、block argument、bitcast 等边界会把相关
value 合并到同一个等价类里。等价类只能有一个最终 data layout。

assignment 遍历 IR 时，每类 op 向求解器贡献两种信息：

```text
result 自然布局:
  这个 op 自己产生的 result 适合用什么 layout 表达。

operand 使用请求:
  这个 op 消费某个 operand 时希望 operand 是什么 layout。
```

有些 producer 生成的是同一个逻辑向量，但可以用多种物理 layout 表达。若它的
所有 consumer 给出的使用请求一致，assignment 会把这个请求反推为 producer
result 的最终布局。否则，producer 保持自己的布局，assignment 在不匹配的 use
site 插入 `pto.vmi.ensure_layout`。mask 使用同样思路，但还会同时求解 predicate
粒度，必要时插入 `ensure_mask_layout` 或 `ensure_mask_granularity`。

最后，pass 会把所有 VMI data/mask type 改写成带 layout 的 type，并同步更新
function type、call site、block argument 和 terminator operand。这个阶段之后，
IR 不再依赖隐藏 plan；后续 pass 和 `vmi-to-vpto` 都只读取 type 上的 layout
和显式 `ensure_*` helper。

### 3.3 `vmi-layout-fold-consumers`

当 consumer 可以直接保持同样的外部效果时，把显式 materialization 折进
consumer。

变换前：

```mlir
%dense = pto.vmi.ensure_layout %x
  : !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>
    -> !pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>>
pto.vmi.store %dense, %dst[%off]
```

变换后：

```mlir
pto.vmi.store %x, %dst[%off]
  : !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>, !pto.ptr<f32, ub>
```

可能的 VPTO 形状：

```text
fold 前：vintlv + vsts + vsts
fold 后：vstsx2，使用交错 store mode
```

### 3.4 `vmi-layout-rematerialize`

通过 clone 低成本、layout-polymorphic 的 producer 来替换 `ensure_*`。

变换前：

```mlir
%s = pto.vmi.broadcast %scale
  : f32 -> !pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>>
%s_split = pto.vmi.ensure_layout %s
  : !pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>>
    -> !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>
```

变换后：

```mlir
%s_split = pto.vmi.broadcast %scale
  : f32 -> !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>
```

预期可 rematerialize 的 producer：

```text
splat constant
broadcast
iota
create_mask
create_group_mask
constant_mask
```

这个 pass 不 rematerialize：

```text
load / masked_load / group_load / group_slot_load
reduce / group_reduce
control-flow results
```

### 3.5 `vmi-layout-sink-materialization`

把匹配的 layout 转换跨过 layout-transparent elementwise op。

变换前：

```mlir
%a_dense = pto.vmi.ensure_layout %a : deinterleaved=2 -> contiguous
%b_dense = pto.vmi.ensure_layout %b : deinterleaved=2 -> contiguous
%y_dense = pto.vmi.addf %a_dense, %b_dense : contiguous
```

变换后：

```mlir
%y_split = pto.vmi.addf %a, %b : deinterleaved=2
%y_dense = pto.vmi.ensure_layout %y_split : deinterleaved=2 -> contiguous
```

效果：

```text
两个 input materialization -> 一个 result materialization
```

这个 pass 不会 sink 穿过 cast、load、store、reduce、group_broadcast 或
control-flow op。

### 3.6 `vmi-legalize-arith-select`

Canonicalization 可能把简单的 `scf.if` 折成 `arith.select`。VMI 希望把
control-flow lowering 保持在结构化控制流里，所以这个 pass 会把 VMI value 上的
`arith.select` 改回 `scf.if`。

```mlir
%r = arith.select %cond, %a, %b
  : !pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>>
```

改成：

```mlir
%r = scf.if %cond
    -> !pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>> {
  scf.yield %a : !pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>>
} else {
  scf.yield %b : !pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>>
}
```

### 3.7 `pto-validate-vmi-layout-ir`

检查 post-assignment gate：

```text
每个 VMI 数据值都有 concrete layout
每个 VMI mask 都有 concrete granularity 和 layout
helper op 有支持的 materialization path
semantic op/layout 组合有支持的 local lowering
vmi-to-vpto 之前没有物理 VPTO value 泄漏到 VMI IR 中
```

非法例子：

```mlir
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8, reassoc}
  : ... -> !pto.vmi.vreg<64xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>

pto.vmi.store %sum, %dst[%off]
  : !pto.vmi.vreg<64xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>,
    !pto.ptr<f32, ub>
```

原因：

```text
dense store 不能把 sparse group_slots 当 dense vector 读取。
应使用 group_store、group_broadcast 或显式支持的 group-to-dense op。
```

### 3.8 `vmi-to-vpto`

把 layout-assigned VMI value 转换成有序物理 VPTO value 列表，并对每个
VMI op 做 local lowering。

例子：

```text
!pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>>
  -> 两个 physical !pto.vreg<64xf32> part

!pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>
  -> 两个 physical !pto.vreg<64xf32> part
     part0 携带 even lanes，part1 携带 odd lanes

!pto.vmi.vreg<256xf32, #pto.vmi.layout<num_groups = 32, slots = 8>>
  -> 四个 physical part
     part0 携带 group 0..7，part1 携带 group 8..15，...
```

`VMILayoutSupport` 不是 pass。它是 assignment、validation、optimization 和
lowering 共享的查询库，用来避免重复实现 layout fact 和 supported
materialization 检查。

## 4. 典型场景

### 4.1 Dense Cast 与 Store

```text
surface:
  load f16，语义上连续
  extf 到 f32
  dense store f32

assignment:
  load result      = contiguous
  extf result      = deinterleaved=2
  store use        = ensure_layout(deinterleaved=2 -> contiguous)

baseline VPTO:
  vlds
  vcvt even / vcvt odd
  vintlv
  vsts + vsts

fold-consumers 后的优化 VPTO:
  vlds
  vcvt even / vcvt odd
  vstsx2，使用 interleaving store
```

这个场景说明为什么需要 `deinterleaved=2`，以及为什么 store-consumer folding
有价值。

### 4.2 Narrow Cast 与 Store

```text
surface:
  load f32
  truncf 到 f16
  dense store f16

assignment:
  load result = deinterleaved=2
  truncf result = contiguous

VPTO:
  vldsx2 deinterleaving load
  vcvt even / vcvt odd
  vor
  vsts
```

这个场景说明 memory op 可以直接产生 consumer 需要的 layout，但不需要保存隐藏
plan。

### 4.3 一个 Producer 同时服务 Dense 和 Group Consumer

```mlir
%x32 = pto.vmi.extf %x16
%sum = pto.vmi.group_reduce_addf %x32, %mask {num_groups = 8, reassoc}
pto.vmi.group_store %sum, %sum_out[%off], %c1 {num_groups = 8}
pto.vmi.store %x32, %dense_out[%off]
```

Assignment 形状：

```text
%x32 layout = deinterleaved=2
group_reduce 直接消费 %x32
dense store 获得 ensure_layout(%x32 -> contiguous)
```

VPTO 形状：

```text
vcvt even/odd
vcgadd + vcgadd + vadd -> group_store result
vintlv + dense stores  -> 产生 dense store 结果
```

这个场景说明为什么需要 use-site materialization。producer 不需要选择一个能同时
满足所有 consumer 的唯一 layout。

### 4.4 按 Group Size 区分的 Group Reduce

对于 `N` 个 f32 lane 和 `G = num_groups`，group size 是 `S = N / G`。

```text
S=8:
  input layout 可以是 contiguous。
  group_reduce result 通常使用 layout<num_groups=G, slots=8>。

S=16:
  如果 input 来自 f16->f32 vcvt，layout 可以是 deinterleaved=2。
  如果 input 从 dense 拆出，layout 可以是 deinterleaved=2, block_elems=8。
  result 通常使用 layout<num_groups=G, slots=8>。

S=32:
  input layout 使用 deinterleaved=4, block_elems=8。
  VPTO 形状是四个部分 group reduction 后接 add tree。
  result 通常使用 layout<num_groups=G, slots=8>。

S=64:
  row-local path 在可行时让每个 group 使用一条 physical row。
  result 可以使用 layout<num_groups=G, slots=1>，避免 unsupported packing。
```

S=32 例子：

```text
assignment:
  source/mask = deinterleaved=4, block_elems=8
  result      = group_slots(num_groups=8, slots=8)

VPTO:
  vdintlv / pdintlv_b32
  vcgadd x4
  使用 PAT_VL8 做 vadd tree
  通过一次 PAT_VL8 store 完成 group_store
```

这个场景说明为什么需要 `block_elems`。

### 4.5 Group Result 继续作为 Dense Rows 使用

Surface 意图：

```mlir
%sum32 = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8, reassoc}
%sum16 = pto.vmi.truncf %sum32
%rows16 = pto.vmi.group_broadcast %sum16 {num_groups = 8}
pto.vmi.store %rows16, %dst[%off]
```

支持的 assignment 形状：

```mlir
%sum32 = pto.vmi.group_reduce_addf ...
  -> !pto.vmi.vreg<128xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>

%rows32 = pto.vmi.group_broadcast %sum32 {num_groups = 8}
  -> !pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>>

%rows32_split = pto.vmi.ensure_layout %rows32
  : contiguous -> deinterleaved=2

%rows16 = pto.vmi.truncf %rows32_split
  : deinterleaved=2 -> contiguous

pto.vmi.store %rows16, %dst[%off]
```

VPTO 形状：

```text
group_reduce:
  vcgadd partials + vadd tree

group_broadcast:
  vselr 风格 selection，把 group slots 展开到 dense row lanes

truncf:
  vcvt even/odd + merge

store:
  vsts
```

这个场景说明为什么 group 结果 layout 必须挂在 value 上：reduce 之后，
cast 和 broadcast 必须知道 group 结果在哪里，而不能回看 producer。

### 4.6 通过 Mask 表达 Tail

VMI 通过 mask 表达 tail，不通过 padding 表达 tail。

```mlir
%mask = pto.vmi.create_mask %active_lanes
%x = pto.vmi.masked_load %src[%off], %mask
%y = pto.vmi.mulf %x, %scale
pto.vmi.masked_store %y, %dst[%off], %mask
```

Grouped tail：

```mlir
%gmask = pto.vmi.create_group_mask %active_elems_per_group
    {num_groups = 8, group_size = 32}
%sum = pto.vmi.group_reduce_addf %x, %gmask {num_groups = 8, reassoc}
```

同一个 semantic mask 面对 f8/f16/f32 user 时，可能需要不同 concrete
granularity。Assignment 会通过 mask helper op 显式表达这些转换。

### 4.7 控制流和函数边界

Concrete layout 必须显式跨过 CFG 和内部 function boundary。

```mlir
%r = scf.if %cond
    -> !pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>> {
  %a_dense = pto.vmi.ensure_layout %a : deinterleaved=2 -> contiguous
  scf.yield %a_dense
} else {
  %b_dense = pto.vmi.ensure_layout %b : deinterleaved=2 -> contiguous
  scf.yield %b_dense
}
```

`vmi-to-vpto` 之后，region result 会变成多个物理 VPTO value：

```text
scf.if -> (!pto.vreg<64xf32>, !pto.vreg<64xf32>)
```

这个场景说明为什么 layout 应该是 type 的一部分，而不是依赖 defining op。

## 5. 当前边界

当前设计方向：

```text
surface VMI:
  描述不带 layout 的逻辑向量语义。

layout assignment:
  选择 layout、mask granularity 和显式 materialization helper。

optimization:
  只在结果 IR 仍然可以 local lowering 时改写显式 helper。

vmi-to-vpto:
  严格 lower 它看到的 assigned/optimized IR。
```

暂不支持或有意收紧的范围：

```text
group_slots value 的普通 dense store:
  非法，除非先经过 group_broadcast 或其他显式 group-to-dense op。

packed group_slots f32->f16 cast:
  非法，除非 assignment 能把它 commute 到 group_broadcast 之后，或者使用
  支持的 row-local slots=1 path。

extract:
  暂不作为支持的 VMI surface。

padding transfer_read:
  当前 tail 设计不需要；tail 使用 mask。

scan / contract / gather / scatter / compress / active_prefix_index:
  dialect surface 中可以存在，但除非补充具体 case，否则不属于第一阶段聚焦的
  layout/lowering 实现集合。
```

设计目标是优先保证语义完整：只要 VMI 接受某个 case，所需的 layout 沟通就必须
在 IR 中显式表达，并且能被 `vmi-to-vpto` local lowering。
