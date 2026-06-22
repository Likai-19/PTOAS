# VMI dialect 设计

## 背景

VPTO 的 `!pto.vreg<NxT>` 是 256 bytes 物理向量寄存器抽象。很多 VPTO op 暴露的是
physical placement：`vcvt` part、pack/unpack、interleave/deinterleave、load/store dist、
predicate granularity 等。TileLang `T.parallel` 或其它前端想表达的是逻辑向量语义，不应该
手写这些 physical placement。

VMI dialect 的目标是提供一层 PTO-friendly 的 semantic vector IR。它不是任何外部向量 dialect
的语法克隆，也不是 VPTO physical dialect。VMI 的设计来源是 PTO virtual vector ISA 需要承接的
逻辑向量语义、layout、mask granularity、memory safety 和控制流 layout join；后续 lowering 只从
VMI 决定 physical layout 和 VPTO op。

本设计采用 `vmi.vreg` 作为 layout carrier，不再引入单独的 `vbundle` type：

```text
semantic VMI
  -> layout-assigned VMI
  -> physical VPTO
```

VMI 的 producer 在核心设计之外。TileLang/PTO lowering、手写 VMI 测试或其它 import 工具都可以
产生 VMI，但它们不能定义 VMI 的 semantic surface。核心设计只要求 producer 在进入 VMI boundary
时生成合法 VMI IR。

## 和旧 VMI layout 设计的关系

旧文档中的核心形式是：

```mlir
!pto.vmi.vreg<NxT, layout>
!pto.vmi.mask<NxG, layout>
```

这个方向是对的：`vmi.vreg` 本身是 virtual aggregate type，可以承载完整 logical vector，
layout 放在它上面比放在 physical `!pto.vreg` 上更合理。

旧设计需要补强的地方主要是 layout descriptor 和 lowering contract，而不是推翻
`vmi.vreg`：

1. 旧 layout descriptor 把 `logical_shape`、`phys_dtype`、`phys_lanes` 放进 attr，和
   `vreg<NxT>` / target registry 存在重复信息。重复字段会产生 verifier 漂移。
2. `axes=[#axis<...>]` 太开放，缺少每个 layout 的精确定义、part ordering 和 lane map。
3. 旧设计要求 `N * bitwidth(T)` 是 256B 整数倍，无法覆盖 tail / 非整 tile。
4. mask 只写成 `mask<LxG, layout>`，但没有定义 data layout、mask layout、mask granularity
   conversion 在宽度转换中的同步规则。
5. 控制流 join 没有定义：`scf.if` 两边 layout 不同、`scf.for` loop-carried layout 如何稳定。
6. memory access map 和 register layout 没有切开，容易把 strided memory view 误当成 vreg
   layout。
7. hard vector semantics 缺失，例如 padding read、active prefix index、dynamic permute、
   compress/expand、scan/reduction/contract 的 VMI 表达和 lowering contract。

因此本设计保留 `vmi.vreg<NxT, layout>` 这个 carrier，但不沿用旧 layout descriptor 的
开放式语义。旧文档没有定义 “logical behavior -> hardware mismatch -> physical
decomposition -> lane map -> propagation/sink” 这条 source contract；这是本文新增的核心约束。

换句话说，本文不是复述旧 `vmi.layout`，而是把旧的开放式 axis descriptor 收紧成一个很小的
public layout 集合。本设计只接受 `contiguous`、`deinterleaved = 2`、`deinterleaved = 4`。
source contract 是新增 layout kind 的准入规则，不是要求实现 generic axes 或任意 lane-map
descriptor。

## 目标

1. VMI surface 表达逻辑向量语义，不暴露 VPTO part/dist/interleave 细节。
2. `vmi.vreg` 是 virtual aggregate type，可以表示大于 256B 的 logical vector。
3. layout 放在 layout-assigned VMI type 上，不再另设 `vbundle`。
4. VMI mask 是一等类型；surface mask 表达 logical predicate，layout-assigned mask 才携带
   concrete predicate granularity `b8/b16/b32`。
5. VMI 支持 tail / 非整 tile；padding physical lane 不可观察。
6. VMI lowering 支持控制流中的 layout join。
7. VMI producer boundary 后的 IR 必须只依赖 VMI semantic op/type 表达逻辑向量语义。

## 非目标

1. 不改变 physical `!pto.vreg<NxT>` 的含义。它仍然是 256 bytes physical register。
2. 不把 VMI 做成任何外部向量 dialect 的逐 op 复制品；VMI 只表达 PTO lowering 需要的 logical
   vector semantics。
3. 不把 scalar lane extract 当作 VMI vector op。scalar lane extract 是 vector-to-scalar
   boundary，必须在进入 VMI 前被 producer 消除，或以明确 diagnostic 退出 PTO 路线。
4. 不把 VPTO load/store dist 暴露成 VMI surface op。dist 是 lowering 选择。

## VMI Producer Boundary Contract

VMI 是 PTO 路线上的 virtual vector ISA。任何 producer 在进入 VMI boundary 后，必须满足下面之一：

1. 逻辑向量语义已经表达为 native VMI semantic op。
2. 逻辑向量语义已经表达为一组 VMI semantic op 的组合，并保持 producer 的 observable semantics。
3. 该行为不是 VMI 负责的向量计算，而是 vector-to-scalar / tensor / debug / transform boundary，
   已经在进入 VMI 前由 producer 消除，或以明确 diagnostic 退出 PTO 路线。

不能把“当前阶段不支持”作为 VMI 设计结果。一个 PTO virtual vector semantic 如果属于 VMI 负责的
逻辑向量语义，文档必须给出 VMI op、组合 lowering、layout contract、memory fallback 或 target
capability diagnostic。diagnostic 只允许表示语义边界或目标能力缺失，不能表示“VMI 没有设计这个能力”。

`pto.vmi -> pto` 的完成条件是：

```text
at VMI producer boundary:
  logical vector semantics are represented by VMI op/type
  no physical VPTO op is introduced by the producer
  no hidden layout/mask/type side table is required to interpret a VMI value

after vmi-layout-assignment:
  every vmi.vreg/vmi.mask has an explicit #pto.vmi.layout
  every mask granularity matches its consumer
  every control-flow yield/iter_arg/result has one stable layout

after vmi-to-vpto:
  no pto.vmi op/type remains
  every logical VMI value has been lowered to ordered physical VPTO values
```

### Capability And Fallback Policy

所有 direct lowering 和 fallback 选择必须来自显式配置，不能依赖 pass 内隐藏全局状态：

```text
TargetCapabilityRegistry:
  element-type storage/compute/convert support
  layout source/sink/conversion support
  memory access capability: OOB, masked, gather/scatter, block-strided
  predicate capability: granularity conversion, prefix-popcount, rearrangement
  reduction/scan/contract capability
  scratch memory spaces, alignment, and lifetime rules

VMIToPTOOptions:
  enableScratchFallback
  enableGuardedScalarFallback
  enableIndexBufferFallback
  allowDebugStrip
  targetVScaleSpecialization
  diagnosticVerbosity
```

fallback 被 option 禁用时，diagnostic 必须报告 `disabled_by_option`。target registry 缺能力时，
diagnostic 必须报告 `missing_capability`。debug-only op 只能由 debug pipeline 消费，或在
`allowDebugStrip` 明确开启时剥离；否则报 `VMI-DEBUG-BOUNDARY`。

fallback resource 也必须显式建模：

```text
scratch fallback:
  memory space, alignment, element type, shape, lifetime, and deallocation point
  must be explicit in the lowering plan
  scratch initialization, such as padding fill, must dominate later scratch load

guarded scalar/vector fallback:
  guard must dominate every memory effect it protects
  invalid lane must not compute a memory effect through an OOB memref address

index-buffer fallback:
  index element width, signedness, and address unit must match the consumer
  buffer lifetime must dominate gather/scatter or compaction use
```

如果无法分配 scratch、无法放置 guard、或 index buffer 宽度不满足目标要求，diagnostic 使用
`VMI-FALLBACK-RESOURCE`，并说明是 resource 缺失而不是语义不可表达。

## 类型模型

### Surface Type

VMI surface type 不显式写 layout：

```mlir
!pto.vmi.vreg<128xf32>
!pto.vmi.vreg<256xf8>
!pto.vmi.vreg<1xf32>

!pto.vmi.mask<128xpred>
!pto.vmi.mask<256xpred>
```

`N` 是 logical lane count，`T` 是 logical element type。surface `mask<Nxpred>` 表示 N 个
logical predicate lane，不预先绑定 VPTO predicate granularity。layout assignment 根据 consumer
选择 concrete granularity：

```text
f32/i32 consumer -> b32
f16/bf16/i16 consumer -> b16
f8/i8 consumer -> b8
```

如果一个 logical mask 被不同 width consumer 使用，VMI lowering 必须按 use 插入
`vmi.ensure_mask_granularity` 或重物化 mask producer，不能假设某个 concrete granularity 可直接
给所有 consumer 使用。

VMI type 以 1-D logical vector 为核心。来自 multi-rank producer value 的语义在进入 VMI boundary 前按 row-major flatten 成：

```mlir
!pto.vmi.vreg<64xf32>
!pto.vmi.mask<64xpred>
```

VMI value 本身只承载 flattened lane sequence，不携带隐式 rank side table。需要 rank 信息的 op
必须在自身 attr 中保存 logical shape / indexing map，例如 `logical_shape = [8, 8]`。这样保持
与既有 `vmi.vreg<NxT>` 设计一致，同时不丢失 transfer、transpose、reshape 等 op 的语义。

shape-sensitive op 的规则是：

```text
elementwise / select:
  operate on flattened lanes and preserve any surrounding op-provided shape context

tile_read / tile_write:
  carry logical_shape and permutation_map attrs

shape_cast / reshape / transpose / contract:
  carry source/result shapes, maps, and iterator metadata as op attrs

block argument / function argument:
  carries only flat vreg type; any later shaped use must provide its own shape attrs
```

因此 logical shape 信息不能保存在 C++ side table，也不能要求 consumer 从 defining op 反查。

Rank-0 logical vector 仍然是 VMI vector value，不是 scalar SSA value：

```mlir
rank-0 logical vector -> !pto.vmi.vreg<1xT>
rank-0 logical predicate -> !pto.vmi.mask<1xpred>
```

只有产生 scalar result 的 extract 才是 vector-to-scalar boundary。rank-0 logical vector load、
bitcast、mask 和 arithmetic 仍然走 VMI，不能因为只有一个 lane 就绕开 VMI verifier。

Scalable logical vector 不能直接进入 VMI type，因为 `vmi.vreg<NxT>` 的 `N` 是 concrete logical lane
count。producer 必须先根据 target profile 和 tiling decision 把 scalable semantics specialize 成固定
`N`；否则在 VMI boundary 报 `VMI-SCALABLE-VECTOR`。这不是 VMI 的临时缺口，而是
固定 256B physical vreg lowering 的前置约束。

### Layout-Assigned Type

`vmi-layout-assignment` 后，所有 VMI data/mask value 都必须带 layout：

```mlir
!pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>>
!pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>
!pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>

!pto.vmi.mask<128xb32, #pto.vmi.layout<contiguous>>
!pto.vmi.mask<128xb32, #pto.vmi.layout<deinterleaved = 2>>
```

这里的 `#pto.vmi.layout` 是唯一的 VMI register layout carrier。它不是 `#pto.vlayout`
的直接复用，也不是 `vbundle` 的 type 参数；但它必须采用同一套精确 lane-map 语义，保证后续
lower 到 physical VPTO 时可验证。

### 非整 Tile

VMI type 不要求 `N * bitwidth(T)` 是 256B 整数倍：

```mlir
!pto.vmi.vreg<100xf32>
!pto.vmi.mask<100xpred>
```

physical lowering 时按 256B part 向上取整。超出 `N` 的 physical lane 是 padding lane：

```text
padding lane:
  may be poison/undef internally
  must not be stored
  must not affect compare/reduction/scan
  must not become visible through layout conversion
```

任何 store、reduction、compress、mask-producing op 都必须用 logical lane count 或 explicit
mask 保护 padding lane。

## Layout 设计来源

VMI layout 的价值必须从逻辑 vector 行为推导，而不是从 layout 名字推导。判断流程是：

```text
1. 前端想表达一个完整的 logical vector 行为。
2. VPTO 底层指令不能把这个 logical vector 天然放进一个 contiguous physical sequence。
3. 但 VPTO 可以把这个 logical vector 拆成一组有固定 lane-map 的 physical parts。
4. 后续常见 op 可以在这些 parts 上逐 part 保持 logical semantics。
5. 边界 consumer 能直接消费这种 parts，或存在可验证的 materialize path。
6. 因此值得把这个 parts relation 提升为 VMI layout。
```

layout 不是“某条指令的名字”，而是一个 representation relation：

```text
Layout L defines:
  logical vector value V[NxT]
  <-> ordered physical parts P0, P1, ...
  with exact map logical lane i -> (part, lane)
```

只有当这个 relation 能让 VMI 保持“用户看到的是一个连续 logical vector”，同时避免前端手写
parts，layout 才有设计价值。

### Register Layout 集合

VMI register layout 不采用复杂通用 descriptor，而是定义为封闭集合：

```text
#pto.vmi.layout<contiguous>
#pto.vmi.layout<deinterleaved = 2>
#pto.vmi.layout<deinterleaved = 4>
```

`deinterleaved = K` 表示一个 logical vector 被拆成 K 个 physical part，第 `p` 个 part 保存
logical lane `p, p + K, p + 2K, ...`。这个名字直接描述元素摆放，不绑定到某条 VPTO op，也不
引入旧 `axes` 的通用维度系统。

不加入 `channel`、`packed_bits`、`blocked`、`stride`、`permutation` 等 layout kind。
这些能力先由 VMI semantic op、memory access plan 或 explicit layout conversion 表达。只有当
一个新 representation 同时满足下面的 source contract，才允许扩展 layout 目录。

### Layout Source Contract

每个 VMI layout kind 必须来自一条明确的 source contract：

```text
logical behavior:
  VMI 想表达的用户级 vector 行为

hardware mismatch:
  为什么 VPTO 不能用一个 contiguous physical sequence 天然承载

physical decomposition:
  VPTO 实际能产生或消费的 physical parts

lane map:
  logical lane -> physical part/lane 的精确定义

propagation rule:
  哪些 VMI op 可以逐 part 保持语义

boundary rule:
  哪些 load/store/pack/convert consumer 可以直接消费，哪些必须 materialize

mask rule:
  对应 mask<NxG, same-layout> 如何生成、转换和消费
```

没有这份 source contract 的 lane movement 不能进入 `#pto.vmi.layout`。

### Source 1: Widen Cast To Larger Logical Vector

逻辑行为：

```mlir
%w = pto.vmi.extf %a
  : !pto.vmi.vreg<128xf16> -> !pto.vmi.vreg<128xf32>
```

用户语义是“128 个 f16 lane 加宽成 128 个连续 f32 lane”。但 128 个 f32 是 512B，超过单个
256B physical vreg。VPTO 的可行 lowering 不是一个 contiguous 512B register，而是两条 part
conversion：

```text
even part:
  physical even[i] = extf(logical[2*i])

odd part:
  physical odd[i]  = extf(logical[2*i+1])
```

因此需要一个 layout 表达“这个 VMI value 仍然是 logical `128xf32`，但 physical representation
是 even/odd 两个 parts”：

```mlir
#pto.vmi.layout<deinterleaved = 2>
```

lane map：

```text
part = i % 2
lane = floor(i / 2)
physical[part][lane] = logical[i]
```

这个 layout 的价值在于后续 elementwise op 不需要 materialize contiguous representation：

```mlir
%s = pto.vmi.addf %w, %b
  : !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>
```

lowering 可以变成两路 add：

```text
add even parts
add odd parts
```

最后如果 store consumer 能把 even/odd parts 交织写回 contiguous memory，就不需要中途
`ensure_layout contiguous`。

同理：

```mlir
%w = pto.vmi.extf %a
  : !pto.vmi.vreg<256xf8> -> !pto.vmi.vreg<256xf32>
```

需要：

```mlir
#pto.vmi.layout<deinterleaved = 4>
```

这里不再使用抽象 stride 命名。`deinterleaved = 4` 的来源是 `f8 -> f32` 的 VPTO part
conversion contract，不是任意 stride 语义。

### Source 2: Narrow / Pack Consumer

逻辑行为：

```mlir
%n = pto.vmi.truncf %x
  : !pto.vmi.vreg<128xf32> -> !pto.vmi.vreg<128xf16>
```

如果 `%x` 已经是 `#pto.vmi.layout<deinterleaved = 2>`，VPTO 可以用 pack/narrow 类
consumer 把 even/odd f32 parts 合成 contiguous f16 result。这里 layout 的来源不是 producer，而是
consumer 能直接接受这种 decomposition：

```text
source layout:
  logical f32 value represented as even/odd f32 parts

consumer:
  narrowing pack consumes those parts

result:
  contiguous f16 logical vector
```

因此 `deinterleaved` 必须同时登记 producer contract 和 inverse/sink contract。否则 layout 只能
产生，不能被合法消耗。

### Source 3: Same-Width Layout Materialization

逻辑行为：

```mlir
%x = pto.vmi.ensure_layout %v
  : !pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>>
    -> !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>
```

这里不新增 surface view op。目标不是产生两个独立 semantic vectors，而是让同一个 logical
vector 继续作为一个 VMI value 存活，只是 physical representation 变成 even/odd parts。IR 中由
`vmi-layout-assignment` 插入
`pto.vmi.ensure_layout`，并由 target registry 证明存在 preserving materialization path。VPTO 的
`vdintlv/vintlv` 类 register rearrangement 可以产生或消费这种 representation。

这和 `vcvt` 产生的 even/odd representation 使用同一个 layout：

```mlir
#pto.vmi.layout<deinterleaved = 2>
```

区别只在 source contract：

```text
logical behavior:
  同宽 logical vector 保持一个 VMI value，但 physical parts 分别保存 even/odd lanes

hardware mismatch:
  VPTO interleave/deinterleave 指令以两个 physical vreg parts 表达

layout:
  deinterleaved=2
```

如果 VMI op 的语义本来就是“返回两个独立 vectors”，例如 AoS -> SoA 后用户分别使用 `%x`
和 `%y`，那不需要 layout，直接产生两个 `vmi.vreg<NxT>`。只有当“一个 logical vector value”
需要以 even/odd parts 长期存活时，才使用 `deinterleaved=2`。

### Channel Split / Merge 不是 Register Layout

channel split/merge 的用户代码通常有两种形态。

第一种是把 interleaved data 当作普通 flat vector：

```text
logical = [r0, g0, b0, a0, r1, g1, b1, a1, ...]
对每个 lane 做同一种逐元素操作
```

这种情况下 `contiguous` representation 就能表达用户语义，不需要 channel layout。

第二种是用户按 channel 编程：

```mlir
%r, %g, %b, %a = pto.vmi.channel_split %rgba
  : !pto.vmi.vreg<128xi8>
    -> !pto.vmi.vreg<32xi8>, !pto.vmi.vreg<32xi8>,
       !pto.vmi.vreg<32xi8>, !pto.vmi.vreg<32xi8>

%r2 = pto.vmi.addi %r, %bias_r : !pto.vmi.vreg<32xi8>
%g2 = pto.vmi.addi %g, %bias_g : !pto.vmi.vreg<32xi8>
%b2 = pto.vmi.addi %b, %bias_b : !pto.vmi.vreg<32xi8>
%a2 = pto.vmi.addi %a, %bias_a : !pto.vmi.vreg<32xi8>
%out = pto.vmi.channel_merge %r2, %g2, %b2, %a2
  : !pto.vmi.vreg<32xi8>, !pto.vmi.vreg<32xi8>,
    !pto.vmi.vreg<32xi8>, !pto.vmi.vreg<32xi8>
    -> !pto.vmi.vreg<128xi8>
```

这里自然的 IR 是多个 semantic VMI values，而不是“一个 VMI value 带 channel layout”。
目标专用 split/merge 能力是 `channel_split/channel_merge` 的 lowering contract；load/store
memory boundary 的 dist/sink contract 也可以作为等价 lowering path。

`channel_split` / `channel_merge` 的语义必须能完全退化成 static shuffle，不能引入额外
layout 规则。`C` 不需要单独 attr：`channel_split` 的 `C` 来自 result 个数，
`channel_merge` 的 `C` 来自 operand 个数。设 input 有 `N = C * M` 个 logical lanes：

```text
channel_split(input, C):
  out[c][i] = input[i * C + c]
  for 0 <= c < C
  for 0 <= i < M

channel_merge(out[0], ..., out[C-1]):
  result[i * C + c] = out[c][i]
  for 0 <= i < M
  for 0 <= c < C
```

如果 `N` 不能被 `C` 整除，或者 merge operands 的 logical lane count 不一致，op verifier
必须拒绝。需要 tail 的场景通过外层 mask / valid lane 语义表达，不能让 channel op 自己发明
padding lane。

因此这两个 op 的价值只是 canonical interface：producer 可以直接表达 channel 语义，
外部 import 工具也可以把识别出的 static shuffle pattern canonicalize 成它们；如果没有
识别或目标没有专用 lowering，保持或退回 `pto.vmi.shuffle` 仍然是等价路径。
当前 direct VPTO lowering 只接受能形成完整 physical channel groups 的形状：flat contiguous
source/result 与 virtual deinterleaved=C channel layout 必须有相同 physical arity，或已经是 matching
deinterleaved=C layout 的 identity forwarding。arity-changing partial group 需要额外 packing/drop
padding plan，不能直接 lowering。

所以 VMI register layout 目录不为 channel-specific representation 引入 layout kind，也不预留
半成品 layout 语义。本文覆盖的用户形态要么是 flat contiguous vector，要么是多个 channel
semantic value；都不需要“一个 VMI value 带 channel layout”。

### Pack / Unpack 不作为长期 Layout

pack/unpack 的逻辑行为通常是 width conversion 或 memory encoding：

```text
wide logical vector -> narrow logical vector
narrow memory payload -> wide logical vector
```

它们的结果可以是 `contiguous` logical vector；pack/unpack 是 producer/sink/conversion
contract，不是必须长期传播的 register layout。只有当目标 ISA 提供 packed-format arithmetic，
并且 VMI 真的要让 packed representation 跨 compute 存活时，才需要另立
`packed_bits` layout。本设计没有 packed-format arithmetic source contract，因此 pack/unpack 不进入
长期 register layout。

### 不应成为 Register Layout 的东西

以下能力虽然来自 VPTO/VISA，但不是 VMI register layout：

| 能力 | 原因 |
|---|---|
| `vsldb/vsstb` block stride | 描述 memory address map；result register 可仍是 contiguous representation |
| gather/scatter index | runtime address map，不是 static logical lane 到 physical part 的关系 |
| dynamic `vselr` | runtime permutation，应是 `pto.vmi.permute` op |
| `vsqz/vusqz` compaction | runtime mask 决定 lane destination，应是 `compress/active_prefix_index` op |
| one-shot `vintlv/vdintlv` | 如果只是 boundary conversion，不应提升成长期 layout；若表示一个 VMI value 的 even/odd parts，则归入 `deinterleaved=2` |

VMI layout 只解决“一个 logical vector value 在寄存器中长期以什么 parts representation 存活”
的问题。memory address、runtime permutation、dynamic compaction 都是其它语义。

### Lane Map

设：

```text
N = logical lane count
lanesPerDataPart(T) = 256B / sizeof(T)
lanesPerMaskPart(b8)  = 256
lanesPerMaskPart(b16) = 128
lanesPerMaskPart(b32) = 64
```

`contiguous`：

```text
chunk = floor(i / lanesPerPart)
lane  = i % lanesPerPart
physical[chunk][lane] = logical[i]
```

`deinterleaved = K`，其中 `K` 只能是 2 或 4：

```text
p     = i % K
q     = floor(i / K)
chunk = floor(q / lanesPerPart)
lane  = q % lanesPerPart
physical[p][chunk][lane] = logical[i]
```

`deinterleaved=2` 和 `deinterleaved=4` 的 physical value ordering 固定为 part-major：

```text
p0_chunk0, p0_chunk1, ..., p1_chunk0, p1_chunk1, ..., p(K-1)_chunk0, ...
```

所有 verifier、type converter、physical lowering 和 control-flow conversion 必须使用同一套
ordering。

### Physical Arity

`vmi-to-vpto` 不能按示例猜 physical value 个数，必须由 type + layout 统一推导。

对 data vreg：

```text
lanesPerPart = 256B / sizeof(T)

contiguous:
  chunks = ceil(N / lanesPerPart)
  physical values = chunks

deinterleaved = K:
  lanesPerLogicalPart = ceil(N / K)
  chunksPerPart = ceil(lanesPerLogicalPart / lanesPerPart)
  physical values = K * chunksPerPart
```

对 mask：

```text
lanesPerPart = lanesPerMaskPart(G)
same formula as data, replacing T with mask granularity G
```

每个 physical value 的有效 lane 由 lane map 反推：

```text
contiguous valid:
  logical = chunk * lanesPerPart + lane
  valid = logical < N

deinterleaved valid:
  logical = K * (chunk * lanesPerPart + lane) + p
  valid = logical < N
```

padding lane 可以是 poison/undef，但 store、mask-producing op、reduction、scan、compress 和
layout conversion 都必须显式带着 `valid` 信息，不能只依赖 physical register 宽度。

### Broadcast 不作为 Register Layout

VMI surface 使用 `broadcast` 表达前端语义：

```mlir
%v = pto.vmi.broadcast %x : f32 -> !pto.vmi.vreg<128xf32>
```

也就是：

```text
for i in 0 .. N:
  v[i] = x
```

这不是 logical lane 到 physical part/lane 的 placement relation，而是一个 value producer
可以延迟 materialize 的事实。`vmi.broadcast` 应保持为 semantic op 或 layout-polymorphic
producer：

```text
consumer wants contiguous:
  materialize scalar into contiguous physical parts

consumer wants deinterleaved=2:
  materialize same scalar into even/odd parts

consumer wants deinterleaved=4:
  materialize same scalar into p0/p1/p2/p3 parts
```

因此 broadcast 不进入 `#pto.vmi.layout` 目录。它由 `vmi-layout-assignment` 按 consumer
layout 重物化或下沉到 consumer lowering，而不是作为 `vreg<NxT, layout>` 的 layout kind。

#### Broadcast Materialization

MLIR SSA value 不能对不同 use 拥有不同 result type。因此 scalar broadcast 的多 layout
适配不是“一个 VMI value 同时带多个 layout”，而是在 layout assignment 中按 use 重物化。

semantic VMI：

```mlir
%b = pto.vmi.broadcast %x : f32 -> !pto.vmi.vreg<128xf32>
%u = pto.vmi.addf %a_contiguous, %b
  : !pto.vmi.vreg<128xf32>
%v = pto.vmi.addf %a_split, %b
  : !pto.vmi.vreg<128xf32>
```

如果 `%u` 需要 `contiguous`，`%v` 需要 `deinterleaved=2`，layout assignment 重写为：

```mlir
%b0 = pto.vmi.broadcast %x
  : f32 -> !pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>>
%u = pto.vmi.addf %a_contiguous, %b0
  : !pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>>

%b1 = pto.vmi.broadcast %x
  : f32 -> !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>
%v = pto.vmi.addf %a_split, %b1
  : !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>
```

physical materialization：

```text
contiguous:
  each physical chunk is filled with scalar x

deinterleaved=2:
  even part is filled with scalar x
  odd part is filled with scalar x

deinterleaved=4:
  p0/p1/p2/p3 parts are all filled with scalar x
```

这要求 `pto.vmi.broadcast` 标记为 rematerializable，并满足 dominance：clone 位置必须被 scalar
operand `%x` dominate。跨控制流时，如果 scalar operand 可在各 predecessor/body 内使用，
优先在 consumer 所在 block 重物化；否则必须在控制流 join 处选择一个具体 layout 并 materialize。

这个规则只对 scalar-to-vector broadcast 是零语义风险的。低 rank vector 到高 rank vector 的
broadcast 可能需要真实 lane replication/shuffle，不能默认按任意 consumer layout 免费重物化；
这类 broadcast 必须携带 broadcast map，并按普通 VMI op 做 layout assignment。

VMI register layout 目录因此是：

```text
contiguous
deinterleaved=2
deinterleaved=4
```

channel split/merge、pack/unpack、memory stride、dynamic permutation、dynamic compaction
不在目录内。它们分别由 VMI semantic op、conversion、memory access plan、`vmi.permute`、
`vmi.compress/active_prefix_index` 承接。

## Pipeline

### 1. VMI Producer Boundary

VMI core pipeline 从合法 VMI semantic IR 开始。Producer 可以是 TileLang/PTO lowering、手写 VMI
测试或其它外部 import 工具，但 producer 不属于 VMI core pipeline。

进入 VMI boundary 时必须满足：

```text
all logical vector semantics are represented by pto.vmi semantic ops
all VMI data/mask values use surface VMI type without layout
no physical VPTO op is introduced
no hidden layout/mask/type side table is required
scalar/tensor/debug/transform boundary has already been handled by producer
```

该 boundary 需要 verifier gate。它验证 VMI IR 自身完整，不验证某个外部 source dialect 的
coverage。

### 2. `vmi-layout-assignment`

该阶段把无 layout VMI type 转换成 layout-assigned VMI type，推荐实现为独立 pass：

```mlir
!pto.vmi.vreg<128xf32>
  -> !pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>>

!pto.vmi.mask<128xpred>
  -> !pto.vmi.mask<128xb32, #pto.vmi.layout<contiguous>>
```

layout assignment 做三件事：

1. 为每个 producer 选择 natural layout。
2. 为每个 consumer 协调 operand/result layout。
3. 在必要处插入：

```mlir
pto.vmi.ensure_layout
pto.vmi.ensure_mask_layout
pto.vmi.ensure_mask_granularity
```

layout assignment 不是局部 pattern 贪心插 conversion，而是约束求解：

```text
nodes:
  every VMI SSA value
  block arguments and region/function results
  rematerializable producers such as scalar broadcast/iota/constant

allowed layouts:
  contiguous
  deinterleaved=2
  deinterleaved=4
  filtered by element type, mask granularity, op capability, and target registry

hard constraints:
  op verifier constraints, such as same-layout elementwise operands
  data/mask layout alignment for predicated ops
  control-flow block argument/yield/call signature equality
  external ABI layout boundary
  source/sink contracts for width conversion, load/store, pack/narrow

soft costs:
  natural producer layout preference
  ensure_layout materialization cost from target registry
  store/load sink cost
  rematerialization cost for broadcast/iota/constant
  scratch/guarded fallback resource cost
```

求解顺序：

```text
1. Build constraints for the whole region/SCC, including control-flow and call edges.
2. Propagate impossible layouts and required mask granularities.
3. Choose a minimum-cost layout for each node.
4. Use deterministic tie-break: prefer existing natural layout, then contiguous.
5. Insert ensure_layout/ensure_mask_layout or rematerialize producers at chosen use sites.
6. Re-run verifier gates; no hidden side table may be needed to interpret the result.
```

如果 hard constraints 冲突，或所有 legal paths 都缺 target capability/resource，报
`VMI-LAYOUT-CONTRACT` 或更具体 diagnostic。diagnostic payload 必须列出 conflict value、producer
natural layout、consumer required layouts、available conversion paths 和被禁用的 fallback。

#### Consumer Layout Demand

“consumer 需要某个 layout”不是前端语义要求，而是 layout assignment 为了让 operands/results
的 lane-map 对齐并减少 layout conversion 选择的共同 representation。

典型例子：

```mlir
%w = pto.vmi.extf %a
  : !pto.vmi.vreg<128xf16> -> !pto.vmi.vreg<128xf32>

%b = pto.vmi.broadcast %scalar
  : f32 -> !pto.vmi.vreg<128xf32>

%s = pto.vmi.addf %w, %b
  : !pto.vmi.vreg<128xf32>
```

`%w` 的 logical 语义是 `128xf32`，但 VPTO `f16 -> f32` 的自然 lowering 产生 even/odd
两路 parts：

```text
w_even[i] = extf(a[2*i])
w_odd[i]  = extf(a[2*i+1])
```

因此 `%w` 的 natural layout 是：

```mlir
#pto.vmi.layout<deinterleaved = 2>
```

`addf` 是 layout-polymorphic elementwise op。它有两个合法选择：

```text
choice A:
  materialize %w to contiguous
  materialize broadcast to contiguous
  do one contiguous add sequence

choice B:
  materialize broadcast directly as deinterleaved=2
  do add on even parts and odd parts separately
  keep result as deinterleaved=2
```

choice B 通常更便宜，因为不需要把 `%w_even/%w_odd` 先 interleave 成 contiguous。broadcast
能直接适配 `deinterleaved=2`，是因为它的 logical lanes 全部等于同一个 scalar：

```text
b_even = [scalar, scalar, ...]
b_odd  = [scalar, scalar, ...]
```

所以这里说 `addf` consumer “需要” `deinterleaved=2`，准确含义是：

```text
layout assignment 选择 deinterleaved=2 作为 addf 的共同 operand/result representation，
因为其中一个 operand 的 natural layout 已经是 deinterleaved=2，并且 broadcast 可零语义风险地重物化到该 layout。
```

### 3. `vmi-to-vpto`

该阶段把 layout-assigned VMI type 做 1:N physical type conversion，推荐实现为独立 pass：

```text
!pto.vmi.vreg<128xf32, contiguous>
  -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

!pto.vmi.vreg<128xf32, deinterleaved=2>
  -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

!pto.vmi.vreg<256xf32, deinterleaved=4>
  -> !pto.vreg<64xf32>, !pto.vreg<64xf32>,
     !pto.vreg<64xf32>, !pto.vreg<64xf32>

!pto.vmi.mask<128xb32, deinterleaved=2>
  -> !pto.mask<b32>, !pto.mask<b32>
```

需要 internal projection/materialization op：

```mlir
%p0, %p1 = pto.vmi.unpack %v
  : !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>
    -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%v = pto.vmi.pack %p0, %p1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32>
    -> !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>
```

`pack/unpack` 不是新的 layout carrier，只是 layouted `vmi.vreg` 到 physical VPTO parts 的
projection/materialization。

`unpack` 必须能作用在任意 SSA value 上，不能依赖 defining op。VMI value 可以来自 block
argument、`scf.if` result、loop iter_arg、function argument 或 call result；这些 value 没有
可 look-through 的 layout materialization defining op。

`pack/unpack` 的 operand/result 个数必须使用 Physical Arity 公式推导。非整 tile 时，最后一个
chunk 的 padding lane 仍属于 physical value，但不属于 logical value。

### Layout Conversion Materialization

`pto.vmi.ensure_layout` / `pto.vmi.ensure_mask_layout` 是 logical-value-preserving conversion：

```text
for every logical lane i:
  dst.logical[i] = src.logical[i]
for every padding lane:
  dst padding remains unobservable
```

source/result layout 完全相同时，`ensure_layout` / `ensure_mask_layout` 是 identity forwarding；
即使存在 partial/tail physical chunk，也不需要 target materialization path。source/result layout
不同时才需要 registry 证明 preserving conversion 及其 full-chunk/tail 处理策略。当前 direct path
允许 equal-arity partial/tail conversion：source/result 的 physical arity 必须相同，且两边都能组成完整
contiguous/deinterleaved=2/4 `intlv` materialization group；arity-changing partial conversion 和 uneven
deinterleaved groups 继续报 unsupported。

合法 materialization path 必须来自 target registry：

```text
same layout:
  no-op

contiguous <-> deinterleaved=2:
  direct interleave/deinterleave register op, load/store dist sink/source,
  or scratch/ordered fallback

contiguous <-> deinterleaved=4:
  direct 4-way layout sink/source, proven staged 2-way sequence,
  or scratch/ordered fallback

deinterleaved=2 <-> deinterleaved=4:
  convert through contiguous only if both legs have preserving paths,
  otherwise use scratch/ordered fallback or report VMI-LAYOUT-CONTRACT
```

`deinterleaved=4` 不能默认假设“两次二路 interleave”就是正确 materialization。只有当 staged
sequence 的 lane map 被 registry 证明等价于：

```text
logical = 4 * lane + p
```

才允许使用。否则必须选择 store sink、scratch buffer 或 diagnostic。

### Verifier Gates

每个 pipeline 边界都必须有 hard verifier，不能把残缺 IR 留给后续 pass 猜测：

```text
at VMI producer boundary:
  every logical vector value is represented by !pto.vmi.vreg / !pto.vmi.mask
  every logical vector operation is represented by pto.vmi semantic op
  no physical VPTO op has been introduced
  no hidden layout/mask/type side table is required to interpret a value

after vmi-layout-assignment:
  every !pto.vmi.vreg / !pto.vmi.mask has #pto.vmi.layout
  layout kind is one of contiguous/deinterleaved=2/deinterleaved=4
  mask granularity matches each consumer
  branch operands, block arguments, function arguments/results, and yields agree on layout
  no hidden layout/mask/type side table is required to interpret a value

before vmi-to-vpto:
  every pto.vmi.ensure_layout / ensure_mask_layout has a registered preserving materialization path
  every fallback path has resource decision and dominance/lifetime proof

after vmi-to-vpto:
  no pto.vmi op or type remains
  no UnrealizedConversionCastOp remains
  no pto.vmi.pack/unpack/ensure_* helper remains
  every physical value arity matches the Physical Arity helper
```

layout、mask、valid-lane 和 physical arity 信息必须存在于 IR type/attr/op operand 中，或可由它们
纯函数推导；不能依赖 C++ side table。违反这些 gate 时使用 `VMI-PASS-INVARIANT` 或更具体的
diagnostic，例如 `VMI-LAYOUT-CONTRACT`、`VMI-MEMORY-ACCESS`、`VMI-RESIDUAL-OP`。

## Layout Assignment 规则

### Elementwise

same-layout operands：

```text
vmi.addf/vmi.mulf/vmi.cmpi/vmi.select
  fan out per physical part
  result keeps operand layout
```

different-layout operands：

```text
choose consumer-demanded layout
insert ensure_layout for other operands
vmi.broadcast can rematerialize in consumer-demanded layout
```

### Width Conversion

典型 natural layout：

```text
vmi.extf 128xf16 -> 128xf32:
  source contiguous f16
  result deinterleaved=2 f32

vmi.extf 256xf8 -> 256xf32:
  source contiguous f8
  result deinterleaved=4 f32

vmi.truncf 128xf32 -> 128xf16:
  source may be deinterleaved=2 f32
  result contiguous f16 if pack/store sink requires contiguous

vmi.truncf 256xf32 -> 256xf8:
  source may be deinterleaved=4 f32
  result contiguous f8 if pack/store sink requires contiguous
```

Direct `vcvt` lowering 可以覆盖同一 contract 下的 partial/tail case：`extf` 的 logical lanes
必须仍然装进一个 contiguous narrow source physical chunk，并自然产生 deinterleaved=2/4 result；
`truncf` 的 deinterleaved=2/4 source parts 必须能 pack 成一个 contiguous narrow result chunk。
这些路径允许 VPTO 对 padding lanes 执行 conversion，但 padding 只能流向 result padding lanes，
不能变成 logical result。

Mask granularity assignment 把 surface `mask<Nxpred>` 转成 concrete
`mask<NxG, layout>`。consumer 决定所需 granularity：

```text
f16 op consumes mask<Nxb16>
f32 op consumes mask<Nxb32>
f8  op consumes mask<Nxb8>
```

如果 data 从 f16 扩到 f32，后续 f32 consumer 需要：

```mlir
!pto.vmi.mask<Nxb32, same-data-layout>
```

不能继续复用 `mask<Nxb16>`。

mask-producing op 的 granularity 不是 producer 固有属性：

```text
vmi.create_mask / constant_mask:
  logical predicate producer; granularity chosen by users
  create_mask 的 logical prefix 语义不受目标 PAT_VL token 集合限制；
  unsupported PAT_VL count 可以用 pto.plt_b* materialize
  constant_mask 的 non-prefix chunk 用 prefix 差分和 predicate boolean ops materialize

vmi.cmpf/cmpi:
  result logical lane count follows compared data
  concrete granularity chosen by mask consumers, not by compare element type alone

multi-use mask:
  choose one concrete granularity for the original SSA value
  insert ensure_mask_granularity or rematerialize cheap mask producers per use
```

`ensure_mask_granularity` 必须 preserve logical predicate lane `mask[i]`。当前 direct lowering 对
concrete `b8/b16/b32` granularity 使用 `pto.punpack` 做 widening，使用 `pto.ppack` 加 `pto.por`
做 narrowing，并按需要串联相邻级别完成 `b8 <-> b32`。如果目标缺少 predicate rearrangement 或
granularity conversion，报 `VMI-LAYOUT-CONTRACT`，不能把 b16/b32 mask 当成同一 physical bit
pattern 直接复用。

### Predication

Region-style mask 不作为长期 region op 保留到 VPTO lowering。producer 必须把 mask thread 到
具体 VMI op：

```text
masked load/store:
  use pto.vmi.masked_load / pto.vmi.masked_store

masked arithmetic with passthru:
  compute candidate result
  merge with passthru by pto.vmi.select(mask, candidate, passthru)

masked reduction/scan:
  inactive and padding lanes are excluded from the logical iteration
```

如果一个 masked op 的 inactive lane 语义要求“不读内存”或“不执行有副作用操作”，不能用
full op + select 伪装；必须使用对应 masked VMI op、ordered fallback，或报 target capability
diagnostic。

### Memory Ops

VMI memory op 表达 memory semantics，不表达 register layout。lowering 先构造 access plan：

```text
base
logical lane count
logical_shape attr, if any
lane-to-address map
contiguity
block-strided row classification
read/write validity mask
padding plan
footprint safety proof
target OOB capability
```

memory access map 不是 register layout。比如 `tile_read` 的 memref stride 可以识别
block-strided rows，并选择 `vsldb`，但 result `vmi.vreg` 的 register layout 仍由
layout assignment 决定。

Producer-specific packed element view 不进入 VMI type。它们必须在 VMI memory op 之前规范化为
element memref + access map：

```text
memref<M x vector<KxT>>
  -> base element type T
  -> logical address = original index * K + vector_lane
```

normalization 必须保留 offset、stride、alignment、memory space 和 alias 信息。无法证明等价
element view 时，报 `VMI-MEMORY-ACCESS`，不能把 packed element memref 伪装成 contiguous VMI
load/store。

direct path examples：

```text
contiguous full-safe:
  vlds/vsts
  !pto.ptr source/destination must be UB-backed; memref source/destination
  must either have unknown memory space at this stage or explicitly use
  #pto.address_space<vec>

32B block-strided rows with block-uniform mask:
  vsldb/vsstb

interleave/deinterleave boundary:
  vldsx2/vstsx2 dist or explicit rearrangement

indexed memory:
  gather/scatter if inactive and duplicate-index semantics match
```

GM-backed VMI memory is semantic input, not a direct vector load/store target.
Current `vmi-to-vpto` direct memory lowering emits `pto.vlds`, `pto.vldsx2`,
`pto.vsts`, or `pto.vstsx2`; those VPTO ops operate on UB-backed vector memory.
If a `pto.vmi.load/store/tile_read/tile_write` still names GM at this stage,
the missing step is an explicit memory movement/materialization plan, scratch
plan, or UB view normalization. Otherwise the pass must report `VMI-UNSUPPORTED`
instead of silently producing illegal VPTO.

### Control Flow

VMI layouted type 可以跨 internal control flow，但 public ABI 不允许 layout leak。

MLIR conversion framework 可以做 region/block/signature 的 structural type conversion，但它不会
自动决定 layout。`vmi-layout-assignment` 必须先把每个 block argument、region yield、branch
operand 和 call boundary 的 layout 固定下来，再交给 `vmi-to-vpto` 做 1:N type conversion。

`scf.if` join：

```text
if all incoming layouts equal:
  keep that layout
else:
  choose consumer-demanded layout, otherwise contiguous
  insert ensure_layout / ensure_mask_layout before yield
```

`scf.for` loop-carried value：

```text
init layout == iter_arg layout == yield layout == loop result layout
```

如果 loop body repeatedly consumes deinterleaved=2/deinterleaved=4，优先保持该 natural layout；如果只有 loop
exit 需要 contiguous，则在 exit 后转换，不在 backedge 每轮转换。

`cf.br` / `cf.cond_br` block arguments：

```text
target block argument has one chosen layout
each predecessor operand is converted to that layout before branch
```

function boundary：

```text
internal VMI functions:
  function argument/result layout is part of layout assignment
  all callsites and returns must agree with the specialized signature layout

external/public ABI:
  must not expose #pto.vmi.layout
  materialize to memory, scalar ABI, or final physical PTO ABI before crossing boundary
```

recursive or mutually recursive VMI functions require SCC fixed-point layout assignment. If a stable signature
layout cannot be found without inserting conversion on every cycle edge, choose `contiguous` at the function
boundary and keep deinterleaved layouts inside the function body.

## VMI Op Families

本节列出 VMI 必须拥有的 semantic op。assembly form 可在 ODS 中微调，但语义边界应保持。
表中用 `/` 写在一起的名字表示多个独立 op，不表示一个 variadic opcode。去重后，正式
semantic op 数量是 75 个。
`ensure_layout`、`ensure_mask_layout`、`ensure_mask_granularity`、`pack`、`unpack` 是内部
layout/materialization helper，不计入 semantic op；如果把 helper 也算作 VMI op，总数是 80 个。

该总表描述目标 semantic surface，不等价于当前第一批实现清单。当前 implementation slice
以 `docs/designs/vmi-implementation-manual.md` 的 Slice 1 为准；例如 `pto.vmi.from_elements`
虽然属于目标 construction family，但没有 scalar lane insert、vreg immediate 或 scratch
materialization plan 前不能宣称 direct lowering 已支持。

```text
construction:                      6
memory:                           10
arithmetic/conversion:            36
permutation/mask/reduction/channel: 23
semantic total:                   75
internal helpers:                  5
total including helpers:          80
```

### Construction

| Op | 语义 |
|---|---|
| `pto.vmi.constant` | logical constant vector，layout assignment 决定 materialization |
| `pto.vmi.broadcast` | scalar 或低 rank value broadcast 到 `vreg<NxT>` |
| `pto.vmi.iota` | 从 scalar base 生成 logical lane index/value vector |
| `pto.vmi.from_elements` | 按 logical lane order 构造 |
| `pto.vmi.create_mask` | prefix 或 logical-shape mask |
| `pto.vmi.constant_mask` | static logical predicate mask, including non-prefix masks |
| `pto.vmi.mask_and/or/xor/not` | logical predicate elementwise operation |

### Memory

```mlir
%v = pto.vmi.load %base[%idx]
  : memref<?xf32> -> !pto.vmi.vreg<128xf32>

pto.vmi.store %v, %base[%idx]
  : !pto.vmi.vreg<128xf32>, memref<?xf32>

%v = pto.vmi.masked_load %base[%idx], %mask, %passthru
  : memref<?xf32>, !pto.vmi.mask<128xpred>,
    !pto.vmi.vreg<128xf32> -> !pto.vmi.vreg<128xf32>

pto.vmi.masked_store %v, %base[%idx], %mask
  : !pto.vmi.vreg<128xf32>, memref<?xf32>, !pto.vmi.mask<128xpred>

%g = pto.vmi.gather %base[%indices], %mask, %passthru
  : memref<?xf32>, !pto.vmi.vreg<128xindex>, !pto.vmi.mask<128xpred>,
    !pto.vmi.vreg<128xf32> -> !pto.vmi.vreg<128xf32>

pto.vmi.scatter %v, %base[%indices], %mask
  : !pto.vmi.vreg<128xf32>, memref<?xf32>,
    !pto.vmi.vreg<128xindex>, !pto.vmi.mask<128xpred>

%e = pto.vmi.expand_load %base[%idx], %mask, %passthru
  : memref<?xf32>, !pto.vmi.mask<128xpred>,
    !pto.vmi.vreg<128xf32> -> !pto.vmi.vreg<128xf32>

pto.vmi.compress_store %v, %base[%idx], %mask
  : !pto.vmi.vreg<128xf32>, memref<?xf32>, !pto.vmi.mask<128xpred>
```

`masked_load` 的 inactive lane 不能产生 memory read。full load + select 只有在 inactive
lane 地址 safe-readable 时才合法。
当前直接 lowering 只覆盖 contiguous result/passthru/mask：full physical chunks 直接 `vlds + vsel`；
partial/tail chunks 必须先证明完整 physical read footprint safe-readable，否则报 `VMI-UNSUPPORTED`。
在第一阶段的矩阵 quant/dequant lowering 中，默认假设 UB 中的行数据按元素连续，tail load 可以安全读满
当前物理 vreg；tail 的对外写入效果仍由 `pto.vmi.create_mask` + `pto.vmi.masked_store`
约束。严格 no-read tail 不是这个默认路径的语义，后续通过 stable gather 模式承接：该模式应把
contiguous tail masked load 转为 `VGATHER2 + Pg` 风格的 per-lane non-faulting load。当前
`vmi-to-vpto` 只预留 `enable-stable-gather-masked-load` 开关；开关打开且遇到
`pto.vmi.masked_load` 时必须给 TODO diagnostic，不能退化成普通 `vlds + vsel`。

普通 `vmi.store` 和 `vmi.masked_store` 的 contiguous tail 可以用 true predicate store 承接：
full physical chunk 使用 all-true mask 或用户 mask，最后一个 partial chunk 使用 prefix valid-lane
mask；因此普通 `vmi.store` direct lowering 要求 value element width 能对应
`pto.mask<b8/b16/b32>`。`masked_store` 先把用户 mask 与 valid-lane mask 做 logical AND。
deinterleaved=2/4 tail store/masked_store 只有在每个 deinterleaved part 的 physical chunk 数相同、可先组成完整
`vintlv/pintlv` group 并 materialize 成 contiguous chunks 时才直接支持；materialized 后 active
lane 为 0 的 padding-only chunk 不发 store。load padding 仍需要独立的 access plan，不能通过未受保护的
full-footprint memory op 偷跑。

`gather/scatter` 使用 logical lane order 解释 `%indices`，index 单位和 memref element type
一致。`gather` inactive lane 返回 `%passthru[i]` 且不能读内存。`scatter` inactive lane 不能写
内存；如果 active lanes 可能写同一地址，direct VPTO lowering 必须证明目标语义与 logical
lane order 等价，否则使用 ordered fallback 或报 `VMI-MEMORY-ACCESS`。

当前 `gather` direct lowering 覆盖一个保守子集：

```text
source:
  !pto.ptr<T, ub>

layout:
  result / indices / mask / passthru all contiguous
  all physical chunks are full, so padding lanes cannot trigger memory reads

type:
  T is 32-bit element type
  indices are signless or unsigned i32
  mask granularity is b32

lowering:
  gathered = pto.vgather2_bc source, indices, mask
  result   = pto.vsel gathered, passthru, mask
```

`VGATHER2_BC` false predicate lanes do not read memory but produce zero result lanes. VMI `gather` requires false
lanes to preserve passthru, so the `vsel` is semantically required, not an optimization artifact. `f16/b16/f8/i8`
gather, tail gather, non-contiguous layout, memref/gm source, and fallback through guarded scalar load or scratch are
future target-capability paths.

当前 `scatter` direct lowering 只在 VMI IR 携带显式 no-conflict proof 时启用：

```mlir
pto.vmi.scatter %v, %base[%indices], %mask {indices_unique}
  : !pto.vmi.vreg<64xf32>, !pto.ptr<f32, ub>,
    !pto.vmi.vreg<64xi32>, !pto.vmi.mask<64xpred>
```

`indices_unique` 的含义是：所有 active logical lanes 的 `%indices` 两两不同。这个 proof 可以来自
producer 的静态分析、前端语义或上游 canonicalization；VMI lowering 不从 runtime 值猜测它。direct
path 的其它限制与 gather 对齐：UB pointer destination、contiguous full physical chunks、32-bit value
element、i32 indices 和 b32 mask。没有 `indices_unique` 时，`vmi-to-vpto` 必须诊断，而不能直接发
`VSCATTER`，因为 `VSCATTER` 对重复 index 的 grant procedure 是目标相关/未定义的，不等价于 VMI
logical lane order。

`expand_load/compress_store` 表达 masked contiguous stream，不是 arbitrary indexed access：

```text
expand_load:
  k = 0
  for i in 0 .. N:
    if mask[i]:
      result[i] = base[idx + k]
      k += 1
    else:
      result[i] = passthru[i]

compress_store:
  k = 0
  for i in 0 .. N:
    if mask[i]:
      base[idx + k] = value[i]
      k += 1
```

Current direct `expand_load` lowering supports two paths. The first is the
degenerate all-active case:

```text
mask == all_true  =>  expand_load(base[idx], mask, passthru) == load(base[idx])
```

The accepted mask must be statically proven all active through
`pto.vmi.create_mask` with constant `active_lanes >= N`, or a dense all-true
`pto.vmi.constant_mask`. The result, passthru, and mask layouts must be
contiguous. Partial/tail chunks still need the same safe full-read proof as
ordinary `vmi.load`; otherwise the direct path reports `VMI-UNSUPPORTED`.

The second direct path covers one full 32-bit UB physical chunk with a runtime
mask:

```text
base'    = pto.addptr base, idx
indices  = pto.vusqz(zero_i32_carrier, mask)
gathered = pto.vgather2_bc base', indices, mask
result   = pto.vsel gathered, passthru, mask
```

It requires contiguous result/passthru/mask layout, 32-bit element type, b32
mask granularity and one full physical chunk. Multi-chunk runtime masks need a
cross-chunk prefix-count carry; f16/b16/f8/i8 need a gather packing contract.
Unsupported cases still require guarded load, scratch fallback, or diagnostic,
and must not be lowered as a plain full load.

Current direct `compress_store` lowering is intentionally narrower than the
surface semantics. It requires contiguous value/mask layout, exactly one full
physical chunk, and a UB `!pto.ptr` destination. The direct sequence is:

```text
store_base = pto.addptr base, idx
sqz = pto.vsqz value, mask
align0 = pto.init_align
align1 = pto.vstur align0, sqz, store_base, "POST_UPDATE"
pto.vstar align1, store_base
```

The paired `vstur` consumer is what makes the later VPTO LLVM emitter select
`VSQZ #st=1`; emitting `vsqz` without that store consumer is only register
compress. Full physical chunk is required in this first path because padding
mask lanes must not be squeezed into memory. Multi-chunk `compress_store`
needs cross-chunk compaction and SQZN/store-state planning; deinterleaved
layouts need logical lane order reconstruction before the store chain.

### Index And Address Contract

`!pto.vmi.vreg<Nxindex>` 是 logical index vector，不是 physical address vector。进入 VPTO 前，
index 必须按 target registry legalize 成目标支持的整数宽度：

```text
index legalization:
  choose target index bitwidth
  prove every lane value fits, or insert preserving extend/trunc/check sequence
  preserve signedness required by the consuming op
```

memory op 的 index 单位是 memref element，不是 byte。byte address 由 memref layout、element
size、base offset 和 lane index 共同计算：

```text
logical element offset -> memref affine/strided map -> byte address
```

`gather/scatter` 的 `%indices`、`expand_load/compress_store` 的 active-prefix offset、`iota` 生成
的 lane index 都必须在同一套 address unit 下解释。不能把 element index 直接当 byte offset，也
不能在没有 range proof 时把 `index` 静默截断成较窄整数。

`active_prefix_index(mask)` 返回当前 lane 之前的 active lane 数：

```text
idx[i] = popcount(mask[0 .. i))
```

因此 `expand_load/compress_store` active lane 使用 `base + idx[i]`。如果目标缺少 prefix-popcount
或 index-vector lowering，必须选择 index-buffer/guarded fallback，或报 `VMI-FALLBACK-RESOURCE`
/ `VMI-LAYOUT-CONTRACT`。

`tile_read/tile_write` 承接 transfer-style padding 和 multi-dimensional access semantics：

```mlir
%tile = pto.vmi.tile_read %view[%c0, %c0], %pad, %mask
  {logical_shape = [8, 8],
   permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
  : memref<8x8xf32, strided<[?, 1], offset: ?>>, f32,
    !pto.vmi.mask<64xpred> -> !pto.vmi.vreg<64xf32>

pto.vmi.tile_write %tile, %view[%c0, %c0], %mask
  {logical_shape = [8, 8],
   permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
  : !pto.vmi.vreg<64xf32>, memref<8x8xf32, strided<[?, 1], offset: ?>>,
    !pto.vmi.mask<64xpred>
```

`tile_read/tile_write` 只承接 memref memory semantics。producer 的 transfer-style read/write 如果作用在
tensor source/destination 上，必须在进入 VMI 前 bufferize 成 memref access plan，或退出 PTO
路线。tensor write-back style 语义是产生新 tensor，不是对 memref 的 memory effect；不能把它
伪装成 `pto.vmi.tile_write`。未处理的 tensor transfer 报 `VMI-TENSOR-BOUNDARY`。

`tile_read` invalid lane 的 result 必须等于 padding，不是后继 op 的 inactive lane。

`tile_read` lowering 必须先构造三个对象：

```text
validMask(result lane):
  logical lane is inside result shape
  and explicit transfer mask maps to true
  and source address is in bounds

paddingValue(result lane):
  scalar padding: same value for every invalid lane
  vector-element padding: select element by suffix coordinate
  broadcast/permuted padding: apply the same result-lane map as data

safeReadProof:
  proves the actual physical load footprint is safe-readable
  independent from validMask
```

`validMask=false` 只说明 result lane 应等于 padding，不说明该 lane 的 source address 可以被读。
因此 `tile_read` 的 preserving lowering 决策是：

```text
safeReadProof == full and validMask all-true:
  direct load

safeReadProof == full and validMask not all-true:
  loaded = full load
  pad = materialize paddingValue in result layout
  result = select(validMask, loaded, pad)

target has true masked/non-faulting load:
  loaded = masked load with inactive lanes not read
  pad = materialize paddingValue in result layout
  result = select(validMask, loaded, pad) unless inactive result is already padding

safeReadProof != full:
  split full-safe and partial paths, or
  fill scratch with paddingValue, guarded-copy only valid lanes, then load scratch, or
  use guarded scalar/vector fallback
```

First implementation stage note:

```text
The padding-preserving branches above are semantic requirements for the full
design, but they are not part of the first-stage VMI implementation. The first
stage may lower only all-valid direct reads, or physical-tail reads whose extra
lanes are outside the logical VMI value and remain unobservable. If invalid
logical lanes require transfer_read paddingValue materialization, true
masked/non-faulting load, scratch, or guarded fallback, lowering must stop with
the implementation diagnostic code VMI-UNSUPPORTED instead of emitting an
approximate full load.
```

如果所有 preserving paths 都因 target capability 或 option 被禁用，报 `VMI-MEMORY-ACCESS`，
payload 必须指出缺的是 unsafe partial `tile_read` padding-preserving path。

`tile_write` 没有 padding value，但有 write-valid mask：

```text
writeMask(source lane):
  logical lane is inside source shape
  and explicit transfer mask maps to true
  and destination address is in bounds
```

`writeMask=false` 的 lane 不能产生 memory effect。只有 full physical footprint safe-writable 且
writeMask all-true 时，才能使用 predicate-ignored store。partial write 必须使用 true masked
store、split/guarded fallback、scatter-like fallback，或报 `VMI-MEMORY-ACCESS`。
当前 direct `vmi.tile_write` 只覆盖 flat contiguous tail：最后一个 partial chunk 使用 prefix
valid-lane predicate 发 `vsts`，同样要求 value element width 能对应 `pto.mask<b8/b16/b32>`。
deinterleaved=2/4 tail 只有在能先完整 materialize 到 contiguous
chunks 时直接支持，padding-only materialized chunk 不发 store；带 transfer mask coordinate remap 的
tile write 仍必须走独立 access plan。

explicit transfer mask 的坐标属于 transfer access space，不一定等于 flattened result/source lane
坐标。non-minor-identity transfer 必须先做 predicate coordinate remap；缺少 remap capability 时，
diagnostic 必须点名 transfer mask coordinate remap，而不是泛化成普通 memory failure。

### Arithmetic And Conversion

VMI 不复用外部 elementwise arithmetic op。需要定义对应 VMI op：

| Semantic | VMI op |
|---|---|
| float binary | `pto.vmi.addf/subf/mulf/divf/minf/maxf` |
| float unary | `pto.vmi.negf/sqrt/exp/ln/relu` |
| integer binary | `pto.vmi.addi/subi/muli` |
| bitwise/shift | `pto.vmi.andi/ori/xori/not/shli/shrui` |
| fused multiply-add | `pto.vmi.fma` |
| float casts | `pto.vmi.extf/truncf` |
| bitcast | `pto.vmi.bitcast` |
| compare/select | `pto.vmi.cmpf/cmpi/select` |

Integer div/rem, arithmetic right shift, integer casts, int-float casts, and
index casts are intentionally not in the current VMI surface. They need
explicit signedness, rounding, saturation, overflow/remainder, and VPTO target
contracts before ODS ops are introduced.

producer constant 转成 `pto.vmi.constant`，包括 dense、splat 和 rank-0 logical vector。
constant 的 element type、shape、splatness 和 poison/undef 属性如果存在，必须保留到 VMI
constant attr；padding physical lane 仍按 VMI padding rule 处理，不能把 padding lane 当成用户
constant lane。

当前 VPTO direct lowering 只把 scalar broadcast 和 splat constant materialize 成
`pto.vdup`。这条路径与逐元素 op 一样要求 physical element width 能对应
`pto.mask<b8/b16/b32>`；其它 element type 或非 splat constant 必须先有明确的 materialization
contract，否则报 `VMI-UNSUPPORTED`。

VMI arithmetic op 必须保留原 `arith` op 的 numeric contract：

```text
floating point:
  fastmath flags
  rounding mode, if present
  NaN / signed-zero / inf behavior implied by flags

integer:
  signedness of div/rem/compare/extend
  overflow flags such as nsw/nuw when present
  truncation and extension width rules

compare/select:
  cmpf/cmpi predicate
  select condition mask granularity and layout
```

lowering 不能因为 VPTO 有更快指令就加强或放松这些属性。比如没有 fastmath 允许时，`fma`
不能拆成 `mulf + addf`，也不能把 `mulf + addf` 合成 `fma`；带 `nsw/nuw` 的 integer op
可以利用 flag 做优化，不带 flag 的 op 必须保持 wraparound/defined overflow 语义。

`pto.vmi.fma` 不能默认拆成 `mulf + addf`。`bitcast` 只有在当前 contiguous/deinterleaved
layout 下 bit grouping physically adjacent、且每个对应 physical chunk 的 logical bit
footprint 相同时才能 direct；padding bits 只能流向 result padding bits。group_slots bitcast
暂不复用这个规则，必须等 slot-wise bitcast contract 定义清楚后再支持。否则需要 layout
conversion、scratch materialization 或 target capability diagnostic。

当前 VPTO direct lowering 对逐元素算术、逻辑、比较和 select 还有一条共同硬约束：物理 element
width 必须能对应到 `pto.mask<b8/b16/b32>`。因此 VMI 语义层可以承载 `index` 或 `f64`
这类类型，但在没有独立 lowering contract 前，`vmi-to-vpto` 必须报 `VMI-UNSUPPORTED`，
不能让 OneToN conversion 或 residual gate 隐式失败。

这条共同约束不是唯一约束。某些目标 VPTO/VISA op 还有自己的 element type contract，
必须在 `vmi-to-vpto` preflight 中单独检查。当前 direct lowering 明确承诺：

```text
addf/subf/mulf: f16/bf16/f32
divf:           f16/f32
minf/maxf:      f16/bf16/f32
negf/absf:      f16/f32
sqrt/exp/ln:    f16/f32
relu:           f16/f32
absi:           signless/signed i8/i16/i32
cmpf:           f16/bf16/f32
cmpi:           signless/signed/unsigned i8/i16/i32
```

因此 bf16/f8 虽然可能是合法 VMI float-like type 且能 materialize b16/b8 predicate mask，
但只要目标 direct op 不承诺该 element type，`vmi-to-vpto` 就必须先报
`VMI-UNSUPPORTED`，直到定义对应 materialization 或 VPTO 目标能力。

当前 direct lowering 将 `pto.vmi.fma %lhs, %rhs, %acc` 映射为每个 physical part 上的
`pto.vmula %acc_part, %lhs_part, %rhs_part, %all_true_mask`。该路径只承诺 f16/bf16/f32
floating-point fused multiply-add；整数 multiply-accumulate、带 rounding/fastmath 变体或需要
不同 accumulator 精度的形式必须单独建模，不能复用这个 op 偷换语义。

### Permutation, Mask, Reduction, Channel

| Semantic | VMI op |
|---|---|
| static lane map | `pto.vmi.shuffle` |
| dynamic indexed lane map | `pto.vmi.permute` |
| logical interleave/deinterleave | `pto.vmi.interleave/deinterleave` |
| shape metadata change | `pto.vmi.shape_cast/reshape/transpose` |
| subvector update | `pto.vmi.slice/insert_slice/insert_element` |
| predicate logic | `pto.vmi.mask_and/or/xor/not` |
| prefix active index | `pto.vmi.active_prefix_index` |
| register compaction/expansion | `pto.vmi.compress/expand` |
| reduction/scan | `pto.vmi.reduction/scan` |
| contraction | `pto.vmi.contract/outerproduct` |
| channel split/merge | `pto.vmi.channel_split/channel_merge` |

`pto.vmi.shuffle` 表达完整 static lane map。当前 VPTO direct lowering 先识别 physical chunk
forwarding：每个 result physical chunk 的所有非 padding lanes 必须来自同一个 source chunk，
且 source lane number 等于 result lane number；result padding lanes 不参与证明，forward 过来的
物理 padding lanes 仍然不可观察。否则在每个 result physical chunk 都来自同一个 source chunk、
result chunk 没有 padding lane、且 source lane index 是 ASC/DESC 连续序列时，用 `pto.vci`
生成 index vector 并发 `pto.vselr`。任意非 affine permutation、以及需要 tail lane 重排但无法安全
materialize tail index vector 的场景，仍然需要通用 index-vector materialization、scratch fallback
或 target capability diagnostic。

`channel_split/channel_merge` 是 PTO-specific semantic op。它们表达用户按 channel 编程时的
多个 logical VMI values，不能降格成
`#pto.vmi.layout` kind。它们必须拥有 static shuffle 等价定义，canonicalization 可以双向进行：
识别出的 shuffle pattern 可以变成 channel op，channel op 也可以合法展开回 shuffle。
Direct lowering 还必须证明 physical group 完整；否则即使 logical shuffle 语义成立，也要报
target capability/materialization diagnostic，而不是让 OneToN pattern 在中途失败。

### Internal Layout Helpers

这些 op 只允许存在于 VMI lowering 的中间阶段，不能作为 VMI semantic surface，也不能残留到
physical VPTO 之后：

| Op | 语义 |
|---|---|
| `pto.vmi.ensure_layout` | data vreg layout-preserving conversion |
| `pto.vmi.ensure_mask_layout` | mask layout-preserving conversion |
| `pto.vmi.ensure_mask_granularity` | logical predicate-preserving granularity conversion |
| `pto.vmi.unpack` | layouted VMI value projection to physical VPTO parts |
| `pto.vmi.pack` | physical VPTO parts materialized as one layouted VMI value |

`active_prefix_index` 语义是：

```text
idx[i] = popcount(mask[0 .. i))
```

VMI surface 不暴露 VPTO `vusqz` 的无意义 source operand；需要 type/ABI carrier 时在
`vmi-to-vpto` late materialize。

当前直接 lowering 只覆盖 contiguous 单物理 chunk。这个 case 可以用 `pto.vusqz` 精确承接：
`vmi-to-vpto` 先 materialize 一个 zero vreg 作为 VPTO `vusqz` 的 source carrier，再把 VMI mask
作为 governing predicate 传入。多物理 chunk 需要把前一 chunk 的 active count carry 到后一 chunk；
deinterleaved layout 还需要按逻辑 lane 顺序重建 prefix，因此不能逐物理 part 独立发 `vusqz`。

`vmi.compress(source, mask)` 语义是按 logical lane order 保留 active source lane 并压缩到结果前缀。
当前直接 lowering 只覆盖 contiguous 单个 full physical chunk，可以用 `pto.vsqz(source, mask)` 承接。
partial/tail chunk 不能直接走 `vsqz`，因为 padding mask lane 如果为 true，padding source lane 可能被
压缩到可观察的 result 前缀。多物理 chunk 需要跨 chunk compaction；`compress_store` 还涉及
`VSQZ #st=1` 与 `VSTUR`/`SQZN` 的配对约束，不能由 register `compress` 自动推出。

`vmi.compress_store(value, base[idx], mask)` 语义是按 logical lane order 把 active lane 写成连续
memory stream。当前直接 lowering 只覆盖 contiguous、单个 full physical chunk 和 UB pointer
destination，并发出 `pto.vsqz -> pto.vstur POST_UPDATE -> pto.vstar` 的完整 store-state chain。非 full
chunk 暂不直接 lowering，因为 padding mask lane 可能被硬件 squeeze 成额外写出；multi-chunk 需要
跨 chunk active count 和 SQZN FIFO/VSTUR 配对计划。

`shape_cast/reshape/transpose` 必须区分 metadata change 和 lane movement：

```text
shape_cast / reshape:
  preserve row-major flattened lane order
  produce explicit result logical_shape attr

transpose / flat_transpose:
  changes logical lane order according to permutation
  must lower through shuffle/permute/layout conversion/direct transpose capability
```

这些 op 的 source/result shape、permutation 和 broadcast map 都是 op attrs。VMI lowering 不能从
producer defining op 或 side table 推断缺失 shape。

低 rank vector 到高 rank vector 的 broadcast 也不能当成 scalar broadcast 免费重物化。它必须
保存 broadcast map：

```text
result[indices] = source[broadcast_map(indices)]
```

只有 scalar-to-vector broadcast 可以按 consumer layout 任意重物化。

`iota` 是 lane index generation 的 VMI 表达：

```text
iota<NxT>(base, ASC):
  result[i] = base + i

iota<NxT>(base, DESC):
  result[i] = base - i
```

第一版 `iota` 的 `T` 跟随 VPTO `vci` 能承接的元素类型：integer 8/16/32 和 f16/f32。
可变 step 不是 surface op 语义的一部分；如果 producer 需要 `base + i * step`，应表达为
`iota(base=0) -> muli/vmi arithmetic -> addi/addf` 组合，或后续单独引入带 step 的 op。
tail physical chunk 的 padding lane 可以承接 iota 的自然延续值，但这些 lane 不是 logical lane；
后续 memory/mask/reduction 等有外部效果的 consumer 必须继续按 valid logical lane 保护。
deinterleaved layout 下的 physical part 需要 strided index materialization：

```text
part p contains logical lanes p, p + factor, p + 2 * factor, ...
ASC value = base + p + factor * local_lane
DESC value = base - p - factor * local_lane
```

因此 direct `vci` 只覆盖 contiguous full-chunk path；deinterleaved path 必须额外物化
`vci(0) * factor + base +/- p`，不能误降成每个 part 内连续的 `vci(base + p)`。当前 lowering
按 physical part 生成 `vci(0) + vmuls(factor) + vadds/vdup/vsub` 序列；padding/tail chunk
仍然需要独立的 padding-safe materialization plan。

`slice/insert_slice` 都按 logical lane order 定义，不读取或写入 padding lane：

```text
slice(offset, size, stride):
  result[j] = source[offset + j * stride]

insert_slice(offset, stride):
  result = dest
  result[offset + j * stride] = update[j]

insert_element(pos):
  result = dest
  result[pos] = scalar
```

`reduction/scan` 的 logical iteration 只覆盖 active logical lanes，padding lanes 不参与：

```text
reduction(op, init, value, mask):
  acc = init
  for i in 0 .. N:
    if mask is absent or mask[i]:
      acc = op(acc, value[i])
  result = acc

scan(op, init, value, mask):
  acc = init
  for i in 0 .. N:
    if mask is absent or mask[i]:
      acc = op(acc, value[i])
      result[i] = acc
    else:
      result[i] = passthru_or_identity
```

Current direct reduction support starts with integer add:

```mlir
%r = pto.vmi.reduce_addi %value, %init, %mask
  : !pto.vmi.vreg<64xi32>, !pto.vmi.vreg<1xi32>,
    !pto.vmi.mask<64xpred> -> !pto.vmi.vreg<1xi32>

%rf = pto.vmi.reduce_addf %value, %init, %mask {reassoc}
  : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<1xf32>,
    !pto.vmi.mask<64xpred> -> !pto.vmi.vreg<1xf32>

%rmax = pto.vmi.reduce_maxf %value, %init, %mask
  : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<1xf32>,
    !pto.vmi.mask<64xpred> -> !pto.vmi.vreg<1xf32>

%rmin = pto.vmi.reduce_minf %value, %init, %mask
  : !pto.vmi.vreg<128xf16>, !pto.vmi.vreg<1xf16>,
    !pto.vmi.mask<128xpred> -> !pto.vmi.vreg<1xf16>
```

`reduce_addi` preserves integer wraparound addition semantics. The direct
lowering requires contiguous layout, full 32-bit source physical chunks,
matching mask chunks, and one rank-0 init/result chunk. It emits `pto.vcadd`
for each masked source chunk, then serially accumulates each chunk result into
the rank-0 accumulator with `pto.vadd` under a `PAT_VL1` predicate. Padding
source lanes are rejected instead of being allowed to participate.

`reduce_addf` is legal only with an explicit `{reassoc}` contract because the
ISA documents pair-wise FP reduction order. The direct lowering supports only
f32, contiguous layout, full source physical chunks, matching b32 mask chunks,
and one rank-0 init/result chunk. It uses the same per-chunk `vcadd` plus
serial `PAT_VL1 vadd` accumulation shape. Without `{reassoc}`, the verifier
rejects the op instead of silently changing ordered floating-point semantics.

`reduce_maxf` and `reduce_minf` preserve VPTO-compatible floating-point min/max
reduction semantics. Direct lowering supports f16/f32, contiguous layout, full
source physical chunks, matching mask chunks, and one rank-0 init/result chunk.
For each physical source chunk, lowering emits `pto.vcmax` or `pto.vcmin`.
The chunk result's lowest lane is then accumulated into the rank-0 accumulator
with `pto.vmax` or `pto.vmin` under a `PAT_VL1` predicate. The index value that
`vcmax/vcmin` writes to the second lane is intentionally not part of the VMI op
result and is discarded by only observing lane 0. Inactive lane identities,
signed zero handling, and NaN behavior follow the underlying `vcmax/vcmin` and
`vmax/vmin` VPTO instructions. Padding source lanes are rejected, because the
logical reduction must not allow padding to become an inactive-lane identity or
a NaN-producing participant.

lowering 可以选择 VPTO reduction/scan 指令、tree decomposition、scratch memory 或 scalarized
ordered fallback，但必须保持 numeric contract。没有目标能力时使用 `VMI-ELEMENT-TYPE` 或
`VMI-LAYOUT-CONTRACT`，不能让未 lower 的逻辑向量 op 残留到 VPTO。

`contract/outerproduct` 在 VMI 中保留 indexing maps、iterator types、accumulator、mask 和
element type，并且不允许绕过 VMI 直接回到其它向量 IR。如果目标有直接 matrix/vector contract
能力，lower 到直接 VPTO sequence；否则按 iterator space 分解成 VMI arithmetic +
reduction/scan，再走普通 VMI lowering。只有当 element type、accumulator 精度或 iterator
semantics 无法由目标表达时，才报 target capability diagnostic。

如果 producer 的 extract-like operation 结果仍是 logical vector，应表达成 `pto.vmi.slice`、
`pto.vmi.shuffle` 或 `pto.vmi.shape_cast`。如果结果是 scalar，则属于 vector-to-scalar boundary，
不进入 VMI vector path，也不产生 `pto.vmi.extract`：

```text
VMI-SCALAR-EXTRACT-BOUNDARY
```

## End-To-End Examples

### f16 Widen Add Store

Semantic VMI：

```mlir
%a = pto.vmi.load %A[%i]
  : memref<?xf16> -> !pto.vmi.vreg<128xf16>
%w = pto.vmi.extf %a
  : !pto.vmi.vreg<128xf16> -> !pto.vmi.vreg<128xf32>
%s = pto.vmi.addf %w, %bias
  : !pto.vmi.vreg<128xf32>
pto.vmi.store %s, %C[%i]
  : !pto.vmi.vreg<128xf32>, memref<?xf32>
```

Layout-assigned VMI：

```mlir
%a = pto.vmi.load %A[%i]
  : memref<?xf16> -> !pto.vmi.vreg<128xf16, #pto.vmi.layout<contiguous>>
%w = pto.vmi.extf %a
  : !pto.vmi.vreg<128xf16, #pto.vmi.layout<contiguous>>
    -> !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>
%s = pto.vmi.addf %w, %bias
  : !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>
pto.vmi.store %s, %C[%i]
  : !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>, memref<?xf32>
```

Physical lowering 可以生成 EVEN/ODD `vcvt`、两路 `vadd`，并在 store sink 使用 interleave
store 或显式 layout conversion。

### f8 To f32

```mlir
%a = pto.vmi.load %A[%i]
  : memref<?xf8> -> !pto.vmi.vreg<256xf8>
%w = pto.vmi.extf %a
  : !pto.vmi.vreg<256xf8> -> !pto.vmi.vreg<256xf32>
%s = pto.vmi.addf %w, %b
  : !pto.vmi.vreg<256xf32>
pto.vmi.store %s, %C[%i]
  : !pto.vmi.vreg<256xf32>, memref<?xf32>
```

layout assignment 可把 `%w/%s` 设为 `#pto.vmi.layout<deinterleaved = 4>`。contiguous store 必须使用
已验证的 layout sink 或先 materialize contiguous representation，不能把 p0/p1/p2/p3 part 当成连续内存写出。

### Block-Strided Tile Read

```mlir
%tile = pto.vmi.tile_read %view[%c0, %c0], %pad, %mask
  {logical_shape = [8, 8],
   permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
  : memref<8x8xf32, strided<[?, 1], offset: ?>>, f32,
    !pto.vmi.mask<64xpred> -> !pto.vmi.vreg<64xf32>
```

如果 access plan 证明每 row 是 32B contiguous block，row 间 stride 可落到 ISA stride 字段，
且 mask block-uniform，lowering 可以选择 `vsldb`。如果 padding 非零，仍需在 load 后用
valid mask 修正 invalid lane。

## Risk Closure Matrix

| 风险 | 设计闭环 | 测试出口 |
|---|---|---|
| producer 直接绕过 VMI 生成 physical VPTO | VMI Producer Boundary Contract + Verifier Gates | `vmi_producer_boundary.mlir`, `vmi_pipeline_hard_gates.mlir` |
| arith numeric contract 被 VPTO 快速路径改写 | fastmath/rounding/overflow/cmp predicate preservation | `vmi_arith_numeric_contract.mlir` |
| layout 设计泛化失控 | closed `contiguous/deinterleaved=2/4` layout set + source contract | `vmi_f16_ext_add_store_deinterleaved2.mlir`, `vmi_f8_ext_add_store_deinterleaved4.mlir` |
| layout assignment 局部贪心导致控制流/多 use 错误 | region/SCC constraint solver + deterministic tie-break | `vmi_layout_assignment_constraint_solver.mlir`, `vmi_cf_and_call_layout_boundary.mlir` |
| 1:N physicalization arity 漂移 | Physical Arity helper + hard gate | `vmi_physical_arity_non_full_deinterleaved.mlir` |
| `deinterleaved=4` materialization 错 lane | registered preserving materialization path | `vmi_ensure_layout_materialization_contract.mlir` |
| mask granularity 过早固化 | surface `mask<Nxpred>` + consumer-driven granularity assignment | `vmi_mask_granularity_width_change.mlir` |
| non-scalar broadcast / transpose 被当成 metadata | explicit broadcast map and lane-movement semantics | `vmi_shape_broadcast_semantics.mlir` |
| transfer padding / OOB read 写成 full load/store | `validMask` / `paddingValue` / `safeReadProof` / `writeMask` decision tree | `vmi_tile_read_padding_decision_tree.mlir`, `vmi_tile_write_oob_no_effect.mlir` |
| index/address 单位或宽度被误用 | index/address legalization contract | `vmi_index_address_legalization.mlir` |
| reduction/scan/contract 回退成 residual logical-vector op | VMI semantic op + direct/decompose/scratch lowering contract | `vmi_reduction_scan_contract_coverage.mlir` |
| shape 信息依赖 hidden side table | flat VMI value + shape-sensitive op attrs | `vmi_shape_broadcast_semantics.mlir`, `vmi_pipeline_hard_gates.mlir` |
| fallback 缺资源时退化成残缺 lowering | explicit fallback resource contract + `VMI-FALLBACK-RESOURCE` | `vmi_fallback_resource_diagnostics.mlir` |
| tensor/debug/scalar boundary 混入 VMI | explicit boundary diagnostics | `vmi_tensor_transfer_boundary.mlir`, `vmi_debug_boundary.mlir`, `vmi_extract_boundary.mlir` |

## Diagnostics

| Code | 场景 |
|---|---|
| `VMI-SCALAR-EXTRACT-BOUNDARY` | scalar lane extract 不是 VMI vector op，必须在进入 VMI 前消除或退出 PTO 路线 |
| `VMI-SCALABLE-VECTOR` | scalable vector 未在进入 VMI 前 specialize 成固定 logical lane count |
| `VMI-ELEMENT-TYPE` | target registry 缺 storage/compute/convert capability |
| `VMI-LAYOUT-CONTRACT` | VMI layout、mask granularity 或控制流/调用边界约束冲突 |
| `VMI-MEMORY-ACCESS` | access plan 无 direct/fallback path |
| `VMI-LAYOUT-CONTRACT` | layout conversion 或 sink 未被 target registry 支持 |
| `VMI-FALLBACK-RESOURCE` | scratch、guard、index buffer 或 fallback index width 资源不可用 |
| `VMI-TENSOR-BOUNDARY` | tensor transfer 必须在进入 VMI 前 bufferize 或退出 PTO 路线 |
| `VMI-DEBUG-BOUNDARY` | debug op 必须在进入 VMI 前消费、剥离或退出 PTO 路线 |
| `VMI-PASS-INVARIANT` | pipeline hard gate 被破坏，例如 hidden side table、残留 conversion cast 或 layout 缺失 |
| `VMI-RESIDUAL-OP` | physicalization 后仍有非法 VMI op/type 或 helper |

diagnostic payload 至少包含 source op、semantic reason、failed contract、available paths、
missing capability 或 disabled fallback option。

## Implementation Plan

具体文件布局、Slice 切分、ODS/type/op/pass/test 落地步骤见
`docs/designs/vmi-implementation-manual.md`。本节只保留高层任务顺序。

1. 定义 `!pto.vmi.vreg<NxT>`、`!pto.vmi.vreg<NxT, layout>`、
   `!pto.vmi.mask<Nxpred>`、`!pto.vmi.mask<NxG, layout>`。
2. 定义 layout 目录：`#pto.vmi.layout<contiguous>`、
   `#pto.vmi.layout<deinterleaved = 2>`、
   `#pto.vmi.layout<deinterleaved = 4>`，
   并实现统一 lane-map / physical-arity helper。
3. 定义 VMI semantic op families：construction、memory、arith、conversion、mask、
   permutation、active-prefix、compress/expand、channel split/merge、reduction/scan/contract。
4. 实现 VMI producer boundary verifier，禁止 producer 直接生成 physical VPTO 或依赖 hidden state。
5. 实现 `vmi-layout-assignment`，包含 op transfer function、cost model、mask granularity
   conversion、control-flow join。
6. 实现 VMI memory lowering：access plan、safe-read/write proof、tile padding materialization、
   transfer mask coordinate remap、masked/guarded/scratch fallback。
7. 实现 `vmi-to-vpto` 1:N type conversion，包含 `pack/unpack` materialization 和 structural
   conversion。
8. 加 target element-type / layout-sink / ISA contract / fallback resource registry。
9. 加 VMI hard gate verifier：覆盖 VMI producer boundary、`vmi-layout-assignment`、
   `vmi-to-vpto` 后的残留 op/type、layout、mask granularity、conversion cast 和 hidden-state
   invariant。
10. 加 VMI diagnostic code registry 和 lit tests。

## Test Checklist

1. `vmi_f16_ext_add_store_deinterleaved2.mlir`
   - `extf` 后 result 是 `vreg<128xf32, deinterleaved=2>`，store 保持 contiguous logical order。
2. `vmi_f8_ext_add_store_deinterleaved4.mlir`
   - `deinterleaved=4` p0/p1/p2/p3 不被误写成 contiguous memory。
3. `vmi_non_full_tile_padding_lanes.mlir`
   - `vreg<100xf32>` padding lane 不可观察。
4. `vmi_mask_granularity_width_change.mlir`
   - surface `mask<Nxpred>` 被不同 width consumer 使用时，正确生成 `mask<Nxb16>` /
     `mask<Nxb32>` 并保持 data layout。
5. `vmi_control_flow_layout_join.mlir`
   - `scf.if/scf.for` layouted VMI type join 稳定。
6. `vmi_tile_read_padding_safe_footprint.mlir`
   - full physical load unsafe 时不偷读 invalid lane。
7. `vmi_block_strided_rows_vsldb.mlir`
   - `tile_read/tile_write` 识别 32B block rows，并拒绝 per-lane mask direct path。
8. `vmi_active_prefix_index_compress.mlir`
   - arbitrary mask compaction 使用 logical prefix order。
9. `vmi_extract_boundary.mlir`
   - scalar extract 输出 `VMI-SCALAR-EXTRACT-BOUNDARY`。
10. `vmi_channel_split_merge_semantic_op.mlir`
    - interleaved channel data 按用户语义拆成多个 VMI values，再通过 merge 写回。
11. `vmi_producer_boundary.mlir`
    - producer boundary 后只有 VMI semantic op/type，不出现 physical VPTO 或 hidden-state 依赖。
12. `vmi_mask_threading.mlir`
    - region-style mask 被 thread 到 masked VMI op 或 `vmi.select` merge，不残留 region mask。
13. `vmi_gather_scatter_memory_semantics.mlir`
    - inactive gather/scatter lane 不读写内存，scatter duplicate-index case 不走非法 direct path。
14. `vmi_reduction_scan_contract_coverage.mlir`
    - reduction/scan/contract 不回退成 residual logical-vector op，按 VMI lowering contract 处理。
15. `vmi_cf_and_call_layout_boundary.mlir`
    - `cf.br/cond_br` block arguments 和 internal call signatures 选择稳定 layout，external ABI 不泄露 layout。
16. `vmi_iota_bitcast_insert_extract_coverage.mlir`
    - lane index、bitcast、vector-result extract-like 和 insert-like 语义都有 VMI 承接。
17. `vmi_memory_view_normalization.mlir`
    - producer-specific vector element view 先规范化为 element view 和 access plan。
18. `vmi_debug_boundary.mlir`
    - debug-only op 不进入 VMI；未被 producer 消费时输出 `VMI-DEBUG-BOUNDARY`。
19. `vmi_arith_numeric_contract.mlir`
    - VMI arithmetic constant、fastmath、cmp predicate、integer signedness/overflow flags 保真。
20. `vmi_shape_broadcast_semantics.mlir`
    - `shape_cast/reshape` 只改 explicit op shape attrs，`transpose/flat_transpose` 和非 scalar broadcast 保持 lane map 语义且不依赖 shape side table。
21. `vmi_physical_arity_non_full_deinterleaved.mlir`
    - 非整 tile 下 `contiguous/deinterleaved=2/4` 的 physical value 个数和 valid lane map 一致。
22. `vmi_ensure_layout_materialization_contract.mlir`
    - `ensure_layout` 保持 logical lane 值，`deinterleaved=4` 只使用 registry 证明过的 materialization path。
23. `vmi_tile_read_padding_decision_tree.mlir`
    - safe full-read + non-all-true valid mask 生成 padding materialization + select；unsafe path 不读 invalid address。
24. `vmi_tile_write_oob_no_effect.mlir`
    - `tile_write` 的 writeMask=false lane 没有 memory effect，不被 lower 成 predicate-ignored store。
25. `vmi_transfer_mask_coordinate_remap.mlir`
    - non-minor-identity `tile_read/tile_write` 的 explicit mask 先映射到 result/source logical lane。
26. `vmi_tile_read_vector_element_padding.mlir`
    - vector-element padding 按 suffix coordinate 展开，invalid lane 使用对应 padding element。
27. `vmi_index_address_legalization.mlir`
    - `vreg<Nxindex>`、gather/scatter indices、active-prefix offset 使用 element units 且宽度合法。
28. `vmi_fallback_resource_diagnostics.mlir`
    - scratch、guarded fallback、index-buffer fallback 缺资源时输出 `VMI-FALLBACK-RESOURCE`。
29. `vmi_tensor_transfer_boundary.mlir`
    - tensor transfer-style producer op 不伪装成 VMI memory op，未 bufferize 时输出 `VMI-TENSOR-BOUNDARY`。
30. `vmi_pipeline_hard_gates.mlir`
    - 各 pass 边界拒绝残留 VMI helper/unrealized cast/hidden state，且 final lowering 不残留 VMI op/type。
31. `vmi_layout_assignment_constraint_solver.mlir`
    - 多 use、rematerializable producer、control-flow join、layout conversion cost 冲突时选择稳定 layout 或输出精确 diagnostic。
