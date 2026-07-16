# VMI ISA 适配 Tile Op 设计

> **状态：** Draft
>
> 本文定义 VMI ISA 面向 Tile Op 语义体系的目标适配方案。目标 ISA 的逐项语义见
> [`docs/isa/vmi-isa`](../isa/vmi-isa/00-architecture-overview.md)。本文聚焦适配原则、
> 关键类型变化、shape/group 表达方式和指令映射决策，不表示相关 ODS、验证器和
> lowering 已全部实现。

## 1. 背景与目标

现有 VMI 使用一维 `vreg`、独立 `mask` 类型和部分 VMI 专用指令名表达逻辑向量
计算。Tile Op 则以二维 tile 形状和一套统一的 `t*` 操作名描述相同或相近的计算。
两套表面同时存在时，会产生以下问题：

- 同一个逻辑操作在 VMI 和 Tile Op 中使用不同名称；
- `group` 隐含在属性中，不能直接从类型看出行边界；
- 一维向量、分组向量和 mask 使用不同类型规则；
- reshape 与按位 reinterpretation 分散在不同操作中；
- 名称相近但语义不同的操作容易被错误合并，例如 `dhist` 与 `thistogram`。

本设计的目标是：

1. VMI 的寄存器值统一使用 `!pto.vmi.tilereg<MxNxdtype>`；
2. 与 Tile Op 具有相同功能的 VMI 指令使用相同的 `t*` 名称；
3. 分组关系直接编码在二维 shape 中，不再依赖 `group` 属性；
4. 一维与二维寄存器视图、同 bit-size 的 dtype reinterpretation 统一由
   `treshape` 表达；
5. mask 也使用 `tilereg`，其 dtype 固定为 `i1`；
6. 没有同语义 Tile Op 的指令保留 VMI 名称，并明确记录不映射的原因。

本设计不把 VMI 值改成 Tile Op 的 `tile_buf`。VMI 的 `tilereg` 仍然是寄存器驻留的
SSA 值；复用 Tile Op 名称只表示两者共享同一份可观察操作语义。

## 2. 映射决策原则

每条 VMI 指令按以下顺序判断：

1. **比较功能，而不是只比较名称。** 核对输入角色、结果含义、逐元素或逐行行为、
   mask、dtype 和数值规则。
2. **完全同义时直接使用 Tile Op 名称。** 例如 `vadd -> tadd`、`vci -> tci`。
3. **核心语义相同但类型表达不同且可精确定义时，使用受约束的 Tile Op 重载。**
   例如 widening `vmull` 使用结果 dtype 为 64-bit 的 `tmul`；
   `vinterpret_cast` 使用总 bit-size 不变的 `treshape`。
4. **原语义由 group 属性决定时，先把 group 转换为二维 shape，再选择对应 row op。**
   例如 grouped `vcadd` 变成 `MxN -> Mx1` 的 `trowsum`。
5. **仅能用多条操作组合、数值行为不同或操作数角色不同，不视为同义。**
   这类操作保留 VMI 名称，例如 `vmula` 和 `dhist`。

因此，映射结果分为四类：

| 类别 | 判定 | 处理方式 |
|---|---|---|
| 直接同义映射 | 操作数角色和可观察语义一致 | 改用 Tile Op 名称 |
| 受约束的语义复用 | 核心语义一致，需要额外 dtype/bit-size 规则 | 使用 Tile Op 名称并定义重载约束 |
| shape 驱动映射 | 原 VMI 的 group/broadcast/reduce 信息需要显式化 | 先改成二维 shape，再映射到 row op |
| VMI-only | 没有单条同语义 Tile Op | 保留 VMI 指令名 |

## 3. 统一类型设计

### 3.1 数据 tile register

VMI 唯一的聚合值类型为：

```mlir
!pto.vmi.tilereg<MxNxdtype>
```

其中 `M`、`N` 是正整数静态维度，元素按 row-major 逻辑顺序排列：

```text
flat_index(m, n) = m * N + n
```

一维逻辑向量使用规范形式 `1xL`：

```mlir
!pto.vmi.tilereg<1x128xf16>
```

分组向量使用二维形式 `MxN`：

```mlir
!pto.vmi.tilereg<8x16xf16>
```

该类型不同于 `!pto.tile_buf`：它没有内存位置、valid region、buffer layout 或 DMA
含义，只表达 VMI 寄存器中的逻辑 tile 值。

### 3.2 Mask tile register

删除独立的 VMI mask 类型。mask 使用相同的 `tilereg`，dtype 固定为 `i1`：

```mlir
!pto.vmi.tilereg<1x128xi1>
!pto.vmi.tilereg<8x16xi1>
```

控制某个数据 tile 的 mask 必须具有相同的 `MxN` shape。比较操作返回同 shape 的
`i1` tile，不再使用 predicate granularity 后缀。

### 3.3 合法 shape

适配后的 shape 保留当前 VMI 逻辑 `vreg` 长度和 group 规则，不额外引入物理寄存器
宽度限制：

| 用途 | 合法 shape | 约束 |
|---|---|---|
| 普通一维数据或 mask | `1xL` | `L > 0` |
| 普通二维数据或 mask | `MxN` | `M > 0` 且 `N > 0` |
| grouped 数据或 mask | `MxN` | `L = M * N`，等价于 group 数 `M` 整除原逻辑长度 `L` |
| 每组一个标量 | `Mx1` | `M > 0` |

当前 VMI group 规则本质上是“正 group 数整除逻辑 lane 数”，因此设计不把合法
group 硬编码为 `{1, 2, 4, 8}`，也不要求 `L` 是某个固定物理 lane 数的倍数。例如：

```text
vreg<128xf32>, group = 8  -> tilereg<8x16xf32>
vreg<512xf16>, group = 2  -> tilereg<2x256xf16>
vreg<384xf32>, group = 6  -> tilereg<6x64xf32>
```

各指令原有的 dtype、目标架构、地址空间和特殊 shape 限制仍需单独验证。

## 4. `treshape`：统一 shape 与 bit view 转换

`pto.vmi.treshape` 是 `tilereg` 之间唯一的 view 转换操作，可以只改变 shape，也可以
同时改变 dtype。它始终保持 row-major bit sequence 不变，并要求源和结果总 bit-size
相同：

```text
Msrc * Nsrc * bitwidth(Tsrc) == Mdst * Ndst * bitwidth(Tdst)
```

### 4.1 一维与二维互转

```mlir
%rows = pto.vmi.treshape %flat
    : !pto.vmi.tilereg<1x128xf32>
      -> !pto.vmi.tilereg<8x16xf32>

%flat_again = pto.vmi.treshape %rows
    : !pto.vmi.tilereg<8x16xf32>
      -> !pto.vmi.tilereg<1x128xf32>
```

shape-only 形式要求元素总数相同，是一维 consumer 与二维 grouped consumer 之间的
标准连接方式。

### 4.2 dtype reinterpretation

原 `bitcast` / `vinterpret_cast` 统一映射成 dtype-changing `treshape`：

```mlir
%bits = pto.vmi.treshape %source
    : !pto.vmi.tilereg<8x16xf32>
      -> !pto.vmi.tilereg<8x16xi32>
```

该操作只重新解释 bit，不做数值转换。需要数值转换时使用 `tcvt`。predicate tile 只
允许在 `i1` dtype 下进行等元素数 shape 转换，不能通过 `treshape` reinterpret 成数据
dtype。

## 5. Group 语义适配

canonical VMI surface 删除 `group` 属性，二维 shape 直接表达 group：

- `M` 是 group 数，即行数；
- `N` 是每组 lane 数，即列数；
- row reduction 消费 `MxN` 并返回 `Mx1`；
- row expansion 消费 `Mx1` 并返回 `MxN`；
- grouped load/store/mask 使用相同的 `MxN` 行结构；
- 一维生产者或消费者通过 `treshape` 与 grouped op 连接。

例如 grouped reduction：

```mlir
%rows = pto.vmi.treshape %flat
    : !pto.vmi.tilereg<1x128xf32>
      -> !pto.vmi.tilereg<8x16xf32>
%mask = pto.vmi.create_group_mask %active_per_row
    : index -> !pto.vmi.tilereg<8x16xi1>
%sum = pto.vmi.trowsum %rows, %mask {reassoc}
    : !pto.vmi.tilereg<8x16xf32>, !pto.vmi.tilereg<8x16xi1>
      -> !pto.vmi.tilereg<8x1xf32>
```

这种表达方式使 group 边界成为类型契约，避免 consumer 再从属性推导行结构。

## 6. 指令映射

canonical 名称仍位于 `pto.vmi` namespace。例如 `vadd -> tadd` 后的操作名是
`pto.vmi.tadd`，而不是把该操作直接替换成操作 `tile_buf` 的 `pto.tadd`。

### 6.1 直接同义映射

| 当前 VMI 名称 | 适配后 VMI 名称 | 映射依据 |
|---|---|---|
| `vload`, `vstore` | `tload`, `tstore` | register 与 UB 之间的 load/store 语义 |
| `vadd`, `vsub`, `vmul`, `vdiv`, `vmax`, `vmin` | `tadd`, `tsub`, `tmul`, `tdiv`, `tmax`, `tmin` | 同 shape 逐元素算术 |
| `vabs`, `vneg`, `vrelu`, `vexp`, `vln`, `vsqrt` | `tabs`, `tneg`, `trelu`, `texp`, `tlog`, `tsqrt` | 同 shape 一元计算 |
| `vand`, `vor`, `vxor`, `vnot`, `vshl`, `vshr` | `tand`, `tor`, `txor`, `tnot`, `tshl`, `tshr` | 同 shape 位运算和移位 |
| `vadds`, `vmuls`, `vmaxs`, `vmins`, `vshls`, `vshrs` | `tadds`, `tmuls`, `tmaxs`, `tmins`, `tshls`, `tshrs` | tile-scalar 逐元素计算 |
| `vsel` | `tsel` | predicate 逐元素选择 |
| `iota`, `vci` | `tci` | 连续索引序列生成 |
| `cmpf`, `cmpi`, `vcmp` | `tcmp` | tile-tile 比较并返回同 shape `i1` tile |
| `vcmps` | `tcmps` | tile-scalar 比较 |
| `vcvt` | `tcvt` | 同 shape 数值类型转换 |
| `vlrelu` | `tlrelu` | leaky-ReLU |
| `vaxpy` | `taxpy` | `acc + x * scalar` |
| `vprelu` | `tprelu` | 每元素 slope 的 PReLU |
| `vgather`, `vgatherb`, `vscatter` | `tgather`, `tgatherb`, `tscatter` | 对应的索引 gather/scatter 语义 |
| `vintlv`, `vdintlv` | `tinterleave`, `tdeinterleave` | 对应的寄存器 interleave/deinterleave 语义 |

### 6.2 shape 驱动映射

| 当前 VMI 形式 | 适配后 VMI 形式 | shape 契约 |
|---|---|---|
| scalar `vbrc` | `texpands` | `T -> MxNxT` |
| grouped `vbrc` | `trowexpand` | `Mx1xT -> MxNxT` |
| `vcadd` | `trowsum` | `MxNxT -> Mx1xT` |
| `vcmax` | `trowmax` | `MxNxT -> Mx1xT` |
| `vcmin` | `trowmin` | `MxNxT -> Mx1xT` |

这些映射不是简单改名：原来由 `group` 或 compact value 隐式表达的信息必须先转成
`MxN` / `Mx1` 类型关系。

### 6.3 受约束的语义复用

#### `vexpdif` 不直接映射为 `trowexpandexpdif`

VMI `vexpdif` 的目标 tilereg 契约为：

```mlir
%result = pto.vmi.vexpdif %x, %max, %mask
    : !pto.vmi.tilereg<MxNxT>, !pto.vmi.tilereg<MxNxf32>,
      !pto.vmi.tilereg<MxNxi1> -> !pto.vmi.tilereg<MxNxf32>
```

其中 `T` 为 `f16` 或 `f32`，逐位置计算：

```text
result[m, n] = exp(f32(x[m, n]) - max[m, n])
```

该契约与 Tile Op `trowexpandexpdif` 存在三项关键差异：

- VMI `max` 是与 `x` 同 shape 的逐 lane 值，不保证每行相同；
- VMI 允许 `x=f16`、`max/result=f32`，而当前 Tile Op 要求输入和结果 dtype 相同；
- VMI 接受任意同 shape predicate 和 `pmode`，Tile Op 形式没有等价的任意 mask
  operand。

因此 canonical VMI 名称继续保留为 `vexpdif`。只有同时满足以下条件时，才允许把
特例作为优化复用 Tile Op `trowexpandexpdif`：

1. `x`、`max` 和结果均为 `f32`；
2. 能证明 `max[m,n] = row_max[m,0]`，即 `max` 直接来自逐行广播；
3. mask 为全有效，或是能由 Tile Op valid shape 精确表达的连续 prefix；
4. inactive position 的行为与 VMI `pmode` 完全一致。

例如下面的 VMI 序列可作为候选匹配：

```mlir
%max_full = pto.vmi.trowexpand %row_max
    : !pto.vmi.tilereg<Mx1xf32> -> !pto.vmi.tilereg<MxNxf32>
%result = pto.vmi.vexpdif %x, %max_full, %mask
    : !pto.vmi.tilereg<MxNxf32>, !pto.vmi.tilereg<MxNxf32>,
      !pto.vmi.tilereg<MxNxi1> -> !pto.vmi.tilereg<MxNxf32>
```

这是一项受约束优化，不是 `vexpdif -> trowexpandexpdif` 的指令重命名。不能证明上述
条件时，仍保留 `vexpdif`；也不能仅因 `tsub + texp` 可计算相同代数表达式就视为存在
同语义单 Tile Op，因为融合计算的舍入和异常值行为可能不同。

#### `vmull -> tmul`

`vmull` 作为 widening `tmul` 重载，不保留独立名称：

```mlir
%wide = pto.vmi.tmul %lhs, %rhs, %mask
    : !pto.vmi.tilereg<MxNxi32>, !pto.vmi.tilereg<MxNxi32>,
      !pto.vmi.tilereg<MxNxi1> -> !pto.vmi.tilereg<MxNxi64>
```

约束如下：

- 两个输入 shape 和 dtype 相同；
- 输入为 `i32`/`ui32`，结果为对应的 `i64`/`ui64`；
- 输入与结果的 `MxN` shape 相同；
- 结果 signedness 与输入一致。

#### `bitcast` / `vinterpret_cast -> treshape`

这两类操作统一使用第 4 节的 `treshape`，以“总 bit-size 相同”作为 verifier 契约。

#### `create_mask` / `create_group_mask` 保留为 VMI predicate 指令

Tile Op 没有与“根据运行时 active lane 数生成连续 prefix predicate”直接同义的单条
指令，因此 `create_mask` 和 `create_group_mask` 不改名，也不映射为 `texpands` 或
`tfillpad`：

```mlir
%flat_mask = pto.vmi.create_mask %active_lanes
    : index -> !pto.vmi.tilereg<1xLxi1>

%group_mask = pto.vmi.create_group_mask %active_per_row
    : index -> !pto.vmi.tilereg<MxNxi1>
```

一维形式生成一个 prefix mask：

```text
mask[0, n] = n < active_lanes
```

二维形式在每一行重复相同的 prefix 规则：

```text
mask[m, n] = n < active_per_row
```

结果 shape 取代原 `num_groups`、`group_size` 等分组信息。`texpands` 的语义是把一个
scalar value 广播到整个目标 tile，只适合映射 scalar `vbrc`；它既不接收
`active_lanes` 来生成动态 prefix，也不是 predicate 专用操作。

#### `vhist -> thistogram`

只有与 Tile Op row-wise histogram 更新契约一致的 `vhist` 映射为 `thistogram`。该
映射不能扩展到完整分布直方图 `dhist`。

### 6.4 VMI-only 指令

| 保留名称 | 语义 | 不映射原因 |
|---|---|---|
| `vselr` | 一维寄存器索引置换 | 没有单条相同置换语义的 Tile Op |
| `vexpdif` | 带同 shape `max` 和 mask 的逐 lane `exp(x - max)` | `trowexpandexpdif` 只接受逐行 scalar，且 dtype 与 predication 契约不同 |
| `vmula` | `result = acc + lhs * rhs`，两个乘数均为 tile | `taxpy` 的一个乘数是 scalar；`tmul + tadd` 不是同一条融合语义 |
| `dhist` | 对带 mask 的 `1xLxui8` 输入生成完整 `1x256xui16` 分布直方图 | `thistogram` 是 row-wise source/index/destination 更新，操作数角色和结果契约不同 |
| `create_mask` | 根据 `active_lanes` 生成 `1xLxi1` prefix predicate | 没有接受动态 active lane 数并返回 prefix predicate 的同义 Tile Op；`texpands` 仅做 scalar broadcast |
| `create_group_mask` | 在 `MxNxi1` 的每一行生成相同宽度的 prefix predicate | 没有同义的逐行动态 prefix predicate Tile Op；二维 shape 仅替代原 group 属性 |

`vmula` 的目标 tilereg 形式为：

```mlir
%result = pto.vmi.vmula %acc, %lhs, %rhs, %mask
    : !pto.vmi.tilereg<MxNxT>, !pto.vmi.tilereg<MxNxT>,
      !pto.vmi.tilereg<MxNxT>, !pto.vmi.tilereg<MxNxi1>
      -> !pto.vmi.tilereg<MxNxT>
```

每个有效位置计算：

```text
result[m, n] = acc[m, n] + lhs[m, n] * rhs[m, n]
```

对于浮点类型，融合乘加的舍入结果可能不同于独立 `tmul` 后再 `tadd`，因此不能只因
可用两条 Tile Op 组合实现相同代数式就认为存在同语义单 op。

`dhist` 的目标 tilereg 形式为：

```mlir
%hist = pto.vmi.dhist %acc, %source, %mask
    : !pto.vmi.tilereg<1x256xui16>, !pto.vmi.tilereg<1xLxui8>,
      !pto.vmi.tilereg<1xLxi1> -> !pto.vmi.tilereg<1x256xui16>
```

```text
result[0, b] = acc[0, b]                         for b in [0, 256)
result[0, source[0, n]] += 1                     when mask[0, n] is true
```


## 待讨论的设计点

### `valid_col` 取代部分 mask 的可行性

> **讨论状态：** 本节只分析可行性，尚未纳入 canonical VMI 类型和指令签名。
> 第 3.2 节的显式 `tilereg<MxNxi1>` mask 及第 6.3 节保留的
> `create_mask` / `create_group_mask` 仍是当前设计结论。

#### 可行性结论

为 `tilereg` 增加 `valid_col` 在技术上可行，也能消除常见尾块计算中重复传递的
prefix mask，但它只能描述每行从第 0 列开始的连续有效前缀：

```text
valid[m, n] = n < valid_col
```

因此，`valid_col` 可以替代“前 `V` 列有效、其余列无效”的 mask，不能完整替代任意
`MxNxi1` predicate。比较产生的离散 predicate、非连续 lane 选择和每行不同的有效
宽度仍需要显式 mask 或另一种 metadata 表达。

#### 候选类型表达

可以沿用 Tile Op `tile_buf` 的 valid-shape 记法，为 `tilereg` 增加可选 `valid`
参数：

```mlir
// valid shape 等于完整 shape，省略 valid。
!pto.vmi.tilereg<8x16xf32>

// 静态 valid_col = 13。
!pto.vmi.tilereg<8x16xf32, valid=8x13>

// valid_col 是运行时动态值。
!pto.vmi.tilereg<8x16xf32, valid=8x?>
```

候选方案第一阶段只允许 valid row 等于静态 `M`，第二维为 `K` 或 `?`：

```text
0 <= K <= N
```

`valid=MxK` 表示所有 `M` 行具有相同的有效列数 `K`。它不表示每行分别携带一个
`valid_col[m]`。

#### 动态 `valid_col` 的绑定

MLIR type parameter 和 attribute 是编译期信息，不能直接引用 `%valid_col` 这样的
运行时 SSA value。因此，类型中的 `?` 只表示“该字段动态”，实际值必须由产生或
包装该 `tilereg` 的 operation operand 绑定。

一种候选的不可变 SSA 表达是增加 metadata view op：

```mlir
%bounded = pto.vmi.with_valid_col %tile, %valid_col
    : !pto.vmi.tilereg<8x16xf32>
      -> !pto.vmi.tilereg<8x16xf32, valid=8x?>
```

`with_valid_col` 不改变寄存器 payload，只返回一个同时携带原数据和运行时
`valid_col` metadata 的新 SSA 值。需要显式读取时可以提供：

```mlir
%valid_col = pto.vmi.get_valid_col %bounded
    : !pto.vmi.tilereg<8x16xf32, valid=8x?> -> index
```

产生 tile register 的操作也可以直接携带绑定 operand，避免单独的 view op：

```mlir
%tile = pto.vmi.tload %source[%offset] valid_col = %valid_col
    : !pto.ptr<f32, ub>
      -> !pto.vmi.tilereg<1x128xf32, valid=1x?>
```

静态 `valid_col` 可直接从类型读取并折叠为常量；动态形式必须在 SSA 定义点、函数
参数 ABI 或 control-flow block argument 中实际携带对应的 `index` metadata，不能依赖
consumer 反向查找某个特定 defining op。

#### 对 mask 参数的影响

如果一条指令省略显式 mask，可以从参与计算的 tile metadata 生成 implicit prefix
mask：

```text
implicit_mask[m, n] = n < effective_valid_col
```

如果仍提供显式 `MxNxi1` mask，则候选组合语义为：

```text
effective_mask = implicit_prefix_mask AND explicit_mask
```

因此更合适的方向是让 mask 从“所有 predicated op 的必选参数”变成“任意 predicate
场景的可选参数”，而不是彻底删除 mask 类型。

| Mask 使用场景 | `valid_col` 是否足够 |
|---|---|
| 一维尾块的连续 prefix mask | 足够 |
| 所有行具有相同有效宽度的 grouped prefix mask | 足够 |
| elementwise/store/reduce 的连续尾部控制 | 足够 |
| `tcmp`/`tcmps` 产生的任意比较结果 | 不足 |
| `tsel` 的离散条件 | 不足 |
| 非连续或稀疏 lane 选择 | 不足 |
| 每行具有不同 `valid_col` | 单个 `valid_col` 不足 |
| `dhist`、gather/scatter 的 mask | 仅当 mask 本身是连续 prefix 时足够 |

如果 `create_mask` / `create_group_mask` 的结果只用于控制连续尾部，未来可以直接绑定
`valid_col`，不必物化 `MxNxi1` tile。如果生成的 predicate 还会被 `tsel`、boolean op
或其他 consumer 当作普通值使用，则仍应通过 `create_mask` / `create_group_mask` 产生
显式 `i1` tile。

#### 传播规则的候选方案

为了让 consumer 不再显式接收尾部 mask，每个产生新 `tilereg` 的操作都需要定义
valid metadata 的传播规则。初步可行规则如下：

- unary 和 tile-scalar op 传播 source 的 `valid_col`；
- store 和 row reduction 使用 source 的 `valid_col` 作为有效计算宽度；
- compare 结果继承参与比较数据的有效区域；
- scalar broadcast 默认产生 full-valid 结果，也可在结果处重新绑定 `valid_col`；
- binary/ternary op 可选择要求输入 valid 区域相等，或把结果定义为输入有效区域的
  intersection，即运行时 `min(valid_col...)`。

最后一条尚需决策。要求相等更容易发现调用错误，但两个动态值是否相等通常无法静态
证明；使用 intersection 能给出确定语义，却可能静默缩小结果有效区域。

#### 与 `pmode` 的边界

`valid_col` 描述值的有效域，mask/`pmode` 描述 inactive position 的计算结果，两者
不完全等价。候选方案中，valid 区域外的 tile element 应视为 unspecified，并且不能
被后续 store 或 reduction 观察。如果程序要求这些位置显式为零或保留 prior
destination value，仍需要显式 mask 与 `pmode`，或单独的 fill/pad 操作。

#### 与 `treshape` 的冲突

动态 valid metadata 会使一维/二维 `treshape` 规则更复杂。一个 `1xL` prefix reshape
成 `MxN` 后，通常不是“所有行具有同一个 valid_col”的矩形区域；相反，一个
`MxK` 矩形区域 flatten 后也不一定形成连续的一维 prefix。因此可行的保守规则是：

- full-valid source reshape 后仍为 full-valid；
- 不改变行列划分的 dtype reinterpretation 可以保留 `valid_col`；
- 其他 partial-valid 的 shape reshape 不自动传播 valid metadata，要求结果显式重新
  绑定；
- 如果必须准确表达每行不同宽度，需要另行设计 `row_valid_cols` metadata，而不能只
  使用单个 `valid_col`。

#### 实现代价与待决问题

动态 `valid_col` 需要把 `tilereg` 视为“寄存器 payload 加一个 scalar metadata”的
SSA 值。后续类型转换可能需要把一个逻辑 `tilereg` 展开为若干物理数据值和一个
`index` 值，并保证 metadata 穿过 `func`、`scf`、分支合流和多结果操作。

在纳入 canonical 设计前，至少需要确定：

1. 类型文本使用 `valid=Mx?` 还是独立的 `valid_cols=?` 字段；
2. 动态值由 producer operand 绑定，还是统一使用 `with_valid_col`；
3. 多输入操作要求 valid 相等，还是采用 intersection；
4. function/control-flow 边界如何携带动态 metadata；
5. partial-valid `treshape` 是拒绝、丢弃 metadata，还是引入 per-row valid 表达；
6. 哪些现有 mask operand 可以改成 optional，哪些必须保留。

综合来看，该方案适合优化一维尾块和统一 grouped prefix mask，但不应作为显式
`tilereg<MxNxi1>` predicate 的完整替代品。

### VMI load/store 直接接受 UB `tile_buf` view 的可行性

> **讨论状态：** 本节只分析是否需要增加以 UB `tile_buf` view 为访存对象的
> `pto.vmi.tload` / `pto.vmi.tstore` 形式，不改变当前第 6.1 节的指令映射和
> [`VMI load/store`](../isa/vmi-isa/01-load-store.md) 中的 `ptr + offset + row_stride`
> canonical 形式。

#### 问题与候选接口

当前 VMI load/store 直接使用 UB pointer、element offset 和可选 row stride：

```mlir
%value = pto.vmi.tload %source[%offset], %row_stride
    : !pto.ptr<T, ub>, index -> !pto.vmi.tilereg<MxNxT>

pto.vmi.tstore %value, %dest[%offset], %row_stride
    : !pto.vmi.tilereg<MxNxT>, !pto.ptr<T, ub>, index
```

当上层已经使用 `tile_buf` 及 `pto.subview` 描述 UB 窗口时，上述形式需要再次展开
base address、offset、shape、row stride 和 valid shape。可以考虑增加以 UB
`tile_buf` view 为访存对象的重载：

```mlir
%window = pto.subview %ub_tile[%row, %col]
    sizes [8, 16] valid [%valid_row, %valid_col]
    : !pto.tile_buf<vec, 32x64xf32>
      -> !pto.tile_buf<vec, 8x16xf32, valid=?x?>

%value = pto.vmi.tload %window
    : !pto.tile_buf<vec, 8x16xf32, valid=?x?>
      -> !pto.vmi.tilereg<8x16xf32>

pto.vmi.tstore %value, %window
    : !pto.vmi.tilereg<8x16xf32>,
      !pto.tile_buf<vec, 8x16xf32, valid=?x?>
```

该形式中的 `tload` / `tstore` 仍然是 UB 与 register-resident `tilereg` 之间的数据
访问，不是 Tile Op `pto.tload` / `pto.tstore` 所表达的 partition view 与 `tile_buf`
之间的搬运。

#### 是否需要独立的 subview operand type

现有 `pto.subview` 的结果仍然是 `!pto.tile_buf<...>`，没有独立的
`!pto.tile_subview` 类型。因此不建议让 verifier 要求 operand 的 defining op 必须是
`pto.subview`。这种 def-use 限制会使经过 `func`、`scf`、cast 或其他 view op 传递的
合法窗口无法使用。

更稳健的候选契约是：

- operand 接受任意 `!pto.tile_buf<vec, MxNxT, ...>`；
- `pto.subview` 是构造该 UB view 的一种方式，不是唯一方式；
- verifier 检查 memory space、shape、dtype、layout 和 valid metadata，而不是检查
  producer 名称；
- 如果必须在类型上区分可访存 view 和 owning tile，则需要另行设计轻量的
  `!pto.vmi.ub_view` 或通用 view interface，不能用 defining-op 特判代替类型契约。

#### 可复用的信息

使用 `tile_buf` view 的主要收益是复用已经存在的访存描述：

| `tile_buf` / subview 信息 | 对 VMI load/store 的作用 |
|---|---|
| memory space `vec` | 证明访问对象位于 UB |
| result shape `MxN` | 推导 `tilereg` 的逻辑 shape |
| element dtype | 检查内存元素与 register 元素类型匹配 |
| subview offsets | 决定窗口起始地址 |
| parent base/stride | 决定各行和各列的物理地址 |
| static/dynamic valid shape | 决定有效读写区域或尾部 mask |
| layout/config | 判断当前 VMI register access 是否能够直接表示该 view |

理想情况下，VMI op 不再重复接收 `%offset` 和 `%row_stride`，而是按 view 的逻辑
坐标读取：

```text
for m in 0 .. M:
  for n in 0 .. N:
    result[m, n] = ub_view[m, n]
```

store 使用相同坐标关系写回。

#### 第一阶段可支持的 view 子集

`tile_buf` 支持的 layout 比当前 VMI grouped load/store 的“行间 stride、行内连续”
模型更广。如果直接接受所有 `tile_buf`，可能隐式引入转置、boxed/fractal 访问或列方向
stride，而这些行为并不等价于当前 VMI load/store。

保守的第一阶段候选约束可以是：

- memory space 必须为 `vec`；
- rank 必须为 2；
- element dtype 与 `tilereg` 相同；
- subview shape 与 `tilereg<MxNxT>` 的 `MxN` 完全相同；
- `slayout=none_box`；
- row-major，且列方向 element stride 为 1；
- 行方向 stride 可为静态或动态值，但必须能从 view descriptor 获得；
- 第一阶段要求 `valid_row = M`，只允许 `valid_col` 小于或等于 `N`。

其他 layout 应明确 verifier failure，而不是把 layout conversion 隐含到 register
load/store 中。未来如果需要支持 col-major、boxed 或 fractal view，应为对应访问模式
定义明确语义，或先通过显式 view/layout conversion 转为支持的 UB view。

#### 与 `valid_col` 讨论的关系

如果上一节的 `tilereg valid_col` 方案成立，`tload` 可以把 UB view 的动态
`valid_col` 直接绑定到结果：

```mlir
%value = pto.vmi.tload %window
    : !pto.tile_buf<vec, 8x16xf32, valid=8x?>
      -> !pto.vmi.tilereg<8x16xf32, valid=8x?>
```

`tstore` 则只写入 source tilereg 与 destination view 的公共有效区域：

```text
effective_valid_col = min(value.valid_col, window.valid_col)
```

如果不采用 tilereg valid metadata，则仍需要以下方案之一：

1. `tload` 同时返回显式 `MxNxi1` mask；
2. consumer 根据 view 的 valid shape 通过 `create_mask` / `create_group_mask` 创建 mask；
3. `tload/tstore` 内部使用 view valid shape，但规定 load 结果无效位置为 unspecified。

动态 `valid_row < M` 不能只用单个 `valid_col` 表达，因此第一阶段应要求所有行有效，
或同时引入完整 valid shape / row mask 设计。

#### Memory effect 与同步边界

`tile_buf` view 是一个具有别名关系的 UB memory object。新增形式需要明确：

- `tload` 对 view 覆盖的 UB 区域产生 read effect；
- `tstore` 对 view 覆盖的 UB 区域产生 write effect；
- 两个来自同一 parent tile 的重叠 subview 必须被 alias analysis 识别；
- 与 Tile Op、DMA、其他 VMI access 之间的 pipeline ordering 和同步不能因接口简化而
  丢失；
- `tstore` 不能被当作只修改 `tilereg` 的 pure op。

如果 lowering 只能通过追踪 `pto.subview` producer 才能恢复 parent stride，那么 view
在 function/control-flow 边界传递时会丢失必要信息。要支持通用场景，UB view 的
base、offset、stride 和 dynamic valid shape 必须是可随 SSA value 传递的一等 descriptor
语义，而不能只存在于局部 producer 链中。

#### 可选方案比较

| 方案 | 优点 | 风险 |
|---|---|---|
| 保留 `ptr + offset + row_stride` | VMI 与 Tile 层解耦，lowering 直接 | 上层必须重复展开 subview metadata |
| `tload/tstore` 重载接受 `tile_buf<vec>` | 语法简洁，可直接复用 subview shape/stride/valid | VMI 与 Tile type 耦合，需要完整 alias/effect/view lowering |
| 显式 `tile_buf_addr` 后调用 pointer 形式 | 不新增 VMI op 类型，地址转换可见 | shape、stride、valid metadata 仍需额外传递 |
| 新增轻量 `!pto.vmi.ub_view` | VMI 访存契约独立且可携带完整 descriptor | 增加新类型和 tile_buf-to-view 转换成本 |

#### 待决问题

在纳入 canonical 设计前，需要确定：

1. 新形式是现有 `tload/tstore` 的 type-based overload，还是使用独立 op 名称；
2. 是否接受所有 `tile_buf<vec>`，还是只接受满足特定 layout 的 view；
3. view 的 runtime base/stride/valid metadata 如何穿过函数和控制流；
4. dynamic valid shape 如何映射到 tilereg metadata 或显式 mask；
5. `tstore` 的 source/destination valid 区域不一致时，是取 intersection 还是报错；
6. pointer 形式是否长期保留，作为底层和不规则访问的显式接口；
7. alias analysis、memory effect 和同步验证是否能覆盖重叠 subview。

从语义上看，该重载是可行的，并能减少上层已经构造 `tile_buf` subview 后再次拆解
地址信息的重复工作；是否采用主要取决于是否希望 VMI memory interface 直接依赖
Tile type，以及现有 view descriptor 能否可靠携带 stride、alias 和 dynamic valid
metadata。
