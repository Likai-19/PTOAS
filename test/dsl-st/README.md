# PTODSL ST Guide

`test/dsl-st/` 放的是基于 PTODSL 的 simulator/ST 用例。

这类测试适合验证：

- PTODSL surface 是否能正确生成目标 IR
- `ptoas --pto-backend=vpto` 后端是否能稳定接受这类新形态
- kernel 在 simulator / runtime 路径上的真实执行结果

如果你要验证的是 parser、verifier、pass dump、IR rewrite 之类“只看编译输出”的行为，优先放到 `test/lit`，不要放这里。

## 与 VPTO 套件的分工

两个套件都吃纯 MLIR，但调度粒度和覆盖面不同，用例不要互相镜像：

- **语义用例放这里**（本目录）。目录级 runner 会把每个模块按用例切片（默认 10 例/片）后并发执行，每片一次
  `msprof op simulator`、每片独立超时；一个模块里的用例越多，越应该待在这里而不是别处。
- **`test/vpto/cases/` 只保留它独有的覆盖面**：`ptoas` CLI 读 `.pto` 文件出 fatobj、
  `launch.cpp`/`main.cpp` 的 C++ 宿主 ABI 与 `kernel.so` 链接、以及 `DEVICE=NPU` 的真机路径。
  它的调度单位是“一个用例目录 / 一个 `.py` 文件”，不可切分。
- 同一个 kernel 在两个套件里各写一份（比如用一层 `from <module> import CASES` 的 shim）不会增加覆盖，
  只会让 VPTO 那一步多出一个不可切分的长尾用例。

## 环境前提

`test/dsl-st/` 现在假设 `ptodsl` 已经通过支持的 `ptoas` 安装合同进入当前
Python 环境，而不是在运行时自己修 `PYTHONPATH` 或 `sys.path`。

推荐准备方式：

```bash
cd $PTOAS_REPO_ROOT
pip install -e . --no-build-isolation
```

如果你在验证 release/CI wheel，也可以直接：

```bash
pip install /path/to/ptoas*.whl
```

快速自检：

```bash
python3 -c "import ptodsl; from ptodsl import pto; print(ptodsl.__file__)"
```

注意：

- `ptoas-bin-*.tar.gz` 只是 compiler-only artifact，不能替代上面的 Python 安装。
- 如果 `import ptodsl` 失败，应先修复安装环境，而不是给测试脚本补 `sys.path`。

## 推荐写法

当前目录下推荐使用 [common.py](common.py) 里的两个 helper：

- `golden_output_case(...)`
- `auto_main(globals())`

目录级运行时，还可以直接使用：

- `python3 test/dsl-st`
- `scripts/sim_dsl.sh test/dsl-st`

对大多数单输出测试，开发者只需要写：

1. kernel
2. 输入构造
3. golden 计算
4. `CASES = [...]`
5. 文件末尾一行 `auto_main(globals())`

## 最小模板

```python
#!/usr/bin/env python3

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto


@pto.jit(name="my_kernel", kernel_kind="vector", target="a5", mode="explicit")
def my_kernel(
    inp_ptr: pto.ptr(pto.f32, "gm"),
    out_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
):
    # write kernel here
    ...


def make_inputs():
    return [np.ones((4, 64), dtype=np.float32)]


def make_expected(inp):
    # compute golden from host inputs
    return inp + 1.0


CASES = [
    golden_output_case(
        "my_kernel_basic",
        my_kernel,
        inputs=make_inputs,
        expected=make_expected,
        rtol=0.0,
        atol=0.0,
    ),
]


auto_main(globals())
```

## `golden_output_case(...)` 约定

`golden_output_case(...)` 默认适合“若干输入 + 一个输出”的测试。

它会自动做这些事情：

- 把 `inputs` 转成 host numpy 数组
- 根据 `expected` 的 shape / dtype 自动分配一个全零输出
- 把这个输出作为最后一个参数传给 kernel
- 默认把最后一个 device tensor 拿出来和 golden 比较

常用参数：

- `inputs`
  - 可以是函数，返回 `list[np.ndarray]`
  - 也可以直接传一个 `list[np.ndarray]`
- `expected`
  - 可以是函数，签名是 `expected(*host_inputs)`
  - 也可以直接传 numpy 数组
- `output_shape`
  - 如果输出 shape 不能直接从 golden 推出来，可以显式指定
- `output_dtype`
  - 如果输出 dtype 需要和 golden 分开控制，可以显式指定
- `output_index`
  - 默认比较最后一个 tensor；如果不是最后一个，改这里
- `grid`
  - launch 的 block 数，默认 `1`
  - 只有真的要覆盖多核行为（例如 kernel 里用 `pto.get_block_idx` 分片）才需要改
  - 此时 golden 要按“所有 block 合起来的结果”来写
- `rtol` / `atol`
  - 浮点结果建议显式写；位级结果一般用 `0.0`

## 什么时候需要自定义 case

如果测试不是“单输出比 golden”这一路径，就不要硬塞进 `golden_output_case(...)`。

常见例子：

- 需要比较多个输出
- 需要读取中间 buffer
- 需要自定义断言信息
- 需要根据运行时结果做结构化检查而不是 `allclose`

这时可以直接继续使用 `run_cases(...)` 的底层接口，自定义：

- `make_case()`
- `check(device_inputs, expected)`

自定义 case dict 里同样可以带 `"grid"`；不写就是 `1`。

## 当前参考用例

可以直接参考：

- [predicate_pack.py](predicate_pack.py)
- [cube_matrix_pipeline.py](cube_matrix_pipeline.py)

它演示了：

- PTODSL kernel authoring
- raw predicate image 的 host golden 写法
- `golden_output_case(...)` 的标准接入方式
- cube matrix pipeline 的端到端 simulator/ST 写法

## 运行方式

单文件打印生成的 MLIR：

```bash
python3 test/dsl-st/predicate_pack.py --emit-mlir
```

单文件走 simulator ST：

```bash
scripts/sim_dsl.sh test/dsl-st/predicate_pack.py
```

自动发现整个目录下的测例并列出 case name：

```bash
python3 test/dsl-st --list
```

自动发现整个目录下的测例并打印合并 MLIR：

```bash
python3 test/dsl-st --emit-mlir
```

自动发现整个目录下的测例并走 simulator ST：

```bash
scripts/sim_dsl.sh test/dsl-st
```

如果只是先做编译链检查，也可以先跑：

```bash
python3 ptodsl/tests/test_jit_compile.py
```

## 编写建议

- 优先让 golden 直接表达语义，不要把 expected 写成难懂的魔数堆。
- 尽量让一个测试只保护一个回归点；如果要覆盖一组紧密相关的形态，可以像 `predicate_pack.py` 一样在同一个 kernel 里并排 materialize。
- 对 predicate / bit-level 结果，优先用 `psts` 这类“直接 materialize raw state”的方式观测，不要绕远路通过别的算子副作用来猜结果。
- 能用 Python 原生字面量的地方就直接用，减少不必要的 `pto.const(...)` 噪音。
- 如果某个写法依赖当前 backend / raw image 约定，最好在 golden 附近留一小段注释，解释为什么 expected 长这样。

## 自动发现约定

`test/dsl-st/` 目录级 runner 会自动加载当前目录下所有顶层 `.py` 用例文件，但会跳过：

- `common.py`
- `__main__.py`
- 以下划线开头的辅助模块
- 以 `test_` 开头的测试文件（unittest/pytest，不是用例模块）

每个被发现的模块都需要定义非空 `CASES` 列表，且所有 case name 在目录内必须唯一。

## Histogram / scatter 的七种合法 lane

`vmi_histogram_scatter_lanes.py` 包含 56 个 kernel、224 个设备场景：

- `1/2/4/8/64/128/256`，`vdhist/vchist` 各覆盖 128/256 bins；
- scatter 覆盖 f32、f16、ui8、i8，随机式无重复偏移与不同奇偶字节值；
- full、empty、prefix、sparse mask，128 lane 的 prefix 为 96；
- 非零 histogram accumulator，以及输出区两侧和所有未写位置的哨兵。

```bash
python3 test/dsl-st/vmi_histogram_scatter_lanes.py --emit-mlir
python3 test/dsl-st/vmi_histogram_scatter_lanes.py
python3 ptodsl/tests/test_vmi_histogram_scatter_lanes.py
```

最后一条无需设备，检查 size 推导、类型构造、操作入口和 mask/offset
长度不匹配的负向用例。非法 logical lane 在 PTODSL 报错；active prefix
不受七值白名单限制。输入分配为短读取保留最小 32B，64/128 ui8 的
VMI load 仍分别只请求 64/128B。

## 并行门禁与 PV 模拟模型

`.gitcode/scripts/run_ptodsl_cases_parallel.sh` 把一个模块切成 unit（大模块按用例切块、
轻量单例模块装箱），每个 unit 一次 `msprof op simulator`，用 `xargs` 动态并发；
`.gitcode/scripts/sim_validation.sh` 的 PTODSL 步骤调用它。每个 unit 都带 discovery
得到的用例数，只有 PASS 行数与之一致才算通过，总数不一致也会让步骤变红。

### 环境变量

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `PTOAS_SIM_MODEL` | `ca`（PR 门禁的 PTODSL 步骤默认 `pv`） | `ca` 用 CANN 自带 CA 模型；`pv` 用私有软链树加载 PV 模型 |
| `PTOAS_SIM_MODEL_ROOT` | `$HOME/.cache/ptoas/sim-pv-<soc>` | pv 私有树的位置 |
| `PTODSL_CASE_JOBS` | `24` | 并发 unit 数 |
| `PTODSL_CASE_BLOCK_SIZE` | `10` | 一个大模块每个 unit 的最大用例数 |
| `PTODSL_CASE_TIMEOUT` | `1800` | 单个 unit 的超时（秒） |
| `PTODSL_CASE_KILL_AFTER` | `60` | 超时后再等多久发 SIGKILL |
| `PTOAS_DSL_PACK_SIZE` | `16` | 每包装几个轻量单例模块；`1` 关闭装箱 |
| `PTOAS_DSL_HEAVY_PATTERN` | cube/cv_mixed/cast_spine/simt/binary_vector/vmi_new_kernels | 命中的单例模块各自独占会话；空或 `-` 表示不设重型家族 |
| `PTOAS_DSL_BIG` | 未设 | cast-spine/quant 类用例放大输入（见对应模块注释） |
| `PTOAS_VMI_FULL_MATRIX` | 未设 | VMI group/histogram 类用例跑完整笛卡尔积（见对应模块注释） |

### 门禁 vs daily

- **PR 门禁**（`.gitcode/scripts/sim_validation.sh` 的 PTODSL 步骤）：`PTOAS_SIM_MODEL=pv`、
  `PTODSL_CASE_JOBS=24`、`PTOAS_DSL_PACK_SIZE=16`。
- **其它步骤 / daily / 手工复现**：默认 `PTOAS_SIM_MODEL=ca`；要用 pv 就显式
  `PTOAS_SIM_MODEL=pv`。所有变量都能用环境覆盖。

### pv 的性质与 CANN 版本锁定

`msprof op simulator` 硬编码 CA 模型，`LD_PRELOAD` 覆盖不掉：`sim_dsl.sh` 的 pv 路径在
私有软链树里把四个 CA 库名影子成 PV 库（`libruntime_camodel.so -> libruntime_cmodel.so` 等），
再让 `ASCEND_HOME_PATH` 指过去。这是对 CANN 内部布局的 hack，需要注意：

- 影子映射依赖具体 CANN 版本的库名（本仓库在 `cann-9.1.0` 上验证）；
- 显式 `PTOAS_SIM_MODEL=pv` **不做退化**：私有树构建失败或缺少 `_pv` 库时 `sim_dsl.sh`
  直接非 0 退出、打印缺失的路径/库；pv 不可用即失败，不会被 ca 掩盖；只有显式
  `PTOAS_SIM_MODEL=ca`（或未设置时的默认值）才走 ca；
- 更换 CANN 版本后要重新确认这四个 `_pv`/`_cmodel` 库仍存在。

## 迁移覆盖说明

`test/vpto/cases/` 里原本由 VPTO runner 逐用例驱动的老用例已随本次迁移删除，改用
`test/dsl-st/` 的 in-memory 模块（kernel MLIR 内嵌，golden/compare 在内存里跑）。有意的例外：

- `onboard-only/simt-keep-resume`：**故意排除、不迁移**。它依赖 `pto.keep` / `pto.resume`
  的持久 SIMT 寄存器语义，模拟器不建模这条路径，而且它本来就是板端（onboard）探针而不是
  simulator 用例。原有覆盖由同目录保留的 `onboard-only/simt-keep-resume-register-swap`
  在板端承担；simulator 侧没有等价模块。
- `test/vpto/cases/` 其余目录只保留 VPTO 独有的覆盖面（`.pto` 文件、C++ 宿主 ABI、`DEVICE=NPU`
  真机路径），不与 `test/dsl-st/` 镜像。

迁移模块开头的 `"""Migrated from test/vpto/cases/...` 出处注释在原目录删除后仍作为历史出处
记录保留，与其余迁移模块一致。

### SIGSEGV 重试（临时兜底，issue #17）

CANN 仿真器有一处已知缺陷（D2H 把设备 VA 当宿主指针），会让随机某个 unit 偶发
`rc=139`（约 1 次 / ~1000 会话）。驱动对此加了**一次性**重试：

- **只对 `rc=139`（SIGSEGV）生效**；超时 `rc=124/137`、compare 失败 `rc=1`、覆盖率
  不符、以及任何其它退出码都**不重试**；
- 只重跑该 unit，最多一次；重跑过的 unit 在 result/汇总里标 `RETRIED`，汇总打印
  `retried=N`；第一次崩溃的日志保存在 `<unit>.sigsegv.log`；
- 重跑后的 unit 仍要满足 `passed == expected` 覆盖率断言；
- 这是**临时兜底**：根因已提单 `cann/pto-as#17`，CANN 修复后必须删除该重试块（驱动里
  `run_once` 之后的 `rc == 139` 分支）。

## Write-mask 审计

迁移模块的 golden 缓冲只有在 `np.empty`（元素确实未定义）时才保留部分 write mask；
用 `zeros`/`full`/`ones` 创建、只写其中一部分的缓冲一律整 buffer 比较。

- 全量审计结论：golden 条目 `456`、保留部分 mask `107`、整 buffer 比较 `349`、
  golden 无法在内存复现 `materialize_failures=4`。部分 mask 的判据都是 golden 用
  `np.empty`（元素确实未定义）创建后再写入；用 `zeros`/`full`/`ones` 创建的缓冲一律整 buffer。
- 规则回归（`np.empty` 保留 mask、已定义缓冲整 buffer、don't-care 写错必须失败）：
  [`test_write_mask_policy.py`](test_write_mask_policy.py)
