# ptoas (PTO Assembler & Optimizer)

## 1. Introduction

**ptoas** is a specialized compiler toolchain built on top of the **LLVM21 VPTO branch (`vpto-dev/llvm-project:feature-vpto-llvm21`)**, designed specifically for **PTO Bytecode** (Programming Tiling Operator Bytecode).

Acting as the bridge between upper-level AI frameworks and underlying NPU/GPGPU/CPU hardware, `ptoas` is built in an **Out-of-Tree** architecture and provides complete C++ and Python interfaces. Its primary responsibilities include:

1. **IR Parsing & Verification**: Parses `.pto` input files and verifies the semantic correctness of PTO Dialect operations (Ops).
2. **Compilation & Optimization (Passes)**: Executes optimization passes targeting the Da Vinci Architecture, such as operator fusion and automatic synchronization insertion.
3. **Code Generation (Lowering)**: Supports lowering PTO IR to `EmitC` / `Linalg` dialects, ultimately generating code that calls the `pto-isa` C++ library.
4. **Python Bindings**: Provides seamlessly integrated Python modules. Through integration with MLIR Core bindings, frameworks such as **PyPTO**, **PTODSL**, and **CuTile** can build, manipulate, and compile PTO Bytecode directly from Python.

---

## 2. Directory Structure

```text
PTOAS/
├── include/
│   └── PTO/               # PTO Dialect headers and TableGen definitions (.td)
├── lib/
│   ├── PTO/               # Dialect core implementation (IR) and Pass logic (Transforms)
│   ├── CAPI/              # C language interface exposure
│   └── Bindings/Python/   # Python Binding C++ implementation (Pybind11)
├── python/                # Python module build scripts and helper code
├── test/
│   └── samples/           # Test cases
├── tools/
│   ├── ptoas/             # ptoas command-line tool entry point (Output: ptoas)
│   └── ptobc/             # ptobc command-line tool entry point (Output: ptobc)
└── CMakeLists.txt         # Top-level build configuration
```

---

## 3. Build Instructions

⚠️ **Important**: This project strictly requires the **LLVM21 VPTO branch `vpto-dev/llvm-project:feature-vpto-llvm21`**.

### 3.0 Environment Variable Configuration

To simplify the build process, **first modify and run the following commands according to your environment**. Subsequent steps reference these variables directly.

```bash
# ================= Configuration (edit here) =================
# Set your workspace root directory
# (recommended: a dedicated directory for LLVM and PTOAS)
export WORKSPACE_DIR=$HOME/llvm-workspace

# LLVM source and build paths
export LLVM_SOURCE_DIR=$WORKSPACE_DIR/llvm-project
export LLVM_BUILD_DIR=$LLVM_SOURCE_DIR/build-shared

# PTOAS source and install paths
export PTO_SOURCE_DIR=$WORKSPACE_DIR/PTOAS
export PTO_INSTALL_DIR=$PTO_SOURCE_DIR/install
# =============================================================

# Create the workspace directory
mkdir -p $WORKSPACE_DIR
```

### 3.1 Prerequisites

* **OS**: Linux (Ubuntu 20.04+ recommended)
* **Compiler**: GCC >= 9 or Clang (C++17 support required)
* **Build System**: CMake >= 3.20, Ninja
* **Python**: 3.8+
* **Python Packages**: `pybind11<3`, `nanobind`, `numpy`

```bash
python3 -m pip install "pybind11<3" nanobind numpy
```

> **Note**: The current LLVM/MLIR Python bindings are not compatible with `pybind11` 3.x.
> If you encounter errors like `def_property family does not currently support keep_alive`
> when building LLVM, run the downgrade command above first.

### 3.2 Step 1: Build LLVM/MLIR (Dependency)

Download the VPTO-adapted LLVM source, check out the `feature-vpto-llvm21` branch, and build with **shared libraries** to ensure correct linking for Python bindings.

```bash
# 1. Clone LLVM
cd $WORKSPACE_DIR
git clone https://github.com/vpto-dev/llvm-project.git
cd $LLVM_SOURCE_DIR

# 2. [Critical] Check out the VPTO adaptation branch
git checkout feature-vpto-llvm21

# 3. Configure CMake (build shared libs with Python bindings enabled)
cmake -G Ninja -S llvm -B $LLVM_BUILD_DIR \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DBUILD_SHARED_LIBS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python3) \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD="host"

# 4. Build LLVM (this step takes a long time)
ninja -C $LLVM_BUILD_DIR
```

### 3.3 Step 2: Build PTOAS (Out-of-Tree)

Clone the PTOAS source and build against the LLVM 21 you just compiled.

```bash
# 1. Clone PTOAS
cd $WORKSPACE_DIR
git clone https://gitcode.com/cann/pto-as.git PTOAS
cd $PTO_SOURCE_DIR

# 2. Build and install via pip
#    The build backend (pyproject.toml) drives CMake + Ninja automatically.
pip install . --no-build-isolation
```

This produces the same artifacts as a manual CMake build:

```text
# CLI tools
$PTO_SOURCE_DIR/build/tools/ptoas/ptoas
$PTO_SOURCE_DIR/build/tools/ptobc/ptobc

# Native extension installed into the MLIR Python package
$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core/
└── mlir
    └── _mlir_libs
        └── _pto.cpython-*.so

# Python dialect files
$PTO_INSTALL_DIR/
└── mlir
    └── dialects
        ├── pto.py
        └── _pto_ops_gen.py
```

### 3.4 Step 3: Supported Python Install Flows

If you want to use Python bindings or PTODSL, prefer the repository-root
`ptoas` package contract instead of manually patching `PYTHONPATH`.

```bash
# 1) Released or CI-built wheel: installs PTOAS + PTODSL together
pip install /path/to/ptoas*.whl

# 2) Non-editable source install from the repository root
cd $PTO_SOURCE_DIR
pip install . --no-build-isolation

# 3) Editable install for PTOAS / PTODSL developers
cd $PTO_SOURCE_DIR
pip install -e . --no-build-isolation
```

After installation, the following imports should work directly:

```python
import ptodsl
from ptodsl import pto, scalar
from mlir.dialects import pto as mlir_pto
```

> Notes:
> - The `ptoas` wheel also installs PTODSL.
> - VMI release wheels publish under the Python project name `ptoas-vmi`; the wheel filename is normalized to `ptoas_vmi-*.whl`.
> - The VMI release line keeps the `ptoas` CLI name; `ptoas --version` prints `ptoas vmi A.B.C`.
> - The `ptoas` and `ptoas-vmi` release wheels are **mutually exclusive**. They both install the same top-level `ptoas` Python package and `ptoas` console script, so do **not** install them into the same Python environment. Mixing them will overwrite files, and uninstalling one can break the other.
> - `ptoas-bin-*.tar.gz` compiler-only tarballs provide CLI/toolchain bits, not a PTODSL-capable Python distribution.
> - `--no-build-isolation` keeps pip from baking a temporary pybind11 path into `CMakeCache.txt`, which would break later `ninja` reconfigure runs after the temporary virtual environment is removed.

If you previously ran `pip install -e .` without the flag and your build is now broken, fix the existing `CMakeCache.txt` with:

```bash
cmake -B build -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
```

---

## 4. Usage

### 4.1 Command-Line Interface (CLI)

```bash
# Parse and print PTO IR
ptoas test/lit/pto/empty_func.pto

# Run the AutoSyncInsert pass
ptoas test/lit/pto/empty_func.pto --enable-insert-sync -o outputfile.cpp

# Specify target hardware architecture (A3 / A5)
ptoas test/lit/pto/empty_func.pto --pto-arch=a5 -o outputfile.cpp

# Specify build level (level3 disables PlanMemory/InsertSync)
ptoas test/lit/pto/empty_func.pto --pto-level=level3 -o outputfile.cpp

# Print the current ptoas release version
ptoas --version
```

### 4.2 Python API

In a supported `ptoas` install environment, both the PTO Dialect and PTODSL
can be imported directly.

```python
from mlir.ir import Context, Module, Location
# [Key] Import pto from mlir.dialects — the standard pattern for out-of-tree bindings
from mlir.dialects import pto
from ptodsl import pto as jit_pto, scalar

with Context() as ctx, Location.unknown():
    pto.register_dialect(ctx, load=True)
    module = Module.create()
    print("PTO Dialect registered successfully!")
    print("PTODSL imported successfully!", jit_pto, scalar)
```

### 4.3 Running Tests

```bash
# Recommended: enter a supported PTOAS / PTODSL install environment first
cd $PTO_SOURCE_DIR
pip install -e . --no-build-isolation

# Run Python binding tests
cd $PTO_SOURCE_DIR/test/samples/MatMul/
python3 ./tmatmulk.py > ./tmatmulk.pto

# Run ptoas tests
$PTO_SOURCE_DIR/build/tools/ptoas/ptoas ./tmatmulk.pto -o ./tmatmulk.cpp
```

### 4.4 On-Board Validation

This flow generates NPU validation test cases from the `.cpp` files produced by ptoas (under `test/samples/`) and runs them on an NPU. The example below reuses `MatMul/tmatmulk.cpp` generated in section 4.3.

> For compile-only validation on a machine without an NPU card, see [docs/no_npu_compile_only_guide_zh.md](docs/no_npu_compile_only_guide_zh.md).

```bash
# 1) Generate the npu_validation test directory
#    (creates npu_validation/ under the current sample directory)

# A2/A3 example:
python3 test/npu_validation/scripts/generate_testcase.py \
  --input test/samples/MatMul/tmatmulk.cpp \
  --run-mode npu \
  --soc-version Ascend910B1

# A5 example:
python3 test/npu_validation/scripts/generate_testcase.py \
  --input test/samples/MatMul/tmatmulk.cpp \
  --run-mode npu \
  --soc-version Ascend950

# 2) Run validation (run.sh requires no additional arguments)
test/samples/MatMul/npu_validation/tmatmulk/run.sh
```

Notes:
- `test/samples/MatMul/npu_validation/tmatmulk/` will contain `tmatmulk_kernel.cpp`, `main.cpp`, `golden.py`, `compare.py`, `run.sh`, and `CMakeLists.txt`.
- `golden.py` generates random inputs by default; outputs default to all zeros (only the count, shape, and data type of inputs/outputs match the kernel parameters).
- `compare.py` compares `golden*.bin` against `output*.bin` and reports an error if they differ.
