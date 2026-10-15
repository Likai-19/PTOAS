# e2e tests

End-to-end numerical-correctness tests for PTODSL vector ops.

## Quick start

```bash
# From repo root inside the hardware docker image:
cd /workspace && PYTHONPATH=/workspace/ptodsl python3 -m pytest ptodsl/tests/e2e -v

# Run one suite
python3 -m pytest ptodsl/tests/e2e/test_binary_elementwise.py -v

# Run a single test
python3 -m pytest ptodsl/tests/e2e/test_binary_elementwise.py -v -k "add-float32-1x64"

# Run all f32 tests
python3 -m pytest ptodsl/tests/e2e/test_binary_elementwise.py -v -k "float32"

# Run only div tests
python3 -m pytest ptodsl/tests/e2e/test_binary_elementwise.py -v -k "div"
```

## Hardware and backend selection

### A3 vpto backend

This is the default, or can be selected with `--backend vpto --target a3`.
The A2 and A3 VPTO lowering pipeline is shared, so A3 e2e coverage validates
the A2/A3 lowering path unless future lowering behavior diverges.

### A3 emitc backend

This can be selected with `--backend emitc --target a3`.

### A5 vpto backend

This can be selected with `--backend vpto --target a5`.

### A5 emitc backend

This can be selected with `--backend emitc --target a5`.

## Test matrix

### Binary elementwise

| Category | Ops | Dtypes | Shapes | Total |
|----------|-----|--------|--------|-------|
| f32 | add, addrelu, sub, mul, div, max, min | float32 | 15 | 105 |
| f16 | add, addrelu, sub, mul, div, max, min | float16 | 6 | 42 |
| i16 bitwise | bit_and, bit_or, bit_xor | int16 | 5 | 15 |
| i16 shifts | bit_shls, bit_shrs | int16 | 5 shapes x 3 shifts | 30 |

### Unary elementwise

| Ops | Dtypes | Shapes | Total |
|-----|--------|--------|-------|
| abs, relu, neg, exp, log, sqrt, rsqrt, recip | float32, float16 | 5 per dtype | 80 |

### Scalar-tile elementwise

| Ops | Dtypes | Shapes / Scalars | Total |
|-----|--------|------------------|-------|
| adds, muls, maxs, mins | float32, float16 | 5 shapes x 3 scalars per dtype | 120 |

### Confirmed hardware e2e count

| Suite | Total |
|-------|-------|
| Binary, bitwise, and shifts | 192 |
| Unary | 80 |
| Scalar-tile | 120 |
| Total | 392 |

### Shape coverage (exercises lowering code paths)

**f32** (`elementsPerRepeat=64`):

| Shape | Lowering Path |
|-------|---------------|
| (1, 32) | modeSmall single-row |
| (4, 32) | modeSmall multi-row |
| (11, 32) | modeSmall multi-row odd |
| (1, 64) | modeNorm1L single-repeat |
| (1, 128) | modeNorm1L multi-repeat aligned |
| (64, 64) | modeNorm1L square aligned |
| (16, 64) | modeNorm1L multi-row aligned |
| (16, 128) | modeNorm1L 16x128 |
| (4, 256) | modeNorm1L 4x256 |
| (1, 1024) | modeNorm1L 16-repeat |
| (16, 256) | modeNorm1L 16x256 |
| (32, 32) | modeSmall 32x32 square small |
| (1, 200) | modeNorm1L 1x200 tail |
| (4, 200) | modeCount1L 4x200 |
| (1, 96) | modeCount1L 1x96 tail |

**f16** (`elementsPerRepeat=128`):

| Shape | Lowering Path |
|-------|---------------|
| (1, 64) | modeSmall 1x64 |
| (4, 64) | modeSmall 4x64 |
| (1, 128) | modeNorm1L single-repeat |
| (16, 128) | modeNorm1L 16x128 |
| (64, 128) | modeNorm1L multi-row |
| (1, 512) | modeNorm1L 4-repeat |

## Adding new test categories

1. Create a new test file in `ptodsl/tests/e2e/` (e.g. `test_elementwise_unary.py`)
2. Add a kernel builder function in `common.py`
3. Parametrize over ops, dtypes, and shapes using the `make_binary_kernel` / `launch_and_check` pattern
4. Run with pytest

## Requirements

- NPU device with torch_npu installed
- ptoas and bisheng on PATH
- Properly built PTOAS with Python bindings (MLIR 19.1)
