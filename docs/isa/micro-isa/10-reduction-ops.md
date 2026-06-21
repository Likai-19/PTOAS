# 10. Reduction Ops

> **Category:** Vector reduction operations
> **Pipeline:** PIPE_V (Vector Core)

Operations that reduce a vector to a scalar or per-group result.

## Common Operand Model

- `%input` is the source vector register value.
- `%mask` is the predicate operand `Pg`; inactive lanes do not participate.
- `%result` is the destination vector register value.
- Reduction results are written into the low-significance portion of the
  destination vector and the remaining destination bits are zero-filled.

---

## Full Vector Reductions

### `pto.vcadd`

- **syntax:** `%result = pto.vcadd %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<MxU>`
- **A5 types:** i8-i32, f16, f32
- **semantics:** Sum all elements. Result in lane 0, others zeroed.

```c
T sum = 0;
for (int i = 0; i < N; i++)
    sum += src[i];
dst[0] = sum;
for (int i = 1; i < N; i++)
    dst[i] = 0;
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result` contains the reduction result in its low element(s).
- **constraints and limitations:** On A5, `i8/u8` inputs produce widened
  `i16/u16` results with half as many lanes (`M = N / 2`), and `i16/u16` inputs
  produce widened `i32/u32` results with half as many lanes. For
  `i32/u32/f16/f32` inputs, `U = T` and `M = N`. If all predicate bits are
  zero, the result is zero.

---

### `pto.vcmax`

- **syntax:** `%result = pto.vcmax %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i16-i32, f16, f32
- **semantics:** Find max element with argmax. The lowest destination element
  stores the maximum value, the second-lowest destination element stores the
  index of the first maximum, and all remaining elements are zero-filled.

```c
T mx = -INF; int idx = 0;
for (int i = 0; i < N; i++)
    if (src[i] > mx) { mx = src[i]; idx = i; }
dst[0] = mx;
dst[1] = idx;
for (int i = 2; i < N; i++)
    dst[i] = 0;
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result[0]` holds the extremum value and `%result[1]` holds the
  index. Other destination elements are zero-filled.
- **constraints and limitations:** If there are multiple maxima, the minimum
  index is written. For floating-point types, inactive lanes are treated as
  `-INF`; if all lanes are inactive, `%result[0]` becomes `-INF`. For integer
  types, inactive lanes are treated as the literal minimum value; if all lanes
  are inactive, `%result[0]` becomes that literal minimum value. The index is
  written into the second destination element slot of the same destination
  vector register.

---

### `pto.vcmin`

- **syntax:** `%result = pto.vcmin %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i16-i32, f16, f32
- **semantics:** Find min element with argmin. The lowest destination element
  stores the minimum value, the second-lowest destination element stores the
  index of the first minimum, and all remaining elements are zero-filled.

```c
T mn = INF; int idx = 0;
for (int i = 0; i < N; i++)
    if (src[i] < mn) { mn = src[i]; idx = i; }
dst[0] = mn;
dst[1] = idx;
for (int i = 2; i < N; i++)
    dst[i] = 0;
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result[0]` holds the extremum value and `%result[1]` holds the
  index. Other destination elements are zero-filled.
- **constraints and limitations:** If there are multiple minima, the minimum
  index is written. For floating-point types, inactive lanes are treated as
  `+INF`; if all lanes are inactive, `%result[0]` becomes `+INF`. For integer
  types, inactive lanes are treated as the literal maximum value; if all lanes
  are inactive, `%result[0]` becomes that literal maximum value. The index is
  written into the second destination element slot of the same destination
  vector register.

---

### `pto.vcbmax`

- **syntax:** `%value, %predicate = pto.vcbmax %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>, !pto.mask<G>`
- **A5 types:** i8-i32, f16, f32
- **semantics:** Find the maximum value and produce a predicate marking every
  participating lane whose value matches that maximum.

```c
T mx = max_active(src, mask);
for (int i = 0; i < N; i++) {
    value[i] = (i == 0) ? mx : 0;
    predicate[i] = mask[i] && matches_max(src[i], mx);
}
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%value[0]` holds the maximum and remaining value elements are
  zero-filled. `%predicate` marks all active lanes matching the maximum.
- **constraints and limitations:** If all lanes are inactive, `%predicate` is
  all zero. Floating-point inactive lanes are treated as `-INF`; integer
  inactive lanes are treated as the literal minimum value. For floating-point
  `+0/-0`, the value result follows the target maximum rule while predicate
  matching marks both zero signs as matching locations.

---

### `pto.vcbmin`

- **syntax:** `%value, %predicate = pto.vcbmin %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>, !pto.mask<G>`
- **A5 types:** i8-i32, f16, f32
- **semantics:** Find the minimum value and produce a predicate marking every
  participating lane whose value matches that minimum.

```c
T mn = min_active(src, mask);
for (int i = 0; i < N; i++) {
    value[i] = (i == 0) ? mn : 0;
    predicate[i] = mask[i] && matches_min(src[i], mn);
}
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%value[0]` holds the minimum and remaining value elements are
  zero-filled. `%predicate` marks all active lanes matching the minimum.
- **constraints and limitations:** If all lanes are inactive, `%predicate` is
  all zero. Floating-point inactive lanes are treated as `+INF`; integer
  inactive lanes are treated as the literal maximum value. Floating-point NaN
  handling follows the target instruction semantics.

---

## Histogram Reductions

### `pto.chistv2`

- **syntax:** `%result = pto.chistv2 %acc, %source, %mask, %bin : !pto.vreg<128xui16>, !pto.vreg<256xui8>, !pto.mask<b8>, i32 -> !pto.vreg<128xui16>`
- **semantics:** Cumulative histogram update over unsigned 8-bit source lanes.
  `%acc` provides the incoming 16-bit bin accumulators and `%result` contains
  the updated accumulators.
- **inputs:** `%source` provides 256 unsigned 8-bit samples, `%mask` selects
  active source lanes, and `%bin` is the target bin/control operand passed to
  the A5 histogram instruction.
- **constraints and limitations:** `%acc` and `%result` are fixed to
  `!pto.vreg<128xui16>`, `%source` is fixed to `!pto.vreg<256xui8>`, and the
  mask granularity is fixed to `b8`.

---

### `pto.dhistv2`

- **syntax:** `%result = pto.dhistv2 %acc, %source, %mask, %bin : !pto.vreg<128xui16>, !pto.vreg<256xui8>, !pto.mask<b8>, i32 -> !pto.vreg<128xui16>`
- **semantics:** Distribution histogram update over unsigned 8-bit source
  lanes. `%acc` provides the incoming 16-bit bin accumulators and `%result`
  contains the updated accumulators.
- **inputs:** `%source` provides 256 unsigned 8-bit samples, `%mask` selects
  active source lanes, and `%bin` is the target bin/control operand passed to
  the A5 histogram instruction.
- **constraints and limitations:** `%acc` and `%result` are fixed to
  `!pto.vreg<128xui16>`, `%source` is fixed to `!pto.vreg<256xui8>`, and the
  mask granularity is fixed to `b8`.

---

## Per-VLane (Group) Reductions

The vector register is organized as **8 VLanes** of 32 bytes each. Group reductions operate within each VLane independently.

```
vreg layout (f32 example, 64 elements total):
VLane 0: [0..7]   VLane 1: [8..15]  VLane 2: [16..23] VLane 3: [24..31]
VLane 4: [32..39] VLane 5: [40..47] VLane 6: [48..55] VLane 7: [56..63]
```

### `pto.vcgadd`

- **syntax:** `%result = pto.vcgadd %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i16-i32, f16, f32
- **semantics:** Sum within each 32-byte VLane. The 8 VLane results are written
  continuously to the low lanes of the destination vector. For f32, results are
  at indices 0, 1, 2, 3, 4, 5, 6, 7.

```c
int K = N / 8;  // elements per VLane
for (int g = 0; g < 8; g++) {
    T sum = 0;
    for (int i = 0; i < K; i++)
        sum += src[g*K + i];
    dst[g] = sum;
}
for (int i = 8; i < N; i++)
    dst[i] = 0;
// For f32: results at dst[0], dst[1], ..., dst[7].
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result` contains one sum per 32-byte VLane group, written
  continuously to the low lanes of the destination vector.
- **constraints and limitations:** This is a per-32-byte VLane-group reduction.
  Inactive lanes are treated as zero.

---

### `pto.vcgmax`

- **syntax:** `%result = pto.vcgmax %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i16-i32, f16, f32
- **semantics:** Max within each VLane.

```c
int K = N / 8;
for (int g = 0; g < 8; g++) {
    T mx = -INF;
    for (int i = 0; i < K; i++)
        if (src[g*K + i] > mx) mx = src[g*K + i];
    dst[g] = mx;
}
for (int i = 8; i < N; i++)
    dst[i] = 0;
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result` contains one maximum per 32-byte VLane group.
- **constraints and limitations:** Grouping is by hardware 32-byte VLane, not by
  arbitrary software subvector.

---

### `pto.vcgmin`

- **syntax:** `%result = pto.vcgmin %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i16-i32, f16, f32
- **semantics:** Min within each VLane.

```c
int K = N / 8;
for (int g = 0; g < 8; g++) {
    T mn = INF;
    for (int i = 0; i < K; i++)
        if (src[g*K + i] < mn) mn = src[g*K + i];
    dst[g] = mn;
}
for (int i = 8; i < N; i++)
    dst[i] = 0;
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result` contains one minimum per 32-byte VLane group.
- **constraints and limitations:** Grouping is by hardware 32-byte VLane, not by
  arbitrary software subvector.

---

## Prefix Operations

### `pto.vcpadd`

- **syntax:** `%result = pto.vcpadd %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** f16, f32
- **semantics:** Inclusive prefix sum (scan).

```c
dst[0] = src[0];
for (int i = 1; i < N; i++)
    dst[i] = dst[i-1] + src[i];
```

**Example:**
```c
// input:  [1, 2, 3, 4, 5, ...]
// output: [1, 3, 6, 10, 15, ...]
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result` is the inclusive prefix-sum vector.
- **constraints and limitations:** Only floating-point element types are
  documented on the current A5 surface here.

---

## Typical Usage

```mlir
// Softmax: find max for numerical stability
%max_vec = pto.vcmax %logits, %mask : !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>
// max is in lane 0, broadcast it
%max_broadcast = pto.vlds %ub_tmp[%c0] {dist = "BRC_B32"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

// Row-wise sum using vcgadd (for 8-row tile)
%row_sums = pto.vcgadd %tile, %mask : !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>
// Results at indices 0, 1, 2, 3, 4, 5, 6, 7

// Full vector sum for normalization
%total = pto.vcadd %values, %mask : !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>
// total[0] contains the sum

// Prefix sum for cumulative distribution
%cdf = pto.vcpadd %pdf, %mask : !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>
```
