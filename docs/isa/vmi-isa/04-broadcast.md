# 4. Broadcast

> **Category:** A (scalar expansion), B (row expansion).
> **Mask:** none.
>
> The former `vbrc` forms use the matching Tile Op names: scalar broadcast is
> `texpands`, while grouped broadcast is `trowexpand`. The scalar form
> (single scalar fanned over `L` lanes) is cheap (`vdup`); the grouped form
> (per-group scalar fan-back) has no single native instruction and is a
> cost-model decision.

---

## `pto.vmi.texpands` / `pto.vmi.trowexpand`

- **semantics:** Broadcast a scalar or group-slot compact value across lanes.

  **Ungrouped:** One value replicated to all `L` lanes.
  ```c
  for (int i = 0; i < L; i++)
      dst[i] = src[0];
  ```

  **Grouped:** Each of the `M` row scalars is fanned back across `N` lanes.
  ```c
  for (int m = 0; m < M; m++)
      for (int n = 0; n < N; n++)
          dst[m][n] = src[m][0];
  ```

- **syntax:**
  ```mlir
  // Ungrouped: scalar → full vector
  %r = pto.vmi.texpands %scalar : f32 -> !pto.vmi.tilereg<1x64xf32>

  // Ungrouped: 1-lane tilereg → full vector
  %r = pto.vmi.texpands %val : !pto.vmi.tilereg<1x1xf32> -> !pto.vmi.tilereg<1x256xf32>

  // Grouped: one scalar per row → dense rows
  %r = pto.vmi.trowexpand %source
      : !pto.vmi.tilereg<128x1xf32> -> !pto.vmi.tilereg<128x8xf32>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `value` | `T`, `!pto.vmi.tilereg<1x1xT>`, or `!pto.vmi.tilereg<Mx1xT>` | Broadcast source |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.tilereg<1xLxT>` or `!pto.vmi.tilereg<MxNxT>` | Broadcast result |

- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**

  | Form | Physical lowering | `#mi` | `dep` |
  |---|---|---|---|
  | Ungrouped (scalar) | `1 × pto.vdup` (register-resident), or `vsts`+`vlds BRC_*` (UB roundtrip) | `1` | `1` |
  | Ungrouped (1-lane tilereg) | `1 × pto.vdup {position="LOWEST"}` per physical reg | `K` | `1` |
  | Grouped (`Mx1 -> MxN`) | **Cost-model decision**: UB roundtrip (`vsts` partials + `vlds BRC_BLK`) **or** `vselr` gather **or** masked recompute | varies | 2–3 |

- **examples:**
  ```mlir
  // Ungrouped: scalar → full vector
  %bc = pto.vmi.texpands %maxe : f32 -> !pto.vmi.tilereg<1x64xf32>
  // → pto.as: pto.vdup %maxe (one op, register-resident)

  // Ungrouped: 1-lane tilereg → full vector (rank-0 broadcast)
  %bc = pto.vmi.texpands %scalar : !pto.vmi.tilereg<1x1xf32> -> !pto.vmi.tilereg<1x256xf32>
  // → pto.as: 4 × pto.vdup {position="LOWEST"} (K=4)

  // Grouped: 128 row scalars → 128 rows × 8 lanes
  %bc = pto.vmi.trowexpand %source
      : !pto.vmi.tilereg<128x1xf32> -> !pto.vmi.tilereg<128x8xf32>
  // → pto.as: 16 × pto.vselr (vselr gather realization)
  ```

- **notes:**
  - Fused `reduce→broadcast` (`trowsum`+`trowexpand`) is the recognized fusion pattern:
    `pto.as` emits them back-to-back and keeps the result as a broadcast axis
    rather than materializing `K` copies.
  - Prefer `vdup` over a UB `BRC` reload for a single scalar.
  - Grouped broadcast has **no single native `pto.mi` op** — `pto.as` picks
    UB roundtrip (default, `vsts` partials + `vlds BRC_BLK`), `vselr` gather
    (when group count and K are tiny), or masked recompute (very small groups).

`texpands` is only the scalar-broadcast mapping of `vbrc`; it is not a mask
creation op. `create_mask` and `create_group_mask` remain VMI predicate
instructions because Tile Op has no direct operation with their prefix-mask
semantics.

`vexpdif` is not renamed to `trowexpandexpdif`. The latter may be selected only
as a constrained optimization when the subtracted value is provably one scalar
per row, all participating dtypes are `f32`, and its valid-region behavior is
equivalent to the VMI mask and `pmode`.
