# 6. Convert

> **Category:** B (`tcvt`), A (`treshape`).
> **Mask:** `Pg` (`tcvt`), none (`treshape`).
>
> One logical `tcvt` whose target dtype IS the layout. `pto.as` expands it into
> the dtype-specific cast chain + part/width staging + matching store
> distribution. The author never spells `EVEN`/`ODD`, `P0`–`P3`, `PK`/`UNPK`,
> or `VL/2` addresses.

---

## `pto.vmi.tcvt`

- **semantics:** Unified elementwise type conversion. The conversion direction
  is derived from source and destination element types:

  | Direction | Condition | Replaces |
  |---|---|---|
  | fp → fp, `|dst| > |src|` | Floating-point widening | `extf` |
  | fp → fp, `|dst| < |src|` | Floating-point narrowing | `truncf` |
  | fp → int | Float to signed integer | `fptosi` |
  | int → fp | Signed integer to float | `sitofp` |
  | int -> int, `|dst| > |src|` | Integer extension (sign from source element type) | `extsi` / `extui` |
  | int → int, `|dst| < |src|` | Saturating integer truncation | `trunci` |

- **syntax:**
  ```mlir
  %r = pto.vmi.tcvt %src {rounding = "H"} : !pto.vmi.tilereg<MxNxT_src> -> !pto.vmi.tilereg<MxNxT_dst>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `src` | `!pto.vmi.tilereg<MxNxT_src>` | Source vector |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.tilereg<MxNxT_dst>` | Converted vector (same `L`, different `T`) |

- **attributes:**

  | Attribute | Values | Valid for | Description |
  |---|---|---|---|
  | `rounding` | `"A"` (away-from-zero), `"H"` (half-up) | fp narrowing | Rounding mode |
  | `saturate` | `"SAT"` | any narrowing | Saturating on overflow |
  | `pmode` | `"zero"`, `"merge"` | all | Inactive-lane behavior |

- **datatypes:** Source and destination from `{f32, f16, bf16, fp8_e4m3, fp8_e5m2, i32, i16, i8, ui32, ui16, ui8}`
- **lowering to `pto.mi`:**

  | Conversion | Physical lowering | `#mi` | `dep` |
  |---|---|---|---|
  | 16↔32 (radix-2) | `2K × vcvt EVEN/ODD` + predicate `ppack`/`punpack` companion | `2K` | `2` |
  | 8↔32 (radix-4) | widen: `UNPK_B8` + `tinterleave` + `vcvt P0` + `punpack`; narrow: `PK4_B32` store (or `vselr` gather) + `ppack` | `2–3` | `2–3` |
  | f32→fp8 quant | `1 cast` + `PK4_B32` | `K` | `1` |
  | f32→int8 quant | 3-stage cast + `PK4_B32` | `~3K` | `3` |
  | int↔int (same width) | `K × vtrc` or `K × vcvt` | `K` | `1` |

- **example:**
  ```mlir
  // fp16 → fp32 widen (radix-2, produces parity EVEN/ODD)
  %w = pto.vmi.tcvt %a
      : !pto.vmi.tilereg<1x128xf16>
      -> !pto.vmi.tilereg<1x128xf32>
  // → pto.as: 2 × pto.vcvt EVEN/ODD + ppack (parity companion)

  // fp32 → fp16 narrow with half-up rounding
  %n = pto.vmi.tcvt %y {rounding = "H"}
      : !pto.vmi.tilereg<1x64xf32> -> !pto.vmi.tilereg<1x64xf16>

  // ui8 -> i16 unsigned extension
  %z = pto.vmi.tcvt %a
      : !pto.vmi.tilereg<1x256xui8> -> !pto.vmi.tilereg<1x256xi16>

  // f32 → fp8 quantized narrow
  %q = pto.vmi.tcvt %s
      : !pto.vmi.tilereg<1x64xf32> -> !pto.vmi.tilereg<1x64xfp8_e4m3>
  ```

- **notes:**
  - `tcvt` **does not change lane count** — `src.L == dst.L` always. The
    physical register count `K` changes because `bitwidth(T)` changes.
  - Integer signedness is determined by the **element type**.
  - The `part`/`parity`/`width` axes are lowering-only; the user never writes
    `EVEN`/`ODD`/`P0..P3`.
  - Radix-4 (8↔32) is **not** a stacked predicate chain and **not** a UB
    roundtrip; the 1↔4 lane spread rides data load/store distribution
    (`UNPK_B*`/`PK4_B32`) or a `vselr` byte-gather.

---

## `pto.vmi.treshape`

- **semantics:** Reshape a tile register while preserving its row-major bit
  sequence. It replaces both one-dimensional/two-dimensional reshape and the
  former `vinterpret_cast`. The result may change shape, dtype, or both, but
  the source and result must have the same total bit size. No numeric
  conversion is performed.

  ```c
  // Same row-major bits, reinterpreted through the destination shape/dtype
  memcpy(&dst, &src, Msrc * Nsrc * sizeof(T_src));
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.treshape %src
      : !pto.vmi.tilereg<MsrcxNsrcxT_src>
        -> !pto.vmi.tilereg<MdstxNdstxT_dst>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `src` | `!pto.vmi.tilereg<MsrcxNsrcxT_src>` | Source tile register |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.tilereg<MdstxNdstxT_dst>` | Reshaped/bit-reinterpreted tile register |

- **attributes:** *(none)*
- **datatypes:** Any supported `T_src`, `T_dst` satisfying
  `Msrc · Nsrc · bitwidth(T_src) == Mdst · Ndst · bitwidth(T_dst)`.
- **lowering to `pto.mi`:**
  ```
  K × pto.vbitcast (or no-op if same physical layout)
  ```
  `#mi = 0` or `K`, `dep = 0` or `1`.

- **notes:**
  - **Category A** — layout-transparent, no new axis produced.
  - This is **not** `tcvt` — no dtype cast chain, no `part`/`parity`/`width`
    axis, no `[pmode]`.
  - The user must ensure semantic legality (e.g., `1x64xf32` → `1x64xi32`
    and `1x128xf16` → `8x16xf16` are valid; `1x64xf32` → `1x64xf16` is not
    because the total bit size changes — use `tcvt` for numeric conversion).

- **example:**
  ```mlir
  %r = pto.vmi.treshape %a : !pto.vmi.tilereg<1x64xf32> -> !pto.vmi.tilereg<1x64xi32>

  %rows = pto.vmi.treshape %flat
      : !pto.vmi.tilereg<1x128xf16> -> !pto.vmi.tilereg<8x16xf16>
  ```
