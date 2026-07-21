# 6. Convert

> **Category:** B (`vcvt`), A (`vinterpret_cast`).
> **Mask:** `Pg` (`vcvt`), none (`vinterpret_cast`).
>
> One logical `vcvt` whose target dtype IS the layout. `pto.as` expands it into
> the dtype-specific cast chain + part/width staging + matching store
> distribution. The author never spells `EVEN`/`ODD`, `P0`–`P3`, `PK`/`UNPK`,
> or `VL/2` addresses.

---

## `pto.vmi.vcvt`

- **semantics:** Unified elementwise type conversion. The conversion direction
  is derived from the source and destination element types; the verifier
  dispatches to one of six kinds:

  1. **FpWiden** — `fp → fp`, `|dst| > |src|` (e.g. `f16 → f32`,
     `bf16 → f32`, `fp8_e4m3 → f16`).

  2. **FpNarrow** — `fp → fp`, `|dst| < |src|` (e.g. `f32 → f16`,
     `f32 → bf16`, `f32 → fp8_e4m3`).

  3. **FpToSi** — `fp → int` (e.g. `f32 → i32`, `f16 → i8`).

  4. **SiToFp** — `int → fp` (e.g. `i32 → f32`, `i8 → f16`).

  5. **IntWiden** — `int → int`, `|dst| > |src|`.

  6. **IntNarrow** — `int → int`, `|dst| < |src|`.

- **syntax:**
  ```mlir
  %r = pto.vmi.vcvt %src {rounding = "H", saturate = "SAT"} : !pto.vmi.vreg<L×T_src> -> !pto.vmi.vreg<L×T_dst>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `src` | `!pto.vmi.vreg<L×T_src>` | Source vector |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<L×T_dst>` | Converted vector (same `L`, different `T`) |

- **attributes:**

  | Attribute | Values | Valid for | Description |
  |---|---|---|---|
  | `rounding` | `"R"` (nearest-even), `"A"` (away-from-zero), `"H"` (half-up), `"Z"` (toward-zero) | fp narrowing | Rounding mode |
  | `saturate` | `"SAT"`, `"NOSAT"` | **required** for fp-narrow / int-narrow / fp→si | `SAT` clamps to ±max of the destination type; `NOSAT` performs a direct bit truncation of the result representation. |

- **datatypes:** Source and destination from `{f32, f16, bf16, fp8_e4m3, fp8_e5m2, i32, i16, i8, ui32, ui16, ui8}`
- **lowering to `pto.mi`:**

  | Conversion | Physical lowering | `#mi` | `dep` |
  |---|---|---|---|
  | 16↔32 (radix-2) | `2K × vcvt EVEN/ODD` + predicate `ppack`/`punpack` companion | `2K` | `2` |
  | 8↔32 (radix-4) | widen: `UNPK_B8` + `vintlv` + `vcvt P0` + `punpack`; narrow: `PK4_B32` store (or `vselr` gather) + `ppack` | `2–3` | `2–3` |
  | f32→fp8 quant | `1 cast` + `PK4_B32` | `K` | `1` |
  | f32→int8 quant | 3-stage cast + `PK4_B32` | `~3K` | `3` |
  | int↔int (same width) | `K × vtrc` or `K × vcvt` | `K` | `1` |

- **example:**
  ```mlir
  // fp16 → fp32 widen (radix-2, produces parity EVEN/ODD)
  %w = pto.vmi.vcvt %a
      : !pto.vmi.vreg<128×f16>
      -> !pto.vmi.vreg<128×f32>
  // → pto.as: 2 × pto.vcvt EVEN/ODD + ppack (parity companion)

  // fp32 → fp16 narrow with half-up rounding
  %n = pto.vmi.vcvt %y {rounding = "H", saturate = "SAT"}
      : !pto.vmi.vreg<64×f32> -> !pto.vmi.vreg<64×f16>

  // ui8 -> i16 unsigned extension
  %z = pto.vmi.vcvt %a
      : !pto.vmi.vreg<256×ui8> -> !pto.vmi.vreg<256×i16>

  // f32 → fp8 quantized narrow (saturate required)
  %q = pto.vmi.vcvt %s {saturate = "SAT"}
      : !pto.vmi.vreg<64×f32> -> !pto.vmi.vreg<64×fp8_e4m3>

  // i32 → i8 int-narrow without saturation (wrap on overflow)
  %t = pto.vmi.vcvt %v {saturate = "NOSAT"}
      : !pto.vmi.vreg<64×i32> -> !pto.vmi.vreg<64×i8>

  // f32 → i32 fp-to-si (saturate required)
  %r = pto.vmi.vcvt %x {saturate = "SAT"}
      : !pto.vmi.vreg<64×f32> -> !pto.vmi.vreg<64×i32>
  ```

- **notes:**
  - `vcvt` **does not change lane count** — `src.L == dst.L` always. The
    physical register count `K` changes because `bitwidth(T)` changes.
  - Integer signedness is determined by the **element type**.
  - The `part`/`parity`/`width` axes are lowering-only; the user never writes
    `EVEN`/`ODD`/`P0..P3`.
  - Radix-4 (8↔32) is **not** a stacked predicate chain and **not** a UB
    roundtrip; the 1↔4 lane spread rides data load/store distribution
    (`UNPK_B*`/`PK4_B32`) or a `vselr` byte-gather.

---

## `pto.vmi.vinterpret_cast`

- **semantics:** Bitwise reinterpretation of a vector register — same bits,
  different element type. No data movement, no layout change.

  ```c
  // Same bits, reinterpreted element-by-element
  memcpy(&dst, &src, L * sizeof(T_src));
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vinterpret_cast %src : !pto.vmi.vreg<L×T_src> -> !pto.vmi.vreg<L×T_dst>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `src` | `!pto.vmi.vreg<L×T_src>` | Source vector |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<L×T_dst>` | Bit-reinterpreted vector |

- **attributes:** *(none)*
- **datatypes:** Any `T_src`, `T_dst` with `L · bitwidth(T_src) == L · bitwidth(T_dst)`
- **lowering to `pto.mi`:**
  ```
  K × pto.vbitcast (or no-op if same physical layout)
  ```
  `#mi = 0` or `K`, `dep = 0` or `1`.

- **notes:**
  - **Category A** — layout-transparent, no new axis produced.
  - This is **not** `vcvt` — no dtype cast chain, no `part`/`parity`/`width`
    axis, no `[pmode]`.
  - The user must ensure semantic legality (e.g., `f32` → `i32` bitcast is
    valid; `f32` → `f16` is not — use `vcvt` for that).

- **example:**
  ```mlir
  %r = pto.vmi.vinterpret_cast %a : !pto.vmi.vreg<64×f32> -> !pto.vmi.vreg<64×i32>
  ```
