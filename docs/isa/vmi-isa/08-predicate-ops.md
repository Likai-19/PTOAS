# 8. Predicate Ops

> **Category:** gen (mask producers — take no input mask).
> **Mask in:** none (they generate masks).
>
> Mask generation is expressed with two ops: `create_mask` (prefix / first-N
> tail) and `create_group_mask` (grouped prefix / grouped first-N tail). Mask
> values use `tilereg` with dtype `i1`; no predicate granularity is spelled in
> the type or op name.
>
> `create_mask` takes a single `index` operand `active_lanes`. When
> `active_lanes ≥ L` it yields an all-active mask; when `active_lanes = N < L`
> it yields a first-N tail mask. `create_group_mask` repeats the first-N pattern
> within each row of the two-dimensional result.
>
> These two predicate instructions have **no direct same-semantic Tile Op** and
> therefore retain their VMI names. In particular, `texpands` is scalar
> broadcast for the former scalar `vbrc`; it does not create prefix masks.

```mlir
%act  = arith.minsi %rem, %cL   // min(rem, L)
%aidx = arith.index_cast %act   // i32 -> index
%mask = pto.vmi.create_mask %aidx : index -> !pto.vmi.tilereg<1x128xi1>
%next = arith.subi %rem, %act   // rem - min(rem, L)
```

---

## `pto.vmi.create_mask`

- **syntax:**
  ```mlir
  %m = pto.vmi.create_mask %active_lanes : index -> !pto.vmi.tilereg<1xLxi1>
  ```
- **semantics:** Create a predicate mask where the first `active_lanes` logical
  lanes are active and the rest are inactive. `active_lanes ≥ L` produces an
  all-active mask; `active_lanes = N` produces a first-N tail mask.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = (i < active_lanes) ? 1 : 0;
  ```

- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `active_lanes` | `index` | Number of leading active lanes |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.tilereg<1xLxi1>` | Predicate mask |

- **example:**
  ```mlir
  // All-active mask (active_lanes >= L)
  %all = pto.vmi.create_mask %c128 : index -> !pto.vmi.tilereg<1x128xi1>

  // First-N tail mask (N = 64)
  %tail = pto.vmi.create_mask %c64 : index -> !pto.vmi.tilereg<1x128xi1>
  ```

---

## `pto.vmi.create_group_mask`

- **syntax:**
  ```mlir
  %m = pto.vmi.create_group_mask %active_elems_per_group
      : index -> !pto.vmi.tilereg<MxNxi1>
  ```
- **semantics:** Create a grouped predicate mask. Each of the `M` rows contains
  `N` lanes; lane `[m,n]` is active iff
  `n < active_elems_per_group`. When
  `active_elems_per_group ≥ N` all lanes are active within every group
  (grouped all-active); otherwise the first `active_elems_per_group` lanes are
  active within each group (grouped first-N tail).

  ```c
  for (int m = 0; m < M; m++)
      for (int n = 0; n < N; n++)
          dst[m][n] = (n < active_elems_per_group) ? 1 : 0;
  ```

- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `active_elems_per_group` | `index` | Active lanes within each group |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.tilereg<MxNxi1>` | Grouped predicate mask |

- **example:**
  ```mlir
  // Grouped all-active: 8 groups, group size 32, all lanes active per group
  %all = pto.vmi.create_group_mask %c32
      : index -> !pto.vmi.tilereg<8x32xi1>

  // Grouped first-N tail: first 25 lanes per group, 8 groups
  %tail = pto.vmi.create_group_mask %c25
      : index -> !pto.vmi.tilereg<8x32xi1>
  ```

---

> **Mask Boolean Ops (`tand` / `tor` / `txor` / `tnot` on masks):**
>
> There is **no dedicated predicate-logic op** (e.g. `pand`/`por`/`pxor`/`pnot`).
> Mask (predicate) boolean operations are **not yet supported**, but are planned.
> The planned approach is to **reuse the elementwise bitwise ops** `pto.vmi.tand` /
> `tor` / `txor` / `tnot` directly on mask operands — their implementations will be
> extended to accept `i1` tileregs (treated as a per-lane bit-wise boolean op on the
> predicate). This also covers the `pnot`-style predicate complement needed by MERGE
> emulation (see [Appendix C](10-appendices.md)).
