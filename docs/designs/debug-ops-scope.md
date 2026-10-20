# Conditional assertions and persistent tensor dumps

## Scope

This change addresses the debug-interface subtask of the instruction-coverage
issue. Other instruction families remain separate work. It introduces no
replacement for existing print or trap operations.

| Operation | Purpose | Backend |
| --- | --- | --- |
| `pto.print` | Formatted scalar text output | Existing support unchanged |
| `pto.tprint` / `pto.tile.print` | Tile text output | Existing support unchanged |
| `pto.trap` | Unconditional trap operation | Existing support unchanged |
| `pto.assert` / `pto.assert_` | Continue when a scalar condition is true; request a trap otherwise | EmitC and VPTO |
| `pto.tdump` / `pto.tile.dump` | Binary tile metadata and payload in caller-owned GM | VPTO only |

The assertion message is printed by EmitC; VPTO currently ignores it. The
interface does not promise identical host error codes or diagnostics across
backends. A false condition emits a backend trap request, but this change does
not promise that every target or simulator stops the kernel or reports a host
error. PTODSL `assert_` emits a device check, unlike Python `assert`, which
runs during host tracing.

## Conditional assertion limitations

The scope of this PR is a conditional diagnostic interface: a true condition
continues execution; a false condition emits the existing backend trap
operation. Reliable termination after failure is not provided by this change.
Kernel termination, suppression of later device writes, and host exception
propagation are not acceptance claims of this PR. Callers must use explicit
control flow to prevent unsafe operations, rather than relying on this check.

The parent issue asks for conditional assert and persistent tensor dump
interfaces; it does not explicitly require fail-fast termination. Reliable
termination remains separate follow-up work, requiring a defined target/runtime
contract and runtime evidence. The false-condition probe remains available to
detect that gap; it must not be hidden or reclassified as a passing test.

## Tensor dump v1

The format remains a 64-byte little-endian header followed by contiguous
row-major payload. Only rank-2 vec/UB tiles are supported. The caller reserves
header plus payload storage, uses separate regions for separate blocks, and
reads the buffer after normal completion and synchronization.

Static valid dimensions are self-describing. Dynamic valid dimensions remain
encoded as `0xFFFFFFFF`; a decoder needs their actual runtime extents from
the caller. This change does not redefine the sentinel or version the format.
Runtime destination sizes must equal the tile's runtime valid dimensions.

Persistence does not imply crash-safe output: a dump followed by trap or a
failed assertion is not guaranteed complete or readable. EmitC dumps,
MAT/ACC staging, padded destinations, automatic allocation, and a general
host decoding tool are outside this change.

## Shared string-literal handling

The EmitC string helper preserves quote and backslash escaping and encodes
control bytes safely. These fixes also affect generated `pto.print` string
literals; they are intentional behavior corrections, not a strictly
behavior-preserving refactor. Embedded NUL is escaped correctly in generated
C++, but C-string consumers still terminate at NUL.

## Validation status

The October 14, 2026 local validation built both the compiler and test tools.
The full lit suite passed 2027 tests with one unsupported simulator-dependent
test. All 33 focused debug-op tests passed. The PTODSL documentation suite
executed 197 tagged tests, including the assertion and dump examples.

The A5 dav_3510 simulator passed the partial-valid dump case with byte-exact
payload comparison and a true assertion condition. With CANN 9.2.0-beta.2,
the fast `simulator/dav_3510/lib` runtime executes TRAP but continues into
the dump. The cycle-accurate `simulator/dav_3510/camodel` runtime reports
`TRAP instr, core break!`, but the dump buffer is nevertheless written.
Neither a TRAP log nor a zero host exit code proves fail-fast termination.
The runtime probe initializes the entire destination to a byte sentinel and
rejects any post-assert modification. Reliable fail-fast remains follow-up work;
no physical-device termination result is claimed.
