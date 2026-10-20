# Tensor dump and conditional assertion runtime validation

The normal case passes a nonzero assertion condition, loads an 8x32 f32 tile,
and dumps only its 4x16 valid region. The host comparison checks the full
shape, valid shape, all header fields, and the contiguous payload. The source
row pitch is 32 elements while the dump row width is 16 elements.

Run through `test/vpto/scripts/run_host_vpto_validation.sh` with
`CASE_NAME=kernels/tdump-vec`, a configured `PTOAS_BIN`, `ASCEND_HOME_PATH`,
`WORK_SPACE`, and simulator or device selection. A successful comparison
also demonstrates that a true assertion condition allows execution to
continue.

## False assertion probe

This probe tracks the known failure-termination limitation; it does not change
the current interface's conditional trap-request contract into a fail-fast
guarantee. A failing probe must be reported, not treated as a passing result.

After building the normal case, run the generated `kernels_tdump-vec` binary
in a separate process with `PTOAS_TEST_ASSERT_FAIL=1`, the same runtime
environment and working directory, and an external timeout. The assertion
occurs before any tile load or dump. Do not run the normal payload comparison
as proof of an assertion failure, and do not assume all backends expose the
same host error code.

The host initializes the entire dump destination to `0xA5` before launch.
If synchronization succeeds in false-condition mode, every destination byte
must remain unchanged; otherwise the probe reports an error and returns 1.
False-condition mode never writes a new `v2.bin`. Use a fresh working
directory so a stale normal-path dump cannot be mistaken for current output.
If synchronization reports an error, verify it is a target trap exception,
not an unrelated launch/runtime failure; the generic error path is not a
passing assertion test. A timeout is also not successful termination.

With CANN 9.2.0-beta.2, the October 14, 2026 fast-runtime probe issued TRAP
and continued into the dump. The cycle-accurate `dav_3510/camodel` probe
reported `TRAP instr, core break!`, but still produced the dump header and
payload. Neither result passes fail-fast acceptance. No physical-device
result is claimed. Do not mark these runs as expected failures or use
the presence of TRAP/core-break logs as proof of termination.
