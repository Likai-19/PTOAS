# kernel-test scripts

This directory is split into two groups:

## User-facing entrypoints

- `run_cycle.sh`
  - Recommended command for cycle collection.
  - Expands selected cases, invokes `kernel-test/run.py` once per case, and prints a cycle report.
- `run_sim.sh`
  - Generic `cannsim` transport for one Python entrypoint.
  - Use this when you want to run a specific script under cannsim directly.
- `run_msprof.sh`
  - Generic `msprof` transport for one Python entrypoint.
  - Use this when you want direct `msprof` artifacts for a specific script.

## Internal helpers

- `helpers/common.sh`
- `helpers/run_sim_entry.sh`
- `helpers/report_cycles.py`

These helper scripts are framework internals. They are called by the user-facing
entrypoints above and are not intended to be run directly.
