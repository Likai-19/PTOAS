#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Flat driver for the PTODSL/DSL ST suite (test/dsl-st): one unit is one
# simulator session. A module with more cases than PTODSL_CASE_BLOCK_SIZE is cut
# into blocks; a single case module in a heavy family keeps its own session; the
# other single case modules are packed PTOAS_DSL_PACK_SIZE to a session. Units
# start longest first through a fixed size xargs pool, each has its own log
# directory, and any failing or timed out unit fails the step.
#
# Discovery runs globally first (test/dsl-st/__main__.py --list) to enforce
# case-name uniqueness and fix the authoritative case set, then per module to
# shard it; the two sets must match exactly. Every unit must report exactly the
# PASS names discovery assigned to it -- a duplicate or a missing name fails --
# and the global passed total must equal the discovered total, so a silently
# skipped or repeated case cannot stay green.
#
# A unit killed by SIGSEGV (exit code 139) is retried once: the CANN simulator
# can dereference a device VA as a host pointer and kill a random unit
# (cann/pto-as#17). Every other exit code -- timeout, compare failure, coverage
# -- is final. This retry is temporary and must be deleted with the CANN fix;
# the summary prints retried=N and the unit is marked RETRIED. See
# test/dsl-st/README.md.
#
# PTOAS_DSL_HEAVY_PATTERN is an extended regex over module names. Unset keeps the
# driver default -- the heavy families measured on the CI runner (cube matmul,
# cv_mixed/fixpipe, cast_spine, simt, binary_vector and vmi_new_kernels, which
# had repeated units over 150 s while the rest sat under 100 s). An empty value
# or "-" matches no module, which disables the heavy families and packs every
# single case module; add a family once its units measure over 100 s.
# PTOAS_DSL_PACK_SIZE=1 turns packing off.
#
# Env: WORKSPACE, BUILD_ROOT, PTODSL_DSL_ROOT, SIM_DSL_SCRIPT, PYTHON_BIN,
# PTODSL_CASE_JOBS (24), PTODSL_CASE_BLOCK_SIZE (10), PTODSL_CASE_TIMEOUT (1800),
# PTODSL_CASE_KILL_AFTER (60), PTODSL_DISCOVERY_TIMEOUT (300),
# PTODSL_FAILED_LOG_LINES (200), PTOAS_DSL_PACK_SIZE (16),
# PTOAS_DSL_HEAVY_PATTERN (see above).

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE="${WORKSPACE:-${REPO_ROOT}}"
BUILD_ROOT="${BUILD_ROOT:-${WORKSPACE}/.work/gitcode-vpto-sim}"
DSL_ROOT="${PTODSL_DSL_ROOT:-${WORKSPACE}/test/dsl-st}"
SIM_DSL_SCRIPT="${SIM_DSL_SCRIPT:-${WORKSPACE}/scripts/sim_dsl.sh}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
JOBS="${PTODSL_CASE_JOBS:-24}"
BLOCK_SIZE="${PTODSL_CASE_BLOCK_SIZE:-10}"
UNIT_TIMEOUT="${PTODSL_CASE_TIMEOUT:-1800}"
KILL_AFTER="${PTODSL_CASE_KILL_AFTER:-60}"
DISCOVERY_TIMEOUT="${PTODSL_DISCOVERY_TIMEOUT:-300}"
FAILED_LOG_LINES="${PTODSL_FAILED_LOG_LINES:-200}"
PACK_SIZE="${PTOAS_DSL_PACK_SIZE:-16}"
HEAVY_PATTERN_DEFAULT="legacy_micro_op_cube_matmul|legacy_micro_op_cv_mixed|legacy_cast_spine"
HEAVY_PATTERN_DEFAULT="${HEAVY_PATTERN_DEFAULT}|legacy_micro_op_simt|legacy_micro_op_binary_vector"
HEAVY_PATTERN_DEFAULT="${HEAVY_PATTERN_DEFAULT}|legacy_vmi_new_kernels"
HEAVY_PATTERN="${PTOAS_DSL_HEAVY_PATTERN-${HEAVY_PATTERN_DEFAULT}}"
# An empty value or "-" disables the heavy families: "^$" matches no module.
if [[ -z "${HEAVY_PATTERN}" || "${HEAVY_PATTERN}" == "-" ]]; then
  HEAVY_PATTERN='^$'
fi
WORK_DIR="${BUILD_ROOT}/ptodsl-cases"
LOG_DIR="${WORK_DIR}/logs"
RESULT_DIR="${WORK_DIR}/results"
CASES_DIR="${WORK_DIR}/cases"
SUMMARY_FILE="${WORK_DIR}/parallel-summary.tsv"
SLICE_DRIVER="${WORK_DIR}/run_case_slice.py"
UNIT_RUNNER="${WORK_DIR}/run_unit.sh"
UNIT_LIST="${WORK_DIR}/unit-list.txt"
UNIT_LIST_NUL="${WORK_DIR}/unit-list.nul"

log() { echo "[ptodsl-parallel] $*"; }
die() { echo "ERROR: $*" >&2; exit 2; }
[[ -d "${DSL_ROOT}" ]] || die "missing DSL ST suite directory: ${DSL_ROOT}"
[[ -f "${SIM_DSL_SCRIPT}" ]] || die "missing serial PTODSL runner: ${SIM_DSL_SCRIPT}"
[[ "${JOBS}" =~ ^[0-9]+$ && "${JOBS}" -ge 1 ]] || die "PTODSL_CASE_JOBS must be >= 1, got ${JOBS}"
[[ "${BLOCK_SIZE}" =~ ^[0-9]+$ && "${BLOCK_SIZE}" -ge 1 ]] \
  || die "PTODSL_CASE_BLOCK_SIZE must be >= 1, got ${BLOCK_SIZE}"
[[ "${UNIT_TIMEOUT}" =~ ^[0-9]+$ && "${UNIT_TIMEOUT}" -ge 1 ]] \
  || die "PTODSL_CASE_TIMEOUT must be >= 1, got ${UNIT_TIMEOUT}"
[[ "${KILL_AFTER}" =~ ^[0-9]+$ && "${KILL_AFTER}" -ge 1 ]] \
  || die "PTODSL_CASE_KILL_AFTER must be >= 1, got ${KILL_AFTER}"
[[ "${PACK_SIZE}" =~ ^[0-9]+$ && "${PACK_SIZE}" -ge 1 ]] || die "PTOAS_DSL_PACK_SIZE must be >= 1, got ${PACK_SIZE}"
command -v timeout >/dev/null 2>&1 || die "timeout is unavailable"
command -v xargs >/dev/null 2>&1 || die "xargs is unavailable"
# nullglob keeps an empty glob from expanding to a literal pattern; every
# empty result is reported explicitly instead.
shopt -s nullglob
rm -rf "${WORK_DIR}"
mkdir -p "${LOG_DIR}" "${RESULT_DIR}" "${CASES_DIR}"

# Global discovery loads every module through common.discover_cases, which
# rejects duplicate case names across the suite, and records the authoritative
# case set. The per-module lists below must cover exactly this set.
ALL_CASES="${WORK_DIR}/all-cases.txt"
PER_MODULE_CASES="${WORK_DIR}/per-module-cases.txt"
if ! global_cases="$(timeout "${DISCOVERY_TIMEOUT}" "${PYTHON_BIN}" \
    "${DSL_ROOT}/__main__.py" --list 2>>"${WORK_DIR}/discovery.log")"; then
  reason="$(tail -n 3 "${WORK_DIR}/discovery.log" 2>/dev/null | tr "\n" " ")"
  die "cannot list the ${DSL_ROOT} cases globally (${reason:-none});" \
    "full log: ${WORK_DIR}/discovery.log"
fi
printf "%s\n" "${global_cases}" | sed "/^$/d" | sort > "${ALL_CASES}"
: > "${PER_MODULE_CASES}"

# One case range of one module, or one whole set of modules, through the product
# case loop. Resetting sys.argv keeps a module that parses its own arguments from
# seeing the driver arguments; a module that raises does not stop the next one.
cat > "${SLICE_DRIVER}" <<\PY
#!/usr/bin/env python3
import importlib.util
import os
import sys
import traceback
from pathlib import Path

args = sys.argv[1:]
if args and args[0] == "--modules":
    targets = [(Path(name).resolve(), None, None) for name in args[1:]]
elif len(args) == 3:
    targets = [(Path(args[0]).resolve(), int(args[1]), int(args[2]))]
else:
    raise SystemExit("usage: run_case_slice.py <module.py> <begin> <end> | --modules <module.py> ...")
if not targets:
    raise SystemExit("no module to run")

sys.path.insert(0, str(targets[0][0].parent))
import common  # noqa: E402

failed = 0
for index, (module_path, begin, end) in enumerate(targets):
    saved_argv = sys.argv
    try:
        sys.argv = [str(module_path)]
        spec = importlib.util.spec_from_file_location("_dsl_st_unit_%d" % index, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load case module: %s" % module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cases = module.CASES if begin is None else module.CASES[begin:end]
        if not cases:
            raise RuntimeError("empty case range of %s" % module_path)
        common.run_cases(cases, argv=[])
    except BaseException:  # pylint: disable=broad-except
        failed += 1
        print("SLICE-MODULE-FAIL %s" % module_path.name, file=sys.stderr)
        traceback.print_exc()
    finally:
        sys.argv = saved_argv
rc = 1 if failed else 0
# Exit without Py_FinalizeEx so torch_npu teardown cannot free the model
# while the simulator background threads still read it (SIGSEGV after the
# cases already finished). Flush first so no PASS line is lost.
sys.stdout.flush()
sys.stderr.flush()
os._exit(rc)
PY

# One session for one spec (unit|kind|payload) under the unit timeout; always
# exits 0 so the pool keeps going, the step status comes from the result files.
cat > "${UNIT_RUNNER}" <<\SH
#!/usr/bin/env bash
set -uo pipefail
IFS="|" read -r expected unit kind rest <<< "$1"
out="${LOG_DIR}/${unit}"
log="${LOG_DIR}/${unit}.log"
started="$(date +%s)"
if [[ "${kind}" == "pack" ]]; then
  read -r -a mods <<< "${rest}"
  payload=(-- --modules "${mods[@]}")
else
  read -r module begin end <<< "${rest}"
  payload=(-- "${module}" "${begin}" "${end}")
fi
run_once() {
  rm -rf "${out}"; mkdir -p "${out}"
  # Keep the ptodsl native build scratch inside the unit output so it is
  # wiped with the run instead of landing in /tmp or ~/.cache.
  export TMPDIR="${out}/tmp"
  mkdir -p "${TMPDIR}"
  timeout --signal=TERM --kill-after="${KILL_AFTER}" "${UNIT_TIMEOUT}" \
    bash "${SIM_DSL_SCRIPT}" --output "${out}" "${SLICE_DRIVER}" "${payload[@]}" \
    > "${log}" 2>&1
  return $?
}
set +e
run_once
rc=$?
retried=0
# Temporary mitigation for cann/pto-as#17: the simulator can dereference a
# device VA as a host pointer and kill one unit with SIGSEGV (rc 139). Retry
# that unit exactly once. Only rc 139 retries; timeout (124/137), compare
# failure (1), coverage and every other exit code are final. Delete this
# block once the CANN fix lands.
if [[ "${rc}" == 139 ]]; then
  retried=1
  cp "${log}" "${LOG_DIR}/${unit}.sigsegv.log" 2>/dev/null || true
  run_once
  rc=$?
fi
set -e
elapsed="$(( $(date +%s) - started ))"
printf "%s %s %s %s %s\n" "${unit}" "${rc}" "${elapsed}" "${expected}" "${retried}" \
  > "${RESULT_DIR}/${unit}.result"
exit 0
SH

: > "${WORK_DIR}/discovery.log"
entries=()
pack_modules=()
pack_names=()
pack_cases=0
pack_index=0
unit_cases_total=0
flush_pack() {
  (( ${#pack_modules[@]} > 0 )) || return 0
  local unit="pack${pack_index}"
  entries+=( "${pack_cases}|${unit}|pack|${pack_modules[*]}" )
  if (( ${#pack_names[@]} > 0 )); then
    printf "%s\n" "${pack_names[@]}" > "${CASES_DIR}/${unit}.txt"
  else
    : > "${CASES_DIR}/${unit}.txt"
  fi
  pack_modules=()
  pack_names=()
  pack_cases=0
  pack_index=$(( pack_index + 1 ))
}
for path in "${DSL_ROOT}"/*.py; do
  [[ -f "${path}" ]] || continue
  module="$(basename -- "${path}" .py)"
  case "${module}" in common | __main__ | _* | test_*) continue ;; esac
  cases=""
  if ! cases="$(timeout "${DISCOVERY_TIMEOUT}" "${PYTHON_BIN}" "${path}" --list 2>>"${WORK_DIR}/discovery.log")"; then
    reason="$(tail -n 3 "${WORK_DIR}/discovery.log" 2>/dev/null | tr "\n" " ")"
    die "cannot list the cases of ${module}.py (${path});" \
      "last discovery output: ${reason:-none}; full log: ${WORK_DIR}/discovery.log"
  fi
  [[ -n "${cases}" ]] || { log "skipping ${module}.py: no case"; continue; }
  printf "%s\n" "${cases}" >> "${PER_MODULE_CASES}"
  count="$(printf "%s\n" "${cases}" | wc -l)"
  unit_cases_total=$(( unit_cases_total + count ))
  if (( count == 1 )) && [[ ! "${module}" =~ ${HEAVY_PATTERN} ]]; then
    pack_modules+=("${path}")
    while IFS= read -r case_name; do pack_names+=("${case_name}"); done <<< "${cases}"
    pack_cases=$(( pack_cases + 1 ))
    (( pack_cases >= PACK_SIZE )) && flush_pack
    continue
  fi
  if (( count <= BLOCK_SIZE )); then
    entries+=( "${count}|${module}|range|${path} 0 ${count}" )
    printf "%s\n" "${cases}" > "${CASES_DIR}/${module}.txt"
  else
    begin=0
    index=0
    while (( begin < count )); do
      end=$(( begin + BLOCK_SIZE ))
      (( end > count )) && end="${count}"
      block="${module}.b${index}"
      entries+=( "$(( end - begin ))|${block}|range|${path} ${begin} ${end}" )
      printf "%s\n" "${cases}" | sed -n "$(( begin + 1 )),${end}p" > "${CASES_DIR}/${block}.txt"
      begin="${end}"
      index=$(( index + 1 ))
    done
  fi
done
flush_pack
# Cross-check the sharded per-module lists against the global discovery. A
# module discovery missed, or a name only one side sees, fails here before any
# simulator session starts.
sort -u "${PER_MODULE_CASES}" > "${PER_MODULE_CASES}.uniq"
if ! cmp -s "${ALL_CASES}" "${PER_MODULE_CASES}.uniq"; then
  diff -u "${ALL_CASES}" "${PER_MODULE_CASES}.uniq" > "${WORK_DIR}/discovery-mismatch.diff" 2>&1 || true
  die "the per-module case lists do not match the global discovery;" \
    "see ${WORK_DIR}/discovery-mismatch.diff"
fi
(( ${#entries[@]} > 0 )) || die "no case module found under ${DSL_ROOT}"

# Longest first: a unit pays at least the fixed simulator start up. The unit
# spec keeps the expected case count as its first field so the summary can
# assert that every unit ran exactly what discovery saw.
printf "%s\n" "${entries[@]}" | sort -t"|" -k1,1nr -k2,2 > "${UNIT_LIST}"
unit_total="$(wc -l < "${UNIT_LIST}")"
# Keep the newline list for the counters, the summary and review, and
# derive a NUL delimited copy for xargs. -0/-P/-n are portable (BSD and
# macOS xargs included), unlike the GNU only -d/-a used before.
tr '\n' '\0' < "${UNIT_LIST}" > "${UNIT_LIST_NUL}"
log "=== PTODSL/DSL ST parallel ==="
log "suite: ${DSL_ROOT}"
log "units=${unit_total} cases=${unit_cases_total}"
log "jobs=${JOBS} block size=${BLOCK_SIZE} pack size=${PACK_SIZE} timeout=${UNIT_TIMEOUT}s (kill after ${KILL_AFTER}s)"
log "heavy pattern: ${HEAVY_PATTERN}"
log "per unit log: ${LOG_DIR}/<unit>.log; msprof output: ${LOG_DIR}/<unit>/"

# Dynamic pool: xargs starts the next unit as each session exits.
export LOG_DIR RESULT_DIR KILL_AFTER UNIT_TIMEOUT SIM_DSL_SCRIPT SLICE_DRIVER
xargs -0 -P "${JOBS}" -n 1 bash "${UNIT_RUNNER}" < "${UNIT_LIST_NUL}"
result_files=("${RESULT_DIR}"/*.result)
result_total=${#result_files[@]}
(( result_total == unit_total )) || { log "only ${result_total} of ${unit_total} units reported"; exit 1; }

: > "${SUMMARY_FILE}"
pass_units=0; fail_units=0; timeout_units=0; passed_cases=0; retried_units=0
pv_units=0; ca_units=0
failed_units=()
for result in "${result_files[@]}"; do
  read -r unit rc seconds expected retried < "${result}"
  retried="${retried:-0}"
  status=FAIL; reason="exit code ${rc}"
  case "${rc}" in
    0) status=PASS ;;
    124) status=TIMEOUT; reason="no result within ${UNIT_TIMEOUT}s" ;;
    137) status=TIMEOUT; reason="still running ${KILL_AFTER}s after the timeout signal" ;;
  esac
  marker=""
  if [[ "${retried}" == 1 ]]; then
    retried_units=$(( retried_units + 1 ))
    marker="RETRIED"
    reason="retried once after SIGSEGV (rc=139); ${reason}"
  fi
  cases="$(awk "/^PASS /{count++} END{print count+0}" "${LOG_DIR}/${unit}.log" 2>/dev/null || true)"
  [[ -n "${cases}" ]] || cases=0
  # Record the simulator model each unit actually ran; a failed pv unit
  # never reports ca, so the totals expose an unwanted degradation.
  model="$(grep -aoE "model: (pv|ca)" "${LOG_DIR}/${unit}.log" 2>/dev/null | head -1 | awk "{print \$2}")"
  case "${model}" in
    pv) pv_units=$(( pv_units + 1 )) ;;
    ca) ca_units=$(( ca_units + 1 )) ;;
  esac
  [[ -n "${expected}" ]] || expected="${cases}"
  reported_names="$(grep "^PASS " "${LOG_DIR}/${unit}.log" 2>/dev/null | awk "{print \$2}" || true)"
  # Coverage gate: a unit is green only when the PASS names it reported are
  # exactly the case names discovery assigned to it. Comparing counts alone
  # lets a duplicate PASS hide a skipped case, so reject duplicates and compare
  # the unique name sets with the expected list recorded in CASES_DIR.
  if [[ "${status}" == "PASS" && -f "${CASES_DIR}/${unit}.txt" ]]; then
    expected_names="$(sed "/^$/d" "${CASES_DIR}/${unit}.txt" | sort)"
    actual_names="$(printf "%s\n" "${reported_names}" | sed "/^$/d" | sort)"
    unique_names="$(printf "%s\n" "${actual_names}" | sed "/^$/d" | sort -u)"
    if [[ "${actual_names}" != "${unique_names}" ]]; then
      status=FAIL
      reason="coverage: duplicate PASS name(s)"
      log "[${unit}] COVERAGE-DUPLICATE: $(printf "%s\n" "${actual_names}" | uniq -d | paste -sd, -)"
    elif [[ "${unique_names}" != "${expected_names}" ]]; then
      status=FAIL
      reason="coverage: PASS names differ from discovery"
      log "[${unit}] COVERAGE-MISMATCH expected=${expected} reported=${cases}"
      log "[${unit}] cases missing: $(comm -23 \
        <(printf "%s\n" "${expected_names}") \
        <(printf "%s\n" "${unique_names}") | paste -sd, -)"
      log "[${unit}] cases unexpected: $(comm -13 \
        <(printf "%s\n" "${expected_names}") \
        <(printf "%s\n" "${unique_names}") | paste -sd, -)"
    fi
  elif [[ "${status}" == "PASS" && "${cases}" -ne "${expected}" ]]; then
    status=FAIL
    reason="coverage: ran ${cases} of ${expected} case(s)"
    log "[${unit}] COVERAGE-UNDER-RUN expected=${expected} passed=${cases}"
  fi
  passed_cases=$(( passed_cases + cases ))
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${unit}" "${status}" "${seconds}" "${cases}" \
    "${expected}" "${marker}" >> "${SUMMARY_FILE}"
  case "${status}" in
    PASS) pass_units=$(( pass_units + 1 )) ;;
    TIMEOUT) timeout_units=$(( timeout_units + 1 )); failed_units+=("${unit}") ;;
    *) fail_units=$(( fail_units + 1 )); failed_units+=("${unit}") ;;
  esac
  if [[ "${status}" == "PASS" ]]; then
    log "[${unit}] PASS cases=${cases}/${expected} ${seconds}s${marker:+ (${marker})}"
    cat "${LOG_DIR}/${unit}.log"
  else
    log "[${unit}] ${status} cases=${cases}/${expected} ${seconds}s (${reason}; log ${LOG_DIR}/${unit}.log)"
    tail -n "${FAILED_LOG_LINES}" "${LOG_DIR}/${unit}.log"
  fi
done
log "=== PTODSL/DSL ST summary ==="
log "units: total=${unit_total} PASS=${pass_units} FAIL=${fail_units} TIMEOUT=${timeout_units}"
log "cases: passed=${passed_cases}/${unit_cases_total}"
log "retried=${retried_units}"
log "models: pv=${pv_units} ca=${ca_units}"
log "per unit results: ${SUMMARY_FILE}"
if (( fail_units + timeout_units > 0 )); then
  log "failed units: ${failed_units[*]}"
  exit 1
fi
if (( passed_cases != unit_cases_total )); then
  log "coverage: ${passed_cases} case(s) passed but ${unit_cases_total} were discovered"
  exit 1
fi
log "all ${passed_cases} case(s) passed"