#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = REPO_ROOT / "docker" / "validate_wheel_payload.py"
CREATE_WHEEL = REPO_ROOT / "docker" / "create_wheel.sh"
LINUX_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "build_wheel.yml"
MAC_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "build_wheel_mac.yml"
WHEEL_IMPORTS = REPO_ROOT / "docker" / "test_wheel_imports.sh"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
CI_SIM_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci_sim.yml"


class ValidateWheelPayloadTests(unittest.TestCase):
    def _make_wheel(
        self,
        root: Path,
        *,
        include_runtime_so: bool,
        include_runtime_entry: bool = True,
        include_bootstrap: bool = True,
        entry_points_text: str = "[console_scripts]\nptoas=ptoas_wheel_bootstrap:main\n",
        wheel_stem: str = "ptoas",
        dist_info_stem: str = "ptoas",
    ) -> Path:
        wheel = root / f"{wheel_stem}-1.2.3-cp311-cp311-linux_x86_64.whl"
        with zipfile.ZipFile(wheel, "w") as zf:
            if include_bootstrap:
                zf.writestr("ptoas_wheel_bootstrap.py", "")
            zf.writestr("ptoas/__init__.py", "")
            zf.writestr("ptoas/_launcher.py", "")
            if include_runtime_entry:
                zf.writestr("ptoas/_runtime_entry.py", "")
            if include_runtime_so:
                zf.writestr("ptoas/_runtime/lib/ptoas.so", "fake")
            zf.writestr(
                f"{dist_info_stem}-1.2.3.dist-info/entry_points.txt",
                entry_points_text,
            )
        return wheel

    def test_validator_accepts_current_runtime_payload_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wheel = self._make_wheel(Path(temp_dir), include_runtime_so=True)
            result = subprocess.run(
                ["python3", str(VALIDATOR), str(wheel)],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("validated wheel payload and launcher contract", result.stdout)

    def test_validator_rejects_missing_runtime_shared_module(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wheel = self._make_wheel(Path(temp_dir), include_runtime_so=False)
            result = subprocess.run(
                ["python3", str(VALIDATOR), str(wheel)],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ptoas/_runtime/lib/ptoas.so", result.stderr)

    def test_validator_rejects_missing_runtime_entry_module(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wheel = self._make_wheel(
                Path(temp_dir),
                include_runtime_so=True,
                include_runtime_entry=False,
            )
            result = subprocess.run(
                ["python3", str(VALIDATOR), str(wheel)],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ptoas/_runtime_entry.py", result.stderr)

    def test_validator_rejects_missing_wheel_bootstrap_module(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wheel = self._make_wheel(
                Path(temp_dir),
                include_runtime_so=True,
                include_bootstrap=False,
            )
            result = subprocess.run(
                ["python3", str(VALIDATOR), str(wheel)],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ptoas_wheel_bootstrap.py", result.stderr)

    def test_validator_rejects_legacy_ptoas_launcher_entrypoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wheel = self._make_wheel(
                Path(temp_dir),
                include_runtime_so=True,
                entry_points_text="[console_scripts]\nptoas=ptoas._launcher:main\n",
            )
            result = subprocess.run(
                ["python3", str(VALIDATOR), str(wheel)],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ptoas_wheel_bootstrap:main", result.stderr)

    def test_validator_accepts_normalized_entrypoint_spacing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wheel = self._make_wheel(
                Path(temp_dir),
                include_runtime_so=True,
                entry_points_text="[console_scripts]\nptoas = ptoas_wheel_bootstrap:main\n",
            )
            result = subprocess.run(
                ["python3", str(VALIDATOR), str(wheel)],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_validator_accepts_vmi_distribution_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wheel = self._make_wheel(
                Path(temp_dir),
                include_runtime_so=True,
                wheel_stem="ptoas_vmi",
                dist_info_stem="ptoas_vmi",
            )
            result = subprocess.run(
                ["python3", str(VALIDATOR), str(wheel)],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("validated wheel payload and launcher contract", result.stdout)

    def test_workflows_and_shell_probe_reuse_shared_validator(self):
        validator_call = 'python "$PTO_SOURCE_DIR/docker/validate_wheel_payload.py" "$PTO_SOURCE_DIR/build/wheel-dist"'
        self.assertIn(
            validator_call,
            LINUX_WORKFLOW.read_text(encoding="utf-8"),
        )
        self.assertIn(
            validator_call,
            MAC_WORKFLOW.read_text(encoding="utf-8"),
        )
        self.assertIn(
            '"${PYTHON_BIN}" "${REPO_ROOT}/docker/validate_wheel_payload.py" "${TEST_WHEEL}"',
            WHEEL_IMPORTS.read_text(encoding="utf-8"),
        )

    def test_release_workflows_separate_cli_and_wheel_versions(self):
        linux_workflow = LINUX_WORKFLOW.read_text(encoding="utf-8")
        mac_workflow = MAC_WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("workflow_dispatch:", linux_workflow)
        self.assertIn("release_kind:", linux_workflow)
        self.assertIn('type: choice', linux_workflow)
        self.assertIn('--mode release)', linux_workflow)
        self.assertIn(
            "if: github.event_name != 'release' || startsWith(github.ref_name, 'v') || startsWith(github.ref_name, 'ptoas-v')",
            linux_workflow,
        )
        self.assertIn(
            "if: (github.event_name == 'release' && (startsWith(github.ref_name, 'v') || startsWith(github.ref_name, 'ptoas-v'))) || github.event_name == 'schedule'",
            linux_workflow,
        )
        self.assertIn('PTOAS_CLI_VERSION="vmi ${PTOAS_VERSION}"', linux_workflow)
        self.assertIn('PTOAS_RELEASE_VERSION_OVERRIDE="${PTOAS_CLI_VERSION}"', linux_workflow)
        self.assertIn('export PTOAS_PYTHON_PACKAGE_VERSION="${PTOAS_VERSION}"', linux_workflow)

        self.assertIn("workflow_dispatch:", mac_workflow)
        self.assertIn("release_kind:", mac_workflow)
        self.assertIn('type: choice', mac_workflow)
        self.assertIn('--mode release)', mac_workflow)
        self.assertIn(
            "if: github.event_name != 'release' || startsWith(github.ref_name, 'v') || startsWith(github.ref_name, 'ptoas-v')",
            mac_workflow,
        )
        self.assertIn(
            "if: (github.event_name == 'release' && (startsWith(github.ref_name, 'v') || startsWith(github.ref_name, 'ptoas-v'))) || github.event_name == 'schedule'",
            mac_workflow,
        )
        self.assertIn('PTOAS_CLI_VERSION="vmi ${PTOAS_VERSION}"', mac_workflow)
        self.assertIn('PTOAS_RELEASE_VERSION_OVERRIDE="${PTOAS_CLI_VERSION}"', mac_workflow)
        self.assertIn('export PTOAS_PYTHON_PACKAGE_VERSION="${PTOAS_VERSION}"', mac_workflow)

    def test_create_wheel_script_validates_package_name(self):
        script = CREATE_WHEEL.read_text(encoding="utf-8")

        self.assertIn(
            "if [[ ! \"${PTOAS_PYTHON_PACKAGE_NAME}\" =~ ^[A-Za-z0-9]+([-_.][A-Za-z0-9]+)*$ ]]; then",
            script,
        )
        self.assertIn(
            "Error: invalid PTOAS_PYTHON_PACKAGE_NAME",
            script,
        )

    def test_wheel_imports_script_keeps_clean_env_ptoas_smoke(self):
        script = WHEEL_IMPORTS.read_text(encoding="utf-8")

        self.assertIn('EXPECTED_PTOAS_CLI_VERSION="${PTOAS_CLI_VERSION:-${PTOAS_VERSION:-}}"', script)
        self.assertIn('env -i \\', script)
        self.assertIn('CLEAN_ENV_PTO="${CLEAN_ENV_PTO}" \\', script)
        self.assertIn('CLEAN_ENV_LOG="${CLEAN_ENV_DIR}/wheel-clean-env-probe.log"', script)
        self.assertIn('CLEAN_ENV_PTO_IR="${CLEAN_ENV_DIR}/wheel-clean-env-probe.pto.ir"', script)
        self.assertIn('def wheel_clean_env_probe():', script)
        self.assertIn('pto.alloc_tile(shape=[1, 16], dtype=pto.f32, addr=0)', script)
        self.assertIn('pto.castptr(pto.const(0, dtype=pto.ui64), pto.ptr(pto.f32, "gm"))', script)
        self.assertIn('pto.tile.load(a_view, a_tile)', script)
        self.assertIn('pto.tile.store(o_tile, o_view)', script)
        self.assertIn('--emit-pto-ir "${CLEAN_ENV_PTO}" -o "${CLEAN_ENV_PTO_IR}"', script)
        self.assertIn('def wheel_clean_env_probe(', script)
        self.assertIn('"${PTOAS_ENTRYPOINT}" --pto-arch=a5 --pto-backend=vpto --pto-level=level3 --enable-tile-op-expand --emit-pto-ir "${CLEAN_ENV_PTO}" -o "${CLEAN_ENV_PTO_IR}"', script)
        self.assertIn('>"${CLEAN_ENV_LOG}" 2>&1', script)
        self.assertIn('grep -q "wheel_clean_env_probe" "${CLEAN_ENV_PTO_IR}"', script)
        self.assertIn('grep -q "pto.tload" "${CLEAN_ENV_PTO_IR}"', script)
        self.assertIn('grep -q "pto.tstore" "${CLEAN_ENV_PTO_IR}"', script)
        self.assertIn('grep -q "candidates = " "${CLEAN_ENV_PTO_IR}"', script)
        self.assertIn('grep -q "TileLib daemon started successfully" "${CLEAN_ENV_LOG}"', script)
        self.assertIn('grep -q "TileLib daemon stopped" "${CLEAN_ENV_LOG}"', script)

    def test_ci_workflows_accept_generic_ptoas_wheel_glob(self):
        self.assertIn("name 'ptoas*.whl'", CI_WORKFLOW.read_text(encoding="utf-8"))
        ci_sim_text = CI_SIM_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("name 'ptoas*.whl'", ci_sim_text)
        self.assertNotIn("name 'ptoas-*.whl'", ci_sim_text)


if __name__ == "__main__":
    unittest.main()
