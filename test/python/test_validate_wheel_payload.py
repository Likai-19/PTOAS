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
LINUX_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "build_wheel.yml"
MAC_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "build_wheel_mac.yml"
WHEEL_IMPORTS = REPO_ROOT / "docker" / "test_wheel_imports.sh"


class ValidateWheelPayloadTests(unittest.TestCase):
    def _make_wheel(self, root: Path, *, include_runtime_so: bool) -> Path:
        wheel = root / "ptoas-1.2.3-cp311-cp311-linux_x86_64.whl"
        with zipfile.ZipFile(wheel, "w") as zf:
            zf.writestr("ptoas/__init__.py", "")
            zf.writestr("ptoas/_launcher.py", "")
            if include_runtime_so:
                zf.writestr("ptoas/_runtime/lib/ptoas.so", "fake")
            zf.writestr(
                "ptoas-1.2.3.dist-info/entry_points.txt",
                "[console_scripts]\nptoas=ptoas._launcher:main\n",
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

    def test_wheel_imports_script_keeps_clean_env_ptoas_smoke(self):
        script = WHEEL_IMPORTS.read_text(encoding="utf-8")

        self.assertIn('env -i \\', script)
        self.assertIn('CLEAN_ENV_PTO="${CLEAN_ENV_PTO}" \\', script)
        self.assertIn('def wheel_clean_env_probe(', script)
        self.assertIn('"${PTOAS_ENTRYPOINT}" "${CLEAN_ENV_PTO}" -o "${CLEAN_ENV_CPP}"', script)
        self.assertIn('grep -q "wheel_clean_env_probe" "${CLEAN_ENV_CPP}"', script)


if __name__ == "__main__":
    unittest.main()
