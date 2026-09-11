from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from kaggle_launch_policy_v2 import (  # noqa: E402
    POLICY,
    PolicyError,
    validate_changed,
    validate_request,
    validate_workflow,
)


class KaggleLaunchPolicyV2Tests(unittest.TestCase):
    def _write_json(self, path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def _request(self) -> dict:
        return {
            "schema_version": 1,
            "request_id": "example-001",
            "execution_policy": POLICY,
            "operation": "save_kernel_once",
            "target": "owner/example",
            "resource": {
                "accelerator": "gpu",
                "machine_shape": "NvidiaTeslaT4",
                "expected_runtime_minutes": 100,
                "hard_timeout_minutes": 180,
            },
            "side_effects": ["create one private Notebook version"],
            "automatic_compute_retries": 0,
        }

    def test_v2_request_has_no_bridge_capacity_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "request.json"
            self._write_json(path, self._request())
            validate_request(path, require_v2=True)

    def test_new_resource_request_requires_v2_marker(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "request.json"
            request = self._request()
            request.pop("execution_policy")
            self._write_json(path, request)
            with self.assertRaisesRegex(PolicyError, "missing_execution_policy_v2"):
                validate_request(path, require_v2=True)

    def test_capacity_fields_are_forbidden(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "request.json"
            request = self._request()
            request["resource"]["max_active_runs"] = 1
            request["resource"]["min_remaining_quota_hours"] = 3.0
            self._write_json(path, request)
            with self.assertRaisesRegex(PolicyError, "bridge_capacity_fields_forbidden"):
                validate_request(path, require_v2=True)

    def test_one_shot_write_without_capacity_gate_passes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "workflow.yml"
            path.write_text(
                "permissions: {}\n"
                "jobs:\n"
                "  launch:\n"
                "    runs-on: ubuntu-24.04\n"
                "    steps:\n"
                "      - run: |\n"
                "          response=api.kernels_push(kernel_dir)\n",
                encoding="utf-8",
            )
            validate_workflow(path)

    def test_capacity_admission_in_write_workflow_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "workflow.yml"
            path.write_text(
                "jobs:\n"
                "  launch:\n"
                "    steps:\n"
                "      - run: |\n"
                "          response=api.kernels_push(kernel_dir)\n"
                "          max_active_runs=1\n"
                "          raise SystemExit('GPU concurrency refused')\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PolicyError, "bridge_capacity_gate_forbidden"):
                validate_workflow(path)

    def test_experiment_write_ci_cannot_depend_on_global_readme_wording(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "95-example-static.yml"
            path.write_text(
                "jobs:\n"
                "  validate:\n"
                "    steps:\n"
                "      - run: |\n"
                "          response=api.kernels_push(kernel_dir)\n"
                "          readme = read(\"README.md\", 131072).decode(\"utf-8\")\n"
                "          raise SystemExit(\"README resource policy marker missing\")\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PolicyError, "global_policy_reverse_dependency_forbidden"):
                validate_workflow(path)

    def test_changed_tree_requires_v2_only_for_new_request(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = root / "base"
            head = root / "head"
            base.mkdir()
            head.mkdir()

            old = self._request()
            old.pop("execution_policy")
            self._write_json(base / "requests/legacy.json", old)
            self._write_json(head / "requests/legacy.json", old)

            new = self._request()
            self._write_json(head / "requests/new.json", new)
            checked = validate_changed(base, head)
            self.assertEqual(checked, ["requests/new.json"])

    def test_modified_legacy_request_cannot_keep_capacity_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = root / "base"
            head = root / "head"
            base.mkdir()
            head.mkdir()

            old = self._request()
            old.pop("execution_policy")
            self._write_json(base / "requests/legacy.json", old)
            modified = json.loads(json.dumps(old))
            modified["resource"]["max_active_runs"] = 1
            self._write_json(head / "requests/legacy.json", modified)
            with self.assertRaisesRegex(PolicyError, "bridge_capacity_fields_forbidden"):
                validate_changed(base, head)


if __name__ == "__main__":
    unittest.main()
