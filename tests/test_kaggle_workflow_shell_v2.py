from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from kaggle_workflow_shell_v2 import (  # noqa: E402
    ShellPolicyError,
    run_blocks,
    validate_changed,
    validate_workflow,
)


class KaggleWorkflowShellV2Tests(unittest.TestCase):
    def _workflow(self, root: Path, text: str, name: str = "test.yml") -> Path:
        path = root / ".github" / "workflows" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_same_step_github_env_value_is_not_visible(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = self._workflow(
                Path(td),
                "jobs:\n"
                "  run:\n"
                "    steps:\n"
                "      - run: |\n"
                "          set -euo pipefail\n"
                "          workdir=\"$(mktemp -d)\"\n"
                "          echo \"WORKDIR=${workdir}\" >> \"$GITHUB_ENV\"\n"
                "          python3 -m venv \"${WORKDIR}/venv\"\n",
            )
            with self.assertRaisesRegex(ShellPolicyError, "same_step_github_env_visibility_forbidden"):
                validate_workflow(path)

    def test_current_shell_assignment_makes_same_step_use_valid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = self._workflow(
                Path(td),
                "jobs:\n"
                "  run:\n"
                "    steps:\n"
                "      - run: |\n"
                "          set -euo pipefail\n"
                "          WORKDIR=\"$(mktemp -d)\"\n"
                "          echo \"WORKDIR=${WORKDIR}\" >> \"$GITHUB_ENV\"\n"
                "          python3 -m venv \"${WORKDIR}/venv\"\n",
            )
            validate_workflow(path)

    def test_lowercase_local_then_next_step_env_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = self._workflow(
                Path(td),
                "jobs:\n"
                "  run:\n"
                "    steps:\n"
                "      - run: |\n"
                "          workdir=\"$(mktemp -d)\"\n"
                "          echo \"WORKDIR=${workdir}\" >> \"$GITHUB_ENV\"\n"
                "          python3 -m venv \"${workdir}/venv\"\n"
                "      - run: |\n"
                "          set -euo pipefail\n"
                "          \"${WORKDIR}/venv/bin/python\" --version\n",
            )
            validate_workflow(path)

    def test_default_parameter_expansion_does_not_claim_visibility(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = self._workflow(
                Path(td),
                "jobs:\n"
                "  run:\n"
                "    steps:\n"
                "      - run: |\n"
                "          workdir=\"$(mktemp -d)\"\n"
                "          echo \"WORKDIR=${workdir}\" >> \"$GITHUB_ENV\"\n"
                "          [[ -z \"${WORKDIR-}\" ]]\n",
            )
            validate_workflow(path)

    def test_changed_only_checks_modified_workflows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base, head = root / "base", root / "head"
            base.mkdir(); head.mkdir()
            good = (
                "jobs:\n  x:\n    steps:\n"
                "      - run: |\n"
                "          workdir=/tmp/a\n"
                "          echo \"WORKDIR=${workdir}\" >> \"$GITHUB_ENV\"\n"
            )
            self._workflow(base, good, "same.yml")
            self._workflow(head, good, "same.yml")
            changed = good + "      - run: echo ok\n"
            self._workflow(head, changed, "changed.yml")
            self.assertEqual(validate_changed(base, head), [".github/workflows/changed.yml"])

    def test_run_block_extraction_stays_step_scoped(self) -> None:
        text = (
            "steps:\n"
            "  - run: |\n"
            "      echo first\n"
            "  - run: |\n"
            "      echo second\n"
        )
        blocks = run_blocks(text)
        self.assertEqual(len(blocks), 2)
        self.assertEqual([line.strip() for line in blocks[0][1]], ["echo first"])
        self.assertEqual([line.strip() for line in blocks[1][1]], ["echo second"])


if __name__ == "__main__":
    unittest.main()
