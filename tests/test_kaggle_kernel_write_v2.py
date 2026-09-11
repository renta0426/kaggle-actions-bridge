from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import kaggle_kernel_write_v2 as writer  # noqa: E402


@dataclass
class Meta:
    ref: str
    current_version_number: int


@dataclass
class Status:
    status: str


class Response:
    def __init__(self, error: str = "") -> None:
        self.error = error


class FakeApi:
    def __init__(self, *, push_response=None, push_exception=None, state="ERROR") -> None:
        self.push_response = push_response
        self.push_exception = push_exception
        self.state = state
        self.push_calls = 0
        self.status_calls = 0

    def kernels_push(self, kernel_dir: str):
        self.push_calls += 1
        if self.push_exception is not None:
            raise self.push_exception
        return self.push_response if self.push_response is not None else Response()

    def kernels_status(self, target: str):
        self.status_calls += 1
        return Status(self.state)

    def quota_view(self):  # pragma: no cover - must never be called
        raise AssertionError("quota_view must not be used by policy v2 writer")

    def kernels_list(self, *args, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("kernels_list must not be used by policy v2 writer")


class KernelWriteV2Tests(unittest.TestCase):
    TARGET = "owner/example"

    def test_success_observes_version_two_with_one_write(self) -> None:
        api = FakeApi(state="RUNNING")
        metas = [Meta(self.TARGET, 1), Meta(self.TARGET, 2)]
        with mock.patch.object(writer, "exact_metadata", side_effect=metas):
            out = writer.push_kernel_version_once(
                api,
                "/tmp/kernel",
                self.TARGET,
                expected_before_version=1,
                expected_after_version=2,
                expected_before_state="ERROR",
                sleep=lambda _: None,
            )
        self.assertEqual(api.push_calls, 1)
        self.assertEqual(out.classification, "write_observed")
        self.assertEqual(out.observed_version, 2)

    def test_platform_rejection_without_side_effect_is_reusable_class(self) -> None:
        api = FakeApi(push_exception=RuntimeError("platform rejected"), state="ERROR")
        with mock.patch.object(writer, "exact_metadata", return_value=Meta(self.TARGET, 1)):
            out = writer.push_kernel_version_once(
                api,
                "/tmp/kernel",
                self.TARGET,
                expected_before_version=1,
                expected_after_version=2,
                expected_before_state="ERROR",
                reconcile_attempts=3,
                sleep=lambda _: None,
            )
        self.assertEqual(api.push_calls, 1)
        self.assertEqual(out.classification, "platform_rejected_no_side_effect")
        self.assertEqual(out.observed_version, 1)

    def test_success_return_without_observed_version_is_ambiguous(self) -> None:
        api = FakeApi(push_response=Response(), state="ERROR")
        with mock.patch.object(writer, "exact_metadata", return_value=Meta(self.TARGET, 1)):
            with self.assertRaisesRegex(writer.KernelWriteError, "ambiguous_write"):
                writer.push_kernel_version_once(
                    api,
                    "/tmp/kernel",
                    self.TARGET,
                    expected_before_version=1,
                    expected_after_version=2,
                    expected_before_state="ERROR",
                    reconcile_attempts=2,
                    sleep=lambda _: None,
                )
        self.assertEqual(api.push_calls, 1)

    def test_wrong_before_version_blocks_before_write(self) -> None:
        api = FakeApi()
        with mock.patch.object(writer, "exact_metadata", return_value=Meta(self.TARGET, 3)):
            with self.assertRaisesRegex(writer.KernelWriteError, "before_version_mismatch"):
                writer.push_kernel_version_once(
                    api,
                    "/tmp/kernel",
                    self.TARGET,
                    expected_before_version=1,
                    expected_after_version=2,
                    sleep=lambda _: None,
                )
        self.assertEqual(api.push_calls, 0)


if __name__ == "__main__":
    unittest.main()
