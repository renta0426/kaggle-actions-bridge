#!/usr/bin/env python3
"""Credential-free policy checks for new/modified Kaggle launch workflows.

Policy v2 deliberately does not decide whether Kaggle has free CPU/GPU/TPU
capacity.  Kaggle is the authority for platform capacity and quota.  This
module prevents bridge-local capacity admission from creeping back into new
or materially modified launch workflows.

The checker has no network dependency and performs no Kaggle operation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

POLICY = "kaggle_native_capacity_v2"

RESOURCE_REQUEST_OPERATIONS = frozenset(
    {
        "kernel_run",
        "kernel_push",
        "save_kernel_once",
        "notebook_run",
        "notebook_push",
        "submission",
        "submit_once",
        "dataset_create",
        "dataset_version",
        "model_create",
        "model_version",
    }
)

DEPRECATED_CAPACITY_FIELDS = frozenset(
    {
        "max_active_runs",
        "min_remaining_quota_hours",
        "min_remaining_cpu_quota_hours",
        "min_remaining_gpu_quota_hours",
        "min_remaining_tpu_quota_hours",
    }
)

# These markers are intentionally narrow.  The goal is to reject launch
# *admission* gates, not read-only diagnostics or exact target reconciliation.
FORBIDDEN_CAPACITY_WORKFLOW_MARKERS = (
    "GPU admission unknown",
    "GPU concurrency refused",
    "pre-write GPU admission refused",
    "CPU admission unknown",
    "CPU concurrency refused",
    "TPU admission unknown",
    "TPU concurrency refused",
    "min_remaining_quota_hours",
    "max_active_runs",
)

KAGGLE_WRITE_MARKERS = (
    "kernels_push(",
    "kaggle kernels push",
    "competitions submit",
    "competition_submit(",
    "dataset_create",
    "dataset_version_create",
    "model_create",
    "model_instance_version_create",
)


class PolicyError(RuntimeError):
    pass


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # fixed local failure only
        raise PolicyError(f"invalid_json:{path}") from exc
    if not isinstance(value, dict):
        raise PolicyError(f"request_not_object:{path}")
    return value


def _operation_is_resource_write(request: dict) -> bool:
    op = str(request.get("operation") or "").strip().lower()
    if op in RESOURCE_REQUEST_OPERATIONS:
        return True
    # Historical manifests use several operation spellings.  Treat an explicit
    # side effect plus a resource stanza as a launch/write request.
    side_effects = request.get("side_effects")
    return isinstance(request.get("resource"), dict) and isinstance(side_effects, list) and bool(side_effects)


def validate_request(path: Path, *, require_v2: bool) -> None:
    request = _load_json(path)
    if not _operation_is_resource_write(request):
        return

    policy = request.get("execution_policy")
    if require_v2 and policy != POLICY:
        raise PolicyError(f"missing_execution_policy_v2:{path}")
    if policy not in (None, POLICY):
        raise PolicyError(f"unknown_execution_policy:{path}")

    resource = request.get("resource")
    if resource is not None and not isinstance(resource, dict):
        raise PolicyError(f"resource_not_object:{path}")
    resource = resource or {}
    forbidden = sorted(DEPRECATED_CAPACITY_FIELDS.intersection(resource))
    if forbidden:
        raise PolicyError(f"bridge_capacity_fields_forbidden:{path}:{','.join(forbidden)}")

    if policy == POLICY:
        accelerator = resource.get("accelerator")
        if accelerator is not None and str(accelerator).lower() not in {"cpu", "gpu", "tpu", "none"}:
            raise PolicyError(f"invalid_accelerator:{path}")
        # One-shot remains an idempotency/security rule, not a capacity rule.
        retries = request.get("automatic_compute_retries", 0)
        if retries != 0:
            raise PolicyError(f"automatic_compute_retry_forbidden:{path}")


def _is_kaggle_write_workflow(text: str) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in KAGGLE_WRITE_MARKERS)


def validate_workflow(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if not _is_kaggle_write_workflow(text):
        return

    problems = [marker for marker in FORBIDDEN_CAPACITY_WORKFLOW_MARKERS if marker.lower() in text.lower()]
    if problems:
        raise PolicyError(f"bridge_capacity_gate_forbidden:{path}:{'|'.join(problems)}")

    # A hard quota gate nearly always uses quota_view() together with a refusal.
    # We allow observation-only quota reads, but reject obvious blocking forms.
    lowered = text.lower()
    if "quota_view(" in lowered and any(token in lowered for token in ("quota refused", "quota unavailable", "remaining<", "remaining <")):
        raise PolicyError(f"blocking_quota_gate_forbidden:{path}")

    # Active-session enumeration may be used for diagnostics.  It must not be
    # coupled to launch admission under policy v2.
    if "kernels_list(" in lowered and "admission" in lowered and "active" in lowered:
        raise PolicyError(f"active_session_admission_forbidden:{path}")


def _all_files(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    if not root.exists():
        return result
    for path in root.rglob("*"):
        if path.is_file():
            result[path.relative_to(root).as_posix()] = path
    return result


def changed_paths(base: Path, head: Path) -> list[tuple[str, bool]]:
    """Return (relative path, is_new) for content changes between two trees."""
    base_files = _all_files(base)
    head_files = _all_files(head)
    out: list[tuple[str, bool]] = []
    for rel, head_path in sorted(head_files.items()):
        base_path = base_files.get(rel)
        if base_path is None:
            out.append((rel, True))
            continue
        try:
            same = head_path.read_bytes() == base_path.read_bytes()
        except OSError as exc:
            raise PolicyError(f"diff_read_failed:{rel}") from exc
        if not same:
            out.append((rel, False))
    return out


def validate_changed(base: Path, head: Path) -> list[str]:
    checked: list[str] = []
    for rel, is_new in changed_paths(base, head):
        path = head / rel
        if rel.startswith("requests/") and rel.endswith(".json"):
            # New resource-write requests must opt into v2.  Modified legacy
            # requests are not required to add the marker immediately, but they
            # may not retain/reintroduce bridge-local capacity fields.
            validate_request(path, require_v2=is_new)
            checked.append(rel)
        elif rel.startswith(".github/workflows/") and rel.endswith((".yml", ".yaml")):
            validate_workflow(path)
            checked.append(rel)
    return checked


def _print_pass(items: Iterable[str]) -> None:
    items = list(items)
    print(f"KAGGLE_LAUNCH_POLICY_V2 PASS checked={len(items)} policy={POLICY}")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_request = sub.add_parser("request")
    p_request.add_argument("path", type=Path)
    p_request.add_argument("--require-v2", action="store_true")

    p_workflow = sub.add_parser("workflow")
    p_workflow.add_argument("path", type=Path)

    p_changed = sub.add_parser("changed")
    p_changed.add_argument("base", type=Path)
    p_changed.add_argument("head", type=Path)

    args = parser.parse_args()
    try:
        if args.command == "request":
            validate_request(args.path, require_v2=args.require_v2)
            _print_pass([str(args.path)])
        elif args.command == "workflow":
            validate_workflow(args.path)
            _print_pass([str(args.path)])
        else:
            _print_pass(validate_changed(args.base, args.head))
    except PolicyError as exc:
        print(f"KAGGLE_LAUNCH_POLICY_V2 FAIL {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
