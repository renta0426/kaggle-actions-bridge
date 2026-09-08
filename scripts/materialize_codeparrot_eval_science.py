#!/usr/bin/env python3
"""Materialize the pinned CodeParrot CPU-evaluation source with bounded infra retries.

This helper is deliberately unrelated to Kaggle compute/output retrieval.  It
fetches a fixed allowlist from one immutable GitHub commit. Only transient public
source transport failures are retried, and failures always identify the exact
repository path that failed.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import ssl
import time
from pathlib import Path
import urllib.error
import urllib.request
from typing import Callable


REPOSITORY = "renta0426/The-Poisoned-Chalice-of-LLM-Evaluation"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
USER_AGENT = "codeparrot-full-eval-science-materializer/1"
TRANSIENT_HTTP = {404, 500, 502, 503, 504}
MAX_ATTEMPTS = 3

BASE_FILES: dict[str, int] = {
    "scripts/run_codeparrot_fresh_evaluation.py": 131072,
    "configs/codeparrot_fresh_confirmation_v1.json": 32768,
    "experiments/codeparrot-fresh-confirmation-v1/full_result_audit_20260908.json": 32768,
    "src/poisoned_chalice/__init__.py": 32768,
    "src/poisoned_chalice/codeparrot_artifacts.py": 65536,
    "src/poisoned_chalice/codeparrot_evaluation.py": 131072,
    "src/poisoned_chalice/codeparrot_fresh.py": 131072,
    "src/poisoned_chalice/codeparrot_postscore.py": 131072,
    "src/poisoned_chalice/novelty_evaluation_v2.py": 131072,
}
TEST_FILES: dict[str, int] = {
    "tests/test_codeparrot_postscore.py": 65536,
    "tests/test_codeparrot_evaluation.py": 65536,
}


def _fetch_once(url: str, maximum: int, *, opener=urllib.request.urlopen) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with opener(request, timeout=30, context=ssl.create_default_context()) as response:
        data = response.read(maximum + 1)
    if not data:
        raise RuntimeError("empty_source")
    if len(data) > maximum:
        raise RuntimeError("source_byte_budget_exceeded")
    return data


def _fetch_bounded(
    *,
    commit: str,
    relative: str,
    maximum: int,
    opener=urllib.request.urlopen,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[bytes, int]:
    url = f"https://raw.githubusercontent.com/{REPOSITORY}/{commit}/{relative}"
    retries = 0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return _fetch_once(url, maximum, opener=opener), retries
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            if status not in TRANSIENT_HTTP or attempt == MAX_ATTEMPTS:
                raise RuntimeError(
                    f"science_source_fetch_failed path={relative} http={status} attempt={attempt}"
                ) from exc
            retries += 1
            sleeper(float(attempt))
        except urllib.error.URLError as exc:
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(
                    f"science_source_fetch_failed path={relative} transport={type(exc.reason).__name__} attempt={attempt}"
                ) from exc
            retries += 1
            sleeper(float(attempt))
        except RuntimeError as exc:
            # Content-contract failures are deterministic and must not be retried.
            raise RuntimeError(
                f"science_source_contract_failed path={relative} reason={exc}"
            ) from exc
    raise AssertionError("unreachable")


def materialize(
    output_dir: Path,
    *,
    commit: str,
    include_tests: bool,
) -> dict[str, object]:
    if not COMMIT_RE.fullmatch(commit):
        raise ValueError("science commit must be an exact lowercase 40-hex SHA")
    if output_dir.exists():
        raise FileExistsError("science output directory already exists")
    files = dict(BASE_FILES)
    if include_tests:
        files.update(TEST_FILES)
    output_dir.mkdir(parents=True, exist_ok=False)
    retries_used = 0
    identities: dict[str, dict[str, object]] = {}
    try:
        for relative, maximum in files.items():
            data, retries = _fetch_bounded(
                commit=commit,
                relative=relative,
                maximum=maximum,
            )
            retries_used += retries
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            if destination.suffix == ".py":
                compile(data, str(destination), "exec")
            identities[relative] = {
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
    except BaseException:
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    print(
        "CODEPARROT_SCIENCE_MATERIALIZE PASS "
        f"commit={commit} files={len(files)} transient_retries={retries_used} include_tests={int(include_tests)}"
    )
    return {
        "commit": commit,
        "files": identities,
        "transient_retries": retries_used,
        "include_tests": include_tests,
    }


def self_test() -> None:
    class FakeResponse:
        def __init__(self, data: bytes): self.data = data
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, limit: int): return self.data[:limit]

    calls: list[int] = []
    sleeps: list[float] = []

    def transient_then_ok(request, timeout, context):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 404, "temporary", {}, None)
        return FakeResponse(b"x=1\n")

    data, retries = _fetch_bounded(
        commit="a" * 40,
        relative="example.py",
        maximum=100,
        opener=transient_then_ok,
        sleeper=sleeps.append,
    )
    assert data == b"x=1\n" and retries == 1 and len(calls) == 2 and sleeps == [1.0]

    calls.clear(); sleeps.clear()
    def nontransient(request, timeout, context):
        calls.append(1)
        raise urllib.error.HTTPError(request.full_url, 429, "rate", {}, None)
    try:
        _fetch_bounded(
            commit="a" * 40,
            relative="example.py",
            maximum=100,
            opener=nontransient,
            sleeper=sleeps.append,
        )
    except RuntimeError as exc:
        assert "path=example.py" in str(exc) and "http=429" in str(exc)
    else:
        raise AssertionError("429 must fail without retry")
    assert len(calls) == 1 and not sleeps

    calls.clear(); sleeps.clear()
    def deterministic_bad_content(request, timeout, context):
        calls.append(1)
        return FakeResponse(b"x" * 101)
    try:
        _fetch_bounded(
            commit="a" * 40,
            relative="oversize.py",
            maximum=100,
            opener=deterministic_bad_content,
            sleeper=sleeps.append,
        )
    except RuntimeError as exc:
        assert "path=oversize.py" in str(exc) and "source_byte_budget_exceeded" in str(exc)
    else:
        raise AssertionError("deterministic content failure must fail")
    assert len(calls) == 1 and not sleeps
    print("CODEPARROT_SCIENCE_MATERIALIZER_SELF_TEST PASS transient_404_retry=1 retry_429=0 deterministic_contract_retry=0 path_diagnostic=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--commit")
    parser.add_argument("--include-tests", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.output_dir is None or args.commit is None:
        parser.error("--output-dir and --commit are required")
    materialize(args.output_dir, commit=args.commit, include_tests=args.include_tests)


if __name__ == "__main__":
    main()
