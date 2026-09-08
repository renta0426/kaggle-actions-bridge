#!/usr/bin/env python3
"""E04 builder v2: preserve nested runtime escapes and validate exact relays."""
from __future__ import annotations

import argparse
import builtins
import importlib.util
import subprocess
import sys
from pathlib import Path

BASE = "scripts/cmi_flu_strategy_e04_prepare.py"


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def load(root: Path):
    path = root / BASE
    spec = importlib.util.spec_from_file_location("cmi_flu_e04_prepare_v1", path)
    if spec is None or spec.loader is None:
        raise SystemExit("E04 v1 builder unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    a = args()
    root = a.repository_root.expanduser().resolve()
    output = a.output.expanduser().resolve()
    base = load(root)
    base.validate_request(root)
    e04, contract, synthetic, task13 = base.load_exact_e04(root)
    frozen = base.load_base(root)

    original_compile = builtins.compile
    def compile_guard(source, filename, mode, *positional, **keywords):
        if filename == "generated_e04_runtime.py":
            return original_compile("pass\n", filename, mode, *positional, **keywords)
        return original_compile(source, filename, mode, *positional, **keywords)
    builtins.compile = compile_guard
    try:
        runtime, package_sha = base.build_runtime(frozen, root, e04, contract, synthetic, task13)
    finally:
        builtins.compile = original_compile
    runtime = runtime.replace('"\n"', '"\\n"')

    config_tuple = '        (CONFIG_TEXT, CONFIG_BLOB, "config"),\n'
    if runtime.count(config_tuple) != 1:
        raise SystemExit("E04 runtime config self-test anchor changed")
    runtime = runtime.replace(config_tuple, "", 1)
    adapter_anchor = '    compile(B21_ADAPTER_SOURCE, "cmi_flu_b21_runtime_adapter.py", "exec")\n'
    if runtime.count(adapter_anchor) != 1:
        raise SystemExit("E04 runtime adapter self-test anchor changed")
    runtime = runtime.replace(
        adapter_anchor,
        '    if git_blob_sha(CONFIG_TEXT.encode("utf-8")) != CONFIG_BLOB:\n'
        '        raise BridgeContractError("config_blob_mismatch")\n' + adapter_anchor,
        1,
    )

    # The frozen B2.1 adapter installs a robust selector on cmi_flu.runner,
    # but that historical wrapper predates the explicit selection_policy
    # parameter now required by E04's science-side provenance check. Preserve
    # the already-installed robust implementation and expose only the explicit
    # robust_v1 compatibility signature expected by the exact E04 blob.
    install_anchor = '        install()\n        from cmi_flu.configuration import load_baseline_config\n'
    if runtime.count(install_anchor) != 1:
        raise SystemExit("E04 B2.1 adapter install anchor changed")
    runtime = runtime.replace(
        install_anchor,
        '        install()\n'
        '        from cmi_flu import runner as _e04_runner\n'
        '        _e04_frozen_robust_compact = _e04_runner.run_compact_task\n'
        '        def _e04_run_compact_task(dataset, *, specs, splits=None, random_state=42, selection_policy="robust_v1"):\n'
        '            if selection_policy != "robust_v1":\n'
        '                raise BundleContractError("E04 requires robust_v1 selection policy")\n'
        '            return _e04_frozen_robust_compact(dataset, specs=specs, splits=splits, random_state=random_state)\n'
        '        _e04_runner.run_compact_task = _e04_run_compact_task\n'
        '        from cmi_flu.configuration import load_baseline_config\n',
        1,
    )

    # Synthetic CI contains no Competition Data, so expose the exception text
    # there for diagnosis. Production keeps only the stable hashed error code.
    failure_line = '        print(f"CMI_FLU_E04_FAILED stage={stage} exception_type={type(exc).__name__} error_code={code}", file=sys.stderr)\n'
    if runtime.count(failure_line) != 1:
        raise SystemExit("E04 runtime failure-log anchor changed")
    runtime = runtime.replace(
        failure_line,
        '        detail = f" detail={str(exc)}" if synthetic else ""\n'
        '        print(f"CMI_FLU_E04_FAILED stage={stage} exception_type={type(exc).__name__} error_code={code}{detail}", file=sys.stderr)\n',
        1,
    )

    original_compile(runtime, "generated_e04_runtime.py", "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(output), "--self-test"], check=True)
    print(
        "CMI_FLU_E04_PREPARE_V2_PASS "
        f"science_commit={base.SCIENCE_COMMIT} e04_blob={base.E04_BLOB} contract_blob={base.CONTRACT_BLOB} "
        f"task13_blob={base.TASK13_BLOB} package_sha256={package_sha} runtime_sha256={base.sha256(runtime.encode())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
