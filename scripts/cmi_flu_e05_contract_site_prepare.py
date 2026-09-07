#!/usr/bin/env python3
"""Build an instrumented E05 005 diagnostic runtime from the exact failed 004 runtime."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

REQUEST_ID = "20260907-cmi-flu-e05-contract-site-005"
TARGET_KERNEL = "renta0426/cmi-flu-e05-contract-site-20260907-005"
OLD_REQUEST_ID = "20260907-cmi-flu-strategy-e05-hai-donor-strain-004"
OLD_TARGET_KERNEL = "renta0426/cmi-flu-e05-hai-donor-strain-20260907-004"
SCIENCE_COMMIT = "3bcddeaf4b028795363a6e12f8af06dd23b3bbdf"
EXPECTED_ERROR_CODE = "74bf25be40ca17429601"
V4_PREPARE = "scripts/cmi_flu_strategy_e05_prepare_v4.py"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_v4(root: Path):
    path = root / V4_PREPARE
    spec = importlib.util.spec_from_file_location("cmi_flu_e05_prepare_v4_for_site_diag", path)
    if spec is None or spec.loader is None:
        raise SystemExit("unable to load E05 v4 builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = {
        "REQUEST_ID": OLD_REQUEST_ID,
        "TARGET_KERNEL": OLD_TARGET_KERNEL,
        "SCIENCE_COMMIT": SCIENCE_COMMIT,
        "E05_BLOB": "78cde9e3a6b6e3b332352218c4b4545f769aa6c4",
        "E05_V2_BLOB": "50951807d28418bb50f4e9dba656fa2b3e4d86ff",
        "HAI_TRANSFER_BLOB": "b671d8bf7f10bebbd65aca2a5bad42e267ee78d5",
    }
    for key, value in expected.items():
        if getattr(module, key, None) != value:
            raise SystemExit(f"E05 v4 builder contract changed:{key}")
    return module


def build_v4_runtime(v4, root: Path, reference_dir: Path) -> str:
    v3 = v4.load_v3(root)
    shim = v4.load_shim(root)
    return v4.patch_runtime(v3, v4.build_v3_runtime(v3, root, reference_dir), shim)


def patch_runtime(runtime: str) -> str:
    if runtime.count(OLD_REQUEST_ID) < 1 or runtime.count(OLD_TARGET_KERNEL) < 1:
        raise SystemExit("E05 004 diagnostic identity anchors missing")
    runtime = runtime.replace(OLD_REQUEST_ID, REQUEST_ID).replace(OLD_TARGET_KERNEL, TARGET_KERNEL)

    import_anchor = "import types\n"
    if runtime.count(import_anchor) != 1:
        raise SystemExit("traceback import anchor changed")
    runtime = runtime.replace(import_anchor, import_anchor + "import traceback\n", 1)

    helper_anchor = "\ndef main() -> int:\n"
    if runtime.count(helper_anchor) != 1:
        raise SystemExit("diagnostic helper anchor changed")
    helper = r'''
def _e05_contract_site(exc: BaseException) -> tuple[str, str]:
    sites = []
    for frame in traceback.extract_tb(exc.__traceback__):
        raw = str(frame.filename).replace("\\", "/")
        module = None
        if "cmi_flu/" in raw:
            module = "cmi_flu/" + raw.rsplit("cmi_flu/", 1)[1]
        elif raw.startswith("<cmi_flu.") and raw.endswith(">"):
            module = raw[1:-1].replace(".", "/") + ".py"
        elif raw.endswith("script.py") or raw.endswith("generated_e05_v4.py"):
            module = "runtime"
        if module is None:
            continue
        module = re.sub(r"[^A-Za-z0-9_./-]", "_", module)[:120]
        function = re.sub(r"[^A-Za-z0-9_.<>-]", "_", str(frame.name))[:80]
        sites.append(f"{module}:{function}:{int(frame.lineno)}")
    site_text = ",".join(sites[-12:]) or "none"
    return site_text, hashlib.sha256(site_text.encode("utf-8")).hexdigest()[:20]
'''
    runtime = runtime.replace(helper_anchor, "\n" + helper.rstrip() + helper_anchor, 1)

    old = '''    except Exception as exc:\n        shutil.rmtree(runtime_root, ignore_errors=True)\n        code = hashlib.sha256(f"{type(exc).__name__}:{str(exc)}".encode("utf-8", errors="replace")).hexdigest()[:20]\n        print(f"CMI_FLU_E05_FAILED stage={stage} exception_type={type(exc).__name__} error_code={code}", file=sys.stderr)\n        return 2\n'''
    if runtime.count(old) != 1:
        raise SystemExit(f"E05 diagnostic failure block count changed:{runtime.count(old)}")
    new = '''    except Exception as exc:\n        shutil.rmtree(runtime_root, ignore_errors=True)\n        code = hashlib.sha256(f"{type(exc).__name__}:{str(exc)}".encode("utf-8", errors="replace")).hexdigest()[:20]\n        sites, sites_sha = _e05_contract_site(exc)\n        print(f"CMI_FLU_E05_CONTRACT_SITE stage={stage} exception_type={type(exc).__name__} error_code={code} sites={sites} sites_sha256={sites_sha}", file=sys.stderr)\n        return 2\n'''
    runtime = runtime.replace(old, new, 1)

    if OLD_REQUEST_ID in runtime or OLD_TARGET_KERNEL in runtime:
        raise SystemExit("prior E05 004 identity remained in diagnostic runtime")
    if runtime.count("CMI_FLU_E05_CONTRACT_SITE") != 1 or runtime.count("def _e05_contract_site") != 1:
        raise SystemExit("E05 diagnostic marker/helper contract failed")
    if "competition_submit" in runtime or "kaggle competitions submit" in runtime:
        raise SystemExit("diagnostic runtime contains submission path")
    compile(runtime, "generated_e05_contract_site_005.py", "exec")
    return runtime


def main() -> int:
    args = parse_args()
    root = args.repository_root.expanduser().resolve()
    reference_dir = args.reference_dir.expanduser().resolve()
    v4 = load_v4(root)
    runtime = patch_runtime(build_v4_runtime(v4, root, reference_dir))
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(out), "--self-test"], check=True)
    print(
        "CMI_FLU_E05_CONTRACT_SITE_BUILD PASS "
        f"request_id={REQUEST_ID} target={TARGET_KERNEL} science_commit={SCIENCE_COMMIT} "
        f"runtime_sha256={sha256(runtime.encode())} expected_error_code={EXPECTED_ERROR_CODE}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
