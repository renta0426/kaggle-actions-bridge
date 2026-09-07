#!/usr/bin/env python3
"""Add privacy-safe failure-site reporting to an E05 runtime.

The emitted marker contains only stage, exception class, the existing hashed
error code, module/function/line locations, and a hash of those locations. It
never emits exception messages, locals, participant values, predictions, or
reference-file rows. This avoids needing another diagnostic compute run for a
new E05 exception class.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def patch_runtime(runtime: str) -> str:
    if "def _e05_safe_failure_site" in runtime or "CMI_FLU_E05_SAFE_FAILURE_SITE" in runtime:
        raise SystemExit("E05 runtime is already safe-site instrumented")
    import_anchor = "import types\n"
    if runtime.count(import_anchor) != 1:
        raise SystemExit("E05 safe-site traceback import anchor changed")
    runtime = runtime.replace(import_anchor, import_anchor + "import traceback\n", 1)

    helper_anchor = "\ndef main() -> int:\n"
    if runtime.count(helper_anchor) != 1:
        raise SystemExit("E05 safe-site helper anchor changed")
    helper = r'''
def _e05_safe_failure_site(exc: BaseException) -> tuple[str, str]:
    sites = []
    for frame in traceback.extract_tb(exc.__traceback__):
        raw = str(frame.filename).replace("\\", "/")
        module = None
        if "cmi_flu/" in raw:
            module = "cmi_flu/" + raw.rsplit("cmi_flu/", 1)[1]
        elif raw.startswith("<cmi_flu.") and raw.endswith(">"):
            module = raw[1:-1].replace(".", "/") + ".py"
        elif raw.endswith("script.py") or "generated_e05" in raw:
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
        raise SystemExit(f"E05 safe-site failure block count changed:{runtime.count(old)}")
    new = '''    except Exception as exc:\n        shutil.rmtree(runtime_root, ignore_errors=True)\n        code = hashlib.sha256(f"{type(exc).__name__}:{str(exc)}".encode("utf-8", errors="replace")).hexdigest()[:20]\n        sites, sites_sha = _e05_safe_failure_site(exc)\n        print(f"CMI_FLU_E05_SAFE_FAILURE_SITE stage={stage} exception_type={type(exc).__name__} error_code={code} sites={sites} sites_sha256={sites_sha}", file=sys.stderr)\n        return 2\n'''
    runtime = runtime.replace(old, new, 1)
    if runtime.count("CMI_FLU_E05_SAFE_FAILURE_SITE") != 1 or runtime.count("def _e05_safe_failure_site") != 1:
        raise SystemExit("E05 safe-site marker/helper contract failed")
    if "competition_submit" in runtime or "kaggle competitions submit" in runtime:
        raise SystemExit("E05 safe-site runtime contains submission path")
    compile(runtime, "generated_e05_safe_failure_site.py", "exec")
    return runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    path = args.runtime.expanduser().resolve()
    patched = patch_runtime(path.read_text(encoding="utf-8"))
    path.write_text(patched, encoding="utf-8")
    print(
        "CMI_FLU_E05_SAFE_FAILURE_SITE_BUILD PASS "
        f"runtime_sha256={hashlib.sha256(patched.encode()).hexdigest()} message=false locals=false row_data=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
