#!/usr/bin/env python3
"""Add privacy-safe failure-site reporting to the E06a production runtime."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def patch_runtime(runtime: str) -> str:
    if "def _e06a_safe_failure_site" in runtime or "CMI_FLU_E06A_SAFE_FAILURE_SITE" in runtime:
        raise SystemExit("E06a runtime is already safe-site instrumented")
    import_anchor = "import types\n"
    if runtime.count(import_anchor) != 1:
        raise SystemExit("E06a safe-site traceback import anchor changed")
    runtime = runtime.replace(import_anchor, import_anchor + "import traceback\n", 1)

    helper_anchor = "\ndef main() -> int:\n"
    if runtime.count(helper_anchor) != 1:
        raise SystemExit("E06a safe-site helper anchor changed")
    helper = r'''
def _e06a_safe_failure_site(exc: BaseException) -> tuple[str, str]:
    sites = []
    for frame in traceback.extract_tb(exc.__traceback__):
        raw = str(frame.filename).replace("\\", "/")
        module = None
        if "cmi_flu/" in raw:
            module = "cmi_flu/" + raw.rsplit("cmi_flu/", 1)[1]
        elif raw.startswith("<cmi_flu.") and raw.endswith(">"):
            module = raw[1:-1].replace(".", "/") + ".py"
        elif raw.endswith("script.py") or "generated_e06a" in raw:
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

    execute_old = '''    except Exception as exc:\n        shutil.rmtree(runtime_root, ignore_errors=True)\n        code = hashlib.sha256(f"{type(exc).__name__}:{str(exc)}".encode("utf-8", errors="replace")).hexdigest()[:20]\n        print(f"CMI_FLU_E06A_FAILED stage={stage} exception_type={type(exc).__name__} error_code={code}", file=sys.stderr)\n        return 2\n'''
    if runtime.count(execute_old) != 1:
        raise SystemExit(f"E06a safe-site execute failure block count changed:{runtime.count(execute_old)}")
    execute_new = '''    except Exception as exc:\n        shutil.rmtree(runtime_root, ignore_errors=True)\n        code = hashlib.sha256(f"{type(exc).__name__}:{str(exc)}".encode("utf-8", errors="replace")).hexdigest()[:20]\n        sites, sites_sha = _e06a_safe_failure_site(exc)\n        print(f"CMI_FLU_E06A_SAFE_FAILURE_SITE stage={stage} exception_type={type(exc).__name__} error_code={code} sites={sites} sites_sha256={sites_sha}", file=sys.stderr)\n        return 2\n'''
    runtime = runtime.replace(execute_old, execute_new, 1)

    locate_old = '''    except Exception as exc:\n        code = hashlib.sha256(f"{type(exc).__name__}:{str(exc)}".encode()).hexdigest()[:20]\n        print(f"CMI_FLU_E06A_FAILED stage=locate_competition_data exception_type={type(exc).__name__} error_code={code}", file=sys.stderr)\n        return 2\n'''
    if runtime.count(locate_old) != 1:
        raise SystemExit(f"E06a safe-site locate failure block count changed:{runtime.count(locate_old)}")
    locate_new = '''    except Exception as exc:\n        code = hashlib.sha256(f"{type(exc).__name__}:{str(exc)}".encode()).hexdigest()[:20]\n        sites, sites_sha = _e06a_safe_failure_site(exc)\n        print(f"CMI_FLU_E06A_SAFE_FAILURE_SITE stage=locate_competition_data exception_type={type(exc).__name__} error_code={code} sites={sites} sites_sha256={sites_sha}", file=sys.stderr)\n        return 2\n'''
    runtime = runtime.replace(locate_old, locate_new, 1)

    if runtime.count("CMI_FLU_E06A_SAFE_FAILURE_SITE") != 2 or runtime.count("def _e06a_safe_failure_site") != 1:
        raise SystemExit("E06a safe-site marker/helper contract failed")
    if "CMI_FLU_E06A_FAILED stage=" in runtime:
        raise SystemExit("E06a uninstrumented failure marker remains")
    if "competition_submit" in runtime or "kaggle competitions submit" in runtime:
        raise SystemExit("E06a safe-site runtime contains submission path")
    compile(runtime, "generated_e06a_safe_failure_site.py", "exec")
    return runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    path = args.runtime.expanduser().resolve()
    patched = patch_runtime(path.read_text(encoding="utf-8"))
    path.write_text(patched, encoding="utf-8")
    print(
        "CMI_FLU_E06A_SAFE_FAILURE_SITE_BUILD PASS "
        f"runtime_sha256={hashlib.sha256(patched.encode()).hexdigest()} "
        "execute_site=true locate_site=true message=false locals=false row_data=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
