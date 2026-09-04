#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

SUCCESS_OLD = '''        print(
            "CMI_FLU_ANCHOR_RESIDUAL_COMPLETE "
            f"request_id={REQUEST_ID} tasks=2 weights={len(weights) + 1}"
        )
        return payload
    except Exception as error:
        safe_failure(output_dir, stage=stage, error=error)
'''
SUCCESS_NEW = '''        shutil.rmtree(runtime_root, ignore_errors=True)
        print(
            "CMI_FLU_ANCHOR_RESIDUAL_COMPLETE "
            f"request_id={REQUEST_ID} tasks=2 weights={len(weights) + 1}"
        )
        return payload
    except Exception as error:
        shutil.rmtree(runtime_root, ignore_errors=True)
        safe_failure(output_dir, stage=stage, error=error)
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    text = args.source.read_text(encoding="utf-8")
    if text.count(SUCCESS_OLD) != 1:
        raise SystemExit(f"anchor-residual cleanup anchor count={text.count(SUCCESS_OLD)}")
    text = text.replace(SUCCESS_OLD, SUCCESS_NEW, 1)
    compile(text, str(args.output), "exec")
    args.output.write_text(text, encoding="utf-8")
    print("CMI_FLU_ANCHOR_RESIDUAL_PATCH_V3 PASS output_hygiene=final_files_only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
