#!/usr/bin/env python3
"""Materialize the connector-verified E11c science relay from bridge payloads."""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

SCIENCE_COMMIT = "0815f4517345e127dbcbcae0f380a54f3a3d15bd"
PAYLOAD = "payloads/cmi-flu-strategy-e11c-robust-tabpfn-001"
E11B_PAYLOAD = "payloads/cmi-flu-strategy-e11b-tabpfn3-001"
FILES = {
    f"{PAYLOAD}/strategy_e11c.py": (
        "src/cmi_flu/strategy_e11c.py",
        "acf876996b5849f7ba794999d45ecb862d1be507",
    ),
    f"{PAYLOAD}/strategy_e11c_synthetic.py": (
        "src/cmi_flu/strategy_e11c_synthetic.py",
        "9b94633192b5254a7d1debe8db10606b48104076",
    ),
    f"{E11B_PAYLOAD}/strategy_e11b.py": (
        "src/cmi_flu/strategy_e11b.py",
        "8a291ae4952786bf16ee667caa5557599310e3fd",
    ),
    f"{E11B_PAYLOAD}/strategy_e11b_synthetic.py": (
        "src/cmi_flu/strategy_e11b_synthetic.py",
        "f25d04d75db584db78d26a232ab84dc5ab1bbf7d",
    ),
    f"{PAYLOAD}/baseline_b021_robust.yaml": (
        "configs/baseline_b021_robust.yaml",
        "170d3211e2795c0730e481056c7bb068accf97c9",
    ),
    f"{PAYLOAD}/strategy_e11c_task11_robust_tabpfn.json": (
        "configs/strategy_e11c_task11_robust_tabpfn.json",
        "3b475f777307238bfcad61c124ab99175207bac2",
    ),
}


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit("E11c science relay output must be fresh")
    output.mkdir(parents=True)
    copied = []
    try:
        for source_rel, (dest_rel, expected_blob) in FILES.items():
            source = root / source_rel
            data = source.read_bytes()
            found = git_blob_sha(data)
            if found != expected_blob:
                raise SystemExit(f"E11c relay source blob mismatch:{source_rel}:{found}")
            if dest_rel.endswith(".py"):
                text = data.decode("utf-8")
                compile(text, dest_rel, "exec")
                lowered = text.casefold()
                if "kaggle competitions submit" in lowered or "competition_submit" in lowered:
                    raise SystemExit(f"E11c relay contains submission path:{dest_rel}")
            dest = output / dest_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            if git_blob_sha(dest.read_bytes()) != expected_blob:
                raise SystemExit(f"E11c relay destination blob mismatch:{dest_rel}")
            copied.append(dest_rel)
        if sorted(copied) != sorted(dest for dest, _ in FILES.values()):
            raise SystemExit("E11c relay file set mismatch")
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
    print(
        "CMI_FLU_E11C_SCIENCE_RELAY PASS "
        f"science_commit={SCIENCE_COMMIT} files={len(copied)} exact_git_blobs=true "
        "private_repo_runtime_access=false submission_path=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
