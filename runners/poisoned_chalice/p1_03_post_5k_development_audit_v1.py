#!/usr/bin/env python3
"""One-shot launcher for the frozen P1-03 post-5k CPU development audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.request

REQUEST_ID = "20260910-poisoned-chalice-p1-03-post-5k-development-audit-v1-001"
TARGET = "renta0426/poisoned-chalice-p1-03-post-5k-development-audit-v1"
BRIDGE_REPO = "renta0426/kaggle-actions-bridge"
RESEARCH_REPO = "renta0426/The-Poisoned-Chalice-of-LLM-Evaluation"
RESEARCH_COMMIT = "feff26fa21184744efc5f4e91a64d22fa1de4b4e"
STAGE1_DATASET = "renta0426/stage1-raw-fim-submission-v1-output"
HIDDEN_CACHE = "renta0426/poisoned-chalice-lumia-hidden-state-cache-5000-v1"
HIDDEN_EVAL = "renta0426/poisoned-chalice-lumia-hidden-state-eval-5000-v1"
NOTEBOOK_NAME = "poisoned-chalice-p1-03-post-5k-development-audit-v1.ipynb"
OUTPUT_PREFIX = "p1_03_post_5k_development_audit_v1"
SNAPSHOT_ROOT = "materialized/poisoned-chalice-p1-03-post-5k-development-audit-v1"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")

EXPECTED_BLOBS = {
    "scripts/run_p1_03_post_5k_development_audit_v1.py": "76e09a937c0b257787bec7269716ff8cb7cb1e6d",
    "scripts/build_p1_03_post_5k_development_audit_notebook_v1.py": "e9272cbee32d03f52dff758f0b87079c3f9bbfa2",
    "configs/p1_03_post_5k_development_audit_v1_20260910.json": "322487c8f0a932160ab825775ca80687ce295579",
}
SNAPSHOT_PARTS = {
    "scripts/run_p1_03_post_5k_development_audit_v1.py": [
        f"{SNAPSHOT_ROOT}/scripts/run_p1_03_post_5k_development_audit_v1.py.part01",
        f"{SNAPSHOT_ROOT}/scripts/run_p1_03_post_5k_development_audit_v1.py.part02",
        f"{SNAPSHOT_ROOT}/scripts/run_p1_03_post_5k_development_audit_v1.py.part03",
        f"{SNAPSHOT_ROOT}/scripts/run_p1_03_post_5k_development_audit_v1.py.part04",
    ],
    "scripts/build_p1_03_post_5k_development_audit_notebook_v1.py": [
        f"{SNAPSHOT_ROOT}/scripts/build_p1_03_post_5k_development_audit_notebook_v1.py.part01",
        f"{SNAPSHOT_ROOT}/scripts/build_p1_03_post_5k_development_audit_notebook_v1.py.part02",
    ],
    "configs/p1_03_post_5k_development_audit_v1_20260910.json": [
        f"{SNAPSHOT_ROOT}/configs/p1_03_post_5k_development_audit_v1_20260910.json",
    ],
}
PERSISTENT_OUTPUTS = [
    f"{OUTPUT_PREFIX}/content_control_audit.json",
    f"{OUTPUT_PREFIX}/content_control_oof_scores.parquet",
    f"{OUTPUT_PREFIX}/fusion_audit.json",
    f"{OUTPUT_PREFIX}/fusion_common_scores.parquet",
    f"{OUTPUT_PREFIX}/portable_scalar_audit.json",
    f"{OUTPUT_PREFIX}/portable_scalar_oof_scores.parquet",
    f"{OUTPUT_PREFIX}/portable_scalar_table.parquet",
    f"{OUTPUT_PREFIX}/run_manifest.json",
    f"{OUTPUT_PREFIX}/static_feature_table.parquet",
]
RESOURCE = {
    "accelerator": "cpu",
    "expected_runtime_minutes": 50,
    "hard_timeout_minutes": 120,
    "max_active_runs": 1,
}

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def slugify(value: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-"))

def bridge_source_sha() -> str:
    value = os.environ.get("SOURCE_SHA") or os.environ.get("GITHUB_SHA") or ""
    if not SHA_RE.fullmatch(value):
        raise RuntimeError("immutable bridge source SHA unavailable")
    return value

def safe_snapshot_path(path: str) -> str:
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise RuntimeError("unsafe bridge snapshot path")
    normalized = str(pure)
    if not normalized.startswith(SNAPSHOT_ROOT + "/"):
        raise RuntimeError("snapshot path outside frozen root")
    return normalized

def fetch_bridge_snapshot(path: str, maximum: int = 131_072) -> bytes:
    safe = safe_snapshot_path(path)
    sha = bridge_source_sha()
    url = f"https://raw.githubusercontent.com/{BRIDGE_REPO}/{sha}/{safe}"
    context = ssl.create_default_context()
    last: Exception | None = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "poisoned-chalice-p1-03-post5k-snapshot/1"})
            with urllib.request.urlopen(request, timeout=30, context=context) as response:
                data = response.read(maximum + 1)
            if not data or len(data) > maximum:
                raise RuntimeError("snapshot byte budget failed")
            return data
        except Exception as exc:
            last = exc
            if attempt == 2:
                break
            time.sleep(attempt + 1)
    raise RuntimeError(f"bounded bridge snapshot fetch failed: {type(last).__name__}")

def reconstruct_science_file(path: str) -> bytes:
    parts = SNAPSHOT_PARTS.get(path)
    if not parts or len(parts) > 16:
        raise RuntimeError(f"snapshot manifest missing: {path}")
    data = b"".join(fetch_bridge_snapshot(part) for part in parts)
    if not data or len(data) > 262_144:
        raise RuntimeError(f"reconstructed science file byte budget failed: {path}")
    expected = EXPECTED_BLOBS[path]
    observed = git_blob_sha(data)
    if observed != expected:
        raise RuntimeError(f"snapshot/research Git blob mismatch: {path}: observed={observed} expected={expected}")
    return data

def exact_request() -> dict:
    return {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "competition": "poisoned-chalice-icse27",
        "operation": "save_kernel_once",
        "target": TARGET,
        "launcher_path": "runners/poisoned_chalice/p1_03_post_5k_development_audit_v1.py",
        "research_repository": RESEARCH_REPO,
        "research_commit": RESEARCH_COMMIT,
        "research_files": [{"path": path, "git_blob_sha": blob} for path, blob in EXPECTED_BLOBS.items()],
        "inputs": {
            "dataset": {"ref": STAGE1_DATASET, "version": 1},
            "kernels": [{"ref": HIDDEN_CACHE, "version": 1}, {"ref": HIDDEN_EVAL, "version": 1}],
        },
        "resource": RESOURCE,
        "side_effects": ["create one private notebook version"],
        "persistent_outputs": PERSISTENT_OUTPUTS,
        "automatic_compute_retries": 0,
        "enable_internet": True,
        "competition_submission": False,
        "select_as_final": False,
        "clean_room": {
            "development_only": True,
            "promotion_allowed": False,
            "fresh_holdout_consumed": False,
            "new_target_model_forward": False,
            "validation_rows_used": False,
            "hidden_stage1_validation_labels_used": False,
            "public_leaderboard_feedback_used": False,
        },
        "scientific_contract": {
            "stage1_oof_reproduction_auc": 0.68013844,
            "hidden_oof_sha256": "c6d35c24087eac4809678bb5cae3d02fef630d07eb6f4033c50b8e285ce56fb4",
            "hidden_cache_manifest_sha256": "380e553cea43c0bb4a649e7d6b696786b4e5178d45ee116efbd5e99cceeab6aa",
            "fusion_hidden_rank_weights": [0.25, 0.5, 0.75],
            "portable_features_per_pooling_variant": 96,
            "content_controls": ["C0_static", "C1_hashed_char_token_ngram", "SimHash_group_diagnostic"],
        },
    }

def validate_request(request: dict) -> None:
    if request != exact_request():
        raise RuntimeError("active request differs from exact frozen P1-03 post-5k contract")

def materialize_research(root: Path) -> None:
    if set(SNAPSHOT_PARTS) != set(EXPECTED_BLOBS):
        raise RuntimeError("snapshot/research manifest key mismatch")
    for path in EXPECTED_BLOBS:
        data = reconstruct_science_file(path)
        dest = root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        if dest.suffix == ".py":
            compile(data, path, "exec")

def write_nbformat_shim(root: Path) -> None:
    (root / "nbformat.py").write_text(r'''import json
class A(dict):
 def __getattr__(self,n):
  try:return self[n]
  except KeyError as e:raise AttributeError(n) from e
 def __setattr__(self,n,v):self[n]=v
class V4:
 def new_notebook(self):return A(cells=[],metadata=A(),nbformat=4,nbformat_minor=5)
 def new_markdown_cell(self,source=""):return A(cell_type="markdown",metadata=A(),source=source)
 def new_code_cell(self,source=""):return A(cell_type="code",execution_count=None,metadata=A(),outputs=[],source=source)
v4=V4()
def write(nb,path):
 with open(path,"w",encoding="utf-8") as f:json.dump(nb,f,ensure_ascii=False,indent=1);f.write("\n")
''', encoding="utf-8")

def validate_kernel(kernel_dir: Path) -> None:
    observed = sorted(p.name for p in kernel_dir.iterdir() if p.is_file())
    if observed != ["kernel-metadata.json", NOTEBOOK_NAME]:
        raise RuntimeError(f"unexpected kernel bundle files: {observed}")
    metadata = load_json(kernel_dir / "kernel-metadata.json")
    exact_meta = {
        "id": TARGET,
        "title": "Poisoned Chalice P1-03 Post-5K Development Audit V1",
        "code_file": NOTEBOOK_NAME,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [STAGE1_DATASET],
        "kernel_sources": [HIDDEN_CACHE, HIDDEN_EVAL],
        "competition_sources": [],
        "model_sources": [],
    }
    if metadata != exact_meta:
        raise RuntimeError("generated kernel metadata differs from exact CPU contract")
    if slugify(metadata["title"]) != TARGET.split("/", 1)[1]:
        raise RuntimeError("kernel title-derived slug differs from frozen target")
    notebook = load_json(kernel_dir / NOTEBOOK_NAME)
    if notebook.get("nbformat") != 4 or len(notebook.get("cells") or []) != 4:
        raise RuntimeError("generated Notebook structure changed")
    code = "\n".join(str(cell.get("source", "")) for cell in notebook["cells"])
    required = [
        "c6d35c24087eac4809678bb5cae3d02fef630d07eb6f4033c50b8e285ce56fb4",
        "380e553cea43c0bb4a649e7d6b696786b4e5178d45ee116efbd5e99cceeab6aa",
        "stage1_shards) != 40",
        "P1_03_POST_5K_DEVELOPMENT_AUDIT COMPLETE",
        "development_only",
    ]
    for marker in required:
        if marker not in code:
            raise RuntimeError(f"generated Notebook marker missing: {marker}")
    for forbidden in ("submission.csv", "competition_submit(", "kaggle competitions submit", "AutoModelForCausalLM"):
        if forbidden in code:
            raise RuntimeError(f"generated Notebook contains forbidden capability: {forbidden}")

def materialize(request: dict, kernel_dir: Path) -> None:
    validate_request(request)
    with tempfile.TemporaryDirectory(prefix="p1-03-post5k-") as td:
        temp = Path(td)
        research = temp / "research"
        research.mkdir()
        materialize_research(research)
        shim = temp / "shim"
        shim.mkdir()
        write_nbformat_shim(shim)
        builder = research / "scripts/build_p1_03_post_5k_development_audit_notebook_v1.py"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(shim)
        completed = subprocess.run(
            [sys.executable, str(builder)], cwd=str(research), env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=45,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Notebook materialization failed rc={completed.returncode} "
                f"output_sha256={sha256(completed.stdout.encode())}"
            )
        generated = research / "notebooks/experiments/poisoned-chalice-p1-03-post-5k-development-audit-v1"
        if not generated.is_dir():
            raise RuntimeError("generated kernel directory missing")
        if kernel_dir.exists():
            shutil.rmtree(kernel_dir)
        kernel_dir.mkdir(parents=True)
        for name in ("kernel-metadata.json", NOTEBOOK_NAME):
            shutil.copy2(generated / name, kernel_dir / name)
    validate_kernel(kernel_dir)
    print(
        "P1_03_POST_5K_MATERIALIZE PASS "
        f"science_commit={RESEARCH_COMMIT} snapshot_source={bridge_source_sha()} cpu=1 retries=0 submissions=0"
    )

def execute(request: dict, kernel_dir: Path, kaggle_bin: str) -> None:
    validate_request(request)
    validate_kernel(kernel_dir)
    completed = subprocess.run(
        [kaggle_bin, "kernels", "push", "-p", str(kernel_dir)],
        env=os.environ.copy(), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
    )
    stdout, stderr = completed.stdout or "", completed.stderr or ""
    if completed.returncode != 0:
        raise RuntimeError(
            f"one-shot SaveKernel failed rc={completed.returncode} "
            f"stdout_bytes={len(stdout.encode())} stderr_bytes={len(stderr.encode())} "
            f"stdout_sha256={sha256(stdout.encode())} stderr_sha256={sha256(stderr.encode())}"
        )
    print(
        f"ONE_SHOT_WRITE_ATTEMPTED true target={TARGET} version=1 accelerator=cpu "
        f"automatic_retries=0 submissions=0 stdout_bytes={len(stdout.encode())} "
        f"stderr_bytes={len(stderr.encode())} stdout_sha256={sha256(stdout.encode())} "
        f"stderr_sha256={sha256(stderr.encode())}"
    )

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--kernel-dir", type=Path)
    parser.add_argument("--kaggle-bin", default="kaggle")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static", action="store_true")
    mode.add_argument("--materialize", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    request = load_json(args.request)
    if args.static:
        validate_request(request)
        with tempfile.TemporaryDirectory(prefix="p1-03-post5k-static-") as td:
            materialize_research(Path(td))
        print(
            "P1_03_POST_5K_STATIC PASS "
            f"science_commit={RESEARCH_COMMIT} snapshot_source={bridge_source_sha()} cpu=1"
        )
        return 0
    if args.kernel_dir is None:
        raise RuntimeError("--kernel-dir required")
    if args.materialize:
        materialize(request, args.kernel_dir)
        return 0
    execute(request, args.kernel_dir, args.kaggle_bin)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
