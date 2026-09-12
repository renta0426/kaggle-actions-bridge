#!/usr/bin/env python3
"""One-shot launcher for STAGE2-DEPLOYMENT-VALIDATION-V1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

REQUEST_ID = "20260912-poisoned-chalice-stage2-deployment-validation-v1-001"
TARGET = "renta0426/poisoned-chalice-stage2-deployment-validation-v1"
TITLE = "Poisoned Chalice Stage2 Deployment Validation V1"
NOTEBOOK_NAME = "poisoned-chalice-stage2-deployment-validation-v1.ipynb"
RESEARCH_COMMIT = "def904f9f526e43648b370693977b932f8d6e64d"
STAGE1_DATASET = "renta0426/stage1-raw-fim-submission-v1-output"
SOURCE_CACHE_KERNEL = "renta0426/poisoned-chalice-lumia-hidden-state-cache-5000-v1"
MODEL_REVISION = "733247c55e3f73af49ce8e9c7949bf14af205928"
CACHE_MANIFEST_SHA256 = "380e553cea43c0bb4a649e7d6b696786b4e5178d45ee116efbd5e99cceeab6aa"
OUTPUT_PREFIX = "stage2_deployment_validation_v1"
EXPECTED_BLOBS = {
    "src/poisoned_chalice/evaluation.py": "cb3afd4f3e3ecafe0e0677eda4fda30e93b04ba8",
    "src/poisoned_chalice/sersem_author_faithful.py": "91f0fcdccd646902ccc939e0835090a517505347",
    "src/poisoned_chalice/sersem_author_runtime.py": "024156b0723cdc084803b400f6705bc76fbfbfc6",
    "src/poisoned_chalice/lumia_hidden_state_pilot.py": "d33fa4f4b0d258083d1503e4b4b883461f41ed8f",
    "src/poisoned_chalice/lumia_hidden_state_cache.py": "61182c7fcde131ed36c0791fa16928dcdc9b6e8b",
    "src/poisoned_chalice/lumia_hidden_state_scale_cache.py": "d69a088addade20475805aa5b69ad5bb0ee5591f",
    "src/poisoned_chalice/lumia_hidden_state_probe.py": "dffe5648eb45f6919dcd002a46f7a2c6d6fb9121",
    "src/poisoned_chalice/stage2_api.py": "16578d3e01b6471e5bfb2b04cebb99e1b3126c7c",
    "src/poisoned_chalice/stage2_deployment_validation.py": "bb5d9dc56f6aed6c6d091e4eb8852172879e5afe",
    "src/poisoned_chalice/stage2_deployment_evaluation.py": "d4ca76ab2c03e2f467de18c6b78e2e352f45a262",
    "scripts/run_stage2_deployment_validation_v1.py": "e5ba9f31aedcb184488c5756ad771f3e25a4e6a0",
    "scripts/build_stage2_deployment_validation_v1_notebook.py": "18c6701a7a1baa6709306c98f88e77d45c30c1c6",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def slugify(value: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-"))


def validate_request(request: dict) -> None:
    expected = {
        "request_id": REQUEST_ID,
        "competition": "poisoned-chalice-icse27",
        "operation": "kernel_run",
        "target": TARGET,
        "launcher_path": "runners/poisoned_chalice/stage2_deployment_validation_v1.py",
        "research_repository": "renta0426/The-Poisoned-Chalice-of-LLM-Evaluation",
        "research_commit": RESEARCH_COMMIT,
        "automatic_compute_retries": 0,
        "enable_internet": True,
        "competition_submission": False,
        "select_as_final": False,
    }
    for key, value in expected.items():
        if request.get(key) != value:
            raise RuntimeError(f"Stage2 deployment request changed: {key}")
    resource = request.get("resource") or {}
    if resource != {
        "accelerator": "gpu", "machine_shape": "NvidiaTeslaT4", "expected_visible_gpu_count": 2,
        "expected_runtime_minutes": 120, "hard_timeout_minutes": 180, "max_active_runs": 1,
        "min_remaining_quota_hours": 2.5,
    }:
        raise RuntimeError("Stage2 resource contract changed")
    if request.get("side_effects") != ["create one private notebook version"]:
        raise RuntimeError("Stage2 side-effect contract changed")
    science = request.get("scientific_contract") or {}
    required = {
        "task_id": "STAGE2-DEPLOYMENT-VALIDATION-V1",
        "model_revision": MODEL_REVISION,
        "source_cache_manifest_sha256": CACHE_MANIFEST_SHA256,
        "source_rows": 5000,
        "target_rows": 5000,
        "prediction_artifact_hashed_before_target_label_join": True,
        "target_performance_feedback_into_fit": False,
        "competition_submission": False,
    }
    for key, value in required.items():
        if science.get(key) != value:
            raise RuntimeError(f"Stage2 scientific contract changed: {key}")


def validate_research(root: Path) -> None:
    for relative, expected in EXPECTED_BLOBS.items():
        path = root / relative
        if not path.is_file() or git_blob_sha(path) != expected:
            raise RuntimeError(f"research Git blob changed: {relative}")
        if path.suffix == ".py":
            compile(path.read_text(encoding="utf-8"), relative, "exec")
    builder = (root / "scripts/build_stage2_deployment_validation_v1_notebook.py").read_text(encoding="utf-8")
    runner = (root / "scripts/run_stage2_deployment_validation_v1.py").read_text(encoding="utf-8")
    for marker in (TARGET, TITLE, STAGE1_DATASET, SOURCE_CACHE_KERNEL, '"competition_sources": []', '"is_private": True', '"enable_gpu": True'):
        if marker not in builder:
            raise RuntimeError(f"Stage2 builder marker missing: {marker}")
    for marker in ('TASK_ID = "STAGE2-DEPLOYMENT-VALIDATION-V1"', MODEL_REVISION, CACHE_MANIFEST_SHA256,
                   'OUTPUT = Path("/kaggle/working/stage2_deployment_validation_v1")', "seal_source_bundle",
                   "select_holdout", "predictions.to_csv(PREDICTIONS, index=False)", "predictions_label_free.sha256",
                   "evaluate_after_seal", '"holdout_retraining_or_reselection": False'):
        if marker not in runner:
            raise RuntimeError(f"Stage2 runner marker missing: {marker}")
    for forbidden in ("kaggle competitions submit", "kagglehub.competition_submit", "kaggle competitions files", "kaggle competitions download"):
        if forbidden in builder.casefold() or forbidden in runner.casefold():
            raise RuntimeError(f"forbidden competition operation present: {forbidden}")


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


def validate_kernel(root: Path) -> None:
    if sorted(path.name for path in root.iterdir() if path.is_file()) != ["kernel-metadata.json", NOTEBOOK_NAME]:
        raise RuntimeError("unexpected Stage2 kernel files")
    metadata = load_json(root / "kernel-metadata.json")
    expected = {
        "id": TARGET, "title": TITLE, "code_file": NOTEBOOK_NAME, "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [STAGE1_DATASET], "kernel_sources": [SOURCE_CACHE_KERNEL], "competition_sources": [],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"kernel metadata changed: {key}")
    if slugify(metadata["title"]) != TARGET.split("/", 1)[1]:
        raise RuntimeError("kernel title-derived slug mismatch")
    notebook = load_json(root / NOTEBOOK_NAME)
    cells = notebook.get("cells") or []
    if notebook.get("nbformat") != 4 or notebook.get("nbformat_minor") != 5 or len(cells) != 4:
        raise RuntimeError("unexpected notebook structure")
    code = "\n".join(str(cell.get("source", "")) for cell in cells)
    for marker in ("STAGE2_DV_INSTALL PASS", "STAGE2_DV_MATERIALIZE PASS", "STAGE2_DEPLOYMENT_VALIDATION_V1_NOTEBOOK COMPLETE", MODEL_REVISION, CACHE_MANIFEST_SHA256, "predictions_label_free.sha256", "evaluate_after_seal"):
        if marker not in code:
            raise RuntimeError(f"generated Stage2 notebook marker missing: {marker}")
    for forbidden in ("kaggle competitions submit", "kagglehub.competition_submit", "kaggle competitions files", "kaggle competitions download"):
        if forbidden in code.casefold():
            raise RuntimeError(f"generated notebook contains forbidden operation: {forbidden}")


def materialize(request: dict, research_root: Path, kernel_dir: Path) -> None:
    validate_request(request)
    validate_research(research_root)
    with tempfile.TemporaryDirectory(prefix="stage2-dv-v1-build-") as td:
        shim = Path(td)
        write_nbformat_shim(shim)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(shim)
        builder = research_root / "scripts/build_stage2_deployment_validation_v1_notebook.py"
        completed = subprocess.run([sys.executable, str(builder)], cwd=str(research_root), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        if completed.returncode != 0:
            raise RuntimeError(f"Stage2 Notebook materialization failed rc={completed.returncode} output_sha256={hashlib.sha256((completed.stdout or '').encode()).hexdigest()}")
    generated = research_root / "notebooks/experiments/poisoned-chalice-stage2-deployment-validation-v1"
    if not generated.is_dir():
        raise RuntimeError("generated Stage2 kernel directory missing")
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True)
    for name in ("kernel-metadata.json", NOTEBOOK_NAME):
        shutil.copy2(generated / name, kernel_dir / name)
    metadata_path = kernel_dir / "kernel-metadata.json"
    metadata = load_json(metadata_path)
    metadata["enable_tpu"] = False
    metadata["machine_shape"] = "NvidiaTeslaT4"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_kernel(kernel_dir)
    print("STAGE2_DEPLOYMENT_VALIDATION_V1_MATERIALIZE PASS target=fresh_5000 submissions=0")


def execute(request: dict, kernel_dir: Path, kaggle_bin: str) -> None:
    validate_request(request)
    validate_kernel(kernel_dir)
    completed = subprocess.run([kaggle_bin, "kernels", "push", "-p", str(kernel_dir)], env=os.environ.copy(), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    stdout, stderr = completed.stdout or "", completed.stderr or ""
    if completed.returncode != 0:
        raise RuntimeError(f"Stage2 deployment launch failed rc={completed.returncode} stdout_sha256={hashlib.sha256(stdout.encode()).hexdigest()} stderr_sha256={hashlib.sha256(stderr.encode()).hexdigest()}")
    print("STAGE2_DEPLOYMENT_VALIDATION_V1_LAUNCH_ACCEPTED " + f"target={TARGET} stdout_bytes={len(stdout.encode())} stderr_bytes={len(stderr.encode())} stdout_sha256={hashlib.sha256(stdout.encode()).hexdigest()} stderr_sha256={hashlib.sha256(stderr.encode()).hexdigest()} retries=0 submissions=0")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--research-root", type=Path)
    parser.add_argument("--kernel-dir", type=Path)
    parser.add_argument("--kaggle-bin", default="kaggle")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static", action="store_true")
    mode.add_argument("--materialize", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    request = load_json(args.request)
    if args.static:
        if args.research_root is None:
            raise RuntimeError("--research-root required for static validation")
        validate_request(request); validate_research(args.research_root)
        print("STAGE2_DEPLOYMENT_VALIDATION_V1_STATIC PASS source=5000 target=5000 submissions=0")
        return 0
    if args.materialize:
        if args.research_root is None or args.kernel_dir is None:
            raise RuntimeError("--research-root and --kernel-dir required")
        materialize(request, args.research_root, args.kernel_dir)
        return 0
    if args.kernel_dir is None:
        raise RuntimeError("--kernel-dir required")
    execute(request, args.kernel_dir, args.kaggle_bin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
