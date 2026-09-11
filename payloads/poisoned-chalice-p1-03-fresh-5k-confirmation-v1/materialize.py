#!/usr/bin/env python3
"""Reconstruct the exact frozen P1-03 fresh 5k confirmation Kaggle Notebook."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import ssl
import subprocess
import sys
import tempfile
import urllib.request

BRIDGE_HISTORY_EVAL_COMMIT = "6dd830e657d1fc004f14ba8ebb8301294fb78f66"
BRIDGE_HISTORY_EVAL_ROOT = "materialized/poisoned-chalice-lumia-hidden-state-eval-5000-v1"
BRIDGE_HISTORY_AUDIT_COMMIT = "5f3d94c9c589cb8c138cae8f93c79464cd1af508"
BRIDGE_HISTORY_AUDIT_PATH = (
    "materialized/poisoned-chalice-p1-03-post-5k-development-audit-v1/"
    "exact/run_p1_03_post_5k_development_audit_v1.py"
)
SCIENCE_COMMIT = "311b7f51a474cc3efb256b5cb7773eefc7f36c26"
SCIENCE_BUILDER_BLOB = "cccf255a13ecbfa25e32cb2fda8cdbd578cae073"
SCIENCE_FRESH_AUDIT_BLOB = "4f9eb6bb32184941e304875dc79cffe8ee7c1a02"
SCIENCE_DEVELOPMENT_AUDIT_BLOB = "3d0a45b20c93554e293b4db9c243cd1133f1fef9"
SCIENCE_EVALUATION_CONFIG_BLOB = "d642dfa51f922613154d7d5f2e95d211631180b5"
SCIENCE_NOTEBOOK_SHA256 = "4236585ba12931e075828bf5764508e237aae863118b4138c1df16a802c09daa"
FRESH_CACHE_MANIFEST_SHA256 = "f2b9c6048f1496b5f0c96a369d9773754f13ebace14aacfce8c0ecec92ba4481"
FRESH_COHORT_SET_SHA256 = "8fd03b98b46ffd9b7ef9a69ea58eaaeb6e0307cdafecf9d1c35a1562bbeaad27"
TARGET = "renta0426/poisoned-chalice-p1-03-fresh-5k-confirmation-v1"
TITLE = "Poisoned Chalice P1-03 Fresh 5K Confirmation V1"
NOTEBOOK = "poisoned-chalice-p1-03-fresh-5k-confirmation-v1.ipynb"

HISTORICAL_EVAL_BLOBS = {
    "src/poisoned_chalice/evaluation.py": "cb3afd4f3e3ecafe0e0677eda4fda30e93b04ba8",
    "src/poisoned_chalice/sersem_author_faithful.py": "91f0fcdccd646902ccc939e0835090a517505347",
    "src/poisoned_chalice/lumia_hidden_state_pilot.py": "d33fa4f4b0d258083d1503e4b4b883461f41ed8f",
    "src/poisoned_chalice/lumia_hidden_state_cache.py": "61182c7fcde131ed36c0791fa16928dcdc9b6e8b",
    "src/poisoned_chalice/lumia_hidden_state_probe.py": "dffe5648eb45f6919dcd002a46f7a2c6d6fb9121",
    "src/poisoned_chalice/lumia_hidden_state_scale.py": "3f323970ab8bac0a243be9cb0fed19b3c40f74f6",
    "scripts/run_lumia_hidden_state_scale_5000_evaluation.py": "d82e37b1ca5c6962c62c409d7944e9b6e7067bc7",
    "configs/p1_03_lumia_hidden_state_scale_5000_v1_20260910.json": "0dccee9b2e562ccab9f4b4a1e4c9f21cb5518b85",
}
HISTORICAL_AUDIT_BLOB = "76e09a937c0b257787bec7269716ff8cb7cb1e6d"


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def raw_get(commit: str, path: str, maximum: int = 512_000) -> bytes:
    url = f"https://raw.githubusercontent.com/renta0426/kaggle-actions-bridge/{commit}/{path}"
    ctx = ssl.create_default_context()
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx), NoRedirect())
    request = urllib.request.Request(url, headers={"User-Agent": "p1-03-fresh-confirm-materializer/1"})
    with opener.open(request, timeout=30) as response:
        data = response.read(maximum + 1)
    if not data or len(data) > maximum:
        raise RuntimeError(f"historical source byte budget failed: {path}")
    return data


def reconstruct_source(parts_dir: Path, stem: str, expected_blob: str) -> bytes:
    parts = sorted(parts_dir.glob(f"{stem}.part*.pyfrag"))
    if len(parts) != 3:
        raise RuntimeError(f"expected exactly three {stem} fragments, got {len(parts)}")
    # Both frozen source splits occur after a line followed by one intentional
    # blank line at the first boundary; the second boundary is an ordinary line
    # boundary. Contents-API fragments omit their terminal newline, so restore
    # those exact bytes explicitly before verifying the Git blob identity.
    raw = [path.read_bytes() for path in parts]
    data = raw[0] + b"\n\n" + raw[1] + b"\n" + raw[2] + b"\n"
    observed = git_blob_sha(data)
    if observed != expected_blob:
        raise RuntimeError(f"{stem} reassembly blob mismatch: {observed}")
    compile(data, stem, "exec")
    return data


def write_nbformat_shim(root: Path) -> None:
    (root / "nbformat.py").write_text(
        r'''import json
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
''',
        encoding="utf-8",
    )


def reconstruct_research(payload_dir: Path, research_root: Path) -> dict:
    for rel, expected_blob in HISTORICAL_EVAL_BLOBS.items():
        data = raw_get(
            BRIDGE_HISTORY_EVAL_COMMIT,
            f"{BRIDGE_HISTORY_EVAL_ROOT}/{rel}",
        )
        if git_blob_sha(data) != expected_blob:
            raise RuntimeError(f"historical eval blob mismatch: {rel}")
        out = research_root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        if rel.endswith(".py"):
            compile(data, rel, "exec")

    old_audit = raw_get(BRIDGE_HISTORY_AUDIT_COMMIT, BRIDGE_HISTORY_AUDIT_PATH)
    if git_blob_sha(old_audit) != HISTORICAL_AUDIT_BLOB:
        raise RuntimeError("historical development audit blob mismatch")
    source = old_audit.decode("utf-8")
    old = "    from sklearn.feature_extraction.text import FeatureUnion, HashingVectorizer\n    from sklearn.linear_model import LogisticRegression\n"
    new = "    from sklearn.feature_extraction.text import HashingVectorizer\n    from sklearn.linear_model import LogisticRegression\n    from sklearn.pipeline import FeatureUnion\n"
    if source.count(old) != 1:
        raise RuntimeError("FeatureUnion repair marker changed")
    current_audit = source.replace(old, new).encode("utf-8")
    if git_blob_sha(current_audit) != SCIENCE_DEVELOPMENT_AUDIT_BLOB:
        raise RuntimeError("repaired development audit does not match frozen science blob")
    audit_out = research_root / "scripts/run_p1_03_post_5k_development_audit_v1.py"
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    audit_out.write_bytes(current_audit)
    compile(current_audit, str(audit_out), "exec")

    fresh_audit = reconstruct_source(payload_dir, "fresh_audit", SCIENCE_FRESH_AUDIT_BLOB)
    fresh_out = research_root / "scripts/run_p1_03_fresh_5k_confirmation_audit_v1.py"
    fresh_out.write_bytes(fresh_audit)

    builder = reconstruct_source(payload_dir, "builder", SCIENCE_BUILDER_BLOB)
    builder_out = research_root / "scripts/build_p1_03_fresh_5k_confirmation_evaluation_notebook_v1.py"
    builder_out.write_bytes(builder)

    return {
        "historical_eval_commit": BRIDGE_HISTORY_EVAL_COMMIT,
        "historical_audit_commit": BRIDGE_HISTORY_AUDIT_COMMIT,
        "science_commit": SCIENCE_COMMIT,
        "science_builder_blob": SCIENCE_BUILDER_BLOB,
        "science_fresh_audit_blob": SCIENCE_FRESH_AUDIT_BLOB,
        "science_development_audit_blob": SCIENCE_DEVELOPMENT_AUDIT_BLOB,
        "science_evaluation_config_blob": SCIENCE_EVALUATION_CONFIG_BLOB,
    }


def validate_kernel(kernel_dir: Path) -> dict:
    files = sorted(path.name for path in kernel_dir.iterdir() if path.is_file())
    if files != ["kernel-metadata.json", NOTEBOOK]:
        raise RuntimeError(f"unexpected kernel files: {files}")
    metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    expected = {
        "id": TARGET,
        "title": TITLE,
        "code_file": NOTEBOOK,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["renta0426/stage1-raw-fim-submission-v1-output"],
        "kernel_sources": ["renta0426/poisoned-chalice-lumia-fresh-cache-5000-v1"],
        "competition_sources": [],
        "model_sources": [],
    }
    if metadata != expected:
        raise RuntimeError(f"kernel metadata mismatch: {metadata}")
    nb_path = kernel_dir / NOTEBOOK
    observed_sha = sha256_file(nb_path)
    if observed_sha != SCIENCE_NOTEBOOK_SHA256:
        raise RuntimeError(f"generated Notebook SHA mismatch: {observed_sha}")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    if nb.get("nbformat") != 4 or len(nb.get("cells") or []) != 5:
        raise RuntimeError("unexpected Notebook structure")
    code = "\n".join(str(cell.get("source", "")) for cell in nb["cells"])
    required = (
        FRESH_CACHE_MANIFEST_SHA256,
        FRESH_COHORT_SET_SHA256,
        "P1-03-fresh-hidden-anchor-evaluation-5000-v1",
        "P1-03-fresh-disjoint-5k-confirmation-evaluation-v1",
        "FROZEN_FUSION_WEIGHTS = (0.75, 0.50)",
        "rank_fusion_hidden_0.75",
        "rank_fusion_hidden_0.50",
        "fusion_intersection_rows",
        "scalar_features_per_pooling_variant",
        "raw_hidden_copied=0",
        "submission=0",
    )
    for marker in required:
        if marker not in code:
            raise RuntimeError(f"generated Notebook marker missing: {marker}")
    for forbidden in (
        'split="validation"', "split='validation'", "competition_submit(",
        "submission.csv", "rank_fusion_hidden_0.25",
    ):
        if forbidden in code:
            raise RuntimeError(f"forbidden generated Notebook marker: {forbidden}")
    return {"notebook_sha256": observed_sha, "metadata": metadata}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-dir", required=True, type=Path)
    parser.add_argument("--kernel-dir", required=True, type=Path)
    args = parser.parse_args()
    payload_dir = args.payload_dir.resolve()
    kernel_dir = args.kernel_dir.resolve()
    with tempfile.TemporaryDirectory(prefix="p1-03-fresh-confirm-research-") as temp:
        research = Path(temp) / "research"
        research.mkdir(parents=True)
        provenance = reconstruct_research(payload_dir, research)
        with tempfile.TemporaryDirectory(prefix="p1-03-fresh-confirm-nbformat-") as shim_temp:
            shim = Path(shim_temp)
            write_nbformat_shim(shim)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(shim)
            proc = subprocess.run(
                [sys.executable, str(research / "scripts/build_p1_03_fresh_5k_confirmation_evaluation_notebook_v1.py")],
                cwd=str(research), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=120, check=False,
            )
        if proc.returncode != 0:
            diagnostic = (proc.stdout + proc.stderr).encode("utf-8", errors="replace")
            raise RuntimeError(
                "frozen science builder failed "
                f"rc={proc.returncode} diagnostic_sha256={hashlib.sha256(diagnostic).hexdigest()}"
            )
        built = research / "notebooks/experiments/poisoned-chalice-p1-03-fresh-5k-confirmation-v1"
        if kernel_dir.exists():
            shutil.rmtree(kernel_dir)
        shutil.copytree(built, kernel_dir)
    validated = validate_kernel(kernel_dir)
    print(json.dumps({
        "status": "pass",
        "target": TARGET,
        "notebook_sha256": validated["notebook_sha256"],
        "provenance": provenance,
        "kaggle_write_calls": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
