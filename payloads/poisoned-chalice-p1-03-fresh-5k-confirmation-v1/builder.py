"""Build the frozen P1-03 fresh-disjoint 5k confirmation evaluation Notebook."""
from __future__ import annotations

from hashlib import sha1
from pathlib import Path
import json
import textwrap

import nbformat

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_SOURCE = (ROOT / "src/poisoned_chalice/evaluation.py").read_text(encoding="utf-8")
FAITHFUL_SOURCE = (ROOT / "src/poisoned_chalice/sersem_author_faithful.py").read_text(encoding="utf-8")
PILOT_SOURCE = (ROOT / "src/poisoned_chalice/lumia_hidden_state_pilot.py").read_text(encoding="utf-8")
CACHE_SOURCE = (ROOT / "src/poisoned_chalice/lumia_hidden_state_cache.py").read_text(encoding="utf-8")
PROBE_SOURCE = (ROOT / "src/poisoned_chalice/lumia_hidden_state_probe.py").read_text(encoding="utf-8")
SCALE_SOURCE = (ROOT / "src/poisoned_chalice/lumia_hidden_state_scale.py").read_text(encoding="utf-8")
HIDDEN_EVAL_BASE_PATH = ROOT / "scripts/run_lumia_hidden_state_scale_5000_evaluation.py"
DEVELOPMENT_AUDIT_PATH = ROOT / "scripts/run_p1_03_post_5k_development_audit_v1.py"
FRESH_AUDIT_PATH = ROOT / "scripts/run_p1_03_fresh_5k_confirmation_audit_v1.py"
SCALE_CONFIG_PATH = ROOT / "configs/p1_03_lumia_hidden_state_scale_5000_v1_20260910.json"

EXPECTED_HIDDEN_EVAL_BASE_BLOB = "d82e37b1ca5c6962c62c409d7944e9b6e7067bc7"
EXPECTED_DEVELOPMENT_AUDIT_BLOB = "3d0a45b20c93554e293b4db9c243cd1133f1fef9"
FRESH_CACHE_MANIFEST_SHA256 = "f2b9c6048f1496b5f0c96a369d9773754f13ebace14aacfce8c0ecec92ba4481"
FRESH_COHORT_SET_SHA256 = "8fd03b98b46ffd9b7ef9a69ea58eaaeb6e0307cdafecf9d1c35a1562bbeaad27"
FRESH_CACHE_TOTAL_SHARD_BYTES = 4512840643

TARGET = "renta0426/poisoned-chalice-p1-03-fresh-5k-confirmation-v1"
TITLE = "Poisoned Chalice P1-03 Fresh 5K Confirmation V1"
CACHE_KERNEL = "renta0426/poisoned-chalice-lumia-fresh-cache-5000-v1"
STAGE1_DATASET = "renta0426/stage1-raw-fim-submission-v1-output"
NOTEBOOK_NAME = "poisoned-chalice-p1-03-fresh-5k-confirmation-v1.ipynb"
OUT_DIR = ROOT / "notebooks/experiments/poisoned-chalice-p1-03-fresh-5k-confirmation-v1"
OUT = OUT_DIR / NOTEBOOK_NAME


def git_blob_sha(data: bytes) -> str:
    return sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


hidden_eval_raw = HIDDEN_EVAL_BASE_PATH.read_bytes()
if git_blob_sha(hidden_eval_raw) != EXPECTED_HIDDEN_EVAL_BASE_BLOB:
    raise RuntimeError("frozen 5k hidden evaluation base blob changed")
development_audit_raw = DEVELOPMENT_AUDIT_PATH.read_bytes()
if git_blob_sha(development_audit_raw) != EXPECTED_DEVELOPMENT_AUDIT_BLOB:
    raise RuntimeError("frozen post-5k development audit blob changed")

hidden_eval_source = hidden_eval_raw.decode("utf-8")
replacements = [
    (
        'EXPECTED_CACHE_MANIFEST_SHA256 = "380e553cea43c0bb4a649e7d6b696786b4e5178d45ee116efbd5e99cceeab6aa"',
        f'EXPECTED_CACHE_MANIFEST_SHA256 = "{FRESH_CACHE_MANIFEST_SHA256}"',
    ),
    (
        'KNOWN_LEGACY_CACHE_TASK_ID = "P1-03-lumia-hidden-state-cache-1000-v1"',
        'KNOWN_LEGACY_CACHE_TASK_ID = "P1-03-lumia-hidden-state-fresh-cache-5000-v1"',
    ),
    (
        'TASK_ID = "P1-03a-lumia-hidden-state-evaluation-5000-v1"',
        'TASK_ID = "P1-03-fresh-hidden-anchor-evaluation-5000-v1"',
    ),
    ("4512897483", str(FRESH_CACHE_TOTAL_SHARD_BYTES)),
    (
        '"known_cache_manifest_task_id_bug_observed": cache_manifest.get("task_id") == KNOWN_LEGACY_CACHE_TASK_ID',
        '"fresh_cache_task_id_observed": cache_manifest.get("task_id") == KNOWN_LEGACY_CACHE_TASK_ID',
    ),
]
for old, new in replacements:
    if hidden_eval_source.count(old) != 1:
        raise RuntimeError(f"hidden evaluation transform marker changed: {old[:80]}")
    hidden_eval_source = hidden_eval_source.replace(old, new)

for marker in (
    FRESH_CACHE_MANIFEST_SHA256,
    'TASK_ID = "P1-03-fresh-hidden-anchor-evaluation-5000-v1"',
    str(FRESH_CACHE_TOTAL_SHARD_BYTES),
    '"fresh_cache_task_id_observed"',
    "run_outer_oof_5000",
    "paired_increment_bootstraps",
):
    if marker not in hidden_eval_source:
        raise RuntimeError(f"fresh hidden evaluation marker missing: {marker}")
if "380e553cea43c0bb4a649e7d6b696786b4e5178d45ee116efbd5e99cceeab6aa" in hidden_eval_source:
    raise RuntimeError("consumed development cache identity survived fresh transform")

FRESH_AUDIT_SOURCE = FRESH_AUDIT_PATH.read_text(encoding="utf-8")
SCALE_CONFIG_TEXT = SCALE_CONFIG_PATH.read_text(encoding="utf-8")
DEVELOPMENT_AUDIT_SOURCE = development_audit_raw.decode("utf-8")

install_cell = r"""
from pathlib import Path
import os
import shutil
import subprocess
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SITE_ROOT = Path("/tmp/p1-03-fresh-confirm-v1-site")
SRC_ROOT = Path("/tmp/p1-03-fresh-confirm-v1-src")
HIDDEN_RUNNER = SRC_ROOT / "scripts/run_fresh_hidden_anchor_5000.py"
AUDIT_RUNNER = SRC_ROOT / "scripts/run_p1_03_fresh_5k_confirmation_audit_v1.py"
HIDDEN_OUTPUT = Path("/tmp/p1_03_fresh_hidden_anchor_eval_5000_v1")
OUTPUT = Path("/kaggle/working/p1_03_fresh_5k_confirmation_v1")
INPUT_ROOT = Path("/kaggle/input")

for path in (SITE_ROOT, SRC_ROOT, HIDDEN_OUTPUT):
    if path.exists():
        shutil.rmtree(path)
if OUTPUT.exists():
    raise RuntimeError("fresh confirmation output already exists")
SITE_ROOT.mkdir(parents=True)
SRC_ROOT.mkdir(parents=True)

packages = [
    "datasets==4.6.1",
    "numpy==2.4.2",
    "pandas==3.0.1",
    "pyarrow==23.0.1",
    "scikit-learn==1.8.0",
    "scipy==1.17.1",
    "joblib==1.5.3",
    "threadpoolctl==3.6.0",
    "huggingface-hub==0.30.0",
]
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q",
        "--disable-pip-version-check", "--target", str(SITE_ROOT), *packages,
    ],
    check=True,
)

probe = "\n".join([
    "import importlib.metadata",
    "import pathlib",
    "import numpy, pandas, pyarrow, scipy, sklearn, datasets, torch",
    "site = pathlib.Path(__import__('os').environ['P1_03_SITE_ROOT']).resolve()",
    "expected = {'numpy':'2.4.2','pandas':'3.0.1','pyarrow':'23.0.1','scipy':'1.17.1','scikit-learn':'1.8.0','datasets':'4.6.1','huggingface-hub':'0.30.0'}",
    "for dist, version in expected.items(): assert importlib.metadata.version(dist) == version, (dist, importlib.metadata.version(dist))",
    "for module in (numpy, pandas, pyarrow, scipy, sklearn, datasets): assert site in pathlib.Path(module.__file__).resolve().parents, (module.__name__, module.__file__)",
    "assert site not in pathlib.Path(torch.__file__).resolve().parents",
    "print('P1_03_FRESH_CONFIRM_IMPORT_GATE PASS')",
])
env = os.environ.copy()
env["PYTHONPATH"] = str(SITE_ROOT)
env["P1_03_SITE_ROOT"] = str(SITE_ROOT)
completed = subprocess.run(
    [sys.executable, "-c", probe],
    env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180,
)
print(completed.stdout)
if completed.returncode != 0:
    raise RuntimeError(f"fresh confirmation import gate failed rc={completed.returncode}")
sys.path.insert(0, str(SITE_ROOT))
"""

write_sources_cell = r"""
package_root = SRC_ROOT / "poisoned_chalice"
config_root = SRC_ROOT / "configs"
data_root = SRC_ROOT / "data/public"
scripts_root = SRC_ROOT / "scripts"
package_root.mkdir(parents=True, exist_ok=True)
config_root.mkdir(parents=True, exist_ok=True)
scripts_root.mkdir(parents=True, exist_ok=True)
(package_root / "__init__.py").write_text("", encoding="utf-8")
(package_root / "evaluation.py").write_text(EVALUATION_SOURCE_TEXT, encoding="utf-8")
(package_root / "sersem_author_faithful.py").write_text(FAITHFUL_SOURCE_TEXT, encoding="utf-8")
(package_root / "lumia_hidden_state_pilot.py").write_text(PILOT_SOURCE_TEXT, encoding="utf-8")
(package_root / "lumia_hidden_state_cache.py").write_text(CACHE_SOURCE_TEXT, encoding="utf-8")
(package_root / "lumia_hidden_state_probe.py").write_text(PROBE_SOURCE_TEXT, encoding="utf-8")
(package_root / "lumia_hidden_state_scale.py").write_text(SCALE_SOURCE_TEXT, encoding="utf-8")
HIDDEN_RUNNER.write_text(HIDDEN_EVAL_SOURCE_TEXT, encoding="utf-8")
(scripts_root / "run_p1_03_post_5k_development_audit_v1.py").write_text(
    DEVELOPMENT_AUDIT_SOURCE_TEXT, encoding="utf-8"
)
AUDIT_RUNNER.write_text(FRESH_AUDIT_SOURCE_TEXT, encoding="utf-8")
(config_root / "p1_03_lumia_hidden_state_scale_5000_v1_20260910.json").write_text(
    SCALE_CONFIG_TEXT_VALUE, encoding="utf-8"
)
"""

execute_cell = r"""
import hashlib
import json
import time

if not INPUT_ROOT.is_dir():
    raise RuntimeError("/kaggle/input unavailable")

cache_candidates = []
for candidate in sorted(INPUT_ROOT.rglob("cache_manifest.json")):
    try:
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    except OSError:
        continue
    if digest == FRESH_CACHE_MANIFEST_SHA256_VALUE:
        cache_candidates.append(candidate)
if len(cache_candidates) != 1:
    raise RuntimeError(
        f"fresh cache exact resolution failed: matching_manifests={len(cache_candidates)} expected=1"
    )
CACHE_ROOT = cache_candidates[0].parent
manifest = json.loads(cache_candidates[0].read_text(encoding="utf-8"))
if manifest.get("task_id") != "P1-03-lumia-hidden-state-fresh-cache-5000-v1":
    raise RuntimeError("fresh cache task identity changed")
if manifest.get("rows") != 5000 or len(manifest.get("shards") or []) != 20:
    raise RuntimeError("fresh cache row/shard contract changed")
if manifest.get("labels_present_during_model_scoring") is not False:
    raise RuntimeError("fresh cache label boundary changed")
selection = json.loads((CACHE_ROOT / "cohort_selection.json").read_text(encoding="utf-8"))
if selection.get("fresh_cohort_set_sha256") != FRESH_COHORT_SET_SHA256_VALUE:
    raise RuntimeError("fresh cohort set identity changed")
if selection.get("fresh_full_stage1_overlap") is not True:
    raise RuntimeError("fresh cohort lost full Stage1 overlap")
if any(int(selection.get(key, -1)) != 0 for key in (
    "fresh_current_hidden_overlap_rows", "fresh_prior_1k_overlap_rows", "fresh_prior_50_overlap_rows"
)):
    raise RuntimeError("fresh cohort disjointness changed")

stage1_paths = sorted(
    path for path in INPUT_ROOT.rglob("features.part*.parquet")
    if "/train_10k/parts/" in path.as_posix()
)
if len(stage1_paths) != 40:
    raise RuntimeError(f"Stage1 exact 10k shard set not found: {len(stage1_paths)}")

print(
    "P1_03_FRESH_CONFIRM_INPUT_GATE PASS "
    f"cache_sha256={FRESH_CACHE_MANIFEST_SHA256_VALUE} "
    f"cohort_sha256={FRESH_COHORT_SET_SHA256_VALUE} stage1_shards=40"
)

from datasets import load_dataset
LANGUAGES = ("Go", "Java", "Python", "Ruby", "Rust")
EXPECTED_ROWS = {"Go":10000, "Java":10000, "Python":10000, "Ruby":10000, "Rust":9040}
DATASET_ID = "Poisoned-Chalice/ICSE-2027-public"
DATASET_REVISION = "2ed5468723efa5457a3665782c6979ea4dbac7c2"
for language in LANGUAGES:
    frame = load_dataset(DATASET_ID, language, split="train", revision=DATASET_REVISION).to_pandas()
    if len(frame) != EXPECTED_ROWS[language]:
        raise RuntimeError(f"public train row count changed for {language}: {len(frame)}")
    if not {"sample_id", "membership"}.issubset(frame.columns):
        raise RuntimeError(f"public label columns missing for {language}")
    label_dir = data_root / language
    label_dir.mkdir(parents=True, exist_ok=True)
    frame[["sample_id", "membership"]].to_parquet(
        label_dir / "train-00000-of-00001.parquet", index=False
    )
    del frame

import torch
if not torch.cuda.is_available():
    raise RuntimeError("CUDA required for frozen nested hidden probe")
if torch.cuda.device_count() < 1 or "T4" not in torch.cuda.get_device_name(0):
    raise RuntimeError(
        f"expected T4 cuda:0, got count={torch.cuda.device_count()} "
        f"name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}"
    )

env = os.environ.copy()
env["PYTHONPATH"] = str(SRC_ROOT) + os.pathsep + str(SITE_ROOT)
env["OMP_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"

phase_a_started = time.perf_counter()
hidden_completed = subprocess.run(
    [
        sys.executable, str(HIDDEN_RUNNER),
        "--cache-dir", str(CACHE_ROOT),
        "--output-dir", str(HIDDEN_OUTPUT),
        "--device", "cuda:0",
    ],
    cwd=str(SRC_ROOT), env=env, text=True,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=6600,
)
print(hidden_completed.stdout)
if hidden_completed.returncode != 0:
    raise RuntimeError(f"fresh hidden-anchor evaluation failed rc={hidden_completed.returncode}")
hidden_manifest = json.loads((HIDDEN_OUTPUT / "run_manifest.json").read_text(encoding="utf-8"))
if hidden_manifest.get("status") != "complete":
    raise RuntimeError("fresh hidden-anchor evaluation incomplete")
if hidden_manifest.get("task_id") != "P1-03-fresh-hidden-anchor-evaluation-5000-v1":
    raise RuntimeError("fresh hidden-anchor evaluation identity changed")
if hidden_manifest.get("cache_manifest_sha256") != FRESH_CACHE_MANIFEST_SHA256_VALUE:
    raise RuntimeError("fresh hidden-anchor evaluated wrong cache")
if hidden_manifest.get("outer_oof_coverage") is not True:
    raise RuntimeError("fresh hidden-anchor OOF coverage incomplete")
print(
    "P1_03_FRESH_CONFIRM_HIDDEN_PHASE PASS "
    f"wall_seconds={time.perf_counter()-phase_a_started:.3f}"
)

phase_b_started = time.perf_counter()
audit_completed = subprocess.run(
    [
        sys.executable, str(AUDIT_RUNNER),
        "--hidden-eval-dir", str(HIDDEN_OUTPUT),
        "--cache-dir", str(CACHE_ROOT),
        "--stage1-root", str(INPUT_ROOT),
        "--output-dir", str(OUTPUT),
    ],
    cwd=str(SRC_ROOT), env=env, text=True,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=7200,
)
print(audit_completed.stdout)
if audit_completed.returncode != 0:
    raise RuntimeError(f"fresh confirmation audit failed rc={audit_completed.returncode}")

expected_outputs = sorted([
    "confirmation_decision.json",
    "content_control_confirmation.json",
    "content_control_scores.parquet",
    "fusion_confirmation.json",
    "fusion_scores.parquet",
    "hidden_decision.json",
    "hidden_fold_selection.json",
    "hidden_metrics.json",
    "hidden_oof_predictions.csv",
    "hidden_run_manifest.json",
    "portable_scalar_confirmation.json",
    "portable_scalar_scores.parquet",
    "portable_scalar_table.parquet",
    "run_manifest.json",
    "static_feature_table.parquet",
])
observed_outputs = sorted(path.name for path in OUTPUT.iterdir())
if observed_outputs != expected_outputs:
    raise RuntimeError(f"unexpected fresh confirmation outputs: {observed_outputs}")
final_manifest = json.loads((OUTPUT / "run_manifest.json").read_text(encoding="utf-8"))
if final_manifest.get("status") != "complete" or final_manifest.get("rows") != 5000:
    raise RuntimeError("fresh confirmation final manifest incomplete")
if final_manifest.get("cache_manifest_sha256") != FRESH_CACHE_MANIFEST_SHA256_VALUE:
    raise RuntimeError("fresh confirmation final cache identity changed")
if final_manifest.get("fusion_intersection_rows") != 5000:
    raise RuntimeError("fresh confirmation did not achieve full Stage1/hidden overlap")
if final_manifest.get("frozen_fusion_weights") != [0.75, 0.50]:
    raise RuntimeError("fresh confirmation fusion weights changed")

print(
    "P1_03_FRESH_5K_CONFIRMATION_V1 COMPLETE "
    f"rows=5000 fusion_intersection=5000 "
    f"phase_b_wall_seconds={time.perf_counter()-phase_b_started:.3f}"
)
"""

cleanup_cell = r"""
shutil.rmtree(SITE_ROOT, ignore_errors=True)
shutil.rmtree(SRC_ROOT, ignore_errors=True)
shutil.rmtree(HIDDEN_OUTPUT, ignore_errors=True)
print("P1_03_FRESH_CONFIRM_CLEANUP PASS raw_hidden_copied=0 submission=0")
"""

nb = nbformat.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata.language_info = {"name": "python", "version": "3.12"}
nb.cells = [
    nbformat.v4.new_markdown_cell(
        "# P1-03 fresh-disjoint 5k confirmation\n\n"
        "This Notebook consumes the completed fresh hidden cache in-place on Kaggle. "
        "It first applies the previously frozen nested 5-fold hidden probe, then evaluates "
        "only the two pre-frozen Stage1/hidden rank fusions (0.75 and 0.50), unchanged C0/C1 "
        "content controls with SimHash group diagnostic, and the frozen mean 96-feature "
        "portable scalar. No validation rows, Public-LB feedback, new target-model forward, "
        "or competition submission are used."
    ),
    nbformat.v4.new_code_cell(textwrap.dedent(install_cell)),
    nbformat.v4.new_code_cell(
        "EVALUATION_SOURCE_TEXT = " + repr(EVALUATION_SOURCE) + "\n"
        "FAITHFUL_SOURCE_TEXT = " + repr(FAITHFUL_SOURCE) + "\n"
        "PILOT_SOURCE_TEXT = " + repr(PILOT_SOURCE) + "\n"
        "CACHE_SOURCE_TEXT = " + repr(CACHE_SOURCE) + "\n"
        "PROBE_SOURCE_TEXT = " + repr(PROBE_SOURCE) + "\n"
        "SCALE_SOURCE_TEXT = " + repr(SCALE_SOURCE) + "\n"
        "HIDDEN_EVAL_SOURCE_TEXT = " + repr(hidden_eval_source) + "\n"
        "DEVELOPMENT_AUDIT_SOURCE_TEXT = " + repr(DEVELOPMENT_AUDIT_SOURCE) + "\n"
        "FRESH_AUDIT_SOURCE_TEXT = " + repr(FRESH_AUDIT_SOURCE) + "\n"
        "SCALE_CONFIG_TEXT_VALUE = " + repr(SCALE_CONFIG_TEXT) + "\n"
        "FRESH_CACHE_MANIFEST_SHA256_VALUE = " + repr(FRESH_CACHE_MANIFEST_SHA256) + "\n"
        "FRESH_COHORT_SET_SHA256_VALUE = " + repr(FRESH_COHORT_SET_SHA256) + "\n"
        + textwrap.dedent(write_sources_cell)
    ),
    nbformat.v4.new_code_cell(textwrap.dedent(execute_cell)),
    nbformat.v4.new_code_cell(textwrap.dedent(cleanup_cell)),
]
for index, cell in enumerate(nb.cells):
    cell["id"] = f"p1-03-fresh-confirm-v1-{index:02d}"

OUT_DIR.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, OUT)
metadata = {
    "id": TARGET,
    "title": TITLE,
    "code_file": NOTEBOOK_NAME,
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [STAGE1_DATASET],
    "kernel_sources": [CACHE_KERNEL],
    "competition_sources": [],
    "model_sources": [],
}
(OUT_DIR / "kernel-metadata.json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(OUT)
