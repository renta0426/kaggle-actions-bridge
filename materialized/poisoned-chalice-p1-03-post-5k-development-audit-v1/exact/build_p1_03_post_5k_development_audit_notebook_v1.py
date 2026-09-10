"""Build the frozen CPU-only P1-03 post-5k development audit Kaggle Notebook."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import textwrap

import nbformat

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_p1_03_post_5k_development_audit_v1.py"
CONFIG_PATH = ROOT / "configs/p1_03_post_5k_development_audit_v1_20260910.json"
RUNNER_SOURCE = RUNNER_PATH.read_text(encoding="utf-8")
CONFIG_TEXT = CONFIG_PATH.read_text(encoding="utf-8")
CONFIG = json.loads(CONFIG_TEXT)

TARGET = "renta0426/poisoned-chalice-p1-03-post-5k-development-audit-v1"
TITLE = "Poisoned Chalice P1-03 Post-5K Development Audit V1"
NOTEBOOK_NAME = "poisoned-chalice-p1-03-post-5k-development-audit-v1.ipynb"
OUT_DIR = ROOT / "notebooks/experiments/poisoned-chalice-p1-03-post-5k-development-audit-v1"
OUT = OUT_DIR / NOTEBOOK_NAME
OUTPUT_DIR = "/kaggle/working/p1_03_post_5k_development_audit_v1"
EXPECTED_CONFIG_SHA256 = hashlib.sha256(CONFIG_TEXT.encode("utf-8")).hexdigest()
EXPECTED_RUNNER_BLOB_SHA = hashlib.sha1(
    b"blob " + str(len(RUNNER_SOURCE.encode("utf-8"))).encode("ascii") + b"\0" + RUNNER_SOURCE.encode("utf-8")
).hexdigest()

assert CONFIG["task_id"] == "P1-03-post-5k-development-audit-v1"
assert CONFIG["development_only"] is True
assert CONFIG["promotion_allowed"] is False
assert CONFIG["resource"]["accelerator"] == "cpu"
assert CONFIG["resource"]["enable_gpu"] is False
assert CONFIG["resource"]["enable_tpu"] is False
assert CONFIG["resource"]["automatic_compute_retries"] == 0
assert CONFIG["inputs"]["stage1_feature_dataset"]["version"] == 1
assert CONFIG["inputs"]["hidden_evaluation_kernel"]["expected_version"] == 1
assert CONFIG["inputs"]["hidden_cache_kernel"]["expected_version"] == 1
assert CONFIG["priority_1_fusion"]["rank_fusion_hidden_weights"] == [0.25, 0.5, 0.75]
assert CONFIG["priority_3_portable_scalar"]["features_per_pooling_variant"] == 96

install_cell = r'''
from pathlib import Path
import os
import shutil
import subprocess
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SITE_ROOT = Path("/tmp/p1-03-post-5k-audit-site")
SRC_ROOT = Path("/tmp/p1-03-post-5k-audit-src")
RUNNER = SRC_ROOT / "run_p1_03_post_5k_development_audit_v1.py"
CONFIG = SRC_ROOT / "p1_03_post_5k_development_audit_v1_20260910.json"
INPUT_ROOT = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working/p1_03_post_5k_development_audit_v1")
for path in (SITE_ROOT, SRC_ROOT):
    if path.exists():
        shutil.rmtree(path)
if OUTPUT.exists():
    raise RuntimeError("development audit output path already exists")
SITE_ROOT.mkdir(parents=True)
SRC_ROOT.mkdir(parents=True)

packages = [
    "datasets==4.6.1",
    "numpy==2.4.2",
    "pandas==3.0.1",
    "pyarrow==23.0.1",
    "scikit-learn==1.8.0",
    "scipy==1.17.1",
    "huggingface-hub==0.30.0",
]
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "--target", str(SITE_ROOT), *packages],
    check=True,
)
probe = r"""
import importlib.metadata
import pathlib
import numpy, pandas, pyarrow, scipy, sklearn, datasets
site = pathlib.Path(__import__('os').environ['AUDIT_SITE_ROOT']).resolve()
expected = {
    'numpy':'2.4.2', 'pandas':'3.0.1', 'pyarrow':'23.0.1',
    'scipy':'1.17.1', 'scikit-learn':'1.8.0', 'datasets':'4.6.1',
    'huggingface-hub':'0.30.0'
}
for dist, version in expected.items():
    assert importlib.metadata.version(dist) == version, (dist, importlib.metadata.version(dist))
for module in (numpy, pandas, pyarrow, scipy, sklearn, datasets):
    assert site in pathlib.Path(module.__file__).resolve().parents, (module.__name__, module.__file__)
print('P1_03_POST_5K_AUDIT_IMPORT_GATE PASS')
"""
env = os.environ.copy()
env["PYTHONPATH"] = str(SITE_ROOT)
env["AUDIT_SITE_ROOT"] = str(SITE_ROOT)
completed = subprocess.run(
    [sys.executable, "-c", probe], env=env, text=True,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240,
)
print(completed.stdout)
if completed.returncode != 0:
    raise RuntimeError(f"dependency import gate failed rc={completed.returncode}")
sys.path.insert(0, str(SITE_ROOT))
'''

write_cell = r'''
import hashlib
import json

RUNNER.write_text(RUNNER_SOURCE_TEXT, encoding="utf-8")
CONFIG.write_text(CONFIG_TEXT_VALUE, encoding="utf-8")
if hashlib.sha256(CONFIG.read_bytes()).hexdigest() != EXPECTED_CONFIG_SHA256_VALUE:
    raise RuntimeError("frozen development config SHA mismatch")
config = json.loads(CONFIG.read_text(encoding="utf-8"))
if config.get("development_only") is not True or config.get("promotion_allowed") is not False:
    raise RuntimeError("development-only boundary changed")
if config.get("resource", {}).get("accelerator") != "cpu":
    raise RuntimeError("development audit accelerator changed")
if config.get("resource", {}).get("automatic_compute_retries") != 0:
    raise RuntimeError("automatic retry contract changed")
print(
    "P1_03_POST_5K_AUDIT_SOURCE_GATE PASS "
    f"config_sha256={EXPECTED_CONFIG_SHA256_VALUE} runner_blob={EXPECTED_RUNNER_BLOB_VALUE}"
)
'''

execute_cell = r'''
import hashlib
import json
import os
import subprocess
import sys
import time

if not INPUT_ROOT.is_dir():
    raise RuntimeError("/kaggle/input is unavailable")

# Exact row-level hidden OOF: current-version attachment is accepted only when
# its immutable file digest matches the completed 5k result.
hidden_matches = []
for path in INPUT_ROOT.rglob("oof_predictions.csv"):
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        continue
    if digest == "c6d35c24087eac4809678bb5cae3d02fef630d07eb6f4033c50b8e285ce56fb4":
        hidden_matches.append(path)
if len(hidden_matches) != 1:
    raise RuntimeError(f"exact hidden OOF input resolution failed: matches={len(hidden_matches)}")

# Exact sealed raw hidden cache manifest. The runner validates every declared shard.
cache_matches = []
for path in INPUT_ROOT.rglob("cache_manifest.json"):
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        continue
    if digest == "380e553cea43c0bb4a649e7d6b696786b4e5178d45ee116efbd5e99cceeab6aa":
        cache_matches.append(path)
if len(cache_matches) != 1:
    raise RuntimeError(f"exact hidden cache input resolution failed: matches={len(cache_matches)}")

# Stage1 source is the frozen private Dataset. Its exact version is checked by the
# bridge before write; here we additionally require the immutable 40x250 train layout.
stage1_shards = sorted(INPUT_ROOT.rglob("train_10k/parts/features.part*.parquet"))
if len(stage1_shards) != 40:
    raise RuntimeError(f"Stage1 10k source resolution failed: shards={len(stage1_shards)}")

print(
    "P1_03_POST_5K_AUDIT_INPUT_GATE PASS "
    f"hidden_oof={hidden_matches[0].name} cache_manifest={cache_matches[0].name} stage1_shards=40"
)

env = os.environ.copy()
env["PYTHONPATH"] = str(SITE_ROOT)
env["OMP_NUM_THREADS"] = "2"
env["MKL_NUM_THREADS"] = "2"
started = time.perf_counter()
completed = subprocess.run(
    [sys.executable, str(RUNNER), "--input-root", str(INPUT_ROOT), "--output-dir", str(OUTPUT)],
    env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=6900,
)
print(completed.stdout)
if completed.returncode != 0:
    raise RuntimeError(f"P1_03_POST_5K_AUDIT_CHILD_FAILED rc={completed.returncode}")

expected_outputs = [
    "content_control_audit.json",
    "content_control_oof_scores.parquet",
    "fusion_audit.json",
    "fusion_common_scores.parquet",
    "portable_scalar_audit.json",
    "portable_scalar_oof_scores.parquet",
    "portable_scalar_table.parquet",
    "run_manifest.json",
    "static_feature_table.parquet",
]
observed = sorted(path.name for path in OUTPUT.iterdir())
if observed != expected_outputs:
    raise RuntimeError(f"unexpected persistent output set: {observed}")
manifest_path = OUTPUT / "run_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("status") != "complete" or manifest.get("development_only") is not True:
    raise RuntimeError("development audit completion boundary failed")
if manifest.get("competition_submission") is not False or manifest.get("no_new_target_model_forward") is not True:
    raise RuntimeError("development audit clean-room boundary failed")
manifest["frozen_config_sha256"] = EXPECTED_CONFIG_SHA256_VALUE
manifest["runner_git_blob_sha"] = EXPECTED_RUNNER_BLOB_VALUE
manifest["notebook_wall_seconds"] = time.perf_counter() - started
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(
    "P1_03_POST_5K_DEVELOPMENT_AUDIT COMPLETE "
    f"rows={manifest.get('content_control_rows')} intersection={manifest.get('fusion_intersection_rows')} "
    f"config_sha256={EXPECTED_CONFIG_SHA256_VALUE}"
)
'''

nb = nbformat.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata.language_info = {"name": "python", "version": "3.12"}
nb.cells = [
    nbformat.v4.new_markdown_cell(
        "# P1-03 post-5k development audit\n\n"
        "CPU-only analysis of three already-completed inputs: frozen Stage1 features, the exact 5k hidden OOF, "
        "and the exact sealed 5k hidden cache. It performs no target-model forward pass and no competition submission. "
        "All fusion and portable-scalar findings on this already-seen 5k cohort are development-only."
    ),
    nbformat.v4.new_code_cell(textwrap.dedent(install_cell)),
    nbformat.v4.new_code_cell(
        "RUNNER_SOURCE_TEXT = " + repr(RUNNER_SOURCE) + "\n"
        "CONFIG_TEXT_VALUE = " + repr(CONFIG_TEXT) + "\n"
        "EXPECTED_CONFIG_SHA256_VALUE = " + repr(EXPECTED_CONFIG_SHA256) + "\n"
        "EXPECTED_RUNNER_BLOB_VALUE = " + repr(EXPECTED_RUNNER_BLOB_SHA) + "\n"
        + textwrap.dedent(write_cell)
    ),
    nbformat.v4.new_code_cell(textwrap.dedent(execute_cell)),
]
for index, cell in enumerate(nb.cells):
    cell["id"] = f"p1-03-post-5k-audit-{index:02d}"

OUT_DIR.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, OUT)
metadata = {
    "id": TARGET,
    "title": TITLE,
    "code_file": NOTEBOOK_NAME,
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_tpu": False,
    "enable_internet": True,
    "dataset_sources": ["renta0426/stage1-raw-fim-submission-v1-output"],
    "kernel_sources": [
        "renta0426/poisoned-chalice-lumia-hidden-state-cache-5000-v1",
        "renta0426/poisoned-chalice-lumia-hidden-state-eval-5000-v1"
    ],
    "competition_sources": [],
    "model_sources": [],
}
(OUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(OUT)
