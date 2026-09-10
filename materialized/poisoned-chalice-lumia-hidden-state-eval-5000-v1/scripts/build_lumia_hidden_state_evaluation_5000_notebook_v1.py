"""Build the frozen P1-03a 5,000-row hidden-state OOF evaluation Kaggle Notebook."""
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
RUN_SCRIPT_SOURCE = (ROOT / "scripts/run_lumia_hidden_state_scale_5000_evaluation.py").read_text(encoding="utf-8")
SCALE_CONFIG_TEXT = (ROOT / "configs/p1_03_lumia_hidden_state_scale_5000_v1_20260910.json").read_text(encoding="utf-8")

TARGET = "renta0426/poisoned-chalice-lumia-hidden-state-eval-5000-v1"
TITLE = "Poisoned Chalice Lumia Hidden State Eval 5000 V1"
CACHE_KERNEL = "renta0426/poisoned-chalice-lumia-hidden-state-cache-5000-v1"
NOTEBOOK_NAME = "poisoned-chalice-lumia-hidden-state-eval-5000-v1.ipynb"
OUT_DIR = ROOT / "notebooks/experiments/poisoned-chalice-lumia-hidden-state-eval-5000-v1"
OUT = OUT_DIR / NOTEBOOK_NAME
EXPECTED_CACHE_MANIFEST_SHA256 = "380e553cea43c0bb4a649e7d6b696786b4e5178d45ee116efbd5e99cceeab6aa"

install_cell = r'''
from pathlib import Path
import os
import shutil
import subprocess
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SITE_ROOT = Path("/tmp/lumia-eval-5000-v1-site")
SRC_ROOT = Path("/tmp/lumia-eval-5000-v1-src")
RUNNER = SRC_ROOT / "scripts/run_lumia_hidden_state_scale_5000_evaluation.py"
OUTPUT = Path("/kaggle/working/lumia_hidden_state_evaluation_5000_v1")
INPUT_ROOT = Path("/kaggle/input")
for path in (SITE_ROOT, SRC_ROOT):
    if path.exists():
        shutil.rmtree(path)
if OUTPUT.exists():
    raise RuntimeError("fresh 5k evaluation target must not already contain output")
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
    [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "--target", str(SITE_ROOT), *packages],
    check=True,
)
probe = r"""
import importlib.metadata
import pathlib
import numpy, pandas, pyarrow, scipy, sklearn, datasets, torch
site = pathlib.Path(__import__('os').environ['LUMIA_SITE_ROOT']).resolve()
expected = {
    'numpy':'2.4.2','pandas':'3.0.1','pyarrow':'23.0.1','scipy':'1.17.1',
    'scikit-learn':'1.8.0','datasets':'4.6.1','huggingface-hub':'0.30.0'
}
for dist, version in expected.items():
    assert importlib.metadata.version(dist) == version, (dist, importlib.metadata.version(dist))
for module in (numpy, pandas, pyarrow, scipy, sklearn, datasets):
    assert site in pathlib.Path(module.__file__).resolve().parents, (module.__name__, module.__file__)
assert site not in pathlib.Path(torch.__file__).resolve().parents
print('LUMIA_EVAL_5000_V1_IMPORT_GATE PASS')
"""
env = os.environ.copy()
env["PYTHONPATH"] = str(SITE_ROOT)
env["LUMIA_SITE_ROOT"] = str(SITE_ROOT)
completed = subprocess.run(
    [sys.executable, "-c", probe],
    env=env,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    timeout=180,
)
print(completed.stdout)
if completed.returncode != 0:
    raise RuntimeError(f"LUMIA_EVAL_5000_V1_IMPORT_GATE_FAILED rc={completed.returncode}")
sys.path.insert(0, str(SITE_ROOT))
'''

write_sources_cell = r'''
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
RUNNER.write_text(RUN_SCRIPT_SOURCE_TEXT, encoding="utf-8")
(config_root / "p1_03_lumia_hidden_state_scale_5000_v1_20260910.json").write_text(SCALE_CONFIG_TEXT_VALUE, encoding="utf-8")
'''

execute_cell = r'''
import hashlib
import json
import time

if not INPUT_ROOT.is_dir():
    raise RuntimeError("/kaggle/input is unavailable")

# Never assume Kaggle's mounted directory name for a Notebook-output input.
# Resolve the sole attached 5k cache by its user-shared exact manifest SHA-256.
cache_candidates = []
for candidate_manifest in sorted(INPUT_ROOT.rglob("cache_manifest.json")):
    try:
        candidate_sha256 = hashlib.sha256(candidate_manifest.read_bytes()).hexdigest()
    except OSError:
        continue
    if candidate_sha256 == EXPECTED_CACHE_MANIFEST_SHA256_VALUE:
        cache_candidates.append(candidate_manifest)
if len(cache_candidates) != 1:
    raise RuntimeError(
        "exact 5k sealed cache input resolution failed: "
        f"matching_manifests={len(cache_candidates)} expected=1"
    )
observed_manifest = cache_candidates[0]
CACHE_ROOT = observed_manifest.parent
observed_manifest_sha256 = hashlib.sha256(observed_manifest.read_bytes()).hexdigest()
manifest = json.loads(observed_manifest.read_text(encoding="utf-8"))
config = manifest.get("config") or {}
if manifest.get("rows") != 5000 or config.get("rows") != 5000:
    raise RuntimeError("mounted 5k cache row contract changed")
if config.get("rows_per_language_label") != 500 or config.get("shard_size") != 250:
    raise RuntimeError("mounted 5k cache cohort/shard contract changed")
if len(manifest.get("shards") or []) != 20:
    raise RuntimeError("mounted 5k cache shard count changed")
if manifest.get("num_layers") != 30 or manifest.get("hidden_size") != 3072:
    raise RuntimeError("mounted 5k cache hidden dimensions changed")
if manifest.get("labels_present_during_model_scoring") is not False or manifest.get("performance_metrics_computed") is not False:
    raise RuntimeError("mounted 5k cache clean-room boundary changed")
# Known metadata-only defect from the frozen cache core.  Accept no other value.
if manifest.get("task_id") != "P1-03-lumia-hidden-state-cache-1000-v1":
    raise RuntimeError("known 5k cache task_id compatibility marker changed")
expected_shards = [CACHE_ROOT / f"shards/shard_{index:04d}.npz" for index in range(20)]
if not all(path.is_file() for path in expected_shards):
    raise RuntimeError("exact 5k sealed cache input is missing one or more declared shards")
print(
    "LUMIA_EVAL_5000_V1_CACHE_INPUT_RESOLVED PASS "
    f"manifest_sha256={observed_manifest_sha256} shards=20 known_task_id_bug=1"
)

# Reconstruct only public training labels from the same pinned public HF revision.
from datasets import load_dataset
import pandas as pd
LANGUAGES = ("Go", "Java", "Python", "Ruby", "Rust")
EXPECTED_ROWS = {"Go":10000,"Java":10000,"Python":10000,"Ruby":10000,"Rust":9040}
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
    frame[["sample_id", "membership"]].to_parquet(label_dir / "train-00000-of-00001.parquet", index=False)
    del frame

import torch
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for the frozen 5k probe evaluation")
if torch.cuda.device_count() < 1 or "T4" not in torch.cuda.get_device_name(0):
    raise RuntimeError(f"expected T4 on cuda:0, got count={torch.cuda.device_count()} name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

env = os.environ.copy()
env["PYTHONPATH"] = str(SRC_ROOT) + os.pathsep + str(SITE_ROOT)
env["OMP_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"
started = time.perf_counter()
completed = subprocess.run(
    [sys.executable, str(RUNNER), "--cache-dir", str(CACHE_ROOT), "--output-dir", str(OUTPUT), "--device", "cuda:0"],
    cwd=str(SRC_ROOT),
    env=env,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    timeout=6600,
)
print(completed.stdout)
if completed.returncode != 0:
    raise RuntimeError(f"LUMIA_EVAL_5000_V1_CHILD_FAILED rc={completed.returncode}")
expected_outputs = ["decision.json", "fold_selection.json", "metrics.json", "oof_predictions.csv", "run_manifest.json"]
observed_outputs = sorted(path.name for path in OUTPUT.iterdir())
if observed_outputs != expected_outputs:
    raise RuntimeError(f"unexpected persistent output set: {observed_outputs}")
run_manifest = json.loads((OUTPUT / "run_manifest.json").read_text(encoding="utf-8"))
if run_manifest.get("status") != "complete" or run_manifest.get("rows") != 5000 or run_manifest.get("outer_oof_coverage") is not True:
    raise RuntimeError("frozen 5k OOF evaluation did not complete with full coverage")
if run_manifest.get("cache_manifest_sha256") != EXPECTED_CACHE_MANIFEST_SHA256_VALUE:
    raise RuntimeError("5k OOF output did not evaluate the exact approved cache")
print(
    "LUMIA_HIDDEN_STATE_EVAL_5000_V1 COMPLETE "
    f"rows=5000 cache_sha256={observed_manifest_sha256} wall_seconds={time.perf_counter()-started:.3f}"
)
'''

nb = nbformat.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata.language_info = {"name": "python", "version": "3.12"}
nb.cells = [
    nbformat.v4.new_markdown_cell(
        "# P1-03a LUMIA hidden-state — frozen 5,000-row OOF scale/stability evaluation\n\n"
        "This Notebook directly attaches the completed 5k cache Notebook as its sole Kaggle input. "
        "It validates the exact cache by manifest SHA-256, verifies every declared shard before loading labels, "
        "then joins only pinned public-train membership and executes the predeclared 5-fold OOF scale protocol. "
        "No validation rows, Public-LB feedback, model-specific holdout, or competition submission are used."
    ),
    nbformat.v4.new_code_cell(textwrap.dedent(install_cell)),
    nbformat.v4.new_code_cell(
        "EVALUATION_SOURCE_TEXT = " + repr(EVALUATION_SOURCE) + "\n"
        "FAITHFUL_SOURCE_TEXT = " + repr(FAITHFUL_SOURCE) + "\n"
        "PILOT_SOURCE_TEXT = " + repr(PILOT_SOURCE) + "\n"
        "CACHE_SOURCE_TEXT = " + repr(CACHE_SOURCE) + "\n"
        "PROBE_SOURCE_TEXT = " + repr(PROBE_SOURCE) + "\n"
        "SCALE_SOURCE_TEXT = " + repr(SCALE_SOURCE) + "\n"
        "RUN_SCRIPT_SOURCE_TEXT = " + repr(RUN_SCRIPT_SOURCE) + "\n"
        "SCALE_CONFIG_TEXT_VALUE = " + repr(SCALE_CONFIG_TEXT) + "\n"
        "EXPECTED_CACHE_MANIFEST_SHA256_VALUE = " + repr(EXPECTED_CACHE_MANIFEST_SHA256) + "\n"
        + textwrap.dedent(write_sources_cell)
    ),
    nbformat.v4.new_code_cell(textwrap.dedent(execute_cell)),
]
for index, cell in enumerate(nb.cells):
    cell["id"] = f"lumia-eval-5000-v1-{index:02d}"

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
    "dataset_sources": [],
    "kernel_sources": [CACHE_KERNEL],
    "competition_sources": [],
    "model_sources": [],
}
(OUT_DIR / "kernel-metadata.json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(OUT)
