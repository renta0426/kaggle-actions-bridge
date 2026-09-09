"""Build the repaired private Kaggle T4 runtime/fidelity gate for hidden-state LUMIA."""
from pathlib import Path
import json
import textwrap

import nbformat

ROOT = Path(__file__).resolve().parents[1]
AUTHOR_HELPERS = (ROOT / "src/poisoned_chalice/sersem_author_faithful.py").read_text(encoding="utf-8")
AUTHOR_RUNTIME = (ROOT / "src/poisoned_chalice/sersem_author_runtime.py").read_text(encoding="utf-8")
PILOT_SOURCE = (ROOT / "src/poisoned_chalice/lumia_hidden_state_pilot.py").read_text(encoding="utf-8")

# The Notebook is deliberately self-contained. Remove package-relative imports
# because the corresponding exact source snapshots are injected as earlier cells.
AUTHOR_RUNTIME = AUTHOR_RUNTIME.replace(
    "from poisoned_chalice.sersem_author_faithful import parse_flake8_errors\n",
    "",
)
PILOT_SOURCE = PILOT_SOURCE.replace(
    "from poisoned_chalice.sersem_author_faithful import (\n"
    "    build_author_character_weights,\n"
    "    sigmoid_z,\n"
    "    token_weights_from_character_weights,\n"
    "    weighted_output_score,\n"
    ")\n",
    "",
)

TARGET = "renta0426/poisoned-chalice-lumia-hs-runtime-50-v2"
TITLE = "Poisoned Chalice Lumia Hs Runtime 50 V2"
NOTEBOOK_NAME = "poisoned-chalice-lumia-hs-runtime-50-v2.ipynb"
OUT_DIR = ROOT / "notebooks/experiments/poisoned-chalice-lumia-hs-runtime-50-v2"
OUT = OUT_DIR / NOTEBOOK_NAME

install_cell = r'''
import importlib.metadata
import os
import subprocess
import sys
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
subprocess.run(["apt-get", "update", "-qq"], check=True)
subprocess.run(
    ["apt-get", "install", "-y", "-qq", "libenchant-2-2", "hunspell-en-us"],
    check=True,
    stdout=subprocess.DEVNULL,
)
packages = [
    "datasets==4.6.1",
    "accelerate==1.13.0",
    "numpy==2.4.2",
    "pandas==3.0.1",
    "pyarrow==23.0.1",
    "scikit-learn==1.8.0",
    "scipy==1.17.1",
    "joblib==1.5.3",
    "threadpoolctl==3.6.0",
    "transformers==4.52.0",
    "tokenizers==0.21.0",
    "huggingface-hub==0.30.0",
    "tree-sitter==0.25.2",
    "tree-sitter-go==0.25.0",
    "tree-sitter-java==0.23.5",
    "tree-sitter-python==0.25.0",
    "tree-sitter-ruby==0.23.1",
    "tree-sitter-rust==0.24.0",
    "pyenchant==3.3.0",
    "flake8==7.3.0",
]
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", *packages],
    check=True,
)
expected_runtime = {
    "numpy": "2.4.2",
    "pandas": "3.0.1",
    "scikit-learn": "1.8.0",
    "scipy": "1.17.1",
    "joblib": "1.5.3",
    "threadpoolctl": "3.6.0",
    "transformers": "4.52.0",
    "tokenizers": "0.21.0",
    "huggingface-hub": "0.30.0",
}
for distribution, expected in expected_runtime.items():
    observed = importlib.metadata.version(distribution)
    assert observed == expected, (distribution, observed, expected)
# transformers 4.52 imports sklearn.metrics from generation.candidate_generator
# whenever scikit-learn is present. Verify the repaired ABI/runtime before data/model work.
from sklearn.metrics import roc_curve  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
print("LUMIA_RUNTIME_V2_INSTALL PASS transformers_model_import=1 sklearn_import=1")
'''

setup_cell = r'''
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import time

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

CONFIG = LumiaRuntimePilotConfig()
OUTPUT = Path(CONFIG.output_dir)
OUTPUT.mkdir(parents=True, exist_ok=True)

assert torch.cuda.is_available(), "CUDA is required"
assert torch.cuda.device_count() == 2, f"expected Kaggle T4 x2 visibility, got {torch.cuda.device_count()}"
gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
assert all("T4" in name for name in gpu_names), gpu_names
assert shutil.which("flake8"), "flake8 CLI missing from Notebook PATH"
assert importlib.metadata.version("flake8") == "7.3.0"
runtime = load_author_runtime(require_exact_versions=True, require_full_components=True)
assert runtime.ast_available and runtime.enchant_available and runtime.flake8_available
print({"gpus": gpu_names, "author_runtime": runtime.installed_versions})
'''

load_data_cell = r'''
parts = []
for language, expected_rows in EXPECTED_TRAIN_ROWS.items():
    frame = load_dataset(
        CONFIG.dataset_id,
        language,
        split="train",
        revision=CONFIG.dataset_revision,
    ).to_pandas()
    assert len(frame) == expected_rows
    frame["language"] = language
    parts.append(frame[["sample_id", "content", "membership", "language"]])
train = pd.concat(parts, ignore_index=True)
validate_public_train(train)
cohort = select_runtime_pilot(train, CONFIG)
assert list(cohort.columns) == ["sample_id", "language", "content"]
cohort_set_hash = hashlib.sha256(
    "\n".join(sorted(cohort.sample_id.astype(str))).encode("utf-8")
).hexdigest()
del train, parts
print({"runtime_rows": len(cohort), "cohort_set_sha256": cohort_set_hash, "labels_in_scoring_frame": False})
'''

model_cell = r'''
torch.manual_seed(CONFIG.seed)
np.random.seed(CONFIG.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG.seed)

tokenizer = AutoTokenizer.from_pretrained(
    CONFIG.model_id,
    revision=CONFIG.model_revision,
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    CONFIG.model_id,
    revision=CONFIG.model_revision,
    device_map="cuda",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).eval()
assert next(model.parameters()).device.type == "cuda"
layer_count = len(_layer_modules(model))
expected_layers = int(getattr(model.config, "num_hidden_layers", layer_count))
assert layer_count == expected_layers and layer_count > 0
attention_impl = getattr(model.config, "_attn_implementation", None)
print({
    "device": str(next(model.parameters()).device),
    "layers": layer_count,
    "hidden_size": int(model.config.hidden_size),
    "max_length": CONFIG.max_length,
    "attention_impl": attention_impl,
})
'''

run_cell = r'''
run_started = time.perf_counter()
summary = run_runtime_pilot(model, tokenizer, cohort, runtime, CONFIG)
summary["cohort_set_sha256"] = cohort_set_hash
summary["runtime"] = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "transformers": importlib.metadata.version("transformers"),
    "tokenizers": importlib.metadata.version("tokenizers"),
    "huggingface_hub": importlib.metadata.version("huggingface-hub"),
    "accelerate": importlib.metadata.version("accelerate"),
    "numpy": importlib.metadata.version("numpy"),
    "pandas": importlib.metadata.version("pandas"),
    "scikit_learn": importlib.metadata.version("scikit-learn"),
    "scipy": importlib.metadata.version("scipy"),
    "joblib": importlib.metadata.version("joblib"),
    "threadpoolctl": importlib.metadata.version("threadpoolctl"),
    "attention_impl": attention_impl,
    "visible_gpu_count": torch.cuda.device_count(),
    "gpu_names": gpu_names,
}
summary["total_cell_wall_seconds"] = time.perf_counter() - run_started
write_json(OUTPUT / "runtime.json", summary)
manifest = {
    "schema_version": 1,
    "status": "complete",
    "task_id": summary["task_id"],
    "operational_attempt": "v2_dependency_repair",
    "model_id": CONFIG.model_id,
    "model_revision": CONFIG.model_revision,
    "dataset_id": CONFIG.dataset_id,
    "dataset_revision": CONFIG.dataset_revision,
    "author_commit": AUTHOR_COMMIT,
    "rows": CONFIG.rows,
    "max_length": CONFIG.max_length,
    "labels_present_during_model_scoring": False,
    "performance_metrics_computed": False,
    "persistent_sample_level_outputs": False,
    "persistent_outputs": ["runtime.json", "run_manifest.json"],
    "runtime_recommendation": summary["runtime_recommendation"],
    "cohort_set_sha256": cohort_set_hash,
}
write_json(OUTPUT / "run_manifest.json", manifest)
assert sorted(path.name for path in OUTPUT.iterdir() if path.is_file()) == ["run_manifest.json", "runtime.json"]
print("LUMIA_HIDDEN_STATE_RUNTIME_PILOT_V2 COMPLETE performance_metrics=0 sample_outputs=0")
print(json.dumps({
    "timing": summary["timing"],
    "token_count": summary["token_count"],
    "num_layers": summary["num_layers"],
    "hidden_size": summary["hidden_size"],
    "diagnostics": summary["diagnostics"],
    "runtime_recommendation": summary["runtime_recommendation"],
}, indent=2, sort_keys=True))
'''

cleanup_cell = r'''
del model, tokenizer, cohort
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("LUMIA_RUNTIME_CLEANUP PASS")
'''

nb = nbformat.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata.language_info = {"name": "python", "version": "3.12"}
nb.cells = [
    nbformat.v4.new_markdown_cell(
        "# LUMIA hidden-state runtime/fidelity gate v2 — 50 public-train rows\n\n"
        "Operational dependency repair of the failed v1 attempt. Scientific inputs, 8192-token scope, "
        "pooling definitions, and the 105-minute scale gate are unchanged. No AUC/TPR/pAUC or submission is computed."
    ),
    nbformat.v4.new_code_cell(textwrap.dedent(install_cell)),
    nbformat.v4.new_code_cell(AUTHOR_HELPERS),
    nbformat.v4.new_code_cell(AUTHOR_RUNTIME),
    nbformat.v4.new_code_cell(PILOT_SOURCE),
    nbformat.v4.new_code_cell(textwrap.dedent(setup_cell)),
    nbformat.v4.new_code_cell(textwrap.dedent(load_data_cell)),
    nbformat.v4.new_code_cell(textwrap.dedent(model_cell)),
    nbformat.v4.new_code_cell(textwrap.dedent(run_cell)),
    nbformat.v4.new_code_cell(textwrap.dedent(cleanup_cell)),
]
for index, cell in enumerate(nb.cells):
    cell["id"] = f"lumia-runtime-v2-{index:02d}"

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
    "keywords": ["gpu", "membership-inference", "lumia", "runtime-pilot", "dependency-repair"],
    "dataset_sources": [],
    "kernel_sources": [],
    "competition_sources": [],
    "model_sources": [],
}
(OUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
print(OUT)
