"""Build the frozen P1-03 1,000-row LUMIA hidden-state cache Kaggle notebook."""
from pathlib import Path
import textwrap

import nbformat

ROOT = Path(__file__).resolve().parents[1]
AUTHOR_HELPERS = (ROOT / "src/poisoned_chalice/sersem_author_faithful.py").read_text(encoding="utf-8")
AUTHOR_RUNTIME = (ROOT / "src/poisoned_chalice/sersem_author_runtime.py").read_text(encoding="utf-8")
PILOT_SOURCE = (ROOT / "src/poisoned_chalice/lumia_hidden_state_pilot.py").read_text(encoding="utf-8")
CACHE_SOURCE = (ROOT / "src/poisoned_chalice/lumia_hidden_state_cache.py").read_text(encoding="utf-8")

TARGET = "renta0426/poisoned-chalice-lumia-hidden-state-cache-1000-v1"
TITLE = "Poisoned Chalice Lumia Hidden State Cache 1000 V1"
NOTEBOOK_NAME = "poisoned-chalice-lumia-hidden-state-cache-1000-v1.ipynb"
OUT_DIR = ROOT / "notebooks/experiments/poisoned-chalice-lumia-hidden-state-cache-1000-v1"
OUT = OUT_DIR / NOTEBOOK_NAME

install_cell = r'''
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

BOOTSTRAP_STARTED = time.perf_counter()
SITE_ROOT = Path("/tmp/lumia-cache-1000-v1-site")
BIN_ROOT = Path("/tmp/lumia-cache-1000-v1-bin")
SRC_ROOT = Path("/tmp/lumia-cache-1000-v1-src")
RUNNER = Path("/tmp/lumia-cache-1000-v1-runner.py")
BOOTSTRAP_JSON = Path("/tmp/lumia-cache-1000-v1-bootstrap.json")

for path in (SITE_ROOT, BIN_ROOT, SRC_ROOT):
    if path.exists():
        shutil.rmtree(path)
for path in (RUNNER, BOOTSTRAP_JSON):
    path.unlink(missing_ok=True)
SITE_ROOT.mkdir(parents=True)
BIN_ROOT.mkdir(parents=True)

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
    "pycodestyle==2.14.0",
    "pyflakes==3.4.0",
    "mccabe==0.7.0",
]
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q",
        "--disable-pip-version-check", "--no-deps", "--force-reinstall",
        "--target", str(SITE_ROOT), *packages,
    ],
    check=True,
)
flake8_wrapper = BIN_ROOT / "flake8"
flake8_wrapper.write_text(
    "#!/bin/sh\nexec " + sys.executable + " -m flake8 \"$@\"\n",
    encoding="utf-8",
)
flake8_wrapper.chmod(0o755)

probe = r"""
import importlib.metadata
import json
import os
import pathlib
import numpy as np
import pandas as pd
import pyarrow
import scipy
import sklearn
import sklearn.metrics
import tokenizers
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
site = pathlib.Path(os.environ["LUMIA_SITE_ROOT"]).resolve()
expected = {
    "numpy": "2.4.2", "pandas": "3.0.1", "pyarrow": "23.0.1",
    "scikit-learn": "1.8.0", "scipy": "1.17.1", "joblib": "1.5.3",
    "threadpoolctl": "3.6.0", "transformers": "4.52.0",
    "tokenizers": "0.21.0", "huggingface-hub": "0.30.0",
}
for distribution, version in expected.items():
    assert importlib.metadata.version(distribution) == version, distribution
for module in (np, pd, pyarrow, scipy, sklearn, transformers, tokenizers):
    assert site in pathlib.Path(module.__file__).resolve().parents, (module.__name__, module.__file__)
assert site not in pathlib.Path(torch.__file__).resolve().parents
assert torch.tensor([1.0, 2.0], device="cpu").numpy().shape == (2,)
print(json.dumps({
    "fresh_child_import_gate": True,
    "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pyarrow.__version__,
    "scipy": scipy.__version__, "sklearn": sklearn.__version__,
    "torch": torch.__version__, "transformers": transformers.__version__,
}, sort_keys=True))
"""
probe_env = os.environ.copy()
probe_env["PYTHONPATH"] = str(SITE_ROOT)
probe_env["PATH"] = str(BIN_ROOT) + os.pathsep + probe_env.get("PATH", "")
probe_env["LUMIA_SITE_ROOT"] = str(SITE_ROOT)
completed = subprocess.run(
    [sys.executable, "-c", probe],
    check=False,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    env=probe_env,
    timeout=180,
)
print(completed.stdout)
if completed.returncode != 0:
    raise RuntimeError(f"LUMIA_CACHE_1000_IMPORT_GATE_FAILED rc={completed.returncode}")
BOOTSTRAP_JSON.write_text(
    json.dumps({
        "bootstrap_wall_seconds": time.perf_counter() - BOOTSTRAP_STARTED,
        "target_site_packages": True,
        "ensurepip_used": False,
        "live_kernel_site_packages_mutated": False,
    }, sort_keys=True) + "\n",
    encoding="utf-8",
)
print("LUMIA_CACHE_1000_INSTALL PASS target_site=1 ensurepip=0 live_kernel_mutated=0")
'''

write_runner_cell = r'''
package_root = SRC_ROOT / "poisoned_chalice"
package_root.mkdir(parents=True, exist_ok=False)
(package_root / "__init__.py").write_text("", encoding="utf-8")
(package_root / "sersem_author_faithful.py").write_text(AUTHOR_HELPERS_SOURCE, encoding="utf-8")
(package_root / "sersem_author_runtime.py").write_text(AUTHOR_RUNTIME_SOURCE, encoding="utf-8")
(package_root / "lumia_hidden_state_pilot.py").write_text(PILOT_SOURCE_TEXT, encoding="utf-8")
(package_root / "lumia_hidden_state_cache.py").write_text(CACHE_SOURCE_TEXT, encoding="utf-8")
'''

runner_source = r'''
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import shutil
import time

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from poisoned_chalice.lumia_hidden_state_cache import (
    LumiaHiddenStateCacheConfig,
    run_hidden_state_cache,
    select_cache_cohort,
)
from poisoned_chalice.lumia_hidden_state_pilot import AUTHOR_COMMIT, EXPECTED_TRAIN_ROWS, _layer_modules, validate_public_train
from poisoned_chalice.sersem_author_runtime import load_author_runtime

CONFIG = LumiaHiddenStateCacheConfig()
OUTPUT = Path(CONFIG.output_dir)
BOOTSTRAP_JSON = Path("/tmp/lumia-cache-1000-v1-bootstrap.json")
CHILD_STARTED = time.perf_counter()
assert not OUTPUT.exists(), "fresh cache target must not already contain output"
assert torch.cuda.is_available(), "CUDA is required"
assert torch.cuda.device_count() == 2, f"expected Kaggle T4 x2 visibility, got {torch.cuda.device_count()}"
gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
assert all("T4" in name for name in gpu_names), gpu_names
assert shutil.which("flake8"), "flake8 wrapper missing"
assert importlib.metadata.version("flake8") == "7.3.0"
runtime = load_author_runtime(require_exact_versions=True, require_full_components=True)
assert runtime.ast_available and runtime.enchant_available and runtime.flake8_available

# Public membership is used only to select the frozen balanced cohort. The
# scoring frame returned by select_cache_cohort physically drops it.
dataset_started = time.perf_counter()
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
cohort = select_cache_cohort(train, CONFIG)
assert list(cohort.columns) == ["sample_id", "language", "content"]
cohort_set_hash = hashlib.sha256(
    "\n".join(sorted(cohort.sample_id.astype(str))).encode("utf-8")
).hexdigest()
dataset_wall = time.perf_counter() - dataset_started
del train, parts, frame

# Deterministic model state and the explicit v5-proven single-GPU placement.
torch.manual_seed(CONFIG.seed)
np.random.seed(CONFIG.seed)
torch.cuda.manual_seed_all(CONFIG.seed)
model_started = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_id, revision=CONFIG.model_revision)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    CONFIG.model_id,
    revision=CONFIG.model_revision,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model = model.to("cuda:0").eval()
parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
buffer_devices = sorted({str(buffer.device) for buffer in model.buffers()})
assert parameter_devices == ["cuda:0"], parameter_devices
assert not buffer_devices or buffer_devices == ["cuda:0"], buffer_devices
assert next(model.parameters()).dtype == torch.float16
layer_count = len(_layer_modules(model))
expected_layers = int(getattr(model.config, "num_hidden_layers", layer_count))
assert layer_count == expected_layers == 30
hidden_size = int(getattr(model.config, "hidden_size", 0))
assert hidden_size == 3072
attention_impl = getattr(model.config, "_attn_implementation", None)
model_wall = time.perf_counter() - model_started
model_allocated = int(torch.cuda.memory_allocated(0))
model_reserved = int(torch.cuda.memory_reserved(0))
torch.cuda.reset_peak_memory_stats(0)

cache = run_hidden_state_cache(model, tokenizer, cohort, runtime, CONFIG)
assert cache["rows"] == 1000
assert cache["num_layers"] == 30 and cache["hidden_size"] == 3072
assert len(cache["shards"]) == 20
assert cache["labels_present_during_model_scoring"] is False
assert cache["performance_metrics_computed"] is False
assert cache["raw_token_layer_activations_persisted"] is False
assert cache["logits_persisted"] is False
scoring_peak_allocated = int(torch.cuda.max_memory_allocated(0))
scoring_peak_reserved = int(torch.cuda.max_memory_reserved(0))

bootstrap = json.loads(BOOTSTRAP_JSON.read_text(encoding="utf-8"))
runtime_payload = {
    "schema_version": 1,
    "task_id": "P1-03-lumia-hidden-state-cache-1000-v1",
    "cohort_set_sha256": cohort_set_hash,
    "rows": 1000,
    "num_layers": 30,
    "hidden_size": 3072,
    "max_length": 8192,
    "labels_present_during_model_scoring": False,
    "performance_metrics_computed": False,
    "bootstrap": bootstrap,
    "timing": {
        "dataset_load_and_selection_seconds": dataset_wall,
        "tokenizer_model_load_and_cuda_transfer_seconds": model_wall,
        **cache["timing"],
        "child_wall_seconds": time.perf_counter() - CHILD_STARTED,
    },
    "memory": {
        "post_model_allocated_bytes": model_allocated,
        "post_model_reserved_bytes": model_reserved,
        "scoring_peak_allocated_bytes": scoring_peak_allocated,
        "scoring_peak_reserved_bytes": scoring_peak_reserved,
    },
    "runtime": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "tokenizers": importlib.metadata.version("tokenizers"),
        "huggingface_hub": importlib.metadata.version("huggingface-hub"),
        "accelerate": importlib.metadata.version("accelerate"),
        "numpy": importlib.metadata.version("numpy"),
        "pandas": importlib.metadata.version("pandas"),
        "pyarrow": importlib.metadata.version("pyarrow"),
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "scipy": importlib.metadata.version("scipy"),
        "attention_impl": attention_impl,
        "visible_gpu_count": torch.cuda.device_count(),
        "gpu_names": gpu_names,
        "scoring_device": "cuda:0",
        "model_parameter_devices": parameter_devices,
        "model_buffer_devices": buffer_devices,
        "isolated_child_interpreter": True,
        "target_site_packages": True,
    },
}
(OUTPUT / "runtime.json").write_text(json.dumps(runtime_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
expected_shards = [f"shards/shard_{index:04d}.npz" for index in range(20)]
persistent_outputs = [
    "cache_manifest.json", "cohort_manifest.jsonl", "sample_metadata.jsonl",
    "runtime.json", "run_manifest.json", *expected_shards,
]
manifest = {
    "schema_version": 1,
    "status": "complete",
    "task_id": "P1-03-lumia-hidden-state-cache-1000-v1",
    "model_id": CONFIG.model_id,
    "model_revision": CONFIG.model_revision,
    "dataset_id": CONFIG.dataset_id,
    "dataset_revision": CONFIG.dataset_revision,
    "author_commit": AUTHOR_COMMIT,
    "rows": 1000,
    "rows_per_language_label": 100,
    "max_length": 8192,
    "shard_size": 50,
    "shard_count": 20,
    "labels_present_during_model_scoring": False,
    "performance_metrics_computed": False,
    "raw_token_layer_activations_persisted": False,
    "logits_persisted": False,
    "cohort_set_sha256": cohort_set_hash,
    "persistent_outputs": persistent_outputs,
    "automatic_compute_retries": 0,
}
(OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
observed_shards = sorted(f"shards/{path.name}" for path in (OUTPUT / "shards").glob("*.npz"))
assert observed_shards == expected_shards
observed_top = sorted(path.name for path in OUTPUT.iterdir())
assert observed_top == [
    "cache_manifest.json", "cohort_manifest.jsonl", "run_manifest.json",
    "runtime.json", "sample_metadata.jsonl", "shards",
]
print("LUMIA_HIDDEN_STATE_CACHE_1000_V1 COMPLETE rows=1000 shards=20 labels_in_scoring=0 metrics=0 token_layer_outputs=0")
print(json.dumps({
    "scoring_timing": cache["timing"],
    "token_count": cache["token_count"],
    "memory": runtime_payload["memory"],
    "shard_bytes": int(sum(item["bytes"] for item in cache["shards"])),
}, indent=2, sort_keys=True))
'''

execute_cell = r'''
RUNNER.write_text(RUNNER_SOURCE, encoding="utf-8")
env = os.environ.copy()
env["PYTHONPATH"] = str(SRC_ROOT) + os.pathsep + str(SITE_ROOT)
env["PATH"] = str(BIN_ROOT) + os.pathsep + env.get("PATH", "")
env["TOKENIZERS_PARALLELISM"] = "false"
env["OMP_NUM_THREADS"] = "1"
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
try:
    completed = subprocess.run(
        [sys.executable, str(RUNNER)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        timeout=3300,
    )
    print(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"LUMIA_CACHE_1000_CHILD_FAILED rc={completed.returncode}")
finally:
    shutil.rmtree(SITE_ROOT, ignore_errors=True)
    shutil.rmtree(BIN_ROOT, ignore_errors=True)
    shutil.rmtree(SRC_ROOT, ignore_errors=True)
    RUNNER.unlink(missing_ok=True)
    BOOTSTRAP_JSON.unlink(missing_ok=True)
print("LUMIA_CACHE_1000_CLEANUP PASS")
'''

nb = nbformat.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata.language_info = {"name": "python", "version": "3.12"}
nb.cells = [
    nbformat.v4.new_markdown_cell(
        "# P1-03 LUMIA hidden-state cache — frozen 1,000-row source experiment\n\n"
        "This is the authorized successor to the successful 50-row v5 runtime gate. "
        "Scientific conditions remain frozen: StarCoder2-3B, FP16, max_length=8192, "
        "all 30 decoder layers, mean/caller/helper pooled views, and matched same-forward "
        "output baselines. Membership is used only for deterministic cohort balancing and "
        "is physically absent during model scoring. No performance metric is computed here."
    ),
    nbformat.v4.new_code_cell(textwrap.dedent(install_cell)),
    nbformat.v4.new_code_cell(
        "AUTHOR_HELPERS_SOURCE = " + repr(AUTHOR_HELPERS) + "\n"
        "AUTHOR_RUNTIME_SOURCE = " + repr(AUTHOR_RUNTIME) + "\n"
        "PILOT_SOURCE_TEXT = " + repr(PILOT_SOURCE) + "\n"
        "CACHE_SOURCE_TEXT = " + repr(CACHE_SOURCE) + "\n"
        + textwrap.dedent(write_runner_cell)
        + "\nRUNNER_SOURCE = " + repr(textwrap.dedent(runner_source)) + "\n"
    ),
    nbformat.v4.new_code_cell(textwrap.dedent(execute_cell)),
]
for index, cell in enumerate(nb.cells):
    cell["id"] = f"lumia-cache-1000-v1-{index:02d}"

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
    "kernel_sources": [],
    "competition_sources": [],
    "model_sources": [],
}
(OUT_DIR / "kernel-metadata.json").write_text(
    __import__("json").dumps(metadata, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(OUT)
