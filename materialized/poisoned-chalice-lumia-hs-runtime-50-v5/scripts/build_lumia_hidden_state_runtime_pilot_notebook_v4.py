"""Build target-directory isolated Kaggle T4 fidelity gate for hidden-state LUMIA."""
from pathlib import Path
import json
import textwrap

import nbformat

ROOT = Path(__file__).resolve().parents[1]
AUTHOR_HELPERS = (ROOT / "src/poisoned_chalice/sersem_author_faithful.py").read_text(encoding="utf-8")
AUTHOR_RUNTIME = (ROOT / "src/poisoned_chalice/sersem_author_runtime.py").read_text(encoding="utf-8")
PILOT_SOURCE = (ROOT / "src/poisoned_chalice/lumia_hidden_state_pilot.py").read_text(encoding="utf-8")

TARGET = "renta0426/poisoned-chalice-lumia-hs-runtime-50-v4"
TITLE = "Poisoned Chalice Lumia Hs Runtime 50 V4"
NOTEBOOK_NAME = "poisoned-chalice-lumia-hs-runtime-50-v4.ipynb"
OUT_DIR = ROOT / "notebooks/experiments/poisoned-chalice-lumia-hs-runtime-50-v4"
OUT = OUT_DIR / NOTEBOOK_NAME

install_cell = r'''
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

SITE_ROOT = Path("/tmp/lumia-runtime-v4-site")
BIN_ROOT = Path("/tmp/lumia-runtime-v4-bin")
SRC_ROOT = Path("/tmp/lumia-runtime-v4-src")
RUNNER = Path("/tmp/lumia-runtime-v4-runner.py")

for path in (SITE_ROOT, BIN_ROOT, SRC_ROOT):
    if path.exists():
        shutil.rmtree(path)
if RUNNER.exists():
    RUNNER.unlink()
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
        "--target", str(SITE_ROOT),
        *packages,
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
import scipy
import sklearn
import torch
import transformers
import tokenizers
import pyarrow
import sklearn.metrics
from transformers import AutoModelForCausalLM, AutoTokenizer

site = pathlib.Path(os.environ["LUMIA_SITE_ROOT"]).resolve()
expected = {
    "numpy": "2.4.2",
    "pandas": "3.0.1",
    "pyarrow": "23.0.1",
    "scikit-learn": "1.8.0",
    "scipy": "1.17.1",
    "joblib": "1.5.3",
    "threadpoolctl": "3.6.0",
    "transformers": "4.52.0",
    "tokenizers": "0.21.0",
    "huggingface-hub": "0.30.0",
}
for distribution, version in expected.items():
    observed = importlib.metadata.version(distribution)
    assert observed == version, (distribution, observed, version)
for module in (np, pd, pyarrow, scipy, sklearn, transformers, tokenizers):
    assert site in pathlib.Path(module.__file__).resolve().parents, (
        module.__name__, module.__file__, site
    )
assert site not in pathlib.Path(torch.__file__).resolve().parents
x = torch.tensor([1.0, 2.0], device="cpu").numpy()
assert x.shape == (2,)
print(json.dumps({
    "site_root": str(site),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "pyarrow": pyarrow.__version__,
    "scipy": scipy.__version__,
    "sklearn": sklearn.__version__,
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "fresh_child_import_gate": True,
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
    raise RuntimeError(f"LUMIA_RUNTIME_V4_TARGET_IMPORT_GATE_FAILED rc={completed.returncode}")
print("LUMIA_RUNTIME_V4_INSTALL PASS target_site=1 ensurepip=0 live_kernel_mutated=0")
'''

write_runner_cell = r'''
package_root = SRC_ROOT / "poisoned_chalice"
package_root.mkdir(parents=True, exist_ok=False)
(package_root / "__init__.py").write_text("", encoding="utf-8")
(package_root / "sersem_author_faithful.py").write_text(AUTHOR_HELPERS_SOURCE, encoding="utf-8")
(package_root / "sersem_author_runtime.py").write_text(AUTHOR_RUNTIME_SOURCE, encoding="utf-8")
(package_root / "lumia_hidden_state_pilot.py").write_text(PILOT_SOURCE_TEXT, encoding="utf-8")
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

from poisoned_chalice.lumia_hidden_state_pilot import (
    AUTHOR_COMMIT,
    EXPECTED_TRAIN_ROWS,
    LumiaRuntimePilotConfig,
    _layer_modules,
    run_runtime_pilot,
    select_runtime_pilot,
    validate_public_train,
    write_json,
)
from poisoned_chalice.sersem_author_runtime import load_author_runtime

CONFIG = LumiaRuntimePilotConfig()
OUTPUT = Path(CONFIG.output_dir)
OUTPUT.mkdir(parents=True, exist_ok=True)

assert torch.cuda.is_available(), "CUDA is required"
assert torch.cuda.device_count() == 2, f"expected Kaggle T4 x2 visibility, got {torch.cuda.device_count()}"
gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
assert all("T4" in name for name in gpu_names), gpu_names
assert shutil.which("flake8"), "flake8 wrapper missing from target runtime PATH"
assert importlib.metadata.version("flake8") == "7.3.0"
runtime = load_author_runtime(require_exact_versions=True, require_full_components=True)
assert runtime.ast_available and runtime.enchant_available and runtime.flake8_available

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

torch.manual_seed(CONFIG.seed)
np.random.seed(CONFIG.seed)
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
    "pyarrow": importlib.metadata.version("pyarrow"),
    "scikit_learn": importlib.metadata.version("scikit-learn"),
    "scipy": importlib.metadata.version("scipy"),
    "joblib": importlib.metadata.version("joblib"),
    "threadpoolctl": importlib.metadata.version("threadpoolctl"),
    "attention_impl": attention_impl,
    "visible_gpu_count": torch.cuda.device_count(),
    "gpu_names": gpu_names,
    "isolated_child_interpreter": True,
    "target_site_packages": True,
}
summary["total_cell_wall_seconds"] = time.perf_counter() - run_started
write_json(OUTPUT / "runtime.json", summary)
manifest = {
    "schema_version": 1,
    "status": "complete",
    "task_id": summary["task_id"],
    "operational_attempt": "v4_target_site_repair",
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
    "isolated_child_interpreter": True,
    "target_site_packages": True,
}
write_json(OUTPUT / "run_manifest.json", manifest)
assert sorted(path.name for path in OUTPUT.iterdir() if path.is_file()) == ["run_manifest.json", "runtime.json"]
print("LUMIA_HIDDEN_STATE_RUNTIME_PILOT_V4 COMPLETE performance_metrics=0 sample_outputs=0")
print(json.dumps({
    "timing": summary["timing"],
    "token_count": summary["token_count"],
    "num_layers": summary["num_layers"],
    "hidden_size": summary["hidden_size"],
    "diagnostics": summary["diagnostics"],
    "runtime_recommendation": summary["runtime_recommendation"],
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
        timeout=3600,
    )
    print(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"LUMIA_RUNTIME_V4_CHILD_FAILED rc={completed.returncode}")
finally:
    shutil.rmtree(SITE_ROOT, ignore_errors=True)
    shutil.rmtree(BIN_ROOT, ignore_errors=True)
    shutil.rmtree(SRC_ROOT, ignore_errors=True)
    RUNNER.unlink(missing_ok=True)
print("LUMIA_RUNTIME_V4_CLEANUP PASS")
'''

nb = nbformat.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata.language_info = {"name": "python", "version": "3.12"}
nb.cells = [
    nbformat.v4.new_markdown_cell(
        "# LUMIA hidden-state runtime/fidelity gate v4 — 50 public-train rows\n\n"
        "Operational repair of the failed v3 ensurepip/venv bootstrap. Scientific inputs, "
        "8192-token scope, pooling definitions, and the 105-minute scale gate are unchanged. "
        "Pinned packages are installed only into a /tmp target directory and all scientific "
        "imports and GPU work execute in a fresh child interpreter."
    ),
    nbformat.v4.new_code_cell(textwrap.dedent(install_cell)),
    nbformat.v4.new_code_cell(
        "AUTHOR_HELPERS_SOURCE = " + repr(AUTHOR_HELPERS) + "\n"
        "AUTHOR_RUNTIME_SOURCE = " + repr(AUTHOR_RUNTIME) + "\n"
        "PILOT_SOURCE_TEXT = " + repr(PILOT_SOURCE) + "\n"
        + textwrap.dedent(write_runner_cell)
        + "\nRUNNER_SOURCE = " + repr(textwrap.dedent(runner_source)) + "\n"
    ),
    nbformat.v4.new_code_cell(textwrap.dedent(execute_cell)),
]
for index, cell in enumerate(nb.cells):
    cell["id"] = f"lumia-runtime-v4-{index:02d}"

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
    "keywords": ["gpu", "membership-inference", "lumia", "runtime-pilot", "target-site-repair"],
    "dataset_sources": [],
    "kernel_sources": [],
    "competition_sources": [],
    "model_sources": [],
}
(OUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
print(OUT)
