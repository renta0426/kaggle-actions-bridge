"""Launch the repaired Gold content-reference calibration on Kaggle T4 x2.

This wrapper reuses the frozen v1 Notebook builder only in materialize mode,
injects the exact runtime-only keyword repair, and owns the single Kaggle write
for the fresh v2 target.  Scientific inputs/configuration remain v1-identical.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

EXECUTION_ID = "shadow-gold-content-reference-v2"
TARGET = "renta0426/shadow-gold-content-reference-v2"
BASE_SCIENCE_COMMIT = "d9e1b66d65c7b38a9367aef538b3997cc2d713dc"
REPAIR_COMMIT = "13d062a8fd94a55a42ae428693a7e59271ed239d"
V1_LAUNCHER_BLOB = "fff2d72e0a00480ad6cb9d5523dd059c81812da7"
REPAIR_BLOB = "0740e0daba743f38349dbac36a303cb441dfae25"
REPAIR_CONFIG_BLOB = "0e6edb478bfca72336e041d5004b802912bb94e9"
MACHINE_SHAPE = "NvidiaTeslaT4"
NOTEBOOK_NAME = "shadow-gold-content-reference-v2.ipynb"
PERSISTENT_OUTPUTS = (
    "content_reference_selected.json",
    "content_reference_metrics.json",
    "content_reference_predictions.jsonl",
    "content_reference_runtime_manifest.json",
)
REPAIR_SOURCE_PATH = "materialized/poisoned-chalice-shadow-gold-content-reference-v2/poisoned_chalice/shadow_gold_content_reference_v2.py"
REPAIR_CONFIG_PATH = "materialized/poisoned-chalice-shadow-gold-content-reference-v2/shadow_gold_content_reference_v2.json"


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def _load_exact(path: Path, expected_blob: str, maximum: int) -> bytes:
    if not path.is_file():
        raise RuntimeError(f"required file missing: {path}")
    data = path.read_bytes()
    if len(data) > maximum or blob_sha(data) != expected_blob:
        raise RuntimeError(f"file identity mismatch: {path}")
    return data


def validate_repair_contract(root: Path) -> tuple[bytes, dict]:
    source = _load_exact(root / REPAIR_SOURCE_PATH, REPAIR_BLOB, 32768)
    config_data = _load_exact(root / REPAIR_CONFIG_PATH, REPAIR_CONFIG_BLOB, 32768)
    compile(source.decode("utf-8"), REPAIR_SOURCE_PATH, "exec")
    source_text = source.decode("utf-8")
    for marker in (
        "content-reference-dual-gpu-kwarg-repair-v1",
        "required_name_fragment",
        "required_gpu_name_fragment",
        "make_dual_gpu_keyword_adapter",
        "run_content_reference_benchmark",
    ):
        if marker not in source_text:
            raise RuntimeError(f"repair source marker missing: {marker}")
    config = json.loads(config_data)
    if config.get("execution_id") != EXECUTION_ID or config.get("repair_from") != "shadow-gold-content-reference-v1":
        raise RuntimeError("repair config identity changed")
    if config.get("repair_version") != "content-reference-dual-gpu-kwarg-repair-v1":
        raise RuntimeError("repair version changed")
    scope = config.get("repair_scope") or {}
    expected_scope = {
        "old_keyword": "required_name_fragment",
        "new_keyword": "required_gpu_name_fragment",
        "dual_gpu_api_changed": False,
        "science_changed": False,
        "training_protocol_changed": False,
        "content_reference_changed": False,
        "candidate_matrix_changed": False,
        "selection_protocol_changed": False,
        "recovery_gate_changed": False,
        "clean_room_changed": False,
    }
    for key, value in expected_scope.items():
        if scope.get(key) != value:
            raise RuntimeError(f"repair scope changed: {key}")
    runtime = config.get("runtime_contract") or {}
    if runtime.get("accelerator") != "gpu" or runtime.get("kaggle_machine_shape") != MACHINE_SHAPE:
        raise RuntimeError("repair runtime resource changed")
    if runtime.get("expected_visible_gpu_count") != 2 or runtime.get("optimizer_steps_per_architecture") != 1280:
        raise RuntimeError("repair runtime scale changed")
    if runtime.get("automatic_compute_retries") != 0 or runtime.get("notebook_internet") is not False:
        raise RuntimeError("repair retry/internet contract changed")
    guards = config.get("scientific_guards") or {}
    for key in (
        "stage2_v3_selection_allowed",
        "competition_feature_promotion_allowed",
        "external_model_holdout_consumed",
        "fourth_external_holdout_consumed",
    ):
        if guards.get(key) is not False:
            raise RuntimeError(f"repair scientific guard changed: {key}")
    return source, config


def load_v1_launcher(path: Path):
    _load_exact(path, V1_LAUNCHER_BLOB, 262144)
    spec = importlib.util.spec_from_file_location("shadow_gold_content_reference_v1_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen v1 builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if getattr(module, "TARGET", None) != "renta0426/shadow-gold-content-reference-v1":
        raise RuntimeError("v1 builder target changed")
    if getattr(module, "RESEARCH_COMMIT", None) != BASE_SCIENCE_COMMIT:
        raise RuntimeError("v1 builder science commit changed")
    return module


def _patch_generated_code(code: str, repair_source: bytes) -> str:
    old_import = "from poisoned_chalice.shadow_gold_content_reference import run_content_reference_benchmark"
    new_import = "from poisoned_chalice.shadow_gold_content_reference_v2 import run_content_reference_benchmark"
    if code.count(old_import) != 1 or new_import in code:
        raise RuntimeError("content-reference import patch anchor changed")
    code = code.replace(old_import, new_import)

    old_execution = "EXECUTION_ID = 'shadow-gold-content-reference-v1'"
    if code.count(old_execution) != 1:
        raise RuntimeError("execution-id patch anchor changed")
    code = code.replace(old_execution, f"EXECUTION_ID = {EXECUTION_ID!r}")

    research_line = f"RESEARCH_COMMIT = {BASE_SCIENCE_COMMIT!r}"
    if code.count(research_line) != 1:
        raise RuntimeError("science-commit patch anchor changed")
    code = code.replace(research_line, research_line + f"\nREPAIR_COMMIT = {REPAIR_COMMIT!r}")

    encoded = base64.b64encode(repair_source).decode("ascii")
    lines = code.splitlines()
    indices = [index for index, line in enumerate(lines) if line.startswith("SOURCE_PAYLOADS = ")]
    if len(indices) != 1:
        raise RuntimeError("source-payload patch anchor changed")
    insert_at = indices[0] + 1
    lines.insert(
        insert_at,
        "SOURCE_PAYLOADS['poisoned_chalice/shadow_gold_content_reference_v2.py'] = " + repr(encoded),
    )
    code = "\n".join(lines) + "\n"

    manifest_anchor = "manifest = dict(manifest); manifest['research_commit'] = RESEARCH_COMMIT; manifest['generic_aggregation_result_commit'] = GENERIC_RESULT_COMMIT; manifest['probe_local_result_commit'] = PROBE_LOCAL_RESULT_COMMIT"
    replacement = manifest_anchor + "; manifest['repair_commit'] = REPAIR_COMMIT; manifest['execution_id'] = EXECUTION_ID; manifest['runtime_repair_only'] = True"
    if code.count(manifest_anchor) != 1:
        raise RuntimeError("manifest patch anchor changed")
    code = code.replace(manifest_anchor, replacement)

    compile(code, NOTEBOOK_NAME, "exec")
    return code


def validate_bundle(kernel_dir: Path) -> None:
    expected_files = {NOTEBOOK_NAME, "kernel-metadata.json"}
    actual = {path.name for path in kernel_dir.iterdir() if path.is_file()}
    if actual != expected_files:
        raise RuntimeError(f"v2 kernel bundle allowlist changed: {sorted(actual)}")
    notebook = json.loads((kernel_dir / NOTEBOOK_NAME).read_text(encoding="utf-8"))
    code = "".join(notebook["cells"][1]["source"])
    compile(code, NOTEBOOK_NAME, "exec")
    for marker in (
        EXECUTION_ID,
        BASE_SCIENCE_COMMIT,
        REPAIR_COMMIT,
        "shadow_gold_content_reference_v2.py",
        "from poisoned_chalice.shadow_gold_content_reference_v2 import run_content_reference_benchmark",
        "runtime_repair_only",
        "reference_leave_one_file_out",
        "known_probe_boundary_used",
        "stage2_v3_selection_allowed",
        "content_reference_metrics.json",
    ):
        if marker not in code:
            raise RuntimeError(f"generated v2 marker missing: {marker}")
    if "from poisoned_chalice.shadow_gold_content_reference import run_content_reference_benchmark" in code:
        raise RuntimeError("v1 un-repaired import remains in generated Notebook")
    metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    expected = {
        "id": TARGET,
        "title": "Gold content-reference calibration v2",
        "code_file": NOTEBOOK_NAME,
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": MACHINE_SHAPE,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"v2 kernel metadata mismatch: {key}")


def materialize(root: Path, kernel_dir: Path, base_launcher_path: Path) -> None:
    repair_source, _ = validate_repair_contract(root)
    base = load_v1_launcher(base_launcher_path)
    base_dir = kernel_dir.parent / "_content_reference_v1_base"
    if base_dir.exists():
        shutil.rmtree(base_dir)
    try:
        base.materialize(root, base_dir)
        base_notebook = json.loads((base_dir / base.NOTEBOOK_NAME).read_text(encoding="utf-8"))
        code = "".join(base_notebook["cells"][1]["source"])
        patched = _patch_generated_code(code, repair_source)
        notebook = dict(base_notebook)
        notebook["cells"] = [dict(cell) for cell in base_notebook["cells"]]
        notebook["cells"][0] = dict(notebook["cells"][0])
        notebook["cells"][0]["source"] = ["# Gold content-reference calibration v2\n", "Runtime-only dual-GPU keyword repair; v1 scientific contract unchanged.\n"]
        notebook["cells"][1] = dict(notebook["cells"][1])
        notebook["cells"][1]["source"] = patched.splitlines(keepends=True)
        if kernel_dir.exists() and any(kernel_dir.iterdir()):
            raise RuntimeError("v2 kernel directory already exists and is not empty")
        kernel_dir.mkdir(parents=True, exist_ok=True)
        (kernel_dir / NOTEBOOK_NAME).write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
        metadata = json.loads((base_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        metadata["id"] = TARGET
        metadata["title"] = "Gold content-reference calibration v2"
        metadata["code_file"] = NOTEBOOK_NAME
        (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        validate_bundle(kernel_dir)
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)
    print("SHADOW_GOLD_CONTENT_REF_V2_MATERIALIZE PASS science_snapshots=15 repair_snapshots=2 science_changed=0 outputs=4 gpu=2 steps=1280 retries=0")


def validate_static(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    count = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
            and node.func.attr == "run"
            and node.args
            and isinstance(node.args[0], (ast.List, ast.Tuple))
        ):
            values = [
                item.value if isinstance(item, ast.Constant) and isinstance(item.value, str) else "<dynamic>"
                for item in node.args[0].elts
            ]
            if any(values[index:index + 2] == ["kernels", "push"] for index in range(len(values) - 1)):
                count += 1
    if count != 1:
        raise RuntimeError(f"v2 launcher must contain exactly one kernels push, found {count}")
    for forbidden in ("competitions submit", "RESEARCH_REPO_READ_TOKEN"):
        if forbidden in source:
            raise RuntimeError(f"forbidden v2 launcher marker: {forbidden}")


def execute(kaggle_bin: Path, kernel_dir: Path) -> None:
    validate_bundle(kernel_dir)
    result = subprocess.run(
        [str(kaggle_bin), "kernels", "push", "-p", str(kernel_dir), "--timeout", "3600"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle repaired content-reference push returned {result.returncode}; outcome ambiguous and retry forbidden"
        )
    print(
        "SHADOW_GOLD_CONTENT_REF_V2_LAUNCH_ACCEPTED "
        "target=renta0426/shadow-gold-content-reference-v2 accelerator=gpu "
        "retries=0 submissions=0 private_repo_access=0 post_push_getkernel=0"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", type=Path)
    parser.add_argument("--base-launcher", type=Path)
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--kernel-dir", type=Path)
    parser.add_argument("--kaggle-bin", type=Path)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if sum((args.static, args.materialize, args.execute)) != 1:
        raise SystemExit("select exactly one operation")
    if args.static:
        if args.launcher is None:
            raise SystemExit("--launcher required")
        validate_static(args.launcher)
        print("SHADOW_GOLD_CONTENT_REF_V2_STATIC PASS write_calls=1 retries=0 submissions=0 repair=kwarg-only")
        return
    if args.materialize:
        if args.snapshot_root is None or args.kernel_dir is None or args.base_launcher is None:
            raise SystemExit("--snapshot-root, --kernel-dir, and --base-launcher required")
        materialize(args.snapshot_root, args.kernel_dir, args.base_launcher)
        return
    if args.kaggle_bin is None or args.kernel_dir is None:
        raise SystemExit("--kaggle-bin and --kernel-dir required")
    execute(args.kaggle_bin, args.kernel_dir)


if __name__ == "__main__":
    main()
