"""Build and launch Gold architecture-transfer execution v2 with a narrow JSONL readback repair.

The frozen v1 launcher is used for local materialization only. This wrapper injects the
exact research repair module and execution manifest, switches the parent import to the
repair layer, retargets the private Kaggle Notebook to a fresh v2 slug, and owns the
sole Kaggle write.
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

BASE_LAUNCHER_BLOB = "4365d199641d4310aec4c8dbf198c20b2347f15b"
SCIENTIFIC_RESEARCH_COMMIT = "bc0661fa0307b3e27b31f2ab4c83e90e74647390"
REPAIR_RESEARCH_COMMIT = "f8678f3d5d0a99763e02906f24c7a34e09e12f5e"
SCIENTIFIC_EXPERIMENT_ID = "shadow-gold-architecture-transfer-v1"
EXECUTION_ID = "shadow-gold-architecture-transfer-v2"
REPAIR_VERSION = "shadow-gold-jsonl-column-order-v2"
TARGET = "renta0426/shadow-gold-architecture-transfer-v2"
NOTEBOOK_NAME = "shadow-gold-architecture-transfer-v2.ipynb"
BASE_NOTEBOOK_NAME = "shadow-gold-architecture-transfer-v1.ipynb"
MACHINE_SHAPE = "NvidiaTeslaT4"
PERSISTENT_OUTPUTS = (
    "gold_attack.json",
    "gold_metrics.json",
    "gold_predictions.jsonl",
    "gold_runtime_manifest.json",
)
REPAIR_MODULE_BRIDGE_PATH = (
    "materialized/poisoned-chalice-shadow-gold-v2/"
    "poisoned_chalice/shadow_gold_kaggle_v2.py"
)
REPAIR_MODULE_BLOB = "df6e67f7d0e67d2585fd6226b9d2592223f808d2"
REPAIR_CONFIG_BRIDGE_PATH = (
    "materialized/poisoned-chalice-shadow-gold-v2/"
    "shadow_gold_architecture_transfer_v2.json"
)
REPAIR_CONFIG_BLOB = "5060bbdb9b96aeef3663e3853fe00578addffc4e"


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def load_base(path: Path):
    data = path.read_bytes()
    if blob_sha(data) != BASE_LAUNCHER_BLOB:
        raise RuntimeError("Gold v1 base launcher blob mismatch")
    spec = importlib.util.spec_from_file_location("shadow_gold_v1_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Gold v1 base launcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.RESEARCH_COMMIT != SCIENTIFIC_RESEARCH_COMMIT:
        raise RuntimeError("Gold v1 scientific commit changed")
    if module.EXPERIMENT_ID != SCIENTIFIC_EXPERIMENT_ID:
        raise RuntimeError("Gold v1 experiment ID changed")
    return module


def load_repair(snapshot_root: Path) -> tuple[bytes, str]:
    module_path = snapshot_root / REPAIR_MODULE_BRIDGE_PATH
    config_path = snapshot_root / REPAIR_CONFIG_BRIDGE_PATH
    module_data = module_path.read_bytes()
    config_data = config_path.read_bytes()
    if len(module_data) > 32768 or blob_sha(module_data) != REPAIR_MODULE_BLOB:
        raise RuntimeError("Gold v2 repair module identity mismatch")
    if len(config_data) > 32768 or blob_sha(config_data) != REPAIR_CONFIG_BLOB:
        raise RuntimeError("Gold v2 repair config identity mismatch")
    source = module_data.decode("utf-8")
    compile(source, REPAIR_MODULE_BRIDGE_PATH, "exec")
    for marker in (
        'SHADOW_GOLD_JSONL_REPAIR_VERSION = "shadow-gold-jsonl-column-order-v2"',
        "canonicalize_exact_schema",
        'name == "gold_features.jsonl"',
        'name == "evaluation_labels.jsonl"',
        "set(actual) != set(expected)",
        "frame.loc[:, expected].copy()",
        "scientific_protocol_changed",
    ):
        if marker not in source:
            raise RuntimeError(f"Gold v2 repair module marker missing: {marker}")
    config = json.loads(config_data)
    if config.get("execution_id") != EXECUTION_ID:
        raise RuntimeError("Gold v2 execution ID changed")
    if config.get("scientific_experiment_id") != SCIENTIFIC_EXPERIMENT_ID:
        raise RuntimeError("Gold v2 scientific experiment identity changed")
    if config.get("repair_from") != SCIENTIFIC_EXPERIMENT_ID:
        raise RuntimeError("Gold v2 repair parent changed")
    if config.get("scientific_protocol_changed") is not False:
        raise RuntimeError("Gold v2 scientific protocol changed")
    repair = config.get("repair") or {}
    if repair.get("version") != REPAIR_VERSION or repair.get("scope") != "persistence_readback_only":
        raise RuntimeError("Gold v2 repair scope changed")
    if any(repair.get(key) is not False for key in (
        "missing_columns_allowed", "extra_columns_allowed", "duplicate_columns_allowed"
    )):
        raise RuntimeError("Gold v2 exact-schema guard weakened")
    unchanged = config.get("unchanged") or {}
    expected = {
        "candidate_rows": 10240,
        "training_corpus_rows": 5120,
        "evaluation_rows": 5120,
        "training_sequences": 20480,
        "optimizer_steps_per_architecture": 320,
        "seed": 2027,
        "global_batch_size": 64,
        "backend": "cuda",
        "kaggle_machine_shape": MACHINE_SHAPE,
        "expected_visible_gpu_count": 2,
        "per_architecture_world_size": 1,
        "distributed_training": False,
        "automatic_compute_retries": 0,
        "notebook_internet": False,
        "competition_rows_used": 0,
        "pretrained_weights_used": False,
        "stage2_v3_selection_allowed": False,
        "external_model_holdout_consumed": False,
    }
    for key, value in expected.items():
        if unchanged.get(key) != value:
            raise RuntimeError(f"Gold v2 unchanged contract changed: {key}")
    return module_data, config_data.decode("utf-8")


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"Gold v2 patch anchor changed: {label} count={text.count(old)}")
    return text.replace(old, new, 1)


def transform_code(code: str, repair_module: bytes, repair_config_text: str) -> str:
    if SCIENTIFIC_RESEARCH_COMMIT not in code:
        raise RuntimeError("Gold v1 scientific commit missing from generated Notebook")
    constants = (
        f"EXECUTION_ID = {EXECUTION_ID!r}\n"
        f"REPAIR_RESEARCH_COMMIT = {REPAIR_RESEARCH_COMMIT!r}\n"
        f"REPAIR_VERSION = {REPAIR_VERSION!r}\n"
        f"REPAIR_PAYLOAD = {base64.b64encode(repair_module).decode('ascii')!r}\n"
        f"REPAIR_CONFIG_TEXT = {repair_config_text!r}\n\n"
    )
    code = _replace_once(
        code,
        "working = Path('/kaggle/working')",
        constants + "working = Path('/kaggle/working')",
        "execution constants",
    )
    init_line = "(runtime_root / 'poisoned_chalice/__init__.py').write_text('', encoding='utf-8')"
    code = _replace_once(
        code,
        init_line,
        init_line
        + "\nrepair_destination = runtime_root / 'poisoned_chalice/shadow_gold_kaggle_v2.py'"
        + "\nrepair_destination.write_bytes(base64.b64decode(REPAIR_PAYLOAD))",
        "repair module write",
    )
    config_line = "config = json.loads(CONFIG_TEXT)"
    repair_validation = r"""
repair_config = json.loads(REPAIR_CONFIG_TEXT)
if repair_config.get('execution_id') != EXECUTION_ID:
    raise RuntimeError('Gold v2 execution ID changed')
if repair_config.get('scientific_experiment_id') != EXPERIMENT_ID:
    raise RuntimeError('Gold v2 scientific experiment ID changed')
if repair_config.get('repair_from') != EXPERIMENT_ID:
    raise RuntimeError('Gold v2 repair parent changed')
if repair_config.get('scientific_protocol_changed') is not False:
    raise RuntimeError('Gold v2 scientific protocol changed')
repair_contract = repair_config.get('repair') or {}
if repair_contract.get('version') != REPAIR_VERSION or repair_contract.get('scope') != 'persistence_readback_only':
    raise RuntimeError('Gold v2 repair contract changed')
if any(repair_contract.get(key) is not False for key in (
    'missing_columns_allowed', 'extra_columns_allowed', 'duplicate_columns_allowed'
)):
    raise RuntimeError('Gold v2 schema guard weakened')
unchanged = repair_config.get('unchanged') or {}
if unchanged.get('optimizer_steps_per_architecture') != 320 or unchanged.get('evaluation_rows') != 5120:
    raise RuntimeError('Gold v2 scientific scale changed')
if unchanged.get('stage2_v3_selection_allowed') is not False or unchanged.get('external_model_holdout_consumed') is not False:
    raise RuntimeError('Gold v2 scientific guard changed')
"""
    code = _replace_once(
        code, config_line, config_line + repair_validation, "repair config validation"
    )
    code = _replace_once(
        code,
        "from poisoned_chalice.shadow_gold_kaggle import run_gold_benchmark",
        "from poisoned_chalice.shadow_gold_kaggle_v2 import run_gold_benchmark",
        "repair import",
    )
    guard_anchor = (
        "    if metrics.get('status') != 'complete' or manifest.get('status') != 'complete':"
    )
    repair_guard = (
        "    if manifest.get('persistence_readback_repair') != REPAIR_VERSION:\n"
        "        raise RuntimeError('Gold v2 readback repair was not applied')\n"
        "    if manifest.get('scientific_protocol_changed') is not False:\n"
        "        raise RuntimeError('Gold v2 scientific protocol changed at runtime')\n"
    )
    code = _replace_once(
        code, guard_anchor, repair_guard + guard_anchor, "runtime repair guard"
    )
    code = _replace_once(
        code,
        "        'experiment_id': EXPERIMENT_ID,\n        'research_commit': RESEARCH_COMMIT,",
        "        'experiment_id': EXPERIMENT_ID,\n"
        "        'execution_id': EXECUTION_ID,\n"
        "        'research_commit': RESEARCH_COMMIT,\n"
        "        'repair_research_commit': REPAIR_RESEARCH_COMMIT,",
        "completion summary identity",
    )
    compile(code, NOTEBOOK_NAME, "exec")
    return code


def validate_generated_code(code: str) -> None:
    tree = ast.parse(code)
    for marker in (
        SCIENTIFIC_RESEARCH_COMMIT,
        REPAIR_RESEARCH_COMMIT,
        EXECUTION_ID,
        REPAIR_VERSION,
        "from poisoned_chalice.shadow_gold_kaggle_v2 import run_gold_benchmark",
        "persistence_readback_repair",
        "scientific_protocol_changed",
        "timeout_seconds=3000",
        "gold_attack.json",
        "gold_metrics.json",
        "gold_predictions.jsonl",
        "gold_runtime_manifest.json",
    ):
        if marker not in code:
            raise RuntimeError(f"Gold v2 generated Notebook marker missing: {marker}")
    if "from poisoned_chalice.shadow_gold_kaggle import run_gold_benchmark" in code:
        raise RuntimeError("Gold v1 direct parent import remained after v2 repair")
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_gold_benchmark"
    ]
    if len(calls) != 1:
        raise RuntimeError("Gold v2 Notebook must invoke run_gold_benchmark exactly once")
    for forbidden in ("competitions submit", "RESEARCH_REPO_READ_TOKEN"):
        if forbidden in code:
            raise RuntimeError(f"forbidden Gold v2 generated marker: {forbidden}")


def validate_bundle(kernel_dir: Path) -> None:
    expected_files = {"kernel-metadata.json", NOTEBOOK_NAME}
    actual = {path.name for path in kernel_dir.iterdir() if path.is_file()}
    if actual != expected_files:
        raise RuntimeError(f"Gold v2 kernel bundle allowlist changed: {sorted(actual)}")
    metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    notebook = json.loads((kernel_dir / NOTEBOOK_NAME).read_text(encoding="utf-8"))
    code = "".join(notebook["cells"][1]["source"])
    compile(code, NOTEBOOK_NAME, "exec")
    validate_generated_code(code)
    expected = {
        "id": TARGET,
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
            raise RuntimeError(f"Gold v2 metadata mismatch: {key}")


def materialize(base_launcher: Path, snapshot_root: Path, kernel_dir: Path) -> None:
    if kernel_dir.exists() and any(kernel_dir.iterdir()):
        raise RuntimeError("Gold v2 kernel directory already exists and is not empty")
    repair_module, repair_config_text = load_repair(snapshot_root)
    base = load_base(base_launcher)
    base_dir = kernel_dir.parent / (kernel_dir.name + "-v1-base")
    if base_dir.exists():
        shutil.rmtree(base_dir)
    try:
        base.materialize(snapshot_root, base_dir)
        notebook = json.loads((base_dir / BASE_NOTEBOOK_NAME).read_text(encoding="utf-8"))
        metadata = json.loads((base_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        code = "".join(notebook["cells"][1]["source"])
        transformed = transform_code(code, repair_module, repair_config_text)
        validate_generated_code(transformed)

        kernel_dir.mkdir(parents=True, exist_ok=False)
        notebook["cells"][0]["source"] = [
            "# Gold controlled architecture-transfer benchmark v2\n",
            "Persistence-readback repair only; the frozen v1 scientific protocol is unchanged.\n",
        ]
        notebook["cells"][0]["id"] = "shadow-gold-v2-intro"
        notebook["cells"][1]["id"] = "shadow-gold-v2-run"
        notebook["cells"][1]["source"] = transformed.splitlines(keepends=True)
        (kernel_dir / NOTEBOOK_NAME).write_text(
            json.dumps(notebook, indent=2) + "\n", encoding="utf-8"
        )

        metadata.update({
            "id": TARGET,
            "title": "Gold controlled architecture-transfer benchmark v2",
            "code_file": NOTEBOOK_NAME,
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": False,
            "machine_shape": MACHINE_SHAPE,
            "dataset_sources": [],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
        })
        (kernel_dir / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validate_bundle(kernel_dir)
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)
    print(
        "SHADOW_GOLD_V2_MATERIALIZE PASS "
        "science_snapshots=10 repair_snapshots=2 outputs=4 gpu=2 steps=320 eval=5120 retries=0"
    )


def literal_commands(source: str) -> list[tuple[str, ...]]:
    commands: list[tuple[str, ...]] = []
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
            and node.func.attr == "run"
            and node.args
            and isinstance(node.args[0], (ast.List, ast.Tuple))
        ):
            commands.append(tuple(
                item.value if isinstance(item, ast.Constant) and isinstance(item.value, str)
                else "<dynamic>"
                for item in node.args[0].elts
            ))
    return commands


def validate_static(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")
    commands = literal_commands(source)

    def pair(command: tuple[str, ...], left: str, right: str) -> bool:
        return any(command[index:index + 2] == (left, right) for index in range(len(command) - 1))

    if sum(pair(command, "kernels", "push") for command in commands) != 1:
        raise RuntimeError("Gold v2 wrapper must contain exactly one kernels push")
    for forbidden in (
        ("competitions", "submit"),
        ("datasets", "create"),
        ("datasets", "version"),
        ("models", "create"),
        ("models", "version"),
        ("kernels", "delete"),
        ("kernels", "cancel"),
    ):
        if any(pair(command, *forbidden) for command in commands):
            raise RuntimeError(f"forbidden Gold v2 write: {' '.join(forbidden)}")


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
            f"Kaggle Gold v2 push returned nonzero ({result.returncode}); "
            "outcome ambiguous and retry forbidden"
        )
    print(
        "SHADOW_GOLD_V2_LAUNCH_ACCEPTED "
        f"target={TARGET} accelerator=gpu machine={MACHINE_SHAPE} "
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
        print("SHADOW_GOLD_V2_STATIC PASS write_calls=1 retries=0 submissions=0 outputs=4")
        return
    if args.base_launcher is None or args.kernel_dir is None:
        raise SystemExit("--base-launcher and --kernel-dir required")
    if args.materialize:
        if args.snapshot_root is None:
            raise SystemExit("--snapshot-root required")
        materialize(args.base_launcher, args.snapshot_root, args.kernel_dir)
        return
    if args.kaggle_bin is None:
        raise SystemExit("--kaggle-bin required")
    execute(args.kaggle_bin, args.kernel_dir)


if __name__ == "__main__":
    main()
