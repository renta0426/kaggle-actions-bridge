#!/usr/bin/env python3
"""Transform the sealed successful CodeParrot 200-row pilot source into the authorized 4,000-row full scorer.

The transformation is deliberately narrow: it preserves the already-successful private scoring cells,
model revision, numerical gates, window contract, and label boundary. It only changes the selected row
set from the first 200 frozen rows to all 4,000 frozen rows, output identity, and authorization metadata.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import nbformat

PILOT_ID = "renta0426/codeparrot-fresh-pilot-v1"
TARGET_ID = "renta0426/codeparrot-fresh-full-v1"
TARGET_TITLE = "CodeParrot Fresh Full V1"
AUDITED_COHORT_SHA256 = "b664e368f9380e230c9cfc0616327d829424749c5473f87575678525db1998fc"
COMPAT_MARKER = "CODEPARROT_PRIVATE_SNAPSHOT_COMPAT_V3"


def _source(cell: nbformat.NotebookNode) -> str:
    value = cell.get("source", "")
    return value if isinstance(value, str) else "".join(value)


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one replacement anchor, found {count}")
    return source.replace(old, new, 1)


def patch_bundle(root: Path) -> Path:
    notebooks = sorted(root.glob("*.ipynb"))
    if len(notebooks) != 1:
        raise RuntimeError(f"expected exactly one pilot notebook, found {len(notebooks)}")
    source_path = notebooks[0]
    notebook = nbformat.read(source_path, as_version=4)
    nbformat.validate(notebook)

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    combined_before = "\n".join(_source(cell) for cell in code_cells)
    required_before = (
        COMPAT_MARKER,
        AUDITED_COHORT_SHA256,
        'EXPERIMENT_ID = "codeparrot-fresh-pilot-v1"',
        'pilot = cohort.iloc[:200].copy()',
        'private_input = scorer_frame(pilot)',
        'expected_internal_ids = {f"row-{index:06d}" for index in range(200)}',
        'sample_features = aggregate_private_features(window_features, 200)',
        'first8_private = scorer_frame(pilot.iloc[:8])',
        'OUTPUT / "pilot_predictions.parquet"',
        '"full_4000_scoring_authorized": False',
        '"performance_metrics_computed": False',
        '"membership_join_performed": False',
        'original_sample_id_passed_to_scorer',
    )
    if not all(marker in combined_before for marker in required_before):
        missing = [marker for marker in required_before if marker not in combined_before]
        raise RuntimeError(f"successful pilot source contract mismatch: {missing}")

    markdown_indexes = [
        i for i, cell in enumerate(notebook.cells)
        if cell.cell_type == "markdown" and "200-row label-blind T4 pilot" in _source(cell)
    ]
    if len(markdown_indexes) != 1:
        raise RuntimeError(f"expected one pilot markdown cell, found {len(markdown_indexes)}")
    notebook.cells[markdown_indexes[0]]["source"] = (
        "# CodeParrot fresh confirmation — 4,000-row label-blind T4 full scoring\n\n"
        "Authorized full extraction on the already-frozen 4,000-row cohort. Predictions are sealed "
        "before any membership join. No performance metric or competition submission is computed here."
    )

    setup_indexes = [
        i for i, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        and 'EXPERIMENT_ID = "codeparrot-fresh-pilot-v1"' in _source(cell)
        and 'OUTPUT = Path("/kaggle/working/codeparrot_fresh_pilot_v1")' in _source(cell)
    ]
    if len(setup_indexes) != 1:
        raise RuntimeError(f"expected one pilot setup cell, found {len(setup_indexes)}")
    setup_index = setup_indexes[0]
    setup = _source(notebook.cells[setup_index])
    setup = _replace_once(
        setup,
        'EXPERIMENT_ID = "codeparrot-fresh-pilot-v1"',
        'EXPERIMENT_ID = "codeparrot-fresh-full-v1"',
        "experiment id",
    )
    setup = _replace_once(
        setup,
        'OUTPUT = Path("/kaggle/working/codeparrot_fresh_pilot_v1")',
        'OUTPUT = Path("/kaggle/working/codeparrot_fresh_full_v1")',
        "output directory",
    )
    setup = _replace_once(setup, '"rows": 200,', '"rows": 4000,', "setup row declaration")
    setup = setup.replace("CodeParrot pilot", "CodeParrot full")
    notebook.cells[setup_index]["source"] = setup
    compile(setup, f"{source_path}:full_setup", "exec")

    load_indexes = [
        i for i, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        and 'pilot = cohort.iloc[:200].copy()' in _source(cell)
        and 'pilot label-free selection failed' in _source(cell)
    ]
    if len(load_indexes) != 1:
        raise RuntimeError(f"expected one pilot selection cell, found {len(load_indexes)}")
    load_index = load_indexes[0]
    load = _source(notebook.cells[load_index])
    old_selection = '''pilot = cohort.iloc[:200].copy()\nif len(pilot) != 200 or pilot.sample_id.duplicated().any():\n    raise RuntimeError("pilot label-free selection failed")'''
    new_selection = '''full = cohort.copy()\nif len(full) != 4000 or full.sample_id.duplicated().any():\n    raise RuntimeError("full label-free selection failed")'''
    load = _replace_once(load, old_selection, new_selection, "full cohort selection")
    load = load.replace("pilot language contract failed", "full language contract failed")
    notebook.cells[load_index]["source"] = load
    compile(load, f"{source_path}:full_load", "exec")

    model_indexes = [
        i for i, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        and "CodeParrot pilot requires CUDA/T4" in _source(cell)
        and "runtime_identity" in _source(cell)
    ]
    if len(model_indexes) == 1:
        model = _source(notebook.cells[model_indexes[0]]).replace(
            "CodeParrot pilot requires CUDA/T4", "CodeParrot full scoring requires CUDA/T4"
        )
        notebook.cells[model_indexes[0]]["source"] = model
        compile(model, f"{source_path}:full_model", "exec")

    run_indexes = [
        i for i, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        and 'private_input = scorer_frame(pilot)' in _source(cell)
        and 'OUTPUT / "pilot_predictions.parquet"' in _source(cell)
        and '"full_4000_scoring_authorized": False' in _source(cell)
    ]
    if len(run_indexes) != 1:
        raise RuntimeError(f"expected one pilot execution cell, found {len(run_indexes)}")
    run_index = run_indexes[0]
    run = _source(notebook.cells[run_index])

    replacements = (
        ('private_input = scorer_frame(pilot)', 'private_input = scorer_frame(full)', 'scoring input'),
        ('range(200)}', 'range(4000)}', 'internal id range'),
        ('aggregate_private_features(window_features, 200)', 'aggregate_private_features(window_features, 4000)', 'aggregation count'),
        ('first8_private = scorer_frame(pilot.iloc[:8])', 'first8_private = scorer_frame(full.iloc[:8])', 'first8 fidelity frame'),
        ('"sample_id": pilot.sample_id.to_numpy()', '"sample_id": full.sample_id.to_numpy()', 'prediction sample id'),
        ('"content_sha256": pilot.content_sha256.to_numpy()', '"content_sha256": full.content_sha256.to_numpy()', 'prediction content hash'),
        ('"language": pilot.language.to_numpy()', '"language": full.language.to_numpy()', 'prediction language'),
        ('len(predictions) != 200', 'len(predictions) != 4000', 'prediction count'),
        ('OUTPUT / "pilot_predictions.parquet"', 'OUTPUT / "full_predictions.parquet"', 'prediction output'),
        ('"schema_version": "codeparrot_fresh_pilot_manifest_v1"', '"schema_version": "codeparrot_fresh_full_manifest_v1"', 'manifest schema'),
        ('"status": "sealed_label_blind_pilot_predictions"', '"status": "sealed_label_blind_full_predictions"', 'manifest status'),
        ('"selection":\n                "first_200_sample_ids_from_label_free_4000_prediction_input"', '"selection":\n                "all_4000_sample_ids_from_frozen_label_free_prediction_input"', 'manifest selection'),
        ('("pilot_fidelity.json", fidelity)', '("full_fidelity.json", fidelity)', 'fidelity output'),
        ('("pilot_profile.json", profile)', '("full_profile.json", profile)', 'profile output'),
        ('("pilot_manifest.json", manifest)', '("full_manifest.json", manifest)', 'manifest output'),
    )
    for old, new, label in replacements:
        run = _replace_once(run, old, new, label)

    row_count = run.count('"rows": 200,')
    if row_count != 3:
        raise RuntimeError(f"run row declarations: expected 3, found {row_count}")
    run = run.replace('"rows": 200,', '"rows": 4000,')
    authorization_count = run.count('"full_4000_scoring_authorized": False')
    if authorization_count != 2:
        raise RuntimeError(f"full authorization markers: expected 2, found {authorization_count}")
    run = run.replace('"full_4000_scoring_authorized": False', '"full_4000_scoring_authorized": True')
    run = run.replace("pilot sample aggregation", "full sample aggregation")
    run = run.replace("non-finite pilot feature", "non-finite full feature")
    run = run.replace("pilot prediction coverage", "full prediction coverage")
    run = run.replace("non-finite pilot predictions", "non-finite full predictions")
    run = run.replace("complete_label_blind_runtime_fidelity_pilot", "complete_label_blind_full_scoring")
    notebook.cells[run_index]["source"] = run
    compile(run, f"{source_path}:full_execution", "exec")

    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []

    combined = "\n".join(_source(cell) for cell in notebook.cells if cell.cell_type == "code")
    required_after = (
        COMPAT_MARKER,
        AUDITED_COHORT_SHA256,
        'EXPERIMENT_ID = "codeparrot-fresh-full-v1"',
        'full = cohort.copy()',
        'private_input = scorer_frame(full)',
        'range(4000)}',
        'aggregate_private_features(window_features, 4000)',
        'first8_private = scorer_frame(full.iloc[:8])',
        'OUTPUT / "full_predictions.parquet"',
        '"schema_version": "codeparrot_fresh_full_manifest_v1"',
        'all_4000_sample_ids_from_frozen_label_free_prediction_input',
        '"full_4000_scoring_authorized": True',
        '"performance_metrics_computed": False',
        '"membership_join_performed": False',
        '"competition_submission": False',
        'original_sample_id_passed_to_scorer',
    )
    if not all(marker in combined for marker in required_after):
        missing = [marker for marker in required_after if marker not in combined]
        raise RuntimeError(f"full scorer contract incomplete: {missing}")
    forbidden_after = (
        'pilot = cohort.iloc[:200].copy()',
        'private_input = scorer_frame(pilot)',
        'OUTPUT / "pilot_predictions.parquet"',
        '"full_4000_scoring_authorized": False',
        "competitions submit",
        "submission.csv",
        "RESEARCH_REPO_READ_TOKEN",
        "GITHUB_PAT",
        "GH_TOKEN",
        "SSH_PRIVATE_KEY",
    )
    if any(marker in combined for marker in forbidden_after):
        bad = [marker for marker in forbidden_after if marker in combined]
        raise RuntimeError(f"forbidden pilot/side-effect marker remains: {bad}")

    metadata_path = root / "kernel-metadata.json"
    if not metadata_path.exists():
        raise RuntimeError("kernel-metadata.json missing from pulled pilot")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("id") != PILOT_ID:
        raise RuntimeError(f"unexpected pulled pilot id: {metadata.get('id')}")
    if metadata.get("kernel_sources") != ["renta0426/codeparrot-fresh-cohort-v1"]:
        raise RuntimeError("pilot cohort source changed")
    if metadata.get("enable_gpu") is not True or metadata.get("enable_tpu") is not False:
        raise RuntimeError("pilot accelerator metadata changed")
    if metadata.get("machine_shape") != "NvidiaTeslaT4":
        raise RuntimeError("pilot machine shape changed")

    target_file = "codeparrot-fresh-full-v1.ipynb"
    target_path = root / target_file
    if source_path != target_path:
        source_path.unlink()
    nbformat.validate(notebook)
    nbformat.write(notebook, target_path)

    metadata["id"] = TARGET_ID
    metadata["title"] = TARGET_TITLE
    metadata["code_file"] = target_file
    metadata["is_private"] = True
    metadata["enable_gpu"] = True
    metadata["enable_tpu"] = False
    metadata["enable_internet"] = True
    metadata["machine_shape"] = "NvidiaTeslaT4"
    metadata["keywords"] = ["gpu", "codeparrot", "membership-inference", "fresh-full"]
    metadata["dataset_sources"] = []
    metadata["kernel_sources"] = ["renta0426/codeparrot-fresh-cohort-v1"]
    metadata["competition_sources"] = []
    metadata["model_sources"] = []
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return target_path


def _synthetic_bundle(root: Path) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_markdown_cell("# 200-row label-blind T4 pilot"),
        nbformat.v4.new_code_cell("# CODEPARROT_PRIVATE_SNAPSHOT_COMPAT_V3\n"),
        nbformat.v4.new_code_cell(
            'from pathlib import Path\n'
            'EXPERIMENT_ID = "codeparrot-fresh-pilot-v1"\n'
            f'EXPECTED_COHORT_SHA256 = "{AUDITED_COHORT_SHA256}"\n'
            'OUTPUT = Path("/kaggle/working/codeparrot_fresh_pilot_v1")\n'
            'print({\n            "rows": 200,\n        })\n'
        ),
        nbformat.v4.new_code_cell(
            'pilot = cohort.iloc[:200].copy()\n'
            'if len(pilot) != 200 or pilot.sample_id.duplicated().any():\n'
            '    raise RuntimeError("pilot label-free selection failed")\n'
        ),
        nbformat.v4.new_code_cell(
            'if False: raise RuntimeError("CodeParrot pilot requires CUDA/T4")\n'
            'runtime_identity = {}\n'
        ),
        nbformat.v4.new_code_cell(
            'private_input = scorer_frame(pilot)\n'
            'expected_internal_ids = {f"row-{index:06d}" for index in range(200)}\n'
            'sample_features = aggregate_private_features(window_features, 200)\n'
            'first8_private = scorer_frame(pilot.iloc[:8])\n'
            'predictions = pd.DataFrame({"sample_id": pilot.sample_id.to_numpy(), "content_sha256": pilot.content_sha256.to_numpy(), "language": pilot.language.to_numpy()})\n'
            'if len(predictions) != 200: raise RuntimeError("pilot prediction coverage failure")\n'
            'predictions.to_parquet(OUTPUT / "pilot_predictions.parquet", index=False)\n'
            'fidelity={"performance_metrics_computed": False, "membership_join_performed": False, "original_sample_id_passed_to_scorer": False}\n'
            'profile={"rows": 200, "status": "complete_label_blind_runtime_fidelity_pilot"}\n'
            'manifest={"schema_version": "codeparrot_fresh_pilot_manifest_v1", "experiment_id": EXPERIMENT_ID, "status": "sealed_label_blind_pilot_predictions", "selection":\n'
            '                "first_200_sample_ids_from_label_free_4000_prediction_input", "rows": 200, "performance_metrics_computed": False, "membership_join_performed": False, "competition_submission": False, "full_4000_scoring_authorized": False}\n'
            'for name,payload in (("pilot_fidelity.json", fidelity),("pilot_profile.json", profile),("pilot_manifest.json", manifest)): pass\n'
            'print({"rows": 200, "full_4000_scoring_authorized": False, "performance_metrics_computed": False, "membership_join_performed": False})\n'
        ),
    ]
    nbformat.write(notebook, root / "codeparrot-fresh-pilot-v1.ipynb")
    (root / "kernel-metadata.json").write_text(json.dumps({
        "id": PILOT_ID,
        "title": "CodeParrot Fresh Pilot V1",
        "code_file": "codeparrot-fresh-pilot-v1.ipynb",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "kernel_sources": ["renta0426/codeparrot-fresh-cohort-v1"],
        "dataset_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "keywords": [],
    }) + "\n", encoding="utf-8")


def self_test() -> None:
    with tempfile.TemporaryDirectory() as temp_value:
        root = Path(temp_value)
        _synthetic_bundle(root)
        target = patch_bundle(root)
        assert target.name == "codeparrot-fresh-full-v1.ipynb"
        patched = nbformat.read(target, as_version=4)
        combined = "\n".join(_source(cell) for cell in patched.cells if cell.cell_type == "code")
        assert 'EXPERIMENT_ID = "codeparrot-fresh-full-v1"' in combined
        assert 'private_input = scorer_frame(full)' in combined
        assert '"full_4000_scoring_authorized": True' in combined
        assert 'OUTPUT / "full_predictions.parquet"' in combined
        metadata = json.loads((root / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert metadata["id"] == TARGET_ID
        assert metadata["code_file"] == target.name
    print("CODEPARROT_FULL_PATCH_V1_SELF_TEST PASS scope=pilot-source-to-frozen-4000 labels=0 metrics=0 submission=0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.notebook_dir is None:
        parser.error("--notebook-dir is required")
    target = patch_bundle(args.notebook_dir)
    print(
        f"CODEPARROT_FULL_PATCH_V1 PASS notebook={target.name} rows=4000 "
        "labels=0 metrics=0 submission=0 full_authorized=1"
    )


if __name__ == "__main__":
    main()
