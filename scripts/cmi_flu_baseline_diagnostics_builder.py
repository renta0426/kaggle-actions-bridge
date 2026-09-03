#!/usr/bin/env python3
"""Build a single-file aggregate-only CMI-Flu baseline diagnostics Kaggle runner."""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import pathlib

EXPECTED_BASE_PACKAGE_SHA256 = "48ff4ed2eadec8b059b8d37fb677af1f249b6486bccfbd55dca2274cfc6f3dc3"
EXPECTED_BASE_SOURCE_COMMIT = "d6297c36366ab5c3ef49b9077c2357277f82a708"
EXPECTED_B2_MERGED_COMMIT = "802d93bac61b97844adf846199863c7ca9604ea1"
DIAGNOSTICS_CMI_COMMIT = "6cc984f6bd2e8e744795c14d13034def2c5dc9e7"
DIAGNOSTICS_BLOB_SHA = "6dd5e9f69095b5db63fded3913f22c0041a30f2a"
NEGATIVE_CONTROLS_BLOB_SHA = "250d255e9f5e768238143b14b89325f4ccb0f111"
REQUEST_ID = "20260903-cmi-flu-baseline-diagnostics-001"
COMPETITION = "cmi-flu-first-prediction-challenge"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-script", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def load_base(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("cmi_flu_b2_base", path)
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load B2 base script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    package = module.package_bytes()
    package_sha = hashlib.sha256(package).hexdigest()
    if package_sha != EXPECTED_BASE_PACKAGE_SHA256:
        raise SystemExit(f"unexpected B2 package SHA-256: {package_sha}")
    source_commit = str(getattr(module, "SOURCE_COMMIT", ""))
    if source_commit != EXPECTED_BASE_SOURCE_COMMIT:
        raise SystemExit(f"unexpected B2 generated source commit: {source_commit}")
    config_text = str(getattr(module, "CONFIG_TEXT", ""))
    if "baseline: b02_taskwise_compact" not in config_text or "verify_md5: true" not in config_text:
        raise SystemExit("unexpected B2 config contract")
    return package, config_text


def chunk_base64(data: bytes, width: int = 96) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return "\n".join(encoded[i:i+width] for i in range(0, len(encoded), width))


def build_script(package: bytes, config_text: str) -> str:
    template = r'''#!/usr/bin/env python3
"""Aggregate-only B1/negative-control diagnostics for CMI-Flu baseline v0.

This runner does not create or submit a competition prediction file.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

REQUEST_ID = "__REQUEST_ID__"
COMPETITION = "__COMPETITION__"
DIAGNOSTICS_CMI_COMMIT = "__DIAGNOSTICS_CMI_COMMIT__"
DIAGNOSTICS_BLOB_SHA = "__DIAGNOSTICS_BLOB_SHA__"
NEGATIVE_CONTROLS_BLOB_SHA = "__NEGATIVE_CONTROLS_BLOB_SHA__"
BASE_GENERATED_SOURCE_COMMIT = "__BASE_GENERATED_SOURCE_COMMIT__"
BASE_B2_MERGED_COMMIT = "__BASE_B2_MERGED_COMMIT__"
PACKAGE_ZIP_SHA256 = "__PACKAGE_ZIP_SHA256__"
PACKAGE_B64 = """__PACKAGE_B64__"""
CONFIG_TEXT = __CONFIG_TEXT__

CORE_FILES = (
    "participants.tsv",
    "investigations_260821.tsv",
    "publicData_cytokine.tsv",
    "publicData_ex_vivo_flow.tsv",
    "publicData_serology_260821.tsv",
    "2025LJI_aim.tsv",
    "2025LJI_cytokine.tsv",
    "2025LJI_ex_vivo_flow.tsv",
    "2025LJI_serology.tsv",
    "sample_submission_part1.csv",
    "md5sum",
)
REFERENCE_PLACEHOLDERS = {
    "cytokine_name_map.csv": "source,target\n",
    "flow_name_revised.csv": "source,target\n",
    "hai_map.csv": "source,target\n",
    "strain_sequences.csv": "virus_strain,sequence\n",
    "vaccine_strains_per_season.txt": "# Not used by aggregate diagnostics.\n",
}


class DiagnosticContractError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/cmi-flu-baseline-diagnostics"))
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def package_bytes() -> bytes:
    encoded = "".join(PACKAGE_B64.split())
    try:
        data = base64.b64decode(encoded, validate=True)
    except Exception as error:
        raise DiagnosticContractError("embedded package is not valid base64") from error
    if hashlib.sha256(data).hexdigest() != PACKAGE_ZIP_SHA256:
        raise DiagnosticContractError("embedded package SHA-256 mismatch")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        bad = archive.testzip()
        names = set(archive.namelist())
    if bad is not None:
        raise DiagnosticContractError("embedded package archive is corrupt")
    required = {
        "cmi_flu/runner.py",
        "cmi_flu/evaluation.py",
        "cmi_flu/negative_controls.py",
        "cmi_flu/features/flow.py",
        "cmi_flu/targets.py",
    }
    if required - names:
        raise DiagnosticContractError("embedded package lacks diagnostic dependencies")
    return data


def self_test() -> int:
    data = package_bytes()
    print(
        "CMI_FLU_BASELINE_DIAGNOSTICS_SELF_TEST PASS "
        f"request_id={REQUEST_ID} package_bytes={len(data)} package_sha256={PACKAGE_ZIP_SHA256}"
    )
    return 0


def locate_competition_data(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    exact = Path("/kaggle/input") / COMPETITION
    candidates.append(exact)
    root = Path("/kaggle/input")
    if root.is_dir():
        try:
            candidates.extend(sorted(p.parent for p in root.rglob("sample_submission_part1.csv")))
        except OSError:
            pass
    candidates.extend((Path.cwd(), Path.cwd() / "data" / "raw"))
    seen: set[Path] = set()
    valid: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        if all((resolved / name).is_file() for name in CORE_FILES):
            valid.append(resolved)
    if not valid:
        raise DiagnosticContractError("CMI-Flu Competition Data mount was not found")
    try:
        exact_resolved = exact.resolve()
    except OSError:
        exact_resolved = exact
    if exact_resolved in valid:
        return exact_resolved
    if explicit is not None:
        resolved_explicit = explicit.expanduser().resolve()
        if resolved_explicit in valid:
            return resolved_explicit
    unique = list(dict.fromkeys(valid))
    if len(unique) != 1:
        raise DiagnosticContractError("multiple candidate Competition Data directories were found")
    return unique[0]


def canonical_strain(value: object) -> str:
    raw = "" if value is None else str(value).strip()
    raw = re.sub(r"(?i)_cell$", "_MDCK", raw)
    raw = re.sub(r"(?i)\s+cell$", "_MDCK", raw)
    return raw


def vaccine_flag(value: object) -> bool:
    if value is None:
        return False
    try:
        import pandas as pd
        if pd.isna(value):
            return False
    except Exception:
        pass
    return str(value).strip().casefold() in {"1", "1.0", "true", "yes", "y"}


def derive_reference_files(input_dir: Path, reference_dir: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    import pandas as pd
    reference_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(
        input_dir / "2025LJI_serology.tsv",
        sep="\t",
        usecols=["assay", "virus_strain", "virus_in_vaccine"],
        low_memory=False,
    )
    hai = source.loc[source["assay"].fillna("").astype(str).str.strip().str.casefold().eq("hai")].copy()
    hai["virus_strain"] = hai["virus_strain"].map(canonical_strain)
    challenge = tuple(sorted(value for value in hai["virus_strain"].dropna().unique() if value))
    vaccine = tuple(sorted(
        value for value in hai.loc[hai["virus_in_vaccine"].map(vaccine_flag), "virus_strain"].dropna().unique() if value
    ))
    if len(challenge) != 12 or len(vaccine) != 3 or not set(vaccine).issubset(challenge):
        raise DiagnosticContractError("derived HAI panels differ from 3/12-strain contract")
    (reference_dir / "all_challenge_virus_strains.txt").write_text("\n".join(challenge) + "\n", encoding="utf-8")
    (reference_dir / "vaccine_strains_2025.txt").write_text("\n".join(vaccine) + "\n", encoding="utf-8")
    for name, content in REFERENCE_PLACEHOLDERS.items():
        (reference_dir / name).write_text(content, encoding="utf-8")
    return vaccine, challenge


def _json_safe(value: Any) -> Any:
    import numpy as np
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _records(frame) -> list[dict[str, Any]]:
    return _json_safe(frame.to_dict(orient="records"))


def _direction_from_task11_target(target) -> int:
    import numpy as np
    from cmi_flu.metrics import safe_spearman
    values: list[float] = []
    for _, group in target.groupby("study_accession", dropna=False, observed=True):
        metric = safe_spearman(group["pre_vacc"], group["target"])
        if metric.status == "ok" and metric.value != 0:
            values.append(metric.value)
    if values:
        median = float(np.median(values))
        if median != 0:
            return 1 if median > 0 else -1
        sign_sum = int(np.sign(values).sum())
        if sign_sum != 0:
            return 1 if sign_sum > 0 else -1
    pooled = safe_spearman(target["pre_vacc"], target["target"])
    if pooled.status == "ok" and pooled.value != 0:
        return 1 if pooled.value > 0 else -1
    return -1


def task11_b1_oof_diagnostic(public_cytokine) -> Mapping[str, Any]:
    import pandas as pd
    from cmi_flu.cv import purged_leave_one_study_out
    from cmi_flu.metrics import evaluate_predictions, grouped_metrics, signed_percentile_rank
    from cmi_flu.targets import build_task_11_target
    target = build_task_11_target(public_cytokine).reset_index(drop=True)
    splits = purged_leave_one_study_out(target["study_accession"], target["subject"])
    parts = []
    directions = {}
    for split in splits:
        train = target.iloc[split.train_indices]
        validation = target.iloc[split.validation_indices].copy()
        direction = _direction_from_task11_target(train)
        directions[split.name] = direction
        validation["prediction"] = signed_percentile_rank(validation["pre_vacc"], direction=direction)
        validation["split"] = split.name
        parts.append(validation[["split", "target", "prediction"]])
    oof = pd.concat(parts, ignore_index=True)
    return {
        "status": "ok",
        "method": "subject-purged leave-one-study-out; sign learned on training studies; held-out-study percentile rank",
        "pooled": evaluate_predictions(oof["target"], oof["prediction"]),
        "folds": _records(grouped_metrics(oof, group_columns=["split"], target_column="target", prediction_column="prediction")),
        "split_count": len(splits),
        "rows": int(len(oof)),
        "target_rows": int(len(target)),
        "studies": int(target["study_accession"].nunique()),
        "directions": directions,
    }


def _flow_anchor_diagnostic(public_flow, *, task: str, population: str, target, mode: str) -> Mapping[str, Any]:
    from cmi_flu.aliases import canonicalize_flow_population
    from cmi_flu.contracts import DataContractError
    from cmi_flu.features.flow import build_flow_baseline_features
    from cmi_flu.metrics import evaluate_predictions, grouped_metrics
    canonical = canonicalize_flow_population(population)
    column = f"flow_rank__{canonical}"
    try:
        features = build_flow_baseline_features(public_flow, populations=[canonical], mode=mode, include_study_rank=True)
    except DataContractError as error:
        return {"status": "not_evaluable", "mode": mode, "reason": str(error), "target_rows": int(len(target)), "matched_rows": 0, "coverage": 0.0}
    if column not in features:
        return {"status": "not_evaluable", "mode": mode, "reason": "rank feature unavailable", "target_rows": int(len(target)), "matched_rows": 0, "coverage": 0.0}
    merge_keys = [key for key in ("participant_id", "subject", "study_accession") if key in target and key in features]
    merged = target.merge(features[[*merge_keys, column]], on=merge_keys, how="inner", validate="one_to_one")
    if merged.empty:
        return {"status": "not_evaluable", "mode": mode, "reason": "zero target/anchor overlap", "target_rows": int(len(target)), "matched_rows": 0, "coverage": 0.0}
    merged = merged.rename(columns={column: "prediction"})
    folds = _records(grouped_metrics(merged, group_columns=["study_accession"], target_column="target", prediction_column="prediction")) if "study_accession" in merged else []
    return {
        "status": "ok", "mode": mode,
        "method": "label-free study/population rank anchor" if mode == "strict" else "label-free study/population/unit/material-stratified rank proxy",
        "pooled": evaluate_predictions(merged["target"], merged["prediction"]),
        "folds": folds,
        "target_rows": int(len(target)), "matched_rows": int(len(merged)),
        "coverage": float(len(merged) / len(target)) if len(target) else 0.0,
        "studies": int(merged["study_accession"].nunique()) if "study_accession" in merged else None,
        "task": task,
    }


def flow_b1_anchor_diagnostics(public_flow) -> Mapping[str, Any]:
    from cmi_flu.targets import build_task_12_target, build_task_13_target
    task12 = build_task_12_target(public_flow)
    task13 = build_task_13_target(public_flow, include_sdy272_asc_proxy=False)
    return {
        "Task1.2": {
            "exact_b1_strict": _flow_anchor_diagnostic(public_flow, task="Task1.2", population="Classical_monocytes", target=task12, mode="strict"),
            "broad_transfer_proxy": _flow_anchor_diagnostic(public_flow, task="Task1.2", population="Classical_monocytes", target=task12, mode="broad"),
        },
        "Task1.3": {
            "exact_b1_strict": _flow_anchor_diagnostic(public_flow, task="Task1.3", population="Antibody-secreting_cells_(ASC)", target=task13, mode="strict"),
            "broad_transfer_proxy": _flow_anchor_diagnostic(public_flow, task="Task1.3", population="Antibody-secreting_cells_(ASC)", target=task13, mode="broad"),
        },
    }


def _hai_b1_proxy(public_serology, *, task: str, day: int, panel_strains: Sequence[str]) -> Mapping[str, Any]:
    from cmi_flu.aliases import canonicalize_strain
    from cmi_flu.metrics import evaluate_predictions, grouped_metrics, percentile_rank
    from cmi_flu.targets import build_hai_long_target, geometric_mean
    paired = build_hai_long_target(public_serology, day=day).copy()
    panel = {canonicalize_strain(strain) for strain in panel_strains}
    paired["virus_strain"] = paired["virus_strain"].map(canonicalize_strain)
    working = paired.loc[paired["virus_strain"].isin(panel)].copy()
    if working.empty:
        return {"status": "not_evaluable", "reason": "no paired HAI rows overlap panel", "requested_strains": len(panel)}
    groups = [key for key in ("participant_id", "subject", "study_accession") if key in working]
    grouped = working.groupby(groups, dropna=False, observed=True).agg(
        anchor=("pre_hai", geometric_mean), target=("post_hai", geometric_mean), panel_size=("virus_strain", "nunique")
    ).reset_index()
    if "study_accession" in grouped:
        grouped["prediction"] = grouped.groupby("study_accession", dropna=False, observed=True)["anchor"].transform(
            lambda values: percentile_rank(values.to_numpy(dtype=float))
        )
        folds = _records(grouped_metrics(grouped, group_columns=["study_accession"], target_column="target", prediction_column="prediction"))
    else:
        grouped["prediction"] = percentile_rank(grouped["anchor"])
        folds = []
    return {
        "status": "proxy_only",
        "method": "same observed public panel subset for pre/post geometric means; study-local baseline rank",
        "warning": "public participants do not carry the complete Challenge panel; this is not the true task target",
        "pooled": evaluate_predictions(grouped["target"], grouped["prediction"]), "folds": folds,
        "participants": int(grouped["participant_id"].nunique()),
        "studies": int(grouped["study_accession"].nunique()) if "study_accession" in grouped else None,
        "requested_strains": len(panel), "available_strains": int(working["virus_strain"].nunique()),
        "panel_size_min": int(grouped["panel_size"].min()), "panel_size_max": int(grouped["panel_size"].max()), "task": task,
    }


def b1_anchor_diagnostics(inputs) -> Mapping[str, Any]:
    tables = inputs.tables
    flow = flow_b1_anchor_diagnostics(tables["public_flow"])
    return {
        "Task1.1": task11_b1_oof_diagnostic(tables["public_cytokine"]),
        "Task1.2": flow["Task1.2"], "Task1.3": flow["Task1.3"],
        "Task1.4": {"status": "not_evaluable", "reason": "no public AIM labels are available"},
        "Task2.1": _hai_b1_proxy(tables["public_serology"], task="Task2.1", day=28, panel_strains=inputs.vaccine_strains),
        "Task2.2": _hai_b1_proxy(tables["public_serology"], task="Task2.2", day=28, panel_strains=inputs.challenge_strains),
        "Task2.3": _hai_b1_proxy(tables["public_serology"], task="Task2.3", day=365, panel_strains=inputs.challenge_strains),
    }


def _compact_task_control(dataset, *, allowed: Sequence[str], name: str, transform: str, random_state: int) -> Mapping[str, Any]:
    from cmi_flu.evaluation import run_compact_task
    from cmi_flu.negative_controls import restricted_ridge_spec
    spec = restricted_ridge_spec(dataset, name=name, allowed_features=allowed, target_transform=transform)
    result = run_compact_task(dataset, specs=[spec], random_state=random_state)
    return {"model": result.selected_spec.to_dict(), "metrics": result.metrics}


def negative_control_diagnostics(config, inputs) -> Mapping[str, Any]:
    from cmi_flu.evaluation import run_hai_compact_for_panels
    from cmi_flu.negative_controls import DEMOGRAPHIC_FEATURES, HAI_CONTEXT_FEATURES, direct_hai_control_dataset, restricted_ridge_spec
    from cmi_flu.runner import build_b02_datasets
    datasets = build_b02_datasets(config, inputs)
    results: dict[str, Any] = {}
    for task, transform in (("Task1.1", "log"), ("Task1.2", "log1p"), ("Task1.3", "log1p")):
        dataset = datasets[task]
        results[task] = {
            "age_only": _compact_task_control(dataset, allowed=("age", "age_missing"), name="age_only_ridge", transform=transform, random_state=config.random_state),
            "demographics_only": _compact_task_control(dataset, allowed=DEMOGRAPHIC_FEATURES, name="demographics_only_ridge", transform=transform, random_state=config.random_state),
        }
    d28 = direct_hai_control_dataset(datasets["HAI_D28"])
    d365 = direct_hai_control_dataset(datasets["HAI_D365"])
    spec28 = restricted_ridge_spec(d28, name="demographics_plus_strain_context_ridge", allowed_features=HAI_CONTEXT_FEATURES, target_transform="identity", clip_min=None)
    spec365 = restricted_ridge_spec(d365, name="demographics_plus_strain_context_ridge", allowed_features=HAI_CONTEXT_FEATURES, target_transform="identity", clip_min=None)
    d28_results = run_hai_compact_for_panels(d28, specs=[spec28], selection_panels={"Task2.1": inputs.vaccine_strains, "Task2.2": inputs.challenge_strains})
    d365_results = run_hai_compact_for_panels(d365, specs=[spec365], selection_panels={"Task2.3": inputs.challenge_strains})
    for task, result in {**d28_results, **d365_results}.items():
        results[task] = {"demographics_plus_strain_context": {"model": result.selected_spec.to_dict(), "metrics": result.metrics}}
    results["Task1.4"] = {"status": "not_evaluable", "reason": "no public AIM labels are available"}
    return results


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metric_value(payload: Mapping[str, Any], *keys: str) -> float | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    try:
        value = float(current)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def render_summary(payload: Mapping[str, Any]) -> str:
    b1 = payload["b1_anchor_diagnostics"]
    controls = payload["negative_controls"]
    lines = [
        "# CMI-Flu baseline v0 diagnostics", "",
        "Aggregate-only diagnostic run; no Kaggle submission was created.", "", "## B1 anchors", "",
    ]
    for task in ("Task1.1", "Task1.2", "Task1.3", "Task1.4", "Task2.1", "Task2.2", "Task2.3"):
        item = b1[task]
        if task in {"Task1.2", "Task1.3"}:
            exact = item["exact_b1_strict"]
            broad = item["broad_transfer_proxy"]
            lines.append(f"- {task}: strict status={exact.get('status')}, strict Spearman={metric_value(exact, 'pooled', 'spearman', 'value')}, broad Spearman={metric_value(broad, 'pooled', 'spearman', 'value')}")
        else:
            lines.append(f"- {task}: status={item.get('status')}, Spearman={metric_value(item, 'pooled', 'spearman', 'value')}")
    lines.extend(["", "## Negative controls", ""])
    for task in ("Task1.1", "Task1.2", "Task1.3"):
        item = controls[task]
        age = metric_value(item["age_only"], "metrics", "pooled", "spearman", "value")
        demo = metric_value(item["demographics_only"], "metrics", "pooled", "spearman", "value")
        lines.append(f"- {task}: age-only Spearman={age}, demographics-only Spearman={demo}")
    for task in ("Task2.1", "Task2.2", "Task2.3"):
        item = controls[task]["demographics_plus_strain_context"]
        score = metric_value(item, "metrics", "panel_proxy", "spearman", "value")
        lines.append(f"- {task}: demographics+strain-context panel-proxy Spearman={score}")
    lines.extend(["", "Task1.4 negative control is not evaluable because public AIM labels are unavailable.", ""])
    return "\n".join(lines)


def safe_failure(output_dir: Path, *, stage: str, error: BaseException) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    code = hashlib.sha256(f"{type(error).__name__}:{str(error)}".encode("utf-8", errors="replace")).hexdigest()[:20]
    payload = {
        "schema_version": 1, "request_id": REQUEST_ID, "competition": COMPETITION,
        "stage": stage, "exception_type": type(error).__name__, "error_code": code,
    }
    (output_dir / "bridge-failure.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"CMI_FLU_BASELINE_DIAGNOSTICS_FAILED stage={stage} exception_type={type(error).__name__} error_code={code}")


def execute(input_dir: Path, output_dir: Path) -> int:
    stage = "initialize"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_root = output_dir / "runtime"
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True)
    try:
        stage = "materialize_package"
        package = package_bytes()
        package_path = runtime_root / "cmi_flu_bundle.zip"
        package_path.write_bytes(package)
        sys.path.insert(0, str(package_path))

        stage = "prepare_runtime_tree"
        data_parent = runtime_root / "data"
        data_parent.mkdir(parents=True, exist_ok=True)
        (data_parent / "raw").symlink_to(input_dir, target_is_directory=True)
        reference_dir = runtime_root / "external" / "google-drive" / "challenge-resources" / "reference_files"
        vaccine_panel, challenge_panel = derive_reference_files(input_dir, reference_dir)
        config_dir = runtime_root / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "baseline_b02_taskwise.yaml"
        config_path.write_text(CONFIG_TEXT, encoding="utf-8")

        stage = "dependency_preflight"
        import joblib, numpy, pandas, scipy, sklearn, yaml
        _ = (joblib.__version__, numpy.__version__, pandas.__version__, scipy.__version__, sklearn.__version__, yaml.__version__)

        stage = "load_inputs"
        from cmi_flu.configuration import load_baseline_config
        from cmi_flu.runner import load_inputs
        config = load_baseline_config(config_path, repository_root=runtime_root)
        if config.baseline != "b02_taskwise_compact" or not config.verify_md5:
            raise DiagnosticContractError("B2 diagnostic config contract mismatch")
        inputs = load_inputs(config)
        if inputs.checksum_report is None or len(inputs.checksum_report.verified) != 28:
            raise DiagnosticContractError("MD5 verification count mismatch")
        if len(vaccine_panel) != 3 or len(challenge_panel) != 12:
            raise DiagnosticContractError("HAI panel contract mismatch")

        stage = "b1_anchor_diagnostics"
        b1 = b1_anchor_diagnostics(inputs)
        stage = "negative_controls"
        controls = negative_control_diagnostics(config, inputs)

        stage = "write_outputs"
        payload = _json_safe({
            "schema_version": 1,
            "request_id": REQUEST_ID,
            "competition": COMPETITION,
            "purpose": "aggregate baseline diagnostics; not a Kaggle submission candidate",
            "provenance": {
                "diagnostics_cmi_commit": DIAGNOSTICS_CMI_COMMIT,
                "diagnostics_blob_sha": DIAGNOSTICS_BLOB_SHA,
                "negative_controls_blob_sha": NEGATIVE_CONTROLS_BLOB_SHA,
                "base_generated_source_commit": BASE_GENERATED_SOURCE_COMMIT,
                "base_b2_merged_commit": BASE_B2_MERGED_COMMIT,
                "base_package_sha256": PACKAGE_ZIP_SHA256,
            },
            "checksum": {"verified_count": len(inputs.checksum_report.verified), "skipped": list(inputs.checksum_report.skipped)},
            "panels": {"vaccine_strains": len(inputs.vaccine_strains), "challenge_strains": len(inputs.challenge_strains)},
            "b1_anchor_diagnostics": b1,
            "negative_controls": controls,
        })
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary_path = output_dir / "summary.md"
        summary_path.write_text(render_summary(payload), encoding="utf-8")
        result = {
            "schema_version": 1,
            "request_id": REQUEST_ID,
            "competition": COMPETITION,
            "purpose": payload["purpose"],
            "md5_verified": len(inputs.checksum_report.verified),
            "vaccine_strains": len(inputs.vaccine_strains),
            "challenge_strains": len(inputs.challenge_strains),
            "metrics_sha256": sha256_file(metrics_path),
            "summary_sha256": sha256_file(summary_path),
            "submission_created": False,
            "submission_attempted": False,
        }
        (output_dir / "bridge-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        shutil.rmtree(runtime_root)
        print(
            "CMI_FLU_BASELINE_DIAGNOSTICS_PASS "
            f"request_id={REQUEST_ID} md5_verified={result['md5_verified']} submission_created=false submission_attempted=false"
        )
        return 0
    except Exception as error:
        if runtime_root.exists():
            shutil.rmtree(runtime_root, ignore_errors=True)
        safe_failure(output_dir, stage=stage, error=error)
        return 2


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    try:
        input_dir = locate_competition_data(args.input_dir)
    except Exception as error:
        safe_failure(args.output_dir, stage="locate_competition_data", error=error)
        return 2
    return execute(input_dir, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
'''
    replacements = {
        "__REQUEST_ID__": REQUEST_ID,
        "__COMPETITION__": COMPETITION,
        "__DIAGNOSTICS_CMI_COMMIT__": DIAGNOSTICS_CMI_COMMIT,
        "__DIAGNOSTICS_BLOB_SHA__": DIAGNOSTICS_BLOB_SHA,
        "__NEGATIVE_CONTROLS_BLOB_SHA__": NEGATIVE_CONTROLS_BLOB_SHA,
        "__BASE_GENERATED_SOURCE_COMMIT__": EXPECTED_BASE_SOURCE_COMMIT,
        "__BASE_B2_MERGED_COMMIT__": EXPECTED_B2_MERGED_COMMIT,
        "__PACKAGE_ZIP_SHA256__": hashlib.sha256(package).hexdigest(),
        "__PACKAGE_B64__": chunk_base64(package),
        "__CONFIG_TEXT__": repr(config_text),
    }
    for old, new in replacements.items():
        if old not in template:
            raise SystemExit(f"template placeholder missing: {old}")
        template = template.replace(old, new)
    compile(template, "generated_diagnostics.py", "exec")
    return template


def main() -> int:
    args = parse_args()
    package, config_text = load_base(args.base_script)
    script = build_script(package, config_text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(script, encoding="utf-8")
    print(
        "CMI_FLU_BASELINE_DIAGNOSTICS_BUILD PASS "
        f"bytes={len(script.encode('utf-8'))} sha256={hashlib.sha256(script.encode('utf-8')).hexdigest()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
