#!/usr/bin/env python3
"""Apply the locked CMI-Flu B2.1 evaluation contract to an extracted B2 package.

This file contains code-only transformations.  It never reads Competition Data.
The resulting runtime-critical source files are verified against Git blob IDs from
the reviewed CMI-Flu B2.1 implementation before they are allowed to run.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

TARGET_BLOBS = {
    "src/cmi_flu/models.py": "f92ac4dff997e38122b5bf7ba5ff5202d283b213",
    "src/cmi_flu/evaluation.py": "0d48febd76b950bd902cbcdbe0fa309a68da12e5",
    "src/cmi_flu/runner.py": "4557377a0025047c45e6aa02618ccc2d93a3bc97",
    "src/cmi_flu/configuration.py": "565201a54c0cb299cfdf1b30429406ac27676579",
}
TARGET_CONFIG_BLOB = "170d3211e2795c0730e481056c7bb068accf97c9"


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one patch anchor, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def replace_between(path: Path, start: str, end: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    left = text.find(start)
    if left < 0:
        raise SystemExit(f"{path}: start anchor missing: {start}")
    right = text.find(end, left + len(start))
    if right < 0:
        raise SystemExit(f"{path}: end anchor missing: {end}")
    path.write_text(text[:left] + replacement + text[right:], encoding="utf-8")


def patch_models(root: Path) -> None:
    path = root / "src/cmi_flu/models.py"
    replace_once(
        path,
        '''@dataclass
class CandidateEvaluation:
    spec: ModelSpec
    oof_predictions: pd.DataFrame
    pooled_metrics: Mapping[str, Any]
    transformed_metrics: Mapping[str, Any]
    fold_metrics: pd.DataFrame
''',
        '''@dataclass
class CandidateEvaluation:
    spec: ModelSpec
    oof_predictions: pd.DataFrame
    pooled_metrics: Mapping[str, Any]
    transformed_metrics: Mapping[str, Any]
    fold_metrics: pd.DataFrame
    raw_oof_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    fold_summary: Mapping[str, Any] = field(default_factory=dict)
    repeat_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
''',
    )

    marker = "\ndef evaluate_model_spec(\n"
    insert = r'''
def aggregate_repeated_oof(oof: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated-CV OOF rows to one prediction per original row."""

    require_columns(
        oof,
        [
            "row_index",
            "target",
            "prediction",
            "target_transformed",
            "prediction_transformed",
        ],
        table_name="OOF predictions",
    )
    counts = oof.groupby("row_index", observed=True).size()
    if counts.empty:
        raise DataContractError("OOF predictions are empty")
    if int(counts.max()) == 1:
        result = oof.copy()
        result["repeat_count"] = 1
        return result

    for column in ("target", "target_transformed"):
        spread = oof.groupby("row_index", observed=True)[column].agg(["min", "max"])
        if not np.allclose(
            spread["min"].to_numpy(dtype=float),
            spread["max"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise DataContractError(f"repeated OOF rows disagree on {column}")

    result = (
        oof.groupby("row_index", as_index=False, observed=True)
        .agg(
            target=("target", "first"),
            prediction=("prediction", "mean"),
            target_transformed=("target_transformed", "first"),
            prediction_transformed=("prediction_transformed", "mean"),
            repeat_count=("prediction", "size"),
        )
        .sort_values("row_index")
        .reset_index(drop=True)
    )
    result["split"] = "aggregated_repeats"
    result["held_out_group"] = None
    return result[
        [
            "row_index",
            "split",
            "held_out_group",
            "target",
            "prediction",
            "target_transformed",
            "prediction_transformed",
            "repeat_count",
        ]
    ]


def repeat_metrics_from_oof(oof: pd.DataFrame) -> pd.DataFrame:
    """Evaluate each complete repeat after concatenating its grouped folds."""

    if "split" not in oof:
        return pd.DataFrame()
    repeat = oof["split"].astype(str).str.extract(
        r"^repeat=(\d+)/fold=\d+$", expand=False
    )
    if repeat.isna().all():
        return pd.DataFrame()
    working = oof.loc[repeat.notna()].copy()
    working["repeat"] = repeat.loc[repeat.notna()].astype(int).to_numpy()
    per_repeat_counts = working.groupby(["repeat", "row_index"], observed=True).size()
    if int(per_repeat_counts.max()) != 1:
        raise DataContractError(
            "a repeated-CV repeat predicts an original row more than once"
        )
    return grouped_metrics(
        working,
        group_columns=["repeat"],
        target_column="target",
        prediction_column="prediction",
    )


def summarize_metric_frame(frame: pd.DataFrame) -> dict[str, float | int | None]:
    """Summarize folds/studies without weighting large cohorts more heavily."""

    if frame.empty:
        return {
            "count": 0,
            "spearman_mean": None,
            "spearman_median": None,
            "spearman_min": None,
            "spearman_max": None,
            "spearman_std": None,
            "rmse_mean": None,
            "rmse_median": None,
            "rmse_max": None,
            "rmse_std": None,
        }
    spearman = pd.to_numeric(frame["spearman"], errors="coerce").to_numpy(dtype=float)
    rmse = pd.to_numeric(frame["rmse"], errors="coerce").to_numpy(dtype=float)
    spearman = spearman[np.isfinite(spearman)]
    rmse = rmse[np.isfinite(rmse)]
    return {
        "count": int(spearman.size),
        "spearman_mean": float(np.mean(spearman)) if spearman.size else None,
        "spearman_median": float(np.median(spearman)) if spearman.size else None,
        "spearman_min": float(np.min(spearman)) if spearman.size else None,
        "spearman_max": float(np.max(spearman)) if spearman.size else None,
        "spearman_std": float(np.std(spearman)) if spearman.size else None,
        "rmse_mean": float(np.mean(rmse)) if rmse.size else None,
        "rmse_median": float(np.median(rmse)) if rmse.size else None,
        "rmse_max": float(np.max(rmse)) if rmse.size else None,
        "rmse_std": float(np.std(rmse)) if rmse.size else None,
    }

'''
    replace_once(path, marker, insert + marker)

    replacement = '''def evaluate_model_spec(
    frame: pd.DataFrame,
    *,
    target_column: str,
    splits: Sequence[NamedSplit],
    spec: ModelSpec,
    excluded_columns: Sequence[str] = (
        "participant_id",
        "subject",
        "study_accession",
        "target",
        "target_log",
    ),
    aggregate_repeats: bool = False,
) -> CandidateEvaluation:
    raw_oof = fit_predict_oof(
        frame,
        target_column=target_column,
        splits=splits,
        spec=spec,
        excluded_columns=excluded_columns,
    )
    oof = aggregate_repeated_oof(raw_oof) if aggregate_repeats else raw_oof.copy()
    pooled = evaluate_predictions(oof["target"], oof["prediction"])
    transformed = evaluate_predictions(
        oof["target_transformed"], oof["prediction_transformed"]
    )
    folds = grouped_metrics(
        raw_oof,
        group_columns=["split"],
        target_column="target",
        prediction_column="prediction",
    )
    repeat_metrics = repeat_metrics_from_oof(raw_oof)
    return CandidateEvaluation(
        spec=spec,
        oof_predictions=oof,
        pooled_metrics=pooled,
        transformed_metrics=transformed,
        fold_metrics=folds,
        raw_oof_predictions=raw_oof,
        fold_summary=summarize_metric_frame(folds),
        repeat_metrics=repeat_metrics,
    )


'''
    replace_between(path, "def evaluate_model_spec(\n", "def evaluate_candidates(\n", replacement)
    replace_once(
        path,
        '''def evaluate_candidates(
    frame: pd.DataFrame,
    *,
    target_column: str,
    splits: Sequence[NamedSplit],
    specs: Iterable[ModelSpec],
    excluded_columns: Sequence[str] = (
        "participant_id",
        "subject",
        "study_accession",
        "target",
        "target_log",
    ),
) -> CandidateSearchResult:
''',
        '''def evaluate_candidates(
    frame: pd.DataFrame,
    *,
    target_column: str,
    splits: Sequence[NamedSplit],
    specs: Iterable[ModelSpec],
    excluded_columns: Sequence[str] = (
        "participant_id",
        "subject",
        "study_accession",
        "target",
        "target_log",
    ),
    aggregate_repeats: bool = False,
) -> CandidateSearchResult:
''',
    )
    text = path.read_text(encoding="utf-8")
    location = text.index("def evaluate_candidates(")
    prefix, suffix = text[:location], text[location:]
    old = '''                    spec=spec,
                    excluded_columns=excluded_columns,
                )
'''
    if suffix.count(old) != 1:
        raise SystemExit(f"{path}: evaluate_candidates call anchor count={suffix.count(old)}")
    suffix = suffix.replace(
        old,
        '''                    spec=spec,
                    excluded_columns=excluded_columns,
                    aggregate_repeats=aggregate_repeats,
                )
''',
        1,
    )
    path.write_text(prefix + suffix, encoding="utf-8")


def patch_evaluation(root: Path) -> None:
    path = root / "src/cmi_flu/evaluation.py"
    replace_once(
        path,
        "from .metrics import evaluate_predictions, within_group_rank_spearman\n",
        "from .metrics import evaluate_predictions, grouped_metrics, within_group_rank_spearman\n",
    )
    replace_once(
        path,
        '''    fit_final_model,
)
''',
        '''    fit_final_model,
    summarize_metric_frame,
)
''',
    )
    replace_once(
        path,
        '''class HAICandidateEvaluation:
    base: CandidateEvaluation
    post_metrics: Mapping[str, Any]
    within_strain_spearman: Mapping[str, Any]
    panel_proxy_metrics: Mapping[str, Any]
    panel_proxy_coverage: Mapping[str, Any]
    enriched_oof: pd.DataFrame
''',
        '''class HAICandidateEvaluation:
    base: CandidateEvaluation
    post_metrics: Mapping[str, Any]
    within_strain_spearman: Mapping[str, Any]
    panel_proxy_metrics: Mapping[str, Any]
    panel_proxy_coverage: Mapping[str, Any]
    panel_proxy_fold_metrics: Sequence[Mapping[str, Any]]
    panel_proxy_fold_summary: Mapping[str, Any]
    enriched_oof: pd.DataFrame
''',
    )

    marker = "\ndef run_compact_task(\n"
    helpers = r'''
def _finite_or(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if np.isfinite(number) else fallback


def _task11_oof_plausibility(evaluation: CandidateEvaluation) -> dict[str, Any]:
    oof = evaluation.oof_predictions
    target = oof["target"].to_numpy(dtype=float)
    target_transformed = oof["target_transformed"].to_numpy(dtype=float)
    prediction_transformed = oof["prediction_transformed"].to_numpy(dtype=float)
    target_std = float(np.std(target))
    rmse = float(evaluation.pooled_metrics["rmse"]["value"])
    rmse_std_ratio = rmse / target_std if target_std > 1e-12 else float("inf")
    low = float(np.min(target_transformed))
    high = float(np.max(target_transformed))
    span = max(high - low, 1e-12)
    allowed_low = low - 2.0 * span
    allowed_high = high + 2.0 * span
    pred_low = float(np.min(prediction_transformed))
    pred_high = float(np.max(prediction_transformed))
    passed = bool(
        np.isfinite(rmse_std_ratio)
        and rmse_std_ratio <= 10.0
        and pred_low >= allowed_low
        and pred_high <= allowed_high
    )
    return {
        "passed": passed,
        "rmse_std_ratio": rmse_std_ratio,
        "rmse_std_ratio_max": 10.0,
        "prediction_transformed_min": pred_low,
        "prediction_transformed_max": pred_high,
        "allowed_transformed_min": allowed_low,
        "allowed_transformed_max": allowed_high,
        "range_expansion": 2.0,
    }


def _task11_challenge_plausibility(
    dataset: TaskDataset, prediction: np.ndarray
) -> dict[str, Any]:
    values = np.asarray(prediction, dtype=float)
    historical = pd.to_numeric(
        dataset.train[dataset.target_column], errors="coerce"
    ).to_numpy(dtype=float)
    historical_max = float(np.max(historical))
    prediction_max = float(np.max(values))
    prediction_min = float(np.min(values))
    max_ratio = prediction_max / historical_max if historical_max > 0 else float("inf")
    passed = bool(
        np.isfinite(values).all()
        and prediction_min >= 0.0
        and np.isfinite(max_ratio)
        and max_ratio <= 10.0
    )
    return {
        "passed": passed,
        "prediction_min": prediction_min,
        "prediction_max": prediction_max,
        "historical_target_max": historical_max,
        "prediction_max_target_ratio": max_ratio,
        "prediction_max_target_ratio_limit": 10.0,
    }


def _rank_compact_candidates(
    dataset: TaskDataset,
    search: CandidateSearchResult,
    *,
    selection_policy: str,
) -> tuple[list[CandidateEvaluation], dict[str, Mapping[str, Any]]]:
    if selection_policy == "legacy":
        return [search.best], {
            search.best.spec.name: {"eligible": True, "policy": "legacy"}
        }
    if selection_policy != "robust_v1":
        raise DataContractError(f"unknown compact selection policy: {selection_policy}")

    diagnostics: dict[str, Mapping[str, Any]] = {}
    eligible: list[CandidateEvaluation] = []
    for evaluation in search.evaluations:
        detail: dict[str, Any] = {"eligible": True, "policy": selection_policy}
        if dataset.task == "Task1.1":
            plausibility = _task11_oof_plausibility(evaluation)
            detail["oof_plausibility"] = plausibility
            if not plausibility["passed"]:
                detail["eligible"] = False
        diagnostics[evaluation.spec.name] = detail
        if detail["eligible"]:
            eligible.append(evaluation)
    if not eligible:
        raise DataContractError(
            f"{dataset.task} has no candidate passing robust selection gates"
        )

    multi_study = int(dataset.train["study_group"].nunique()) >= 2
    if multi_study:
        def key(evaluation: CandidateEvaluation) -> tuple[float, float, float, float, float, str]:
            summary = evaluation.fold_summary
            return (
                _finite_or(summary.get("spearman_mean"), -np.inf),
                _finite_or(summary.get("spearman_median"), -np.inf),
                _finite_or(summary.get("spearman_min"), -np.inf),
                _finite_or(evaluation.pooled_metrics["spearman"]["value"], -np.inf),
                -_finite_or(evaluation.pooled_metrics["rmse"]["value"], np.inf),
                evaluation.spec.name,
            )
    else:
        def key(evaluation: CandidateEvaluation) -> tuple[float, float, str]:
            return (
                _finite_or(evaluation.pooled_metrics["spearman"]["value"], -np.inf),
                -_finite_or(evaluation.pooled_metrics["rmse"]["value"], np.inf),
                evaluation.spec.name,
            )
    return sorted(eligible, key=key, reverse=True), diagnostics

'''
    replace_once(path, marker, helpers + marker)

    compact = '''def run_compact_task(
    dataset: TaskDataset,
    *,
    specs: Sequence[ModelSpec],
    splits: Sequence[NamedSplit] | None = None,
    random_state: int = 42,
    selection_policy: str = "legacy",
) -> TaskRunResult:
    dataset.validate()
    chosen_splits = list(splits or default_splits_for_task(dataset, random_state=random_state))
    search = evaluate_candidates(
        dataset.train,
        target_column=dataset.target_column,
        splits=chosen_splits,
        specs=specs,
        excluded_columns=dataset.excluded_columns,
        aggregate_repeats=selection_policy == "robust_v1",
    )
    ranked, selection_diagnostics = _rank_compact_candidates(
        dataset, search, selection_policy=selection_policy
    )

    selected: CandidateEvaluation | None = None
    fitted_model: Any | None = None
    challenge_prediction: np.ndarray | None = None
    for evaluation in ranked:
        candidate_model, candidate_prediction = fit_final_model(
            dataset.train,
            dataset.challenge,
            target_column=dataset.target_column,
            spec=evaluation.spec,
            excluded_columns=dataset.excluded_columns,
        )
        if selection_policy == "robust_v1" and dataset.task == "Task1.1":
            challenge_check = _task11_challenge_plausibility(
                dataset, candidate_prediction
            )
            selection_diagnostics[evaluation.spec.name] = {
                **selection_diagnostics[evaluation.spec.name],
                "challenge_plausibility": challenge_check,
            }
            if not challenge_check["passed"]:
                continue
        selected = evaluation
        fitted_model = candidate_model
        challenge_prediction = candidate_prediction
        break
    if selected is None or fitted_model is None or challenge_prediction is None:
        raise DataContractError(
            f"{dataset.task} has no candidate passing final prediction guardrails"
        )

    challenge = dataset.challenge[["participant_id"]].copy()
    challenge["prediction"] = challenge_prediction
    summaries = [
        {
            "model": evaluation.spec.to_dict(),
            "pooled_metrics": evaluation.pooled_metrics,
            "transformed_metrics": evaluation.transformed_metrics,
            "fold_summary": evaluation.fold_summary,
            "repeat_metrics": evaluation.repeat_metrics.to_dict(orient="records"),
            "raw_oof_rows": int(len(evaluation.raw_oof_predictions)),
            "top_line_oof_rows": int(len(evaluation.oof_predictions)),
            "selection_diagnostics": selection_diagnostics.get(
                evaluation.spec.name, {}
            ),
        }
        for evaluation in search.evaluations
    ]
    return TaskRunResult(
        task=dataset.task,
        selected_spec=selected.spec,
        challenge_predictions=challenge,
        oof_predictions=selected.oof_predictions,
        metrics={
            "pooled": selected.pooled_metrics,
            "transformed": selected.transformed_metrics,
            "fold_summary": selected.fold_summary,
            "repeat_metrics": selected.repeat_metrics.to_dict(orient="records"),
            "raw_oof_rows": int(len(selected.raw_oof_predictions)),
            "top_line_oof_rows": int(len(selected.oof_predictions)),
            "selection_policy": selection_policy,
            "selection_diagnostics": selection_diagnostics.get(
                selected.spec.name, {}
            ),
            "split_count": len(chosen_splits),
            "training_rows": len(dataset.train),
            "training_subjects": int(dataset.train["subject_group"].nunique()),
            "training_studies": int(dataset.train["study_group"].nunique()),
        },
        fold_metrics=selected.fold_metrics,
        candidate_summaries=summaries,
        candidate_failures=search.failures,
        fitted_model=fitted_model,
    )


'''
    replace_between(path, "def run_compact_task(\n", "def _post_hai_metrics(\n", compact)

    replace_once(
        path,
        '''        return {
            "metrics": evaluate_predictions([], []),
            "participants": 0,
''',
        '''        return {
            "metrics": evaluate_predictions([], []),
            "fold_metrics": [],
            "fold_summary": summarize_metric_frame(pd.DataFrame()),
            "participants": 0,
''',
    )
    replace_once(
        path,
        '''    return {
        "metrics": evaluate_predictions(grouped["panel_target"], grouped["panel_prediction"]),
        "participants": int(grouped["participant_id"].nunique()),
''',
        '''    folds = grouped_metrics(
        grouped,
        group_columns=["split"],
        target_column="panel_target",
        prediction_column="panel_prediction",
    )
    return {
        "metrics": evaluate_predictions(grouped["panel_target"], grouped["panel_prediction"]),
        "fold_metrics": folds.to_dict(orient="records"),
        "fold_summary": summarize_metric_frame(folds),
        "participants": int(grouped["participant_id"].nunique()),
''',
    )
    replace_once(
        path,
        '''    coverage = {key: value for key, value in panel_proxy.items() if key != "metrics"}
''',
        '''    coverage = {
        key: value
        for key, value in panel_proxy.items()
        if key not in {"metrics", "fold_metrics", "fold_summary"}
    }
''',
    )
    replace_once(
        path,
        '''        panel_proxy_metrics=panel_proxy["metrics"],
        panel_proxy_coverage=coverage,
        enriched_oof=oof,
''',
        '''        panel_proxy_metrics=panel_proxy["metrics"],
        panel_proxy_coverage=coverage,
        panel_proxy_fold_metrics=panel_proxy["fold_metrics"],
        panel_proxy_fold_summary=panel_proxy["fold_summary"],
        enriched_oof=oof,
''',
    )

    select_hai = '''def _select_hai_candidate(
    evaluations: Sequence[HAICandidateEvaluation],
    *,
    selection_policy: str = "legacy",
) -> HAICandidateEvaluation:
    if not evaluations:
        raise DataContractError("no valid HAI candidates remain")
    if selection_policy not in {"legacy", "robust_v1"}:
        raise DataContractError(f"unknown HAI selection policy: {selection_policy}")

    if selection_policy == "legacy":
        def key(evaluation: HAICandidateEvaluation) -> tuple[float, float, float, str]:
            return (
                _finite_or(evaluation.panel_proxy_metrics["spearman"]["value"], -np.inf),
                _finite_or(evaluation.within_strain_spearman["value"], -np.inf),
                -_finite_or(evaluation.post_metrics["rmse"]["value"], np.inf),
                evaluation.base.spec.name,
            )
    else:
        def key(evaluation: HAICandidateEvaluation) -> tuple[float, float, float, float, float, float, str]:
            summary = evaluation.panel_proxy_fold_summary
            return (
                _finite_or(summary.get("spearman_mean"), -np.inf),
                _finite_or(summary.get("spearman_median"), -np.inf),
                _finite_or(summary.get("spearman_min"), -np.inf),
                _finite_or(evaluation.panel_proxy_metrics["spearman"]["value"], -np.inf),
                _finite_or(evaluation.within_strain_spearman["value"], -np.inf),
                -_finite_or(evaluation.post_metrics["rmse"]["value"], np.inf),
                evaluation.base.spec.name,
            )
    return max(evaluations, key=key)


'''
    replace_between(path, "def _select_hai_candidate(\n", "def run_hai_multitask(\n", select_hai)
    replace_once(path, '''    panel: Sequence[str],
) -> HAIRunResult:
''', '''    panel: Sequence[str],
    selection_policy: str = "legacy",
) -> HAIRunResult:
''')
    replace_once(path, "    selected = _select_hai_candidate(enriched)\n", "    selected = _select_hai_candidate(enriched, selection_policy=selection_policy)\n")
    replace_once(path, '''            "panel_proxy_coverage": evaluation.panel_proxy_coverage,
''', '''            "panel_proxy_coverage": evaluation.panel_proxy_coverage,
            "panel_proxy_fold_metrics": evaluation.panel_proxy_fold_metrics,
            "panel_proxy_fold_summary": evaluation.panel_proxy_fold_summary,
''')
    replace_once(path, '''            "panel_proxy_coverage": selected.panel_proxy_coverage,
            "selection_panel_strains": list(panel),
''', '''            "panel_proxy_coverage": selected.panel_proxy_coverage,
            "panel_proxy_fold_metrics": selected.panel_proxy_fold_metrics,
            "panel_proxy_fold_summary": selected.panel_proxy_fold_summary,
            "selection_policy": selection_policy,
            "selection_panel_strains": list(panel),
''')
    replace_once(path, '''    splits: Sequence[NamedSplit] | None = None,
) -> dict[str, HAIRunResult]:
''', '''    splits: Sequence[NamedSplit] | None = None,
    selection_policy: str = "legacy",
) -> dict[str, HAIRunResult]:
''')
    replace_once(path, '''            panel=panel,
        )
''', '''            panel=panel,
            selection_policy=selection_policy,
        )
''')
    replace_once(path, '''    selection_panel_strains: Sequence[str] | None = None,
) -> HAIRunResult:
''', '''    selection_panel_strains: Sequence[str] | None = None,
    selection_policy: str = "legacy",
) -> HAIRunResult:
''')
    replace_once(path, '''        splits=splits,
    )["result"]
''', '''        splits=splits,
        selection_policy=selection_policy,
    )["result"]
''')


def patch_runner(root: Path) -> None:
    path = root / "src/cmi_flu/runner.py"
    replace_once(
        path,
        '''    datasets = build_b02_datasets(config, inputs)

    compact_results = {
''',
        '''    datasets = build_b02_datasets(config, inputs)
    selection_policy = str(config.section("selection", required=False).get("policy", "legacy"))

    compact_results = {
''',
    )
    text = path.read_text(encoding="utf-8")
    old = '''            random_state=config.random_state,
        ),'''
    if text.count(old) < 3:
        raise SystemExit(f"{path}: compact runner anchors found {text.count(old)}")
    text = text.replace(old, '''            random_state=config.random_state,
            selection_policy=selection_policy,
        ),''', 3)
    path.write_text(text, encoding="utf-8")
    replace_once(path, '''        selection_panels={
            "Task2.1": inputs.vaccine_strains,
            "Task2.2": inputs.challenge_strains,
        },
    )
''', '''        selection_panels={
            "Task2.1": inputs.vaccine_strains,
            "Task2.2": inputs.challenge_strains,
        },
        selection_policy=selection_policy,
    )
''')
    replace_once(path, '''            selection_panels={"Task2.3": inputs.challenge_strains},
        )
''', '''            selection_panels={"Task2.3": inputs.challenge_strains},
            selection_policy=selection_policy,
        )
''')
    replace_once(path, '''    if config.baseline == "b02_taskwise_compact":
        return run_b02(config, inputs)
''', '''    if config.baseline in {"b02_taskwise_compact", "b021_taskwise_robust"}:
        return run_b02(config, inputs)
''')


def patch_configuration(root: Path) -> None:
    path = root / "src/cmi_flu/configuration.py"
    replace_once(
        path,
        '    if baseline not in {"b01_anchor", "b02_taskwise_compact"}:\n',
        '''    if baseline not in {
        "b01_anchor",
        "b02_taskwise_compact",
        "b021_taskwise_robust",
    }:
''',
    )


def create_config(root: Path) -> None:
    source_path = root / "configs/baseline_b02_taskwise.yaml"
    destination = root / "configs/baseline_b021_robust.yaml"
    source = source_path.read_text(encoding="utf-8")
    if source.count("baseline: b02_taskwise_compact\n") != 1:
        raise SystemExit("B2 config baseline anchor mismatch")
    source = source.replace("baseline: b02_taskwise_compact\n", "baseline: b021_taskwise_robust\n", 1)
    source = source.replace("filename: b02_taskwise_compact.csv\n", "filename: b021_taskwise_robust.csv\n", 1)
    anchor = "hai:\n  target_representation: residual\n  run_stress_tests: true\n"
    if source.count(anchor) != 1:
        raise SystemExit("B2 config HAI anchor mismatch")
    source = source.replace(anchor, anchor + "selection:\n  policy: robust_v1\n", 1)
    destination.write_text(source, encoding="utf-8")


def validate_target_blobs(root: Path) -> None:
    for relative, expected in TARGET_BLOBS.items():
        data = (root / relative).read_bytes()
        actual = git_blob_sha(data)
        if actual != expected:
            raise SystemExit(f"B2.1 source mismatch {relative}: expected={expected} actual={actual}")
    config = root / "configs/baseline_b021_robust.yaml"
    actual = git_blob_sha(config.read_bytes())
    if actual != TARGET_CONFIG_BLOB:
        raise SystemExit(f"B2.1 config mismatch: expected={TARGET_CONFIG_BLOB} actual={actual}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.workspace.resolve()
    patch_models(root)
    patch_evaluation(root)
    patch_runner(root)
    patch_configuration(root)
    create_config(root)
    validate_target_blobs(root)
    print("CMI_FLU_B21_SCIENTIFIC_PATCH PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
