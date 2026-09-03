from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def _finite_or(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if np.isfinite(number) else fallback


def _metric_summary(frame: pd.DataFrame) -> dict[str, float | int | None]:
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


def _aggregate_repeated_oof(oof: pd.DataFrame) -> pd.DataFrame:
    counts = oof.groupby("row_index", observed=True).size()
    if counts.empty:
        raise RuntimeError("OOF predictions are empty")
    if int(counts.max()) == 1:
        result = oof.copy()
        result["repeat_count"] = 1
        return result
    for column in ("target", "target_transformed"):
        spread = oof.groupby("row_index", observed=True)[column].agg(["min", "max"])
        if not np.allclose(spread["min"], spread["max"], rtol=0.0, atol=1e-12):
            raise RuntimeError(f"repeated OOF rows disagree on {column}")
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
            "row_index", "split", "held_out_group", "target", "prediction",
            "target_transformed", "prediction_transformed", "repeat_count",
        ]
    ]


def _repeat_metrics(raw_oof: pd.DataFrame, grouped_metrics: Any) -> pd.DataFrame:
    repeat = raw_oof["split"].astype(str).str.extract(r"^repeat=(\d+)/fold=\d+$", expand=False)
    if repeat.isna().all():
        return pd.DataFrame()
    working = raw_oof.loc[repeat.notna()].copy()
    working["repeat"] = repeat.loc[repeat.notna()].astype(int).to_numpy()
    counts = working.groupby(["repeat", "row_index"], observed=True).size()
    if int(counts.max()) != 1:
        raise RuntimeError("a repeated-CV repeat predicts an original row more than once")
    return grouped_metrics(
        working,
        group_columns=["repeat"],
        target_column="target",
        prediction_column="prediction",
    )


def _task11_oof_plausibility(evaluation: Any) -> dict[str, Any]:
    oof = evaluation.oof_predictions
    target = oof["target"].to_numpy(dtype=float)
    transformed_target = oof["target_transformed"].to_numpy(dtype=float)
    transformed_prediction = oof["prediction_transformed"].to_numpy(dtype=float)
    target_std = float(np.std(target))
    rmse = float(evaluation.pooled_metrics["rmse"]["value"])
    ratio = rmse / target_std if target_std > 1e-12 else float("inf")
    low, high = float(np.min(transformed_target)), float(np.max(transformed_target))
    span = max(high - low, 1e-12)
    allowed_low, allowed_high = low - 2.0 * span, high + 2.0 * span
    pred_low, pred_high = float(np.min(transformed_prediction)), float(np.max(transformed_prediction))
    passed = bool(
        np.isfinite(ratio) and ratio <= 10.0
        and pred_low >= allowed_low and pred_high <= allowed_high
    )
    return {
        "passed": passed,
        "rmse_std_ratio": ratio,
        "rmse_std_ratio_max": 10.0,
        "prediction_transformed_min": pred_low,
        "prediction_transformed_max": pred_high,
        "allowed_transformed_min": allowed_low,
        "allowed_transformed_max": allowed_high,
        "range_expansion": 2.0,
    }


def _task11_challenge_plausibility(dataset: Any, prediction: np.ndarray) -> dict[str, Any]:
    values = np.asarray(prediction, dtype=float)
    historical = pd.to_numeric(dataset.train[dataset.target_column], errors="coerce").to_numpy(dtype=float)
    historical_max = float(np.max(historical))
    prediction_max, prediction_min = float(np.max(values)), float(np.min(values))
    ratio = prediction_max / historical_max if historical_max > 0 else float("inf")
    passed = bool(np.isfinite(values).all() and prediction_min >= 0.0 and np.isfinite(ratio) and ratio <= 10.0)
    return {
        "passed": passed,
        "prediction_min": prediction_min,
        "prediction_max": prediction_max,
        "historical_target_max": historical_max,
        "prediction_max_target_ratio": ratio,
        "prediction_max_target_ratio_limit": 10.0,
    }


def install() -> None:
    import cmi_flu.evaluation as ev
    import cmi_flu.runner as runner
    from cmi_flu.aliases import canonicalize_strain
    from cmi_flu.cv import purged_leave_one_study_out
    from cmi_flu.metrics import evaluate_predictions, grouped_metrics
    from cmi_flu.models import evaluate_candidates, fit_final_model
    from cmi_flu.targets import geometric_mean

    def compact(dataset: Any, *, specs: Sequence[Any], splits: Sequence[Any] | None = None, random_state: int = 42) -> Any:
        dataset.validate()
        chosen = list(splits or ev.default_splits_for_task(dataset, random_state=random_state))
        search = evaluate_candidates(
            dataset.train,
            target_column=dataset.target_column,
            splits=chosen,
            specs=specs,
            excluded_columns=dataset.excluded_columns,
        )
        repeated = int(dataset.train["study_group"].nunique()) < 2
        prepared: list[tuple[Any, pd.DataFrame, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], pd.DataFrame]] = []
        diagnostics: dict[str, Mapping[str, Any]] = {}
        for evaluation in search.evaluations:
            raw = evaluation.oof_predictions
            top = _aggregate_repeated_oof(raw) if repeated else raw.copy()
            pooled = evaluate_predictions(top["target"], top["prediction"])
            transformed = evaluate_predictions(top["target_transformed"], top["prediction_transformed"])
            fold_summary = _metric_summary(evaluation.fold_metrics)
            repeats = _repeat_metrics(raw, grouped_metrics)
            detail: dict[str, Any] = {"eligible": True, "policy": "robust_v1"}
            if dataset.task == "Task1.1":
                probe = type("Probe", (), {"oof_predictions": top, "pooled_metrics": pooled})()
                detail["oof_plausibility"] = _task11_oof_plausibility(probe)
                detail["eligible"] = bool(detail["oof_plausibility"]["passed"])
            diagnostics[evaluation.spec.name] = detail
            if detail["eligible"]:
                prepared.append((evaluation, top, pooled, transformed, fold_summary, repeats))
        if not prepared:
            raise RuntimeError(f"{dataset.task} has no candidate passing robust selection gates")
        multi_study = int(dataset.train["study_group"].nunique()) >= 2
        def rank_key(item: tuple[Any, pd.DataFrame, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], pd.DataFrame]) -> tuple[Any, ...]:
            evaluation, _, pooled, _, summary, _ = item
            if multi_study:
                return (
                    _finite_or(summary.get("spearman_mean"), -np.inf),
                    _finite_or(summary.get("spearman_median"), -np.inf),
                    _finite_or(summary.get("spearman_min"), -np.inf),
                    _finite_or(pooled["spearman"]["value"], -np.inf),
                    -_finite_or(pooled["rmse"]["value"], np.inf),
                    evaluation.spec.name,
                )
            return (
                _finite_or(pooled["spearman"]["value"], -np.inf),
                -_finite_or(pooled["rmse"]["value"], np.inf),
                evaluation.spec.name,
            )
        prepared.sort(key=rank_key, reverse=True)
        selected = None
        fitted = None
        challenge_prediction = None
        selected_top = None
        selected_pooled = None
        selected_transformed = None
        selected_summary = None
        selected_repeats = None
        for item in prepared:
            evaluation, top, pooled, transformed, summary, repeats = item
            model, prediction = fit_final_model(
                dataset.train,
                dataset.challenge,
                target_column=dataset.target_column,
                spec=evaluation.spec,
                excluded_columns=dataset.excluded_columns,
            )
            if dataset.task == "Task1.1":
                check = _task11_challenge_plausibility(dataset, prediction)
                diagnostics[evaluation.spec.name] = {**diagnostics[evaluation.spec.name], "challenge_plausibility": check}
                if not check["passed"]:
                    continue
            selected, fitted, challenge_prediction = evaluation, model, prediction
            selected_top, selected_pooled, selected_transformed = top, pooled, transformed
            selected_summary, selected_repeats = summary, repeats
            break
        if selected is None:
            raise RuntimeError(f"{dataset.task} has no candidate passing final prediction guardrails")
        challenge = dataset.challenge[["participant_id"]].copy()
        challenge["prediction"] = challenge_prediction
        summaries = []
        for evaluation in search.evaluations:
            match = next((item for item in prepared if item[0].spec.name == evaluation.spec.name), None)
            top = evaluation.oof_predictions if match is None else match[1]
            pooled = evaluation.pooled_metrics if match is None else match[2]
            transformed = evaluation.transformed_metrics if match is None else match[3]
            summary = _metric_summary(evaluation.fold_metrics) if match is None else match[4]
            repeats = _repeat_metrics(evaluation.oof_predictions, grouped_metrics) if match is None else match[5]
            summaries.append({
                "model": evaluation.spec.to_dict(),
                "pooled_metrics": pooled,
                "transformed_metrics": transformed,
                "fold_summary": summary,
                "repeat_metrics": repeats.to_dict(orient="records"),
                "raw_oof_rows": int(len(evaluation.oof_predictions)),
                "top_line_oof_rows": int(len(top)),
                "selection_diagnostics": diagnostics.get(evaluation.spec.name, {}),
            })
        return ev.TaskRunResult(
            task=dataset.task,
            selected_spec=selected.spec,
            challenge_predictions=challenge,
            oof_predictions=selected_top,
            metrics={
                "pooled": selected_pooled,
                "transformed": selected_transformed,
                "fold_summary": selected_summary,
                "repeat_metrics": selected_repeats.to_dict(orient="records"),
                "raw_oof_rows": int(len(selected.oof_predictions)),
                "top_line_oof_rows": int(len(selected_top)),
                "selection_policy": "robust_v1",
                "selection_diagnostics": diagnostics.get(selected.spec.name, {}),
                "split_count": len(chosen),
                "training_rows": len(dataset.train),
                "training_subjects": int(dataset.train["subject_group"].nunique()),
                "training_studies": int(dataset.train["study_group"].nunique()),
            },
            fold_metrics=selected.fold_metrics,
            candidate_summaries=summaries,
            candidate_failures=search.failures,
            fitted_model=fitted,
        )

    def panel_fold_summary(enriched_oof: pd.DataFrame, panel: Sequence[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        panel_set = {canonicalize_strain(value) for value in panel}
        work = enriched_oof.copy()
        work["virus_strain"] = work["virus_strain"].map(canonicalize_strain)
        work = work.loc[work["virus_strain"].isin(panel_set)].copy()
        grouped = (
            work.groupby(["split", "participant_id"], dropna=False, observed=True)
            .agg(
                panel_target=("post_hai", geometric_mean),
                panel_prediction=("post_prediction", geometric_mean),
            )
            .reset_index()
        )
        folds = grouped_metrics(
            grouped,
            group_columns=["split"],
            target_column="panel_target",
            prediction_column="panel_prediction",
        )
        return folds.to_dict(orient="records"), _metric_summary(folds)

    def hai(dataset: Any, *, specs: Sequence[Any], selection_panels: Mapping[str, Sequence[str]], splits: Sequence[Any] | None = None) -> dict[str, Any]:
        dataset.validate()
        chosen = list(splits or purged_leave_one_study_out(dataset.train["study_group"].astype(str), dataset.train["subject_group"].astype(str)))
        search = evaluate_candidates(
            dataset.train,
            target_column=dataset.target_column,
            splits=chosen,
            specs=specs,
            excluded_columns=dataset.excluded_columns,
        )
        results: dict[str, Any] = {}
        for name, panel_strains in selection_panels.items():
            panel = ev._resolve_hai_panel(dataset, panel_strains)
            enriched = [ev._enrich_hai_oof(dataset, evaluation, panel_strains=panel) for evaluation in search.evaluations]
            ranked = []
            extra: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
            for candidate in enriched:
                folds, summary = panel_fold_summary(candidate.enriched_oof, panel)
                extra[candidate.base.spec.name] = (folds, summary)
                key = (
                    _finite_or(summary.get("spearman_mean"), -np.inf),
                    _finite_or(summary.get("spearman_median"), -np.inf),
                    _finite_or(summary.get("spearman_min"), -np.inf),
                    _finite_or(candidate.panel_proxy_metrics["spearman"]["value"], -np.inf),
                    _finite_or(candidate.within_strain_spearman["value"], -np.inf),
                    -_finite_or(candidate.post_metrics["rmse"]["value"], np.inf),
                    candidate.base.spec.name,
                )
                ranked.append((key, candidate))
            ranked.sort(key=lambda pair: pair[0], reverse=True)
            selected = ranked[0][1]
            fitted, target_prediction = fit_final_model(
                dataset.train,
                dataset.challenge,
                target_column=dataset.target_column,
                spec=selected.base.spec,
                excluded_columns=dataset.excluded_columns,
            )
            post_prediction = dataset.target_prediction_to_post_hai(target_prediction, frame=dataset.challenge)
            challenge = dataset.challenge[["participant_id", "virus_strain"]].copy()
            challenge["prediction"] = post_prediction
            summaries = []
            for candidate in enriched:
                folds, summary = extra[candidate.base.spec.name]
                summaries.append({
                    "model": candidate.base.spec.to_dict(),
                    "target_metrics": candidate.base.pooled_metrics,
                    "post_hai_metrics": candidate.post_metrics,
                    "within_strain_spearman": candidate.within_strain_spearman,
                    "panel_proxy_metrics": candidate.panel_proxy_metrics,
                    "panel_proxy_coverage": candidate.panel_proxy_coverage,
                    "panel_proxy_fold_metrics": folds,
                    "panel_proxy_fold_summary": summary,
                })
            selected_folds, selected_summary = extra[selected.base.spec.name]
            results[str(name)] = ev.HAIRunResult(
                day=dataset.day,
                selected_spec=selected.base.spec,
                challenge_strain_predictions=challenge,
                oof_predictions=selected.enriched_oof,
                metrics={
                    "target": selected.base.pooled_metrics,
                    "post_hai": selected.post_metrics,
                    "within_strain_spearman": selected.within_strain_spearman,
                    "panel_proxy": selected.panel_proxy_metrics,
                    "panel_proxy_coverage": selected.panel_proxy_coverage,
                    "panel_proxy_fold_metrics": selected_folds,
                    "panel_proxy_fold_summary": selected_summary,
                    "selection_policy": "robust_v1",
                    "selection_panel_strains": list(panel),
                    "split_count": len(chosen),
                    "training_rows": len(dataset.train),
                    "training_subjects": int(dataset.train["subject_group"].nunique()),
                    "training_studies": int(dataset.train["study_group"].nunique()),
                },
                candidate_summaries=summaries,
                candidate_failures=search.failures,
                fitted_model=fitted,
            )
        return results

    runner.run_compact_task = compact
    runner.run_hai_compact_for_panels = hai
