from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


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


def install() -> None:
    """Expose frozen B2.1 robust HAI behavior through the newer evaluation API surface."""

    import cmi_flu.evaluation as evaluation
    import cmi_flu.runner as runner
    from cmi_flu.aliases import canonicalize_strain
    from cmi_flu.metrics import grouped_metrics
    from cmi_flu.targets import geometric_mean

    robust_hai = runner.run_hai_compact_for_panels
    legacy_evaluate_hai_spec = evaluation.evaluate_hai_spec

    def compatible_hai(
        dataset: Any,
        *,
        specs: Sequence[Any],
        selection_panels: Mapping[str, Sequence[str]],
        splits: Sequence[Any] | None = None,
        selection_policy: str = "robust_v1",
    ) -> dict[str, Any]:
        if selection_policy != "robust_v1":
            raise RuntimeError(
                "HAI B2.1 compatibility shim only permits selection_policy=robust_v1"
            )
        return robust_hai(
            dataset,
            specs=specs,
            selection_panels=selection_panels,
            splits=splits,
        )

    def compatible_evaluate_hai_spec(
        dataset: Any,
        *,
        spec: Any,
        splits: Sequence[Any],
        panel_strains: Sequence[str] | None = None,
    ) -> Any:
        result = legacy_evaluate_hai_spec(
            dataset,
            spec=spec,
            splits=splits,
            panel_strains=panel_strains,
        )
        if hasattr(result, "panel_proxy_fold_summary") and hasattr(
            result, "panel_proxy_fold_metrics"
        ):
            return result

        panel = evaluation._resolve_hai_panel(dataset, panel_strains)
        panel_set = {canonicalize_strain(value) for value in panel}
        work = result.enriched_oof.copy()
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
        result.panel_proxy_fold_metrics = folds.to_dict(orient="records")
        result.panel_proxy_fold_summary = _metric_summary(folds)
        return result

    evaluation.run_hai_compact_for_panels = compatible_hai
    evaluation.evaluate_hai_spec = compatible_evaluate_hai_spec
