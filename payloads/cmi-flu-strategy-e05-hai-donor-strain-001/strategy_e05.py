"""Strategy v2 E05: low-capacity donor-by-strain HAI response models.

E05 is deliberately narrow.  It compares four pre-registered conditions for
Task2.1/Task2.2 under the same leakage-resistant split families:

1. frozen B2.1 task-specific HAI model,
2. frozen Phase-A sequence control,
3. Ridge(alpha=10) with g(pre-HAI) + Z(donor) + S(strain) main effects,
4. the identical Ridge design plus exactly eight Z x S interactions.

The public API returns aggregate diagnostics only.  Participant-level OOF and
challenge predictions are transient and are never part of the returned object.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .aliases import canonicalize_strain
from .configuration import BaselineConfig
from .contracts import DataContractError, require_columns, require_finite
from .cv import NamedSplit, purged_leave_one_group_out, purged_leave_one_study_out
from .datasets import HAIModelDataset, build_hai_model_dataset
from .evaluation import (
    aggregate_hai_task_predictions,
    evaluate_hai_panel_proxy,
    evaluate_hai_spec,
)
from .hai_transfer import add_hai_ontology_features
from .metrics import (
    evaluate_predictions,
    percentile_rank,
    safe_spearman,
    within_group_rank_spearman,
)
from .models import ModelSpec, build_estimator, fit_final_model
from .strategy_e01_v2 import _metric_summary, _paired_comparison
from .targets import geometric_mean


EXPERIMENT = "strategy_v2_e05_hai_donor_strain"
RIDGE_ALPHA = 10.0
HINGE_HAI_VALUES: tuple[float, ...] = (10.0, 40.0, 160.0)
MAX_INTERACTIONS = 8
SIMULTANEOUS_HOLDOUTS = 4
SIMULTANEOUS_MIN_SUBJECTS = 3

B21_MODELS = {
    "Task2.1": "et_subtype_d3_l5",
    "Task2.2": "et_subtype_d5_l10",
}
SEQUENCE_CONTROLS = {
    "Task2.1": ("ontology_sequence_local", "et_subtype_d3_l5"),
    "Task2.2": ("ontology_sequence_target_domain", "et_subtype_d3_l5"),
}
CONDITIONS: tuple[str, ...] = (
    "b21_reference",
    "phase_a_fixed_sequence",
    "ridge_main_effects",
    "ridge_donor_by_strain_interactions",
)
Z_COLUMNS: tuple[str, ...] = (
    "z_hai_mean_rank",
    "z_hai_variance_rank",
    "z_hai_breadth40_rank",
    "z_age_rank",
)
S_NUMERIC_COLUMNS: tuple[str, ...] = (
    "s_vaccine_numeric",
    "s_vaccine_unknown",
    "s_sequence_distance",
)
S_CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "s_subtype",
    "s_substrate",
)
G_COLUMNS: tuple[str, ...] = (
    "g_log2_pre_hai",
    "g_hinge_10",
    "g_hinge_40",
    "g_hinge_160",
)
INTERACTION_COLUMNS: tuple[str, ...] = tuple(
    column
    for z in Z_COLUMNS
    for column in (
        f"interaction__{z}__vaccine",
        f"interaction__{z}__distance",
    )
)
MAIN_NUMERIC_COLUMNS: tuple[str, ...] = (
    *G_COLUMNS,
    *Z_COLUMNS,
    *S_NUMERIC_COLUMNS,
)
MAIN_CATEGORICAL_COLUMNS: tuple[str, ...] = S_CATEGORICAL_COLUMNS


@dataclass(frozen=True)
class _ConditionRun:
    name: str
    oof: pd.DataFrame
    challenge_strain_predictions: pd.DataFrame
    metrics: Mapping[str, Any]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _find_model(config: BaselineConfig, name: str) -> ModelSpec:
    matches = [spec for spec in config.model_specs("hai") if spec.name == name]
    if len(matches) != 1:
        raise DataContractError(f"E05 expected exactly one hai/{name} model; found {len(matches)}")
    return matches[0]


def _canonical_panel(values: Sequence[str]) -> tuple[str, ...]:
    panel = tuple(dict.fromkeys(canonicalize_strain(value) for value in values))
    if not panel or any(not strain for strain in panel):
        raise DataContractError("E05 task panel is empty after canonicalization")
    return panel


def _subject_level_rank(
    frame: pd.DataFrame,
    value_column: str,
) -> np.ndarray:
    """Rank one donor-level X-only value within study, then broadcast to strain rows."""

    require_columns(
        frame,
        ["study_group", "subject_group", value_column],
        table_name=f"E05 donor-rank {value_column}",
    )
    working = frame[["study_group", "subject_group", value_column]].copy()
    working[value_column] = pd.to_numeric(working[value_column], errors="coerce")

    records: list[pd.DataFrame] = []
    for (study, subject), group in working.groupby(
        ["study_group", "subject_group"], dropna=False, observed=True, sort=False
    ):
        finite = group[value_column].dropna().to_numpy(dtype=float)
        if finite.size and not np.allclose(finite, finite[0], rtol=0.0, atol=1e-12):
            raise DataContractError(
                f"E05 donor feature {value_column} varies by strain for {study}/{subject}"
            )
        value = float(finite[0]) if finite.size else np.nan
        records.append(
            pd.DataFrame(
                {
                    "study_group": [str(study)],
                    "subject_group": [str(subject)],
                    value_column: [value],
                }
            )
        )
    donors = pd.concat(records, ignore_index=True)
    donors["__rank"] = np.nan
    for _, positions in donors.groupby(
        "study_group", dropna=False, observed=True, sort=False
    ).indices.items():
        pos = np.asarray(positions, dtype=int)
        values = donors.iloc[pos][value_column].to_numpy(dtype=float)
        finite_mask = np.isfinite(values)
        if not finite_mask.any():
            continue
        ranks = percentile_rank(values[finite_mask])
        donors.loc[donors.index[pos[finite_mask]], "__rank"] = ranks

    mapping = {
        (str(row["study_group"]), str(row["subject_group"])): row["__rank"]
        for row in donors.to_dict(orient="records")
    }
    return np.asarray(
        [
            mapping[(str(study), str(subject))]
            for study, subject in zip(
                frame["study_group"], frame["subject_group"], strict=True
            )
        ],
        dtype=float,
    )


def _e05_design(frame: pd.DataFrame, *, interactions: bool) -> pd.DataFrame:
    """Construct the locked g/Z/S design; no target or free donor/strain ID enters."""

    required = [
        "study_group",
        "subject_group",
        "log2_pre_hai",
        "hai_log2_mean",
        "hai_log2_std",
        "hai_breadth_ge_40",
        "age",
        "strain_subtype",
        "strain_substrate",
        "ontology_official_component_vaccine",
        "ontology_seq_distance_to_study_vaccine",
    ]
    require_columns(frame, required, table_name="E05 HAI feature source")

    design = pd.DataFrame(index=frame.index)
    pre = pd.to_numeric(frame["log2_pre_hai"], errors="coerce").to_numpy(dtype=float)
    require_finite(pre, name="E05.log2_pre_hai")
    design["g_log2_pre_hai"] = pre
    for value in HINGE_HAI_VALUES:
        name = f"g_hinge_{int(value)}"
        design[name] = np.maximum(pre - np.log2(value), 0.0)

    donor_source = frame.copy()
    donor_source["__hai_variance"] = (
        pd.to_numeric(donor_source["hai_log2_std"], errors="coerce") ** 2
    )
    rank_sources = {
        "z_hai_mean_rank": "hai_log2_mean",
        "z_hai_variance_rank": "__hai_variance",
        "z_hai_breadth40_rank": "hai_breadth_ge_40",
        "z_age_rank": "age",
    }
    for output, source in rank_sources.items():
        design[output] = _subject_level_rank(donor_source, source)

    membership = (
        frame["ontology_official_component_vaccine"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.casefold()
    )
    allowed = membership.isin({"yes", "no", "unknown"})
    if not bool(allowed.all()):
        examples = sorted(set(membership.loc[~allowed].astype(str)))[:5]
        raise DataContractError(f"E05 unknown vaccine-membership labels: {examples}")
    design["s_vaccine_numeric"] = membership.eq("yes").astype(float)
    design["s_vaccine_unknown"] = membership.eq("unknown").astype(float)
    design["s_sequence_distance"] = pd.to_numeric(
        frame["ontology_seq_distance_to_study_vaccine"], errors="coerce"
    )
    design["s_subtype"] = frame["strain_subtype"].fillna("unknown").astype(str)
    design["s_substrate"] = frame["strain_substrate"].fillna("unknown").astype(str)

    if interactions:
        for z in Z_COLUMNS:
            design[f"interaction__{z}__vaccine"] = (
                pd.to_numeric(design[z], errors="coerce")
                * design["s_vaccine_numeric"].to_numpy(dtype=float)
            )
            design[f"interaction__{z}__distance"] = (
                pd.to_numeric(design[z], errors="coerce")
                * design["s_sequence_distance"].to_numpy(dtype=float)
            )
        if tuple(column for column in design.columns if column.startswith("interaction__")) != INTERACTION_COLUMNS:
            raise DataContractError("E05 interaction column contract changed")
        if len(INTERACTION_COLUMNS) != MAX_INTERACTIONS:
            raise DataContractError("E05 interaction count must remain exactly eight")

    expected = [
        *MAIN_NUMERIC_COLUMNS,
        *MAIN_CATEGORICAL_COLUMNS,
        *(INTERACTION_COLUMNS if interactions else ()),
    ]
    if list(design.columns) != expected:
        raise DataContractError(
            f"E05 feature order mismatch: got={list(design.columns)}, expected={expected}"
        )
    return design.reset_index(drop=True)


def _hierarchical_sample_weights(frame: pd.DataFrame) -> np.ndarray:
    """Equal study mass, equal subject mass within study, equal strain mass within subject."""

    require_columns(
        frame,
        ["study_group", "subject_group", "virus_strain"],
        table_name="E05 weight frame",
    )
    keys = frame[["study_group", "subject_group", "virus_strain"]].copy()
    keys["study_group"] = keys["study_group"].astype(str)
    keys["subject_group"] = keys["subject_group"].astype(str)
    keys["virus_strain"] = keys["virus_strain"].map(canonicalize_strain)
    n_studies = int(keys["study_group"].nunique())
    if n_studies < 1:
        raise DataContractError("E05 weights require at least one study")

    subjects_per_study = keys.groupby("study_group", observed=True)["subject_group"].nunique()
    strains_per_subject = keys.groupby(
        ["study_group", "subject_group"], observed=True
    )["virus_strain"].nunique()
    rows_per_cell = keys.groupby(
        ["study_group", "subject_group", "virus_strain"], observed=True
    ).size()

    weights = np.empty(len(keys), dtype=float)
    for position, row in enumerate(keys.itertuples(index=False)):
        study = str(row.study_group)
        subject = str(row.subject_group)
        strain = str(row.virus_strain)
        weights[position] = 1.0 / (
            n_studies
            * int(subjects_per_study.loc[study])
            * int(strains_per_subject.loc[(study, subject)])
            * int(rows_per_cell.loc[(study, subject, strain)])
        )
    require_finite(weights, name="E05 hierarchical sample weights")
    if (weights <= 0).any():
        raise DataContractError("E05 hierarchical sample weights must be positive")
    weights *= len(weights) / float(weights.sum())
    return weights


def _fit_ridge(
    train_design: pd.DataFrame,
    target: np.ndarray,
    weights: np.ndarray,
    prediction_design: pd.DataFrame,
    *,
    interactions: bool,
) -> tuple[Pipeline, np.ndarray]:
    numeric = [*MAIN_NUMERIC_COLUMNS, *INTERACTION_COLUMNS] if interactions else list(MAIN_NUMERIC_COLUMNS)
    categorical = list(MAIN_CATEGORICAL_COLUMNS)
    spec = ModelSpec(
        name=(
            "e05_ridge_interactions_a10"
            if interactions
            else "e05_ridge_main_effects_a10"
        ),
        family="ridge",
        params={"alpha": RIDGE_ALPHA},
        target_transform="identity",
    )
    estimator = build_estimator(
        spec,
        numeric_columns=numeric,
        categorical_columns=categorical,
    )
    values = np.asarray(target, dtype=float)
    require_finite(values, name="E05 Ridge target")
    estimator.fit(
        train_design,
        values,
        regressor__sample_weight=np.asarray(weights, dtype=float),
    )
    prediction = np.asarray(estimator.predict(prediction_design), dtype=float).reshape(-1)
    require_finite(prediction, name=f"E05 {spec.name} prediction")
    return estimator, prediction


def _ridge_oof(
    dataset: HAIModelDataset,
    *,
    splits: Sequence[NamedSplit],
    panel_strains: Sequence[str],
    interactions: bool,
) -> pd.DataFrame:
    """Weighted Ridge OOF on direct log2 D28 post-HAI with training-only preprocessors."""

    dataset.validate()
    frame = dataset.train.reset_index(drop=True)
    design = _e05_design(frame, interactions=interactions)
    target = pd.to_numeric(frame["target_log2_post"], errors="coerce").to_numpy(dtype=float)
    require_finite(target, name="E05 target_log2_post")

    parts: list[pd.DataFrame] = []
    for split in splits:
        train_idx = np.asarray(split.train_indices, dtype=int)
        validation_idx = np.asarray(split.validation_indices, dtype=int)
        weights = _hierarchical_sample_weights(frame.iloc[train_idx])
        _, log2_prediction = _fit_ridge(
            design.iloc[train_idx],
            target[train_idx],
            weights,
            design.iloc[validation_idx],
            interactions=interactions,
        )
        post_prediction = np.exp2(np.clip(log2_prediction, -20.0, 30.0))
        validation = frame.iloc[validation_idx]
        part = validation[
            [
                "participant_id",
                "subject_group",
                "study_group",
                "virus_strain",
                "post_hai",
                "log2_pre_hai",
            ]
        ].copy()
        part["split"] = split.name
        part["post_prediction"] = post_prediction
        parts.append(part)

    if not parts:
        raise DataContractError("E05 Ridge produced no OOF folds")
    oof = pd.concat(parts, ignore_index=True)
    oof["virus_strain"] = oof["virus_strain"].map(canonicalize_strain)
    panel = set(_canonical_panel(panel_strains))
    if oof.loc[oof["virus_strain"].isin(panel)].empty:
        raise DataContractError("E05 Ridge OOF has no overlap with task panel")
    return oof


def _ridge_final_challenge(
    dataset: HAIModelDataset,
    *,
    interactions: bool,
) -> pd.DataFrame:
    train = dataset.train.reset_index(drop=True)
    challenge = dataset.challenge.reset_index(drop=True)
    train_design = _e05_design(train, interactions=interactions)
    challenge_design = _e05_design(challenge, interactions=interactions)
    target = pd.to_numeric(train["target_log2_post"], errors="coerce").to_numpy(dtype=float)
    weights = _hierarchical_sample_weights(train)
    _, log2_prediction = _fit_ridge(
        train_design,
        target,
        weights,
        challenge_design,
        interactions=interactions,
    )
    prediction = np.exp2(np.clip(log2_prediction, -20.0, 30.0))
    result = challenge[["participant_id", "virus_strain"]].copy()
    result["prediction"] = prediction
    return result


def _condition_metrics(
    oof: pd.DataFrame,
    *,
    panel_strains: Sequence[str],
) -> Mapping[str, Any]:
    require_columns(
        oof,
        [
            "split",
            "participant_id",
            "subject_group",
            "study_group",
            "virus_strain",
            "post_hai",
            "post_prediction",
        ],
        table_name="E05 condition OOF",
    )
    panel = set(_canonical_panel(panel_strains))
    working = oof.copy()
    working["virus_strain"] = working["virus_strain"].map(canonicalize_strain)
    selection = working.loc[working["virus_strain"].isin(panel)].copy()
    if selection.empty:
        raise DataContractError("E05 condition has no task-panel OOF rows")
    proxy = evaluate_hai_panel_proxy(working, panel_strains=tuple(panel))
    within = within_group_rank_spearman(
        selection,
        group_column="virus_strain",
        target_column="post_hai",
        prediction_column="post_prediction",
    ).to_dict()
    return _json_safe(
        {
            "post_hai": evaluate_predictions(
                selection["post_hai"], selection["post_prediction"]
            ),
            "within_strain_spearman": within,
            "panel_proxy": proxy["metrics"],
            "panel_proxy_fold_summary": proxy["fold_summary"],
            "panel_proxy_fold_metrics": proxy["fold_metrics"],
            "panel_proxy_coverage": {
                key: value
                for key, value in proxy.items()
                if key not in {"metrics", "fold_metrics", "fold_summary"}
            },
        }
    )


def _control_run(
    dataset: HAIModelDataset,
    *,
    spec: ModelSpec,
    splits: Sequence[NamedSplit],
    panel_strains: Sequence[str],
    name: str,
    fit_challenge: bool,
) -> _ConditionRun:
    evaluated = evaluate_hai_spec(
        dataset,
        spec=spec,
        splits=splits,
        panel_strains=panel_strains,
    )
    if fit_challenge:
        _, target_prediction = fit_final_model(
            dataset.train,
            dataset.challenge,
            target_column=dataset.target_column,
            spec=spec,
            excluded_columns=dataset.excluded_columns,
        )
        post_prediction = dataset.target_prediction_to_post_hai(
            target_prediction,
            frame=dataset.challenge,
        )
        challenge = dataset.challenge[["participant_id", "virus_strain"]].copy()
        challenge["prediction"] = np.asarray(post_prediction, dtype=float)
    else:
        challenge = pd.DataFrame(columns=["participant_id", "virus_strain", "prediction"])
    return _ConditionRun(
        name=name,
        oof=evaluated.enriched_oof.copy(),
        challenge_strain_predictions=challenge,
        metrics=_json_safe(
            {
                "post_hai": evaluated.post_metrics,
                "within_strain_spearman": evaluated.within_strain_spearman,
                "panel_proxy": evaluated.panel_proxy_metrics,
                "panel_proxy_fold_summary": evaluated.panel_proxy_fold_summary,
                "panel_proxy_fold_metrics": evaluated.panel_proxy_fold_metrics,
                "panel_proxy_coverage": evaluated.panel_proxy_coverage,
            }
        ),
    )


def _ridge_run(
    dataset: HAIModelDataset,
    *,
    splits: Sequence[NamedSplit],
    panel_strains: Sequence[str],
    interactions: bool,
    fit_challenge: bool,
) -> _ConditionRun:
    oof = _ridge_oof(
        dataset,
        splits=splits,
        panel_strains=panel_strains,
        interactions=interactions,
    )
    challenge = (
        _ridge_final_challenge(dataset, interactions=interactions)
        if fit_challenge
        else pd.DataFrame(columns=["participant_id", "virus_strain", "prediction"])
    )
    return _ConditionRun(
        name=(
            "ridge_donor_by_strain_interactions"
            if interactions
            else "ridge_main_effects"
        ),
        oof=oof,
        challenge_strain_predictions=challenge,
        metrics=_condition_metrics(oof, panel_strains=panel_strains),
    )


def _panel_donor_frame(
    oof: pd.DataFrame,
    *,
    panel_strains: Sequence[str],
    anchor: bool = False,
) -> pd.DataFrame:
    require_columns(
        oof,
        [
            "participant_id",
            "subject_group",
            "study_group",
            "virus_strain",
            "post_hai",
            "log2_pre_hai",
        ],
        table_name="E05 donor panel frame",
    )
    panel = set(_canonical_panel(panel_strains))
    work = oof.copy()
    work["virus_strain"] = work["virus_strain"].map(canonicalize_strain)
    work = work.loc[work["virus_strain"].isin(panel)].copy()
    if work.empty:
        raise DataContractError("E05 donor panel has no rows")
    if anchor:
        work["__prediction"] = np.exp2(
            pd.to_numeric(work["log2_pre_hai"], errors="coerce").to_numpy(dtype=float)
        )
    else:
        require_columns(work, ["post_prediction"])
        work["__prediction"] = pd.to_numeric(work["post_prediction"], errors="coerce")

    grouped = (
        work.groupby(
            ["study_group", "participant_id", "subject_group"],
            dropna=False,
            observed=True,
        )
        .agg(
            target=("post_hai", geometric_mean),
            prediction=("__prediction", geometric_mean),
        )
        .reset_index()
    )
    if anchor:
        ranked = np.full(len(grouped), np.nan, dtype=float)
        for _, positions in grouped.groupby(
            "study_group", dropna=False, observed=True
        ).indices.items():
            pos = np.asarray(positions, dtype=int)
            ranked[pos] = percentile_rank(
                grouped.iloc[pos]["prediction"].to_numpy(dtype=float)
            )
        grouped["prediction"] = ranked
    return grouped


def _anchor_metrics(
    oof: pd.DataFrame,
    *,
    panel_strains: Sequence[str],
) -> Mapping[str, Any]:
    frame = _panel_donor_frame(oof, panel_strains=panel_strains, anchor=True)
    return _metric_summary(frame, prediction_scale="rank_score")


def _simultaneous_subject_strain_splits(
    frame: pd.DataFrame,
    *,
    panel_strains: Sequence[str],
) -> list[NamedSplit]:
    """Choose up to four study+strain cells from X/metadata only, never outcomes."""

    require_columns(
        frame,
        ["study_group", "subject_group", "virus_strain"],
        table_name="E05 simultaneous split source",
    )
    panel = set(_canonical_panel(panel_strains))
    work = frame[["study_group", "subject_group", "virus_strain"]].copy()
    work["virus_strain"] = work["virus_strain"].map(canonicalize_strain)
    work = work.loc[work["virus_strain"].isin(panel)].copy()
    if work.empty:
        raise DataContractError("E05 simultaneous split has zero panel overlap")
    cells = (
        work.groupby(["study_group", "virus_strain"], observed=True)
        .agg(subjects=("subject_group", "nunique"), rows=("subject_group", "size"))
        .reset_index()
    )
    cells = cells.loc[cells["subjects"] >= SIMULTANEOUS_MIN_SUBJECTS].copy()
    cells = cells.sort_values(
        ["subjects", "study_group", "virus_strain"],
        ascending=[False, True, True],
        kind="mergesort",
    ).head(SIMULTANEOUS_HOLDOUTS)
    splits: list[NamedSplit] = []
    full_study = frame["study_group"].astype(str).to_numpy()
    full_subject = frame["subject_group"].astype(str).to_numpy()
    full_strain = frame["virus_strain"].map(canonicalize_strain).astype(str).to_numpy()
    for row in cells.to_dict(orient="records"):
        held_study = str(row["study_group"])
        held_strain = canonicalize_strain(row["virus_strain"])
        validation_mask = (full_study == held_study) & (full_strain == held_strain)
        held_subjects = np.unique(full_subject[validation_mask])
        train_mask = (
            (full_study != held_study)
            & (full_strain != held_strain)
            & (~np.isin(full_subject, held_subjects))
        )
        train_idx = np.flatnonzero(train_mask)
        validation_idx = np.flatnonzero(validation_mask)
        if train_idx.size < 2 or validation_idx.size < SIMULTANEOUS_MIN_SUBJECTS:
            continue
        if set(full_subject[train_idx]).intersection(full_subject[validation_idx]):
            raise DataContractError("E05 simultaneous holdout leaks subjects")
        if held_strain in set(full_strain[train_idx]):
            raise DataContractError("E05 simultaneous holdout leaks held strain")
        splits.append(
            NamedSplit(
                name=f"study={held_study}/strain={held_strain}",
                train_indices=train_idx,
                validation_indices=validation_idx,
                held_out_group=f"{held_study}|{held_strain}",
                purged_subjects=tuple(sorted(set(held_subjects))),
            )
        )
    if not splits:
        raise DataContractError("E05 simultaneous split produced no usable cells")
    return splits


def _split_families(
    dataset: HAIModelDataset,
    *,
    panel_strains: Sequence[str],
) -> Mapping[str, Sequence[NamedSplit]]:
    frame = dataset.train.reset_index(drop=True)
    panel = set(_canonical_panel(panel_strains))
    study = purged_leave_one_study_out(
        frame["study_group"].astype(str),
        frame["subject_group"].astype(str),
    )
    season = purged_leave_one_group_out(
        frame["vaccine_season"].astype(str),
        frame["subject_group"].astype(str),
        excluded_values=["unknown", "<na>", "nan"],
        prefix="vaccine_season",
    )
    overlap = sorted(
        set(frame["virus_strain"].map(canonicalize_strain)).intersection(panel)
    )
    strain = purged_leave_one_group_out(
        frame["virus_strain"].map(canonicalize_strain).astype(str),
        frame["subject_group"].astype(str),
        included_values=overlap,
        prefix="strain",
    )
    simultaneous = _simultaneous_subject_strain_splits(
        frame, panel_strains=tuple(panel)
    )
    return {
        "subject_purged_study_out": study,
        "subject_purged_season_out": season,
        "subject_purged_strain_out": strain,
        "fixed_subject_strain_simultaneous": simultaneous,
    }


def _challenge_anchor(
    dataset: HAIModelDataset,
    *,
    panel_strains: Sequence[str],
    task: str,
) -> pd.DataFrame:
    rows = dataset.challenge[["participant_id", "virus_strain", "log2_pre_hai"]].copy()
    rows["prediction"] = np.exp2(
        pd.to_numeric(rows["log2_pre_hai"], errors="coerce").to_numpy(dtype=float)
    )
    return aggregate_hai_task_predictions(
        rows[["participant_id", "virus_strain", "prediction"]],
        panel_strains=panel_strains,
        task=task,
    )


def _challenge_rank_agreement(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> Mapping[str, Any]:
    require_columns(reference, ["participant_id", "prediction"])
    require_columns(candidate, ["participant_id", "prediction"])
    aligned = reference.merge(
        candidate,
        on="participant_id",
        suffixes=("_reference", "_candidate"),
        how="inner",
        validate="one_to_one",
    )
    if len(aligned) != len(reference) or len(aligned) != len(candidate):
        raise DataContractError("E05 challenge rank agreement does not align all donors")
    ref = pd.to_numeric(aligned["prediction_reference"], errors="coerce").to_numpy(dtype=float)
    cand = pd.to_numeric(aligned["prediction_candidate"], errors="coerce").to_numpy(dtype=float)
    require_finite(ref, name="E05 challenge anchor")
    require_finite(cand, name="E05 challenge candidate")
    ref_rank = percentile_rank(ref)
    cand_rank = percentile_rank(cand)
    delta = np.abs(ref_rank - cand_rank)
    n = len(aligned)
    one_rank_step = 1.0 / (n - 1) if n > 1 else float("inf")
    return _json_safe(
        {
            "donors": int(n),
            "rank_spearman": safe_spearman(ref, cand).to_dict(),
            "mean_absolute_percentile_difference": float(delta.mean()),
            "max_absolute_percentile_difference": float(delta.max()),
            "one_rank_step": one_rank_step,
            "at_least_one_rank_position_changed": bool(
                n > 1 and float(delta.max()) >= one_rank_step - 1e-12
            ),
        }
    )


def _challenge_condition(
    run: _ConditionRun,
    *,
    dataset: HAIModelDataset,
    panel_strains: Sequence[str],
    task: str,
    anchor: pd.DataFrame,
) -> Mapping[str, Any]:
    aggregated = aggregate_hai_task_predictions(
        run.challenge_strain_predictions,
        panel_strains=panel_strains,
        task=task,
    )
    return _challenge_rank_agreement(anchor, aggregated)


def _study_comparisons(
    runs: Mapping[str, _ConditionRun],
    *,
    panel_strains: Sequence[str],
) -> Mapping[str, Any]:
    donor_frames = {
        name: _panel_donor_frame(run.oof, panel_strains=panel_strains)
        for name, run in runs.items()
    }
    anchor = _panel_donor_frame(
        runs["b21_reference"].oof,
        panel_strains=panel_strains,
        anchor=True,
    )
    result: dict[str, Any] = {}
    for name in CONDITIONS:
        result[f"{name}_vs_anchor"] = _paired_comparison(
            donor_frames[name],
            anchor,
            candidate_name=name,
            reference_name="pre_hai_panel_geometric_mean_rank",
        )
    result["interaction_vs_main_effects"] = _paired_comparison(
        donor_frames["ridge_donor_by_strain_interactions"],
        donor_frames["ridge_main_effects"],
        candidate_name="ridge_donor_by_strain_interactions",
        reference_name="ridge_main_effects",
    )
    return _json_safe(result)


def _route_result(
    comparisons: Mapping[str, Any],
    challenge: Mapping[str, Any],
    runs: Mapping[str, _ConditionRun],
) -> Mapping[str, Any]:
    paired = comparisons["ridge_donor_by_strain_interactions_vs_anchor"]
    delta = paired.get("study_mean_delta_strict")
    changed = bool(
        challenge["ridge_donor_by_strain_interactions"][
            "at_least_one_rank_position_changed"
        ]
    )
    improved = delta is not None and float(delta) > 0.0

    interaction_rmse = (
        runs["ridge_donor_by_strain_interactions"]
        .metrics["panel_proxy"]
        .get("rmse", {})
        .get("value")
    )
    main_rmse = (
        runs["ridge_main_effects"]
        .metrics["panel_proxy"]
        .get("rmse", {})
        .get("value")
    )
    calibration_only = bool(
        not changed
        and interaction_rmse is not None
        and main_rmse is not None
        and float(interaction_rmse) < float(main_rmse)
    )
    if changed and improved:
        route = "e05_rank_signal"
    elif calibration_only:
        route = "e06_calibration_candidate_only"
    else:
        route = "no_positive_e05_signal"
    return {
        "positive_e05_signal": bool(changed and improved),
        "historical_proxy_study_mean_delta_vs_anchor": delta,
        "challenge_rank_changed_by_at_least_one_position": changed,
        "interaction_panel_proxy_rmse": interaction_rmse,
        "main_effects_panel_proxy_rmse": main_rmse,
        "routing": route,
        "note": (
            "E05 promotion still requires full condition/stress review; this routing "
            "does not alter the frozen competition incumbent or trigger submission."
        ),
    }


def _split_summary(splits: Sequence[NamedSplit]) -> Mapping[str, Any]:
    return {
        "split_count": len(splits),
        "held_out_groups": [str(split.held_out_group) for split in splits],
        "validation_rows": [int(len(split.validation_indices)) for split in splits],
        "purged_subjects": [int(len(split.purged_subjects)) for split in splits],
    }


def _assert_aligned_datasets(
    reference: HAIModelDataset,
    *candidates: HAIModelDataset,
) -> None:
    keys = ["participant_id", "virus_strain", "study_group", "subject_group"]
    require_columns(reference.train, keys, table_name="E05 reference train")
    require_columns(reference.challenge, keys, table_name="E05 reference challenge")
    for index, candidate in enumerate(candidates):
        require_columns(candidate.train, keys, table_name=f"E05 candidate {index} train")
        require_columns(candidate.challenge, keys, table_name=f"E05 candidate {index} challenge")
        for partition in ("train", "challenge"):
            left = getattr(reference, partition)[keys].astype(str).reset_index(drop=True)
            right = getattr(candidate, partition)[keys].astype(str).reset_index(drop=True)
            if left.shape != right.shape or not left.equals(right):
                raise DataContractError(
                    f"E05 enriched dataset row order differs for candidate={index} partition={partition}"
                )


def _run_task(
    task: str,
    *,
    config: BaselineConfig,
    base_dataset: HAIModelDataset,
    ridge_dataset: HAIModelDataset,
    sequence_dataset: HAIModelDataset,
    panel_strains: Sequence[str],
) -> Mapping[str, Any]:
    panel = _canonical_panel(panel_strains)
    _assert_aligned_datasets(base_dataset, ridge_dataset, sequence_dataset)
    split_families = _split_families(ridge_dataset, panel_strains=panel)
    study_splits = split_families["subject_purged_study_out"]

    b21_spec = _find_model(config, B21_MODELS[task])
    sequence_spec = _find_model(config, SEQUENCE_CONTROLS[task][1])
    study_runs: dict[str, _ConditionRun] = {
        "b21_reference": _control_run(
            base_dataset,
            spec=b21_spec,
            splits=study_splits,
            panel_strains=panel,
            name="b21_reference",
            fit_challenge=True,
        ),
        "phase_a_fixed_sequence": _control_run(
            sequence_dataset,
            spec=sequence_spec,
            splits=study_splits,
            panel_strains=panel,
            name="phase_a_fixed_sequence",
            fit_challenge=True,
        ),
        "ridge_main_effects": _ridge_run(
            ridge_dataset,
            splits=study_splits,
            panel_strains=panel,
            interactions=False,
            fit_challenge=True,
        ),
        "ridge_donor_by_strain_interactions": _ridge_run(
            ridge_dataset,
            splits=study_splits,
            panel_strains=panel,
            interactions=True,
            fit_challenge=True,
        ),
    }

    comparisons = _study_comparisons(study_runs, panel_strains=panel)
    anchor = _challenge_anchor(ridge_dataset, panel_strains=panel, task=task)
    challenge = {
        name: _challenge_condition(
            run,
            dataset=ridge_dataset,
            panel_strains=panel,
            task=task,
            anchor=anchor,
        )
        for name, run in study_runs.items()
    }

    stress: dict[str, Any] = {}
    for family, splits in split_families.items():
        if family == "subject_purged_study_out":
            stress[family] = {
                "splits": _split_summary(splits),
                "conditions": {
                    name: run.metrics for name, run in study_runs.items()
                },
                "anchor_metrics": _anchor_metrics(
                    study_runs["b21_reference"].oof,
                    panel_strains=panel,
                ),
            }
            continue

        family_runs = {
            "b21_reference": _control_run(
                base_dataset,
                spec=b21_spec,
                splits=splits,
                panel_strains=panel,
                name="b21_reference",
                fit_challenge=False,
            ),
            "phase_a_fixed_sequence": _control_run(
                sequence_dataset,
                spec=sequence_spec,
                splits=splits,
                panel_strains=panel,
                name="phase_a_fixed_sequence",
                fit_challenge=False,
            ),
            "ridge_main_effects": _ridge_run(
                ridge_dataset,
                splits=splits,
                panel_strains=panel,
                interactions=False,
                fit_challenge=False,
            ),
            "ridge_donor_by_strain_interactions": _ridge_run(
                ridge_dataset,
                splits=splits,
                panel_strains=panel,
                interactions=True,
                fit_challenge=False,
            ),
        }
        stress[family] = {
            "splits": _split_summary(splits),
            "conditions": {
                name: run.metrics for name, run in family_runs.items()
            },
            "anchor_metrics": _anchor_metrics(
                family_runs["b21_reference"].oof,
                panel_strains=panel,
            ),
        }

    return _json_safe(
        {
            "task": task,
            "panel_size": len(panel),
            "study_out_conditions": {
                name: run.metrics for name, run in study_runs.items()
            },
            "anchor_reference": {
                "name": "pre_hai_panel_geometric_mean_rank",
                "metrics": _anchor_metrics(
                    study_runs["b21_reference"].oof,
                    panel_strains=panel,
                ),
            },
            "study_out_paired_comparisons": comparisons,
            "challenge_rank_agreements_vs_anchor": challenge,
            "stress_tests": stress,
            "routing": _route_result(comparisons, challenge, study_runs),
        }
    )


def run_strategy_e05(
    config: BaselineConfig,
    inputs: Any,
    *,
    sequence_reference: pd.DataFrame,
    vaccine_reference: pd.DataFrame,
) -> Mapping[str, Any]:
    """Run the locked E05 experiment and return aggregate-only diagnostics."""

    if str(config.section("hai").get("target_representation", "residual")) != "residual":
        raise DataContractError("E05 frozen controls require hai.target_representation=residual")
    if str(config.section("selection", required=False).get("policy", "legacy")) != "robust_v1":
        raise DataContractError("E05 requires selection.policy=robust_v1")

    tables = inputs.tables
    base = build_hai_model_dataset(
        tables["public_serology"],
        tables["challenge_serology"],
        tables["participants"],
        tables["investigations"],
        day=28,
        target_representation="residual",
        challenge_panel_strains=inputs.challenge_strains,
    )
    ridge = add_hai_ontology_features(
        base,
        vaccine_reference=vaccine_reference,
        sequence_reference=sequence_reference,
        vaccine_2025=inputs.vaccine_strains,
        condition="ontology_sequence_local",
    )
    sequence_local = ridge
    sequence_target = add_hai_ontology_features(
        base,
        vaccine_reference=vaccine_reference,
        sequence_reference=sequence_reference,
        vaccine_2025=inputs.vaccine_strains,
        condition="ontology_sequence_target_domain",
    )

    task21 = _run_task(
        "Task2.1",
        config=config,
        base_dataset=base,
        ridge_dataset=ridge,
        sequence_dataset=sequence_local,
        panel_strains=inputs.vaccine_strains,
    )
    task22 = _run_task(
        "Task2.2",
        config=config,
        base_dataset=base,
        ridge_dataset=ridge,
        sequence_dataset=sequence_target,
        panel_strains=inputs.challenge_strains,
    )
    return _json_safe(
        {
            "experiment": EXPERIMENT,
            "tasks": {"Task2.1": task21, "Task2.2": task22},
            "feature_contract": {
                "ridge_alpha": RIDGE_ALPHA,
                "g": {
                    "baseline": "log2_pre_hai",
                    "fixed_hinge_hai_values": list(HINGE_HAI_VALUES),
                },
                "z": [
                    "within-study ranked baseline log2-HAI mean",
                    "within-study ranked baseline log2-HAI variance",
                    "within-study ranked baseline breadth>=40",
                    "within-study ranked age",
                ],
                "s": [
                    "strain subtype",
                    "study-season vaccine component membership with explicit unknown indicator",
                    "strain substrate",
                    "pre-fixed sequence distance to study vaccine",
                ],
                "interaction_count": len(INTERACTION_COLUMNS),
                "interaction_columns": list(INTERACTION_COLUMNS),
                "free_participant_embedding": False,
                "raw_strain_id_one_hot": False,
                "x_only_within_study_rank_transduction": True,
                "imputation_and_standardization_fit_on_training_rows_only": True,
            },
            "weight_contract": {
                "study_total_weight": "equal",
                "subject_total_weight_within_study": "equal",
                "strain_total_weight_within_subject": "equal",
                "rows_are_not_counted_as_independent_donors": True,
            },
            "split_contract": {
                "subject_purged_study_out": True,
                "subject_purged_season_out": True,
                "subject_purged_strain_out": True,
                "fixed_subject_strain_simultaneous_holdouts": SIMULTANEOUS_HOLDOUTS,
                "simultaneous_min_subjects": SIMULTANEOUS_MIN_SUBJECTS,
                "simultaneous_selection_uses_outcomes": False,
            },
            "controls": {
                "Task2.1": {
                    "b21_model": B21_MODELS["Task2.1"],
                    "phase_a_sequence_condition": SEQUENCE_CONTROLS["Task2.1"][0],
                    "phase_a_sequence_model": SEQUENCE_CONTROLS["Task2.1"][1],
                },
                "Task2.2": {
                    "b21_model": B21_MODELS["Task2.2"],
                    "phase_a_sequence_condition": SEQUENCE_CONTROLS["Task2.2"][0],
                    "phase_a_sequence_model": SEQUENCE_CONTROLS["Task2.2"][1],
                },
            },
            "leaderboard_used_for_selection": False,
            "competition_submission_attempted": False,
            "output_policy": "aggregate_only_no_participant_ids_or_row_level_predictions",
        }
    )