"""Phase A HAI strain/domain-transfer diagnostics.

The public experiment API intentionally returns only aggregate participant metrics plus
public strain-reference metadata. Row-level OOF and challenge predictions are transient.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .aliases import canonicalize_strain
from .configuration import BaselineConfig
from .contracts import DataContractError, require_columns, require_unique
from .cv import purged_leave_one_group_out
from .datasets import HAIModelDataset, build_hai_model_dataset
from .evaluation import (
    HAICandidateEvaluation,
    HAIRunResult,
    aggregate_hai_task_predictions,
    evaluate_hai_spec,
    run_hai_compact_for_panels,
)
from .features.serology import parse_influenza_strain
from .metrics import percentile_rank, safe_spearman
from .models import ModelSpec

CONDITIONS: tuple[str, ...] = (
    "b21_reference",
    "ontology_metadata",
    "ontology_sequence_local",
    "ontology_sequence_target_domain",
)
NUCLEOTIDE_IUPAC = frozenset("ACGTUNRYKMSWBDHV")
PROMOTION_MEAN_TOLERANCE = 0.03
PROMOTION_MIN_TOLERANCE = 0.10


def canonical_component_identity(value: object) -> str:
    """Canonical strain identity with only known substrate suffixes removed."""

    strain = canonicalize_strain(value)
    return re.sub(r"(?i)(?:_MDCK|_CELL|_EGG)$", "", strain).strip()


def canonical_season_start(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    match = re.search(r"(?:19|20)\d{2}", text)
    return int(match.group(0)) if match else None


def normalize_sequence(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return "".join(character for character in str(value).upper() if character.isalpha())


def _sequence_k(sequence: str) -> int:
    if not sequence:
        return 0
    nucleotide_fraction = sum(character in NUCLEOTIDE_IUPAC for character in sequence) / len(
        sequence
    )
    return 9 if nucleotide_fraction >= 0.95 else 3


def sequence_kmer_distance(left: object, right: object) -> float:
    """Return deterministic alignment-free k-mer Jaccard distance."""

    first = normalize_sequence(left)
    second = normalize_sequence(right)
    if not first or not second:
        return float("nan")
    first_k = _sequence_k(first)
    second_k = _sequence_k(second)
    if first_k <= 0 or first_k != second_k:
        return float("nan")
    k = first_k
    if len(first) < k or len(second) < k:
        return float("nan")
    first_kmers = {first[index : index + k] for index in range(len(first) - k + 1)}
    second_kmers = {second[index : index + k] for index in range(len(second) - k + 1)}
    union = first_kmers | second_kmers
    if not union:
        return float("nan")
    return float(1.0 - len(first_kmers & second_kmers) / len(union))


def _strain_family(value: object) -> str:
    parsed = parse_influenza_strain(value)
    subtype = str(parsed["strain_subtype"])
    if subtype != "B":
        return subtype
    lineage = parsed.get("strain_lineage")
    return f"B/{lineage}" if lineage else "B/unknown"


def build_sequence_lookup(reference: pd.DataFrame) -> dict[str, Mapping[str, Any]]:
    """Prefer complete then longer organizer sequence for each canonical strain."""

    require_columns(
        reference,
        ["virus_strain", "sequence", "sequence_status"],
        table_name="strain sequence reference",
    )
    working = reference[["virus_strain", "sequence", "sequence_status"]].copy()
    working["virus_strain"] = working["virus_strain"].map(canonicalize_strain)
    working["sequence"] = working["sequence"].map(normalize_sequence)
    working["sequence_status"] = working["sequence_status"].fillna("").astype(str).str.strip()
    working = working.loc[
        working["virus_strain"].astype(str).ne("") & working["sequence"].astype(str).ne("")
    ].copy()
    working["_complete"] = working["sequence_status"].str.casefold().eq("complete")
    working["_length"] = working["sequence"].str.len()
    working = working.sort_values(
        ["virus_strain", "_complete", "_length"],
        ascending=[True, False, False],
        kind="mergesort",
    )
    chosen = working.drop_duplicates("virus_strain", keep="first")
    return {
        row["virus_strain"]: {
            "sequence": row["sequence"],
            "sequence_status": row["sequence_status"],
            "complete": bool(row["_complete"]),
            "family": _strain_family(row["virus_strain"]),
        }
        for row in chosen.to_dict(orient="records")
    }


def load_vaccine_strain_reference(path: str | Path) -> pd.DataFrame:
    """Read the organizer two-column season/strain file without assuming a delimiter."""

    rows: list[dict[str, Any]] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip().strip('"') for field in re.split(r"\t|,", line) if field.strip()]
        if len(fields) < 2:
            continue
        if any("season" in field.casefold() for field in fields) and any(
            "strain" in field.casefold() for field in fields
        ):
            continue
        season_candidates = [field for field in fields if canonical_season_start(field) is not None]
        strain_candidates = [
            field
            for field in fields
            if "/" in field or field.upper().startswith(("H1N1", "H3N2", "VIC", "YAM"))
        ]
        if season_candidates and strain_candidates:
            season, strain = season_candidates[0], strain_candidates[-1]
        else:
            season, strain = fields[0], fields[-1]
        season_start = canonical_season_start(season)
        strain = canonicalize_strain(strain)
        if season_start is None or not strain:
            continue
        rows.append(
            {
                "vaccine_season": season,
                "vaccine_season_start": season_start,
                "virus_strain": strain,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise DataContractError("vaccine strain reference produced no usable season/strain rows")
    require_unique(
        result,
        ["vaccine_season_start", "virus_strain"],
        table_name="vaccine strain reference",
    )
    return result


def _vaccine_map(reference: pd.DataFrame) -> dict[int, tuple[str, ...]]:
    require_columns(
        reference,
        ["vaccine_season_start", "virus_strain"],
        table_name="vaccine strain reference",
    )
    working = reference.copy()
    working["vaccine_season_start"] = pd.to_numeric(
        working["vaccine_season_start"], errors="coerce"
    )
    working = working.dropna(subset=["vaccine_season_start"])
    working["vaccine_season_start"] = working["vaccine_season_start"].astype(int)
    working["virus_strain"] = working["virus_strain"].map(canonicalize_strain)
    return {
        int(season): tuple(dict.fromkeys(group["virus_strain"].astype(str)))
        for season, group in working.groupby("vaccine_season_start", observed=True)
    }


def _same_family_candidates(strain: str, candidates: Sequence[str]) -> list[str]:
    family = _strain_family(strain)
    return [candidate for candidate in candidates if _strain_family(candidate) == family]


def _nearest_sequence_distance(
    strain: str,
    candidates: Sequence[str],
    sequence_lookup: Mapping[str, Mapping[str, Any]],
) -> tuple[float, str | None]:
    strain = canonicalize_strain(strain)
    source = sequence_lookup.get(strain)
    if source is None:
        return float("nan"), None
    best_distance = float("inf")
    best_name: str | None = None
    for candidate in _same_family_candidates(strain, candidates):
        target = sequence_lookup.get(canonicalize_strain(candidate))
        if target is None:
            continue
        distance = sequence_kmer_distance(source["sequence"], target["sequence"])
        if np.isfinite(distance) and (
            distance < best_distance
            or (
                np.isclose(distance, best_distance)
                and best_name is not None
                and canonicalize_strain(candidate) < best_name
            )
        ):
            best_distance = float(distance)
            best_name = canonicalize_strain(candidate)
    return (best_distance, best_name) if best_name is not None else (float("nan"), None)


def _substrate_relation(strain: str, candidates: Sequence[str]) -> str:
    parsed = parse_influenza_strain(strain)
    component = canonical_component_identity(strain)
    equivalent = [
        candidate
        for candidate in candidates
        if canonical_component_identity(candidate) == component
    ]
    if not equivalent:
        return "not_component"
    substrate = parsed.get("strain_substrate")
    if substrate in {None, "unspecified"}:
        return "unknown"
    candidate_substrates = {
        parse_influenza_strain(candidate).get("strain_substrate") for candidate in equivalent
    }
    if substrate in candidate_substrates:
        return "match"
    if candidate_substrates == {"unspecified"} or None in candidate_substrates:
        return "unknown"
    return "mismatch"


def _isolation_year_gap(strain: str, candidates: Sequence[str]) -> float:
    parsed = parse_influenza_strain(strain)
    year = parsed.get("strain_isolation_year")
    if year is None:
        return float("nan")
    gaps: list[int] = []
    for candidate in _same_family_candidates(strain, candidates):
        candidate_year = parse_influenza_strain(candidate).get("strain_isolation_year")
        if candidate_year is not None:
            gaps.append(abs(int(year) - int(candidate_year)))
    return float(min(gaps)) if gaps else float("nan")


def _ontology_feature_rows(
    frame: pd.DataFrame,
    *,
    vaccine_by_season: Mapping[int, Sequence[str]],
    sequence_lookup: Mapping[str, Mapping[str, Any]],
    vaccine_2025: Sequence[str],
    include_sequence: bool,
    include_target_domain: bool,
) -> pd.DataFrame:
    require_columns(frame, ["virus_strain", "vaccine_season"], table_name="HAI model frame")
    unique = frame[["virus_strain", "vaccine_season"]].drop_duplicates().copy()
    records: list[dict[str, Any]] = []
    vaccine_2025 = tuple(canonicalize_strain(value) for value in vaccine_2025)
    vaccine_2025_components = {canonical_component_identity(value) for value in vaccine_2025}
    for row in unique.to_dict(orient="records"):
        strain = canonicalize_strain(row["virus_strain"])
        season_start = canonical_season_start(row["vaccine_season"])
        season_components = tuple(vaccine_by_season.get(season_start, ())) if season_start else ()
        component = canonical_component_identity(strain)
        exact_match = "unknown" if not season_components else (
            "yes" if strain in season_components else "no"
        )
        component_match = "unknown" if not season_components else (
            "yes"
            if component in {canonical_component_identity(value) for value in season_components}
            else "no"
        )
        record: dict[str, Any] = {
            "virus_strain": row["virus_strain"],
            "vaccine_season": row["vaccine_season"],
            "ontology_component": component,
            "ontology_official_exact_vaccine": exact_match,
            "ontology_official_component_vaccine": component_match,
            "ontology_vaccine_substrate_relation": (
                "unknown" if not season_components else _substrate_relation(strain, season_components)
            ),
            "ontology_isolation_year_gap_to_vaccine": (
                _isolation_year_gap(strain, season_components)
                if season_components
                else float("nan")
            ),
        }
        if include_sequence:
            sequence_entry = sequence_lookup.get(strain)
            record["ontology_sequence_available"] = int(sequence_entry is not None)
            record["ontology_sequence_complete"] = int(
                bool(sequence_entry and sequence_entry.get("complete"))
            )
            distance, _ = (
                _nearest_sequence_distance(strain, season_components, sequence_lookup)
                if season_components
                else (float("nan"), None)
            )
            record["ontology_seq_distance_to_study_vaccine"] = distance
        if include_target_domain:
            record["ontology_component_2025_vaccine"] = (
                "yes" if component in vaccine_2025_components else "no"
            )
            distance_2025, _ = _nearest_sequence_distance(
                strain, vaccine_2025, sequence_lookup
            )
            record["ontology_seq_distance_to_2025_vaccine"] = distance_2025
        records.append(record)
    result = pd.DataFrame(records)
    require_unique(
        result,
        ["virus_strain", "vaccine_season"],
        table_name="HAI ontology feature rows",
    )
    return result


def add_hai_ontology_features(
    dataset: HAIModelDataset,
    *,
    vaccine_reference: pd.DataFrame,
    sequence_reference: pd.DataFrame,
    vaccine_2025: Sequence[str],
    condition: str,
) -> HAIModelDataset:
    if condition not in CONDITIONS:
        raise DataContractError(f"unknown HAI transfer condition: {condition}")
    if condition == "b21_reference":
        return dataset
    sequence_lookup = build_sequence_lookup(sequence_reference)
    vaccine_by_season = _vaccine_map(vaccine_reference)
    include_sequence = condition in {
        "ontology_sequence_local",
        "ontology_sequence_target_domain",
    }
    include_target_domain = condition == "ontology_sequence_target_domain"

    def enrich(frame: pd.DataFrame) -> pd.DataFrame:
        features = _ontology_feature_rows(
            frame,
            vaccine_by_season=vaccine_by_season,
            sequence_lookup=sequence_lookup,
            vaccine_2025=vaccine_2025,
            include_sequence=include_sequence,
            include_target_domain=include_target_domain,
        )
        return frame.merge(
            features,
            on=["virus_strain", "vaccine_season"],
            how="left",
            validate="many_to_one",
        )

    enriched = replace(
        dataset,
        train=enrich(dataset.train),
        challenge=enrich(dataset.challenge),
        metadata={**dict(dataset.metadata), "hai_transfer_condition": condition},
    )
    enriched.validate()
    return enriched


def _task_specs(config: BaselineConfig) -> list[ModelSpec]:
    specs = config.model_specs("hai")
    if not specs:
        raise DataContractError("HAI model set is empty")
    return specs


def _metric_summary(evaluation: HAICandidateEvaluation) -> dict[str, Any]:
    return {
        "post_hai": evaluation.post_metrics,
        "within_strain_spearman": evaluation.within_strain_spearman,
        "panel_proxy": evaluation.panel_proxy_metrics,
        "panel_proxy_coverage": evaluation.panel_proxy_coverage,
        "panel_proxy_fold_summary": evaluation.panel_proxy_fold_summary,
        "panel_proxy_fold_metrics": evaluation.panel_proxy_fold_metrics,
    }


def _stress_tests(
    dataset: HAIModelDataset,
    *,
    selected_spec: ModelSpec,
    panel_strains: Sequence[str],
) -> Mapping[str, Any]:
    results: dict[str, Any] = {}
    try:
        season_splits = purged_leave_one_group_out(
            dataset.train["vaccine_season"].astype(str),
            dataset.train["subject_group"].astype(str),
            excluded_values=["unknown", "<na>", "nan"],
            prefix="vaccine_season",
        )
        season_summary = _metric_summary(
            evaluate_hai_spec(
                dataset,
                spec=selected_spec,
                splits=season_splits,
                panel_strains=panel_strains,
            )
        )
        season_summary["split_count"] = len(season_splits)
        results["purged_leave_one_vaccine_season_out"] = season_summary
    except (DataContractError, ValueError, TypeError, RuntimeError) as error:
        results["purged_leave_one_vaccine_season_out"] = {
            "status": "not_run",
            "reason": f"{type(error).__name__}: {error}",
        }

    overlap = sorted(
        set(dataset.train["virus_strain"].map(canonicalize_strain)).intersection(
            canonicalize_strain(value) for value in panel_strains
        )
    )
    try:
        strain_splits = purged_leave_one_group_out(
            dataset.train["virus_strain"].map(canonicalize_strain).astype(str),
            dataset.train["subject_group"].astype(str),
            included_values=overlap,
            prefix="strain",
        )
        strain_summary = _metric_summary(
            evaluate_hai_spec(
                dataset,
                spec=selected_spec,
                splits=strain_splits,
                panel_strains=panel_strains,
            )
        )
        strain_summary.update(
            {"split_count": len(strain_splits), "overlap_strains": len(overlap)}
        )
        results["purged_leave_one_strain_out"] = strain_summary
    except (DataContractError, ValueError, TypeError, RuntimeError) as error:
        results["purged_leave_one_strain_out"] = {
            "status": "not_run",
            "reason": f"{type(error).__name__}: {error}",
            "overlap_strains": len(overlap),
        }
    return results


def _run_condition(
    config: BaselineConfig,
    *,
    day28: HAIModelDataset,
    day365: HAIModelDataset,
    vaccine_strains: Sequence[str],
    challenge_strains: Sequence[str],
) -> Mapping[str, Any]:
    specs = _task_specs(config)
    selection_policy = str(
        config.section("selection", required=False).get("policy", "legacy")
    )
    if selection_policy != "robust_v1":
        raise DataContractError("HAI strain transfer requires selection.policy=robust_v1")
    day28_runs = run_hai_compact_for_panels(
        day28,
        specs=specs,
        selection_panels={"Task2.1": vaccine_strains, "Task2.2": challenge_strains},
        selection_policy=selection_policy,
    )
    day365_runs = run_hai_compact_for_panels(
        day365,
        specs=specs,
        selection_panels={"Task2.3": challenge_strains},
        selection_policy=selection_policy,
    )
    runs: dict[str, HAIRunResult] = {**day28_runs, **day365_runs}
    panels = {
        "Task2.1": tuple(vaccine_strains),
        "Task2.2": tuple(challenge_strains),
        "Task2.3": tuple(challenge_strains),
    }
    datasets = {"Task2.1": day28, "Task2.2": day28, "Task2.3": day365}
    return {
        "runs": runs,
        "summary": {
            task: {
                "selected_model": run.selected_spec.to_dict(),
                "metrics": run.metrics,
                "candidate_summaries": run.candidate_summaries,
                "candidate_failures": run.candidate_failures,
                "stress": _stress_tests(
                    datasets[task],
                    selected_spec=run.selected_spec,
                    panel_strains=panels[task],
                ),
            }
            for task, run in runs.items()
        },
    }


def _challenge_rank_agreement(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> Mapping[str, Any]:
    require_columns(
        reference, ["participant_id", "prediction"], table_name="reference task prediction"
    )
    require_columns(
        candidate, ["participant_id", "prediction"], table_name="candidate task prediction"
    )
    aligned = reference.merge(
        candidate,
        on="participant_id",
        how="inner",
        suffixes=("_reference", "_candidate"),
        validate="one_to_one",
    )
    if len(aligned) != 40:
        raise DataContractError(f"HAI challenge agreement expected 40 donors; found {len(aligned)}")
    ref = aligned["prediction_reference"].to_numpy(dtype=float)
    cand = aligned["prediction_candidate"].to_numpy(dtype=float)
    ref_rank = percentile_rank(ref)
    cand_rank = percentile_rank(cand)
    difference = np.abs(ref_rank - cand_rank)
    return {
        "n": 40,
        "rank_spearman": safe_spearman(ref, cand).to_dict(),
        "mean_absolute_percentile_difference": float(np.mean(difference)),
        "max_absolute_percentile_difference": float(np.max(difference)),
    }


def _finite(value: Any, fallback: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if np.isfinite(result) else fallback


def _promotion(
    conditions: Mapping[str, Mapping[str, Any]],
    *,
    task: str,
    candidate: str,
    strain_floor_condition: str | None = None,
) -> Mapping[str, Any]:
    reference = conditions["b21_reference"][task]
    proposed = conditions[candidate][task]
    reference_standard = reference["metrics"]["panel_proxy_fold_summary"]
    proposed_standard = proposed["metrics"]["panel_proxy_fold_summary"]
    reference_strain = reference["stress"].get("purged_leave_one_strain_out", {})
    proposed_strain = proposed["stress"].get("purged_leave_one_strain_out", {})
    reference_strain_mean = _finite(
        reference_strain.get("panel_proxy_fold_summary", {}).get("spearman_mean")
    )
    proposed_strain_mean = _finite(
        proposed_strain.get("panel_proxy_fold_summary", {}).get("spearman_mean")
    )
    reference_mean = _finite(reference_standard.get("spearman_mean"))
    proposed_mean = _finite(proposed_standard.get("spearman_mean"))
    reference_min = _finite(reference_standard.get("spearman_min"))
    proposed_min = _finite(proposed_standard.get("spearman_min"))
    checks = {
        "strain_mean_strictly_higher": bool(
            np.isfinite(reference_strain_mean)
            and np.isfinite(proposed_strain_mean)
            and proposed_strain_mean > reference_strain_mean
        ),
        "held_study_mean_within_tolerance": bool(
            np.isfinite(reference_mean)
            and np.isfinite(proposed_mean)
            and proposed_mean >= reference_mean - PROMOTION_MEAN_TOLERANCE
        ),
        "held_study_min_within_tolerance": bool(
            np.isfinite(reference_min)
            and np.isfinite(proposed_min)
            and proposed_min >= reference_min - PROMOTION_MIN_TOLERANCE
        ),
    }
    if strain_floor_condition is not None:
        floor = conditions[strain_floor_condition][task]["stress"].get(
            "purged_leave_one_strain_out", {}
        )
        floor_mean = _finite(floor.get("panel_proxy_fold_summary", {}).get("spearman_mean"))
        checks["strain_mean_at_least_local_sequence"] = bool(
            np.isfinite(floor_mean)
            and np.isfinite(proposed_strain_mean)
            and proposed_strain_mean >= floor_mean
        )
    return {
        "candidate": candidate,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "reference": {
            "held_study_mean": reference_mean,
            "held_study_min": reference_min,
            "purged_strain_mean": reference_strain_mean,
        },
        "candidate_metrics": {
            "held_study_mean": proposed_mean,
            "held_study_min": proposed_min,
            "purged_strain_mean": proposed_strain_mean,
        },
    }


def challenge_only_sequence_neighbors(
    public_serology: pd.DataFrame,
    *,
    challenge_strains: Sequence[str],
    vaccine_2025: Sequence[str],
    sequence_reference: pd.DataFrame,
) -> Sequence[Mapping[str, Any]]:
    require_columns(public_serology, ["assay", "virus_strain"], table_name="public serology")
    public = public_serology.loc[
        public_serology["assay"].fillna("").astype(str).str.strip().str.casefold().eq("hai"),
        "virus_strain",
    ].map(canonicalize_strain)
    public_strains = sorted(set(public.dropna().astype(str)))
    challenge = tuple(canonicalize_strain(value) for value in challenge_strains)
    only = sorted(set(challenge) - set(public_strains))
    lookup = build_sequence_lookup(sequence_reference)
    results: list[Mapping[str, Any]] = []
    for strain in only:
        entry = lookup.get(strain)
        public_distance, public_neighbor = _nearest_sequence_distance(
            strain, public_strains, lookup
        )
        vaccine_distance, vaccine_neighbor = _nearest_sequence_distance(
            strain, vaccine_2025, lookup
        )
        results.append(
            {
                "challenge_strain": strain,
                "sequence_available": entry is not None,
                "sequence_status": None if entry is None else entry.get("sequence_status"),
                "nearest_public_strain": public_neighbor,
                "nearest_public_distance": (
                    None if not np.isfinite(public_distance) else public_distance
                ),
                "nearest_2025_vaccine_strain": vaccine_neighbor,
                "nearest_2025_vaccine_distance": (
                    None if not np.isfinite(vaccine_distance) else vaccine_distance
                ),
            }
        )
    return results


def run_hai_strain_transfer_experiment(
    config: BaselineConfig,
    inputs: Any,
    *,
    sequence_reference: pd.DataFrame,
    vaccine_reference: pd.DataFrame,
) -> Mapping[str, Any]:
    """Run the locked aggregate-only Phase A HAI strain-transfer experiment."""

    hai_config = config.section("hai")
    if str(hai_config.get("target_representation", "residual")) != "residual":
        raise DataContractError("HAI strain transfer requires residual target representation")
    if str(config.section("selection", required=False).get("policy", "legacy")) != "robust_v1":
        raise DataContractError("HAI strain transfer requires robust_v1 selection")

    tables = inputs.tables
    base_day28 = build_hai_model_dataset(
        tables["public_serology"],
        tables["challenge_serology"],
        tables["participants"],
        tables["investigations"],
        day=28,
        target_representation="residual",
        challenge_panel_strains=inputs.challenge_strains,
    )
    base_day365 = build_hai_model_dataset(
        tables["public_serology"],
        tables["challenge_serology"],
        tables["participants"],
        tables["investigations"],
        day=365,
        target_representation="residual",
        challenge_panel_strains=inputs.challenge_strains,
    )

    condition_results: dict[str, Mapping[str, Any]] = {}
    for condition in CONDITIONS:
        day28 = add_hai_ontology_features(
            base_day28,
            vaccine_reference=vaccine_reference,
            sequence_reference=sequence_reference,
            vaccine_2025=inputs.vaccine_strains,
            condition=condition,
        )
        day365 = add_hai_ontology_features(
            base_day365,
            vaccine_reference=vaccine_reference,
            sequence_reference=sequence_reference,
            vaccine_2025=inputs.vaccine_strains,
            condition=condition,
        )
        condition_results[condition] = _run_condition(
            config,
            day28=day28,
            day365=day365,
            vaccine_strains=inputs.vaccine_strains,
            challenge_strains=inputs.challenge_strains,
        )

    agreements: dict[str, Any] = {}
    reference_runs = condition_results["b21_reference"]["runs"]
    panels = {
        "Task2.1": inputs.vaccine_strains,
        "Task2.2": inputs.challenge_strains,
        "Task2.3": inputs.challenge_strains,
    }
    for condition in CONDITIONS[1:]:
        agreements[condition] = {}
        for task, panel in panels.items():
            reference_task = aggregate_hai_task_predictions(
                reference_runs[task].challenge_strain_predictions,
                panel_strains=panel,
                task=task,
            )
            candidate_task = aggregate_hai_task_predictions(
                condition_results[condition]["runs"][task].challenge_strain_predictions,
                panel_strains=panel,
                task=task,
            )
            agreements[condition][task] = _challenge_rank_agreement(
                reference_task, candidate_task
            )

    public_summary = {
        condition: result["summary"] for condition, result in condition_results.items()
    }
    promotion = {
        task: {
            "scientific_sequence_local": _promotion(
                public_summary,
                task=task,
                candidate="ontology_sequence_local",
            ),
            "competition_target_domain": _promotion(
                public_summary,
                task=task,
                candidate="ontology_sequence_target_domain",
                strain_floor_condition="ontology_sequence_local",
            ),
        }
        for task in ("Task2.1", "Task2.2", "Task2.3")
    }

    return {
        "experiment": "phase_a_hai_strain_transfer",
        "conditions": public_summary,
        "promotion": promotion,
        "challenge_rank_agreements": agreements,
        "challenge_only_sequence_neighbors": challenge_only_sequence_neighbors(
            tables["public_serology"],
            challenge_strains=inputs.challenge_strains,
            vaccine_2025=inputs.vaccine_strains,
            sequence_reference=sequence_reference,
        ),
        "sequence_reference_rows": int(len(sequence_reference)),
        "sequence_lookup_strains": int(len(build_sequence_lookup(sequence_reference))),
        "vaccine_reference_rows": int(len(vaccine_reference)),
        "leaderboard_used_for_selection": False,
        "competition_submission_attempted": False,
        "output_policy": "aggregate_only_no_participant_ids_or_row_level_predictions_public_strain_names_allowed",
    }
