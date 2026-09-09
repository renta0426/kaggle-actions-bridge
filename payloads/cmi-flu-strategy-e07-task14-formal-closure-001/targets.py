"""Canonical baseline extraction and Part 1 target construction."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from .aliases import (
    canonicalize_aim_stimulation,
    canonicalize_cytokine,
    canonicalize_flow_population,
    canonicalize_strain,
    canonicalize_timepoint,
)
from .contracts import (
    DataContractError,
    coerce_numeric,
    require_columns,
    require_positive,
    require_unique,
)

ID_COLUMNS: tuple[str, ...] = ("participant_id", "subject", "study_accession")


def _present_id_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in ID_COLUMNS if column in frame.columns]
    if "participant_id" not in columns:
        raise DataContractError("assay table is missing participant_id")
    return columns


def _canonical_timepoints(frame: pd.DataFrame, timepoint_column: str) -> pd.DataFrame:
    result = frame.copy()
    result[timepoint_column] = result[timepoint_column].map(canonicalize_timepoint)
    return result


def _aggregate_values(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    value_column: str,
    output_column: str,
) -> pd.DataFrame:
    grouped = (
        frame.groupby(list(group_columns), dropna=False, observed=True)[value_column]
        .mean()
        .rename(output_column)
        .reset_index()
    )
    require_unique(grouped, group_columns, table_name=output_column)
    return grouped


def select_pre_vacc(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    value_column: str = "value",
    timepoint_column: str = "timepoint",
    output_column: str = "pre_vacc",
) -> pd.DataFrame:
    """Select literal Pre-vacc, falling back to the arithmetic mean of timepoints <= 0."""

    require_columns(
        frame,
        [*group_columns, value_column, timepoint_column],
        table_name="baseline source",
    )
    working = _canonical_timepoints(frame, timepoint_column)
    working[value_column] = coerce_numeric(
        working[value_column], name=value_column, allow_missing=True
    )
    working = working.dropna(subset=[value_column])

    literal = working.loc[working[timepoint_column] == "Pre-vacc"].copy()
    literal_result = _aggregate_values(
        literal,
        group_columns=group_columns,
        value_column=value_column,
        output_column=output_column,
    )
    literal_result["baseline_source"] = "literal_pre_vacc"

    numeric_timepoint = pd.to_numeric(working[timepoint_column], errors="coerce")
    fallback = working.loc[numeric_timepoint.notna() & (numeric_timepoint <= 0)].copy()
    fallback_result = _aggregate_values(
        fallback,
        group_columns=group_columns,
        value_column=value_column,
        output_column=output_column,
    )
    fallback_result["baseline_source"] = "arithmetic_mean_nonpositive_timepoints"

    if literal_result.empty:
        result = fallback_result
    elif fallback_result.empty:
        result = literal_result
    else:
        literal_keys = pd.MultiIndex.from_frame(literal_result[list(group_columns)])
        fallback_keys = pd.MultiIndex.from_frame(fallback_result[list(group_columns)])
        fallback_result = fallback_result.loc[~fallback_keys.isin(literal_keys)]
        result = pd.concat([literal_result, fallback_result], ignore_index=True)

    if result.empty:
        raise DataContractError("no literal Pre-vacc or numeric timepoint <= 0 values were found")
    require_unique(result, group_columns, table_name="canonical baseline")
    return result


def select_timepoint(
    frame: pd.DataFrame,
    timepoint: int | float | str,
    *,
    group_columns: Sequence[str],
    value_column: str = "value",
    timepoint_column: str = "timepoint",
    output_column: str = "post_value",
) -> pd.DataFrame:
    require_columns(frame, [*group_columns, value_column, timepoint_column])
    working = _canonical_timepoints(frame, timepoint_column)
    target = canonicalize_timepoint(timepoint)
    selected = working.loc[working[timepoint_column] == target].copy()
    if selected.empty:
        raise DataContractError(f"no rows found for canonical timepoint {target!r}")
    selected[value_column] = coerce_numeric(
        selected[value_column], name=value_column, allow_missing=True
    )
    selected = selected.dropna(subset=[value_column])
    return _aggregate_values(
        selected,
        group_columns=group_columns,
        value_column=value_column,
        output_column=output_column,
    )


def build_task_11_target(cytokine: pd.DataFrame) -> pd.DataFrame:
    """Build Task 1.1: raw D1 IP10 divided by canonical Pre-vacc IP10."""

    require_columns(cytokine, ["participant_id", "timepoint", "analyte", "value"])
    working = cytokine.copy()
    working["analyte"] = working["analyte"].map(canonicalize_cytokine)
    working = working.loc[working["analyte"] == "CXCL10"].copy()
    ids = _present_id_columns(working)
    groups = [*ids, "analyte"]
    baseline = select_pre_vacc(working, group_columns=groups)
    day1 = select_timepoint(working, 1, group_columns=groups, output_column="day1")
    target = baseline.merge(day1, on=groups, how="inner", validate="one_to_one")
    require_positive(target["pre_vacc"], name="Task1.1 pre_vacc IP10")
    require_positive(target["day1"], name="Task1.1 day1 IP10")
    target["target"] = target["day1"] / target["pre_vacc"]
    target["target_log"] = np.log(target["target"])
    return target[[*ids, "pre_vacc", "day1", "target", "target_log", "baseline_source"]]


def _prepare_flow(frame: pd.DataFrame, *, include_sdy272_asc_proxy: bool) -> pd.DataFrame:
    require_columns(frame, ["participant_id", "timepoint", "name", "value"])
    working = frame.copy()
    working["name"] = working["name"].map(canonicalize_flow_population)
    if include_sdy272_asc_proxy:
        required = ["study_accession", "population_definition"]
        require_columns(working, required, table_name="flow ASC proxy source")
        definition = working["population_definition"].fillna("").astype(str).str.upper()
        proxy = (
            working["study_accession"].astype(str).eq("SDY272")
            & working["name"].astype(str).str.casefold().eq("b_cells")
            & definition.str.contains("CD19+", regex=False)
            & definition.str.contains(r"(?:MS4A1|CD20)-", regex=True)
            & definition.str.contains("CD27+", regex=False)
            & definition.str.contains("CD38++", regex=False)
        )
        working.loc[proxy, "name"] = "Antibody-secreting_cells_(ASC)"
    return working


def build_flow_absolute_target(
    flow: pd.DataFrame,
    *,
    population: str,
    day: int,
    include_sdy272_asc_proxy: bool = False,
) -> pd.DataFrame:
    working = _prepare_flow(flow, include_sdy272_asc_proxy=include_sdy272_asc_proxy)
    canonical_population = canonicalize_flow_population(population)
    working = working.loc[working["name"] == canonical_population].copy()
    ids = _present_id_columns(working)
    target = select_timepoint(
        working,
        day,
        group_columns=[*ids, "name"],
        output_column="target",
    )
    return target[[*ids, "target"]]


def build_task_12_target(flow: pd.DataFrame) -> pd.DataFrame:
    return build_flow_absolute_target(flow, population="Classical_monocytes", day=1)


def build_task_13_target(
    flow: pd.DataFrame,
    *,
    include_sdy272_asc_proxy: bool = False,
) -> pd.DataFrame:
    return build_flow_absolute_target(
        flow,
        population="Antibody-secreting_cells_(ASC)",
        day=7,
        include_sdy272_asc_proxy=include_sdy272_asc_proxy,
    )


def build_task_14_anchor(aim: pd.DataFrame) -> pd.DataFrame:
    """Build the unlabeled Task 1.4 anchor from Pre-vacc Conserved AIM."""

    require_columns(aim, ["participant_id", "timepoint", "stimulation", "value"])
    working = aim.copy()
    working["stimulation"] = working["stimulation"].map(canonicalize_aim_stimulation)
    working = working.loc[working["stimulation"] == "Conserved"].copy()
    ids = _present_id_columns(working)
    optional_measure = [column for column in ("name",) if column in working.columns]
    baseline = select_pre_vacc(
        working,
        group_columns=[*ids, "stimulation", *optional_measure],
        output_column="anchor",
    )
    if optional_measure:
        baseline = (
            baseline.groupby(ids, dropna=False, observed=True)
            .agg(anchor=("anchor", "mean"), baseline_source=("baseline_source", "first"))
            .reset_index()
        )
    return baseline[[*ids, "anchor", "baseline_source"]]


def _prepare_hai(serology: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        serology,
        ["participant_id", "timepoint", "virus_strain", "assay", "value"],
        table_name="serology",
    )
    working = serology.copy()
    assay = working["assay"].fillna("").astype(str).str.strip().str.casefold()
    working = working.loc[assay == "hai"].copy()
    working["virus_strain_raw"] = working["virus_strain"].astype("string")
    working["virus_strain"] = working["virus_strain"].map(canonicalize_strain)
    working["value"] = coerce_numeric(working["value"], name="HAI value", allow_missing=True)
    working = working.dropna(subset=["value"])
    require_positive(working["value"], name="HAI value")
    return working


def build_hai_long_target(
    serology: pd.DataFrame,
    *,
    day: int,
) -> pd.DataFrame:
    """Pair canonical baseline and post-vaccination HAI per participant and strain."""

    working = _prepare_hai(serology)
    ids = _present_id_columns(working)
    groups = [*ids, "virus_strain"]
    baseline = select_pre_vacc(working, group_columns=groups, output_column="pre_hai")
    post = select_timepoint(
        working,
        day,
        group_columns=groups,
        output_column="post_hai",
    )
    paired = baseline.merge(post, on=groups, how="inner", validate="one_to_one")
    require_positive(paired["pre_hai"], name=f"D{day} paired pre HAI")
    require_positive(paired["post_hai"], name=f"D{day} paired post HAI")
    paired["log2_pre_hai"] = np.log2(paired["pre_hai"])
    paired["log2_post_hai"] = np.log2(paired["post_hai"])
    paired["log2_fold"] = paired["log2_post_hai"] - paired["log2_pre_hai"]
    paired["target_day"] = int(day)

    optional = [column for column in ("virus_in_vaccine",) if column in working.columns]
    if optional:
        metadata = (
            working[groups + optional]
            .drop_duplicates(groups + optional)
            .groupby(groups, dropna=False, observed=True)[optional]
            .first()
            .reset_index()
        )
        paired = paired.merge(metadata, on=groups, how="left", validate="one_to_one")
    return paired


def geometric_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise DataContractError("geometric mean requires at least one value")
    require_positive(array, name="geometric mean input")
    return float(np.exp(np.mean(np.log(array))))


def aggregate_hai_panel(
    frame: pd.DataFrame,
    *,
    panel_strains: Sequence[str],
    value_column: str,
    group_columns: Sequence[str] = ("participant_id",),
    output_column: str = "panel_geometric_mean",
    require_complete: bool = True,
) -> pd.DataFrame:
    """Aggregate strain-level values over an explicitly supplied challenge panel."""

    require_columns(frame, [*group_columns, "virus_strain", value_column])
    canonical_panel = tuple(canonicalize_strain(strain) for strain in panel_strains)
    if not canonical_panel or len(set(canonical_panel)) != len(canonical_panel):
        raise DataContractError("panel_strains must be non-empty and unique after canonicalization")

    working = frame.copy()
    working["virus_strain"] = working["virus_strain"].map(canonicalize_strain)
    working[value_column] = coerce_numeric(
        working[value_column], name=value_column, allow_missing=True
    )
    working = working.loc[working["virus_strain"].isin(canonical_panel)].dropna(
        subset=[value_column]
    )
    require_unique(
        working,
        [*group_columns, "virus_strain"],
        table_name="HAI panel strain values",
    )

    counts = working.groupby(list(group_columns), dropna=False, observed=True)[
        "virus_strain"
    ].nunique()
    if require_complete:
        incomplete = counts[counts != len(canonical_panel)]
        if not incomplete.empty:
            examples = incomplete.head(5).to_dict()
            raise DataContractError(
                f"incomplete HAI panel for {len(incomplete)} groups; examples={examples}"
            )

    aggregated = (
        working.groupby(list(group_columns), dropna=False, observed=True)[value_column]
        .apply(geometric_mean)
        .rename(output_column)
        .reset_index()
    )
    aggregated["panel_size_observed"] = (
        working.groupby(list(group_columns), dropna=False, observed=True)["virus_strain"]
        .nunique()
        .to_numpy()
    )
    return aggregated


def build_hai_anchor(
    serology: pd.DataFrame,
    *,
    panel_strains: Sequence[str],
    output_column: str,
) -> pd.DataFrame:
    working = _prepare_hai(serology)
    ids = _present_id_columns(working)
    baseline = select_pre_vacc(
        working,
        group_columns=[*ids, "virus_strain"],
        output_column="pre_hai",
    )
    return aggregate_hai_panel(
        baseline,
        panel_strains=panel_strains,
        value_column="pre_hai",
        group_columns=ids,
        output_column=output_column,
        require_complete=True,
    )
