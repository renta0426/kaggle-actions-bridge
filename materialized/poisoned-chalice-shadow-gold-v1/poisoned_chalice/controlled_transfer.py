"""Construct an exact-membership controlled transfer benchmark.

A model trained from scratch on ``train_corpus`` has unambiguous sample-level
membership: evaluation rows with ``membership == 1`` are selected from that
corpus, while rows with ``membership == 0`` come from disjoint source/near-
duplicate clusters. Existing membership labels are rejected so a benchmark
cannot accidentally inherit labels belonging to some other target model.

The splitter is intentionally model-independent. The same frozen split and
exposure schedule can be used to train two different causal-LM architectures;
an attack fitted or selected on model A can then be evaluated unchanged on
model B, which is the transfer experiment missing from the Stage 1-only setup.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd


CONTROLLED_BENCHMARK_VERSION = "controlled-membership-transfer-v1"
_FORBIDDEN_LABEL_COLUMNS = {
    "label",
    "labels",
    "membership",
    "is_member",
    "member",
    "target",
    "y",
}
_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|[^\s]", re.UNICODE)
_UINT64_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class ControlledTransferConfig:
    content_column: str = "content"
    language_column: str = "language"
    group_column: str | None = None
    seed: int = 2027
    member_fraction: float = 0.5
    eval_rows_per_language_per_class: int = 1_000
    min_eval_rows_per_language_per_class: int = 50
    min_characters: int = 64
    max_characters: int = 32_000
    length_bin_edges: tuple[int, ...] = (
        0,
        128,
        256,
        512,
        1_024,
        2_048,
        4_096,
        8_192,
        16_384,
        32_001,
    )
    shingle_size: int = 5
    fingerprint_size: int = 32
    lsh_bands: int = 8
    near_duplicate_threshold: float = 0.85
    max_lsh_bucket_rows: int = 256

    def __post_init__(self) -> None:
        if not self.content_column or not self.language_column:
            raise ValueError("content_column and language_column are required")
        if not 0 < self.member_fraction < 1:
            raise ValueError("member_fraction must be strictly between 0 and 1")
        positive = {
            "eval_rows_per_language_per_class": self.eval_rows_per_language_per_class,
            "min_eval_rows_per_language_per_class": self.min_eval_rows_per_language_per_class,
            "min_characters": self.min_characters,
            "max_characters": self.max_characters,
            "shingle_size": self.shingle_size,
            "fingerprint_size": self.fingerprint_size,
            "lsh_bands": self.lsh_bands,
            "max_lsh_bucket_rows": self.max_lsh_bucket_rows,
        }
        for name, value in positive.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_characters > self.max_characters:
            raise ValueError("min_characters must not exceed max_characters")
        if self.min_eval_rows_per_language_per_class > self.eval_rows_per_language_per_class:
            raise ValueError("minimum evaluation rows cannot exceed the requested cap")
        if self.fingerprint_size % self.lsh_bands:
            raise ValueError("fingerprint_size must be divisible by lsh_bands")
        if not 0 < self.near_duplicate_threshold <= 1:
            raise ValueError("near_duplicate_threshold must be in (0, 1]")
        edges = tuple(int(value) for value in self.length_bin_edges)
        if len(edges) < 2 or edges[0] > self.min_characters or edges[-1] <= self.max_characters:
            raise ValueError("length_bin_edges must cover the configured character range")
        if any(right <= left for left, right in zip(edges, edges[1:])):
            raise ValueError("length_bin_edges must be strictly increasing")


@dataclass(frozen=True)
class ControlledTransferBenchmark:
    train_corpus: pd.DataFrame
    evaluation: pd.DataFrame
    manifest: Mapping[str, Any]


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _stable_digest(value: str, *, size: int = 16) -> str:
    return hashlib.blake2b(value.encode("utf-8"), digest_size=size).hexdigest()


def _stable_order_key(seed: int, value: str) -> str:
    return _stable_digest(f"{seed}\0{value}", size=16)


def _lexical_tokens(content: str) -> list[str]:
    return _TOKEN_PATTERN.findall(content)


def _shingle_hashes(content: str, shingle_size: int) -> frozenset[int]:
    tokens = _lexical_tokens(content)
    if not tokens:
        return frozenset()
    if len(tokens) < shingle_size:
        payloads = ["\x1f".join(tokens)]
    else:
        payloads = [
            "\x1f".join(tokens[start : start + shingle_size])
            for start in range(len(tokens) - shingle_size + 1)
        ]
    return frozenset(
        int.from_bytes(
            hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest(),
            "big",
        )
        for payload in payloads
    )


def _bottom_k_fingerprint(shingles: frozenset[int], size: int) -> tuple[int, ...]:
    ordered = sorted(shingles)
    if len(ordered) >= size:
        return tuple(ordered[:size])
    return tuple(ordered + [_UINT64_MASK] * (size - len(ordered)))


def _jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    if not left and not right:
        return 1.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def _validate_columns(frame: pd.DataFrame, config: ControlledTransferConfig) -> None:
    normalised = {str(column).strip().casefold() for column in frame.columns}
    forbidden = sorted(normalised & _FORBIDDEN_LABEL_COLUMNS)
    if forbidden:
        raise ValueError(
            "controlled benchmark input must not contain pre-existing membership labels: "
            f"{forbidden}"
        )
    required = {config.content_column, config.language_column}
    if config.group_column is not None:
        required.add(config.group_column)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"controlled benchmark input is missing columns: {missing}")


def _prepare_rows(
    frame: pd.DataFrame,
    config: ControlledTransferConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    _validate_columns(frame, config)
    columns = [config.content_column, config.language_column]
    if config.group_column is not None:
        columns.append(config.group_column)
    prepared = frame[columns].copy()
    rename = {
        config.content_column: "content",
        config.language_column: "language",
    }
    if config.group_column is not None:
        rename[config.group_column] = "source_group"
    prepared = prepared.rename(columns=rename)
    prepared["content"] = prepared.content.astype(str)
    prepared["language"] = prepared.language.astype(str).str.strip()
    if prepared.language.eq("").any():
        raise ValueError("language values must be non-empty")
    if config.group_column is not None:
        if prepared.source_group.isna().any() or prepared.source_group.astype(str).str.strip().eq("").any():
            raise ValueError("source group values must be non-empty when group_column is set")
        prepared["source_group"] = prepared.source_group.astype(str)
    prepared["character_count"] = prepared.content.str.len().astype(int)
    input_rows = len(prepared)
    prepared = prepared[
        prepared.character_count.between(config.min_characters, config.max_characters)
    ].copy()
    filtered_rows = input_rows - len(prepared)
    prepared["content_hash"] = prepared.content.map(lambda value: _stable_digest(value, size=32))
    before_exact = len(prepared)
    prepared = (
        prepared.sort_values(["content_hash", "language"], kind="mergesort")
        .drop_duplicates("content_hash", keep="first")
        .reset_index(drop=True)
    )
    exact_duplicates_removed = before_exact - len(prepared)
    if prepared.empty:
        raise ValueError("no rows remain after controlled benchmark filtering")
    edges = np.asarray(config.length_bin_edges, dtype=int)
    prepared["length_bin"] = np.searchsorted(
        edges, prepared.character_count.to_numpy(int), side="right"
    ) - 1
    prepared["benchmark_id"] = prepared.content_hash.map(lambda value: f"ct-{value[:24]}")
    return prepared, {
        "input_rows": input_rows,
        "filtered_by_length": filtered_rows,
        "exact_duplicates_removed": exact_duplicates_removed,
        "unique_rows": len(prepared),
    }


def _cluster_rows(
    rows: pd.DataFrame,
    config: ControlledTransferConfig,
) -> tuple[np.ndarray, dict[str, int]]:
    union_find = _UnionFind(len(rows))
    group_unions = 0
    if config.group_column is not None:
        groups: dict[str, list[int]] = defaultdict(list)
        for index, value in enumerate(rows.source_group.astype(str)):
            groups[value].append(index)
        for indices in groups.values():
            for other in indices[1:]:
                union_find.union(indices[0], other)
                group_unions += 1

    shingle_sets = [
        _shingle_hashes(content, config.shingle_size)
        for content in rows.content
    ]
    fingerprints = [
        _bottom_k_fingerprint(shingles, config.fingerprint_size)
        for shingles in shingle_sets
    ]
    band_width = config.fingerprint_size // config.lsh_bands
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for row_index, signature in enumerate(fingerprints):
        for band in range(config.lsh_bands):
            start = band * band_width
            key = (band, signature[start : start + band_width])
            buckets[key].append(row_index)

    candidates: set[tuple[int, int]] = set()
    for key, indices in buckets.items():
        if len(indices) > config.max_lsh_bucket_rows:
            raise ValueError(
                "an LSH bucket exceeded the bounded comparison budget; "
                f"key_band={key[0]} rows={len(indices)}"
            )
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                candidates.add((min(left, right), max(left, right)))

    near_duplicate_pairs = 0
    for left, right in sorted(candidates):
        if _jaccard(shingle_sets[left], shingle_sets[right]) >= config.near_duplicate_threshold:
            union_find.union(left, right)
            near_duplicate_pairs += 1

    root_to_cluster: dict[int, int] = {}
    clusters = np.empty(len(rows), dtype=np.int64)
    for index in range(len(rows)):
        root = union_find.find(index)
        if root not in root_to_cluster:
            root_to_cluster[root] = len(root_to_cluster)
        clusters[index] = root_to_cluster[root]
    return clusters, {
        "source_group_unions": group_unions,
        "lsh_candidate_pairs": len(candidates),
        "near_duplicate_pairs": near_duplicate_pairs,
        "clusters": len(root_to_cluster),
    }


def _cluster_stratum_counts(rows: pd.DataFrame) -> dict[int, Counter[tuple[str, int]]]:
    result: dict[int, Counter[tuple[str, int]]] = {}
    for cluster_id, group in rows.groupby("cluster_id", sort=False):
        result[int(cluster_id)] = Counter(
            zip(group.language.astype(str), group.length_bin.astype(int))
        )
    return result


def _assign_clusters(rows: pd.DataFrame, config: ControlledTransferConfig) -> dict[int, int]:
    cluster_counts = _cluster_stratum_counts(rows)
    total = Counter(zip(rows.language.astype(str), rows.length_bin.astype(int)))
    target = {stratum: count * config.member_fraction for stratum, count in total.items()}
    current: Counter[tuple[str, int]] = Counter()
    assignments: dict[int, int] = {}
    ordered_clusters = sorted(
        cluster_counts,
        key=lambda cluster_id: _stable_order_key(config.seed, f"cluster-{cluster_id}"),
    )
    for cluster_id in ordered_clusters:
        counts = cluster_counts[cluster_id]
        affected = set(counts)
        member_cost = sum(
            abs((current[stratum] + counts[stratum]) - target[stratum])
            for stratum in affected
        )
        nonmember_cost = sum(
            abs(current[stratum] - target[stratum])
            for stratum in affected
        )
        if member_cost < nonmember_cost:
            assignment = 1
        elif member_cost > nonmember_cost:
            assignment = 0
        else:
            assignment = int(
                int(_stable_order_key(config.seed + 1, f"cluster-{cluster_id}"), 16) % 2 == 0
            )
        assignments[cluster_id] = assignment
        if assignment:
            current.update(counts)
    if set(assignments.values()) != {0, 1}:
        raise ValueError("controlled split produced only one membership class")
    return assignments


def _ordered_indices(frame: pd.DataFrame, seed: int, salt: str) -> list[int]:
    return sorted(
        frame.index.tolist(),
        key=lambda index: _stable_order_key(
            seed, f"{salt}\0{frame.at[index, 'benchmark_id']}"
        ),
    )


def _matched_evaluation(rows: pd.DataFrame, config: ControlledTransferConfig) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    pair_counter = 0
    for language in sorted(rows.language.unique()):
        pairs: list[tuple[int, int, int]] = []
        language_rows = rows[rows.language == language]
        for length_bin in sorted(language_rows.length_bin.unique()):
            stratum = language_rows[language_rows.length_bin == length_bin]
            member = stratum[stratum.membership == 1]
            nonmember = stratum[stratum.membership == 0]
            member_indices = _ordered_indices(member, config.seed + 2, "member")
            nonmember_indices = _ordered_indices(nonmember, config.seed + 3, "nonmember")
            for left, right in zip(member_indices, nonmember_indices):
                pairs.append((left, right, int(length_bin)))
        pairs = sorted(
            pairs,
            key=lambda pair: _stable_order_key(
                config.seed + 4,
                f"{rows.at[pair[0], 'benchmark_id']}\0{rows.at[pair[1], 'benchmark_id']}",
            ),
        )[: config.eval_rows_per_language_per_class]
        if len(pairs) < config.min_eval_rows_per_language_per_class:
            raise ValueError(
                f"insufficient matched evaluation rows for {language}: {len(pairs)} per class"
            )
        for member_index, nonmember_index, _ in pairs:
            pair_id = f"pair-{pair_counter:08d}"
            pair_counter += 1
            member_row = rows.loc[[member_index]].copy()
            member_row["matched_pair_id"] = pair_id
            nonmember_row = rows.loc[[nonmember_index]].copy()
            nonmember_row["matched_pair_id"] = pair_id
            selected.extend([member_row, nonmember_row])
    evaluation = pd.concat(selected, ignore_index=True)
    order = sorted(
        evaluation.index,
        key=lambda index: _stable_order_key(
            config.seed + 5, evaluation.at[index, "benchmark_id"]
        ),
    )
    return evaluation.loc[order].reset_index(drop=True)


def _distribution_manifest(evaluation: pd.DataFrame) -> dict[str, Any]:
    by_language = {
        str(language): {
            str(int(label)): int(count)
            for label, count in group.groupby("membership").size().items()
        }
        for language, group in evaluation.groupby("language", sort=True)
    }
    by_language_length = {
        f"{language}|{int(length_bin)}|{int(label)}": int(count)
        for (language, length_bin, label), count in evaluation.groupby(
            ["language", "length_bin", "membership"], sort=True
        ).size().items()
    }
    return {
        "by_language_and_class": by_language,
        "by_language_length_and_class": by_language_length,
    }


def build_controlled_transfer_benchmark(
    frame: pd.DataFrame,
    config: ControlledTransferConfig | None = None,
) -> ControlledTransferBenchmark:
    """Build a reproducible member-training corpus and matched evaluation set."""

    runtime = config or ControlledTransferConfig()
    rows, preparation = _prepare_rows(frame, runtime)
    clusters, clustering = _cluster_rows(rows, runtime)
    rows = rows.copy()
    rows["cluster_id"] = clusters
    assignment = _assign_clusters(rows, runtime)
    rows["membership"] = rows.cluster_id.map(assignment).astype(np.int8)

    evaluation = _matched_evaluation(rows, runtime)
    member_clusters = set(rows.loc[rows.membership == 1, "cluster_id"].astype(int))
    nonmember_clusters = set(rows.loc[rows.membership == 0, "cluster_id"].astype(int))
    if member_clusters & nonmember_clusters:
        raise RuntimeError("cluster leakage between controlled member and non-member pools")
    member_hashes = set(rows.loc[rows.membership == 1, "content_hash"])
    nonmember_hashes = set(rows.loc[rows.membership == 0, "content_hash"])
    if member_hashes & nonmember_hashes:
        raise RuntimeError("exact content leakage between controlled member and non-member pools")

    train_corpus = rows[rows.membership == 1][
        ["benchmark_id", "content", "language", "character_count", "length_bin", "cluster_id"]
    ].copy()
    train_corpus = train_corpus.sort_values("benchmark_id", kind="mergesort").reset_index(drop=True)
    evaluation = evaluation[
        [
            "benchmark_id",
            "content",
            "language",
            "character_count",
            "length_bin",
            "cluster_id",
            "membership",
            "matched_pair_id",
        ]
    ].copy()

    train_ids = set(train_corpus.benchmark_id)
    evaluation_member_ids = set(evaluation.loc[evaluation.membership == 1, "benchmark_id"])
    evaluation_nonmember_ids = set(evaluation.loc[evaluation.membership == 0, "benchmark_id"])
    if not evaluation_member_ids <= train_ids:
        raise RuntimeError("controlled member evaluation rows are not all in the training corpus")
    if evaluation_nonmember_ids & train_ids:
        raise RuntimeError("controlled non-member evaluation rows leaked into the training corpus")

    manifest = {
        "status": "complete",
        "benchmark_version": CONTROLLED_BENCHMARK_VERSION,
        "config": asdict(runtime),
        "preparation": preparation,
        "clustering": clustering,
        "train_rows": len(train_corpus),
        "evaluation_rows": len(evaluation),
        "evaluation_member_rows": int((evaluation.membership == 1).sum()),
        "evaluation_nonmember_rows": int((evaluation.membership == 0).sum()),
        "member_clusters": len(member_clusters),
        "nonmember_clusters": len(nonmember_clusters),
        "cluster_overlap": 0,
        "exact_content_hash_overlap": 0,
        "source_membership_labels_used": False,
        "sample_ids_used_as_features": False,
        "evaluation_distribution": _distribution_manifest(evaluation),
    }
    return ControlledTransferBenchmark(
        train_corpus=train_corpus,
        evaluation=evaluation.reset_index(drop=True),
        manifest=manifest,
    )


def make_exposure_schedule(
    train_corpus: pd.DataFrame,
    repeats: int,
    *,
    seed: int = 2027,
) -> pd.DataFrame:
    """Return a deterministic sample schedule with exactly ``repeats`` exposures."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if "benchmark_id" not in train_corpus.columns:
        raise ValueError("train_corpus is missing columns: ['benchmark_id']")
    if train_corpus.benchmark_id.duplicated().any():
        raise ValueError("benchmark_id must be unique in train_corpus")
    rows = []
    for exposure_round in range(repeats):
        identifiers = sorted(
            train_corpus.benchmark_id.astype(str),
            key=lambda value: _stable_order_key(
                seed + exposure_round, f"exposure\0{value}"
            ),
        )
        rows.extend(
            {
                "benchmark_id": identifier,
                "exposure_round": exposure_round,
                "sequence_index": offset,
            }
            for offset, identifier in enumerate(identifiers)
        )
    schedule = pd.DataFrame(rows)
    counts = schedule.groupby("benchmark_id").size()
    if not counts.eq(repeats).all():
        raise RuntimeError("exposure schedule is not balanced")
    return schedule


__all__ = [
    "CONTROLLED_BENCHMARK_VERSION",
    "ControlledTransferBenchmark",
    "ControlledTransferConfig",
    "build_controlled_transfer_benchmark",
    "make_exposure_schedule",
]
