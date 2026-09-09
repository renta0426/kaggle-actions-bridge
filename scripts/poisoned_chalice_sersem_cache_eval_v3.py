#!/usr/bin/env python3
"""Standalone CPU SERSEM evaluation with historical-tokenizer alignment.

P1-03 v2 failed closed because the historical Starter++ token cache was produced
under transformers 5.0.0, while direct retokenization under the released SERSEM
runtime (transformers 4.52.0 / tokenizers 0.21.0) changes tokenization for a
material fraction of public files. V3 consumes target-token character offsets
prepared by ``poisoned_chalice_historical_token_alignment.py`` under the
historical runtime, while keeping all SERSEM structural weighting in the pinned
author runtime.

This remains a cache-conditioned component study: historical Starter++ uses
whole/prefix/middle/suffix windows of max length 768, not the author's full
single-sequence max_length=8192 LUMIA experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

AUTHOR_REPOSITORY = "Serdark4ra/SERSEM-Poisoned-Chalice-Competition-2026"
AUTHOR_COMMIT = "413f56040e5b4805bcf15ed794dec56bc4e16b41"
DATASET_REVISION = "2ed5468723efa5457a3665782c6979ea4dbac7c2"
MODEL_REVISION = "733247c55e3f73af49ce8e9c7949bf14af205928"
HISTORICAL_TRANSFORMERS = "5.0.0"
HISTORICAL_TOKENIZERS = "0.22.2"

W_BOILERPLATE = 0.1
W_STANDARD = 1.0
W_IDENTIFIER_LONG = 3.0
W_STRING = 5.0
W_FORMATTING_ERROR = 5.0
W_COMMENT = 10.0
W_MULTILINGUAL = 10.0
W_PSYCHOLOGICAL = 10.0

COMMENT_FB = re.compile(r"(//[^\n]*|#[^\n]*|/\*.*?\*/)", re.DOTALL)
STRING_FB = re.compile(r"(\".*?\"|'.*?')", re.DOTALL)
IDENTIFIER_FB = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")
PSYCH = re.compile(r"(TODO|FIXME|HACK|Note to self)", re.IGNORECASE)
ERRATIC_SPACING = re.compile(r"([^ ]  +[^ ])")
MIXED_INDENT = re.compile(r"^(\t+ +| +\t+)", re.MULTILINE)

EXPECTED_VERSIONS = {
    "numpy": "2.4.2",
    "pandas": "3.0.1",
    "pyarrow": "23.0.1",
    "scikit-learn": "1.8.0",
    "scipy": "1.17.1",
    "transformers": "4.52.0",
    "tokenizers": "0.21.0",
    "tree-sitter": "0.25.2",
    "tree-sitter-go": "0.25.0",
    "tree-sitter-java": "0.23.5",
    "tree-sitter-python": "0.25.0",
    "tree-sitter-ruby": "0.23.1",
    "tree-sitter-rust": "0.24.0",
    "pyenchant": "3.3.0",
    "flake8": "7.3.0",
}


def _np():
    import numpy as np
    return np


def _pd():
    import pandas as pd
    return pd


def _runtime_versions() -> dict[str, str]:
    observed: dict[str, str] = {}
    mismatches: list[str] = []
    for package, expected in EXPECTED_VERSIONS.items():
        try:
            value = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            value = "<missing>"
        observed[package] = value
        if value != expected:
            mismatches.append(package)
    if mismatches:
        raise RuntimeError("runtime_version_mismatch:" + ",".join(sorted(mismatches)))
    return observed


def _split_identifier(identifier: str) -> list[str]:
    words: list[str] = []
    for part in identifier.split("_"):
        subparts = re.findall(r"[A-Za-z][a-z0-9]*|[A-Z]+(?=[A-Z][a-z0-9]|\b)", part)
        if subparts:
            words.extend(subparts)
        elif part:
            words.append(part)
    return words


def _regex_base_weights(text: str):
    np = _np()
    weights = np.full(len(text), W_STANDARD, dtype=np.float64)
    for match in COMMENT_FB.finditer(text):
        weights[match.start():match.end()] = W_COMMENT
    for match in STRING_FB.finditer(text):
        if match.end() - match.start() > 15:
            weights[match.start():match.end()] = W_STRING
    for match in IDENTIFIER_FB.finditer(text):
        if match.end() - match.start() > 10:
            view = weights[match.start():match.end()]
            view[view == W_STANDARD] = W_IDENTIFIER_LONG
    return weights


def _generic_format_errors(text: str) -> list[tuple[int, int]]:
    errors: list[tuple[int, int]] = []
    for index, line in enumerate(text.splitlines()):
        line_number = index + 1
        if MIXED_INDENT.match(line):
            match = MIXED_INDENT.search(line)
            if match is not None:
                errors.append((line_number, match.start() + 1))
        for match in ERRATIC_SPACING.finditer(line):
            prefix = line[:match.start()]
            if "//" not in prefix and "#" not in prefix:
                errors.append((line_number, match.start() + 2))
    return errors


def _flake8_errors(text: str) -> list[tuple[int, int]]:
    executable = shutil.which("flake8")
    if executable is None:
        raise RuntimeError("flake8_unavailable")
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=True) as handle:
            handle.write(text)
            handle.flush()
            completed = subprocess.run(
                [executable, handle.name],
                capture_output=True,
                text=True,
                check=False,
            )
    except Exception:
        return []
    errors: list[tuple[int, int]] = []
    for line in completed.stdout.splitlines():
        match = re.match(r"^.+?:(\d+):(\d+): (.+)", line)
        if match:
            errors.append((int(match.group(1)), int(match.group(2))))
    return errors


def _apply_psych_and_lint(text: str, base, errors: Sequence[tuple[int, int]]):
    np = _np()
    weights = np.asarray(base, dtype=np.float64).copy()
    psychological_matches = 0
    for match in PSYCH.finditer(text):
        psychological_matches += 1
        weights[match.start():match.end()] = W_PSYCHOLOGICAL
    lines = text.splitlines(keepends=True)
    starts: list[int] = []
    total = 0
    for line in lines:
        starts.append(total)
        total += len(line)
    for line_num, col_num in errors:
        if 1 <= line_num <= len(lines):
            offset = starts[line_num - 1] + (col_num - 1)
            start = max(0, offset - 2)
            stop = min(len(text), offset + 3)
            view = weights[start:stop]
            view[view < W_FORMATTING_ERROR] = W_FORMATTING_ERROR
    return weights, psychological_matches


def caller_literal_character_weights(text: str):
    """Literal released LumiaAttack caller: language defaults to None."""
    base = _regex_base_weights(text)
    errors = _generic_format_errors(text)
    weights, psych = _apply_psych_and_lint(text, base, errors)
    return weights, {
        "ast_backend": "regex_fallback_due_to_language_none",
        "linter_backend": "generic_fallback_due_to_language_none",
        "lint_error_count": len(errors),
        "psychological_matches": psych,
    }


def _parsers() -> dict[str, Any]:
    # The released ASTExtractor imports all five grammar modules in one try.
    import tree_sitter
    import tree_sitter_go as tsgo
    import tree_sitter_java as tsjava
    import tree_sitter_python as tspython
    import tree_sitter_ruby as tsruby
    import tree_sitter_rust as tsrust

    modules = {
        "Go": tsgo,
        "Java": tsjava,
        "Python": tspython,
        "Ruby": tsruby,
        "Rust": tsrust,
    }
    return {
        language: tree_sitter.Parser(tree_sitter.Language(module.language()))
        for language, module in modules.items()
    }


def _known_word_checker():
    import enchant
    dictionary = enchant.Dict("en_US")

    def is_known(word: str) -> bool:
        return bool(dictionary.check(word))

    return is_known


def helper_language_aware_character_weights(text: str, language: str, parser: Any, word_is_known):
    """Released AST/linter helper behavior when its optional language is supplied."""
    np = _np()
    weights = np.full(len(text), W_STANDARD, dtype=np.float64)
    byte_text = text.encode("utf-8")
    ast_success = False
    try:
        tree = parser.parse(byte_text)

        def visit(node: Any) -> None:
            nonlocal ast_success
            node_type = str(node.type)
            start = len(byte_text[: int(node.start_byte)].decode("utf-8", errors="replace"))
            stop = len(byte_text[: int(node.end_byte)].decode("utf-8", errors="replace"))
            if "comment" in node_type:
                weights[start:stop] = W_COMMENT
            elif "string" in node_type:
                if stop - start > 15:
                    weights[start:stop] = W_STRING
            elif "identifier" in node_type:
                identifier = text[start:stop]
                words = _split_identifier(identifier)
                unknown_count = sum(
                    1 for word in words if len(word) >= 4 and not word_is_known(word)
                )
                multilingual = (
                    unknown_count > 0 and len(words) > 0 and unknown_count / len(words) >= 0.5
                )
                view = weights[start:stop]
                mask = view == W_STANDARD
                if multilingual:
                    view[mask] = W_MULTILINGUAL
                elif stop - start > 10:
                    view[mask] = W_IDENTIFIER_LONG
            for child in node.children:
                visit(child)

        visit(tree.root_node)
        ast_success = True
    except Exception:
        ast_success = False

    if not ast_success:
        weights = _regex_base_weights(text)
        ast_backend = "regex_fallback_after_parse_failure"
    else:
        ast_backend = "tree_sitter"

    if language == "Python":
        flake_errors = _flake8_errors(text)
        if flake_errors:
            errors = flake_errors
            linter_backend = "flake8"
        else:
            errors = _generic_format_errors(text)
            linter_backend = "generic_fallback_after_empty_flake8"
    else:
        errors = _generic_format_errors(text)
        linter_backend = "generic_fallback"

    weights, psych = _apply_psych_and_lint(text, weights, errors)
    return weights, {
        "ast_backend": ast_backend,
        "linter_backend": linter_backend,
        "lint_error_count": len(errors),
        "psychological_matches": psych,
    }


def token_weights(text: str, offsets: Sequence[Sequence[int]], character_weights):
    np = _np()
    char_weights = np.asarray(character_weights, dtype=np.float64)
    if char_weights.shape != (len(text),):
        raise RuntimeError("character_weight_length_mismatch")
    result = np.full(len(offsets), W_BOILERPLATE, dtype=np.float64)
    for index, raw in enumerate(offsets):
        start, stop = int(raw[0]), int(raw[1])
        if start < 0 or stop < start or stop > len(text):
            raise RuntimeError("invalid_token_offset")
        if start == stop:
            continue
        span_weight = float(np.max(char_weights[start:stop]))
        if span_weight <= W_STANDARD:
            token_text = text[start:stop].strip()
            if len(token_text) > 2 and IDENTIFIER_FB.match(token_text):
                result[index] = W_STANDARD
            else:
                result[index] = W_BOILERPLATE
        else:
            result[index] = span_weight
    return result


def _sigmoid(values):
    np = _np()
    z = np.asarray(values, dtype=np.float64)
    output = np.empty_like(z)
    positive = z >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    output[~positive] = exp_z / (1.0 + exp_z)
    return output


def _weighted_score(z, weights) -> float:
    np = _np()
    values = _sigmoid(z)
    w = np.asarray(weights, dtype=np.float64)
    if values.shape != w.shape or values.ndim != 1 or values.size == 0:
        raise RuntimeError("score_weight_alignment_failure")
    if not np.isfinite(values).all() or not np.isfinite(w).all():
        raise RuntimeError("nonfinite_score_input")
    if (w < 0).any() or float(w.sum()) <= 0:
        raise RuntimeError("invalid_weight_mass")
    return float(np.average(values, weights=w))


def _window_key(row: Any) -> tuple[str, int]:
    return str(row.position), int(row.window_start)


def _score_sample(token_rows, alignment_rows, text: str, caller_chars, helper_chars):
    np = _np()
    alignments = {_window_key(row): row for row in alignment_rows.itertuples(index=False)}
    token_items = list(token_rows.itertuples(index=False))
    token_items.sort(key=lambda row: (int(row.window_start), str(row.position)))
    if len(alignments) != len(token_items):
        raise RuntimeError("alignment_window_coverage_failure")

    baseline_windows: list[float] = []
    caller_windows: list[float] = []
    helper_windows: list[float] = []
    caller_union: dict[int, tuple[int, float, float]] = {}
    helper_union: dict[int, tuple[int, float, float]] = {}

    for row in token_items:
        alignment = alignments.get(_window_key(row))
        if alignment is None:
            raise RuntimeError("alignment_window_missing")
        z = np.asarray(row.correct_z, dtype=np.float64)
        observed = np.asarray(row.target_token_ids, dtype=np.int64)
        offsets = np.asarray(alignment.target_offsets, dtype=np.int64)
        if offsets.ndim != 2 or offsets.shape[1] != 2:
            raise RuntimeError("alignment_offset_shape_failure")
        if len(z) != len(observed) or len(z) != len(offsets):
            raise RuntimeError("alignment_target_length_failure")
        if int(alignment.target_token_count) != len(z):
            raise RuntimeError("alignment_manifest_target_count_failure")

        caller_weights = token_weights(text, offsets, caller_chars)
        helper_weights = token_weights(text, offsets, helper_chars)
        baseline_windows.append(float(_sigmoid(z).mean()))
        caller_windows.append(_weighted_score(z, caller_weights))
        helper_windows.append(_weighted_score(z, helper_weights))

        start = int(row.window_start) + 1
        for local_index, (global_index, z_value, caller_weight, helper_weight) in enumerate(
            zip(range(start, start + len(z)), z, caller_weights, helper_weights),
            start=1,
        ):
            previous = caller_union.get(global_index)
            if previous is None or local_index > previous[0]:
                caller_union[global_index] = (local_index, float(z_value), float(caller_weight))
                helper_union[global_index] = (local_index, float(z_value), float(helper_weight))

    if not baseline_windows:
        raise RuntimeError("sample_without_cached_windows")
    caller_union_z = [value for _, value, _ in caller_union.values()]
    caller_union_w = [weight for _, _, weight in caller_union.values()]
    helper_union_z = [value for _, value, _ in helper_union.values()]
    helper_union_w = [weight for _, _, weight in helper_union.values()]
    return (
        float(np.mean(baseline_windows)),
        float(np.mean(caller_windows)),
        _weighted_score(caller_union_z, caller_union_w),
        float(np.mean(helper_windows)),
        _weighted_score(helper_union_z, helper_union_w),
    )


def _labels(series):
    pd = _pd()
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    normalized = series.astype(str).str.strip().str.lower().str.replace("_", "-", regex=False)
    mapped = normalized.map(
        {
            "member": 1,
            "non-member": 0,
            "nonmember": 0,
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
        }
    )
    if mapped.isna().any():
        raise RuntimeError("unrecognized_membership_label")
    return mapped.astype(int)


def _low_fpr_metrics(y_true, score) -> dict[str, float]:
    np = _np()
    from sklearn.metrics import roc_auc_score, roc_curve

    y = np.asarray(y_true, dtype=int)
    values = np.asarray(score, dtype=np.float64)
    fpr, tpr, _ = roc_curve(y, values)
    result = {
        "auc": float(roc_auc_score(y, values)),
        "pauc_01": float(roc_auc_score(y, values, max_fpr=0.01)),
    }
    for rate in (0.001, 0.005, 0.01, 0.02):
        result[f"tpr_at_{rate:g}_fpr"] = float(tpr[fpr <= rate].max())
    return result


def _bootstrap(frame, score, replicates: int, seed: int):
    np = _np()
    rng = np.random.default_rng(seed)
    work = frame[["language", "label"]].copy()
    work["score"] = np.asarray(score, dtype=np.float64)
    strata = [group.index.to_numpy() for _, group in work.groupby(["language", "label"])]
    labels = work.label.to_numpy()
    scores = work.score.to_numpy()
    rows: list[dict[str, float]] = []
    for _ in range(replicates):
        sampled = np.concatenate(
            [rng.choice(index, size=len(index), replace=True) for index in strata]
        )
        rows.append(_low_fpr_metrics(labels[sampled], scores[sampled]))
    result: dict[str, dict[str, float]] = {}
    for metric in rows[0]:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        result[metric] = {
            "mean": float(values.mean()),
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
    return result


def _conservative_tp_count(y_true, score, target_fpr: float = 0.01) -> int:
    np = _np()
    y = np.asarray(y_true, dtype=int)
    values = np.asarray(score, dtype=np.float64)
    negatives = np.sort(values[y == 0])[::-1]
    allowed = max(1, int(math.floor(target_fpr * len(negatives))))
    threshold = negatives[allowed - 1]
    return int(((y == 1) & (values > threshold)).sum())


def _load_cache(cache_dir: Path):
    pd = _pd()
    files = [cache_dir / f"token_statistics.part{index:03d}.parquet" for index in range(40)]
    if not all(path.is_file() for path in files):
        raise RuntimeError("token_cache_shard_set_incomplete")
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    required = {"sample_id", "language", "position", "window_start", "target_token_ids", "correct_z"}
    if not required.issubset(frame.columns):
        raise RuntimeError("token_cache_schema_mismatch")
    key = ["sample_id", "position", "window_start"]
    if frame.duplicated(key).any():
        raise RuntimeError("token_cache_duplicate_window_key")
    return frame


def evaluate(
    dataset_parquet: Path,
    cache_dir: Path,
    alignment_parquet: Path,
    alignment_manifest_json: Path,
    output_json: Path,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    pd = _pd()
    versions = _runtime_versions()
    token_rows = _load_cache(cache_dir)
    alignment = pd.read_parquet(alignment_parquet)
    alignment_manifest = json.loads(alignment_manifest_json.read_text(encoding="utf-8"))

    scored_ids = set(token_rows.sample_id.astype(str))
    if len(scored_ids) != 10_000:
        raise RuntimeError("unexpected_cached_sample_count")
    alignment_required = {
        "sample_id",
        "language",
        "position",
        "window_start",
        "target_offsets",
        "target_token_count",
        "full_token_count",
    }
    if not alignment_required.issubset(alignment.columns):
        raise RuntimeError("alignment_schema_mismatch")
    key = ["sample_id", "position", "window_start"]
    if alignment.duplicated(key).any():
        raise RuntimeError("alignment_duplicate_window_key")
    token_key = set(map(tuple, token_rows[key].astype({"sample_id": str, "position": str}).to_numpy()))
    alignment_key = set(map(tuple, alignment[key].astype({"sample_id": str, "position": str}).to_numpy()))
    if token_key != alignment_key:
        raise RuntimeError("alignment_exact_window_set_mismatch")
    if alignment_manifest.get("mismatch_windows") != 0:
        raise RuntimeError("alignment_manifest_reports_mismatch")
    historical_runtime = alignment_manifest.get("runtime", {})
    if historical_runtime.get("transformers") != HISTORICAL_TRANSFORMERS:
        raise RuntimeError("alignment_historical_transformers_mismatch")
    if historical_runtime.get("tokenizers") != HISTORICAL_TOKENIZERS:
        raise RuntimeError("alignment_historical_tokenizers_mismatch")

    dataset = pd.read_parquet(dataset_parquet)
    required = {"sample_id", "language", "content", "membership"}
    if not required.issubset(dataset.columns):
        raise RuntimeError("public_train_schema_mismatch")
    if len(dataset) != 49_040 or dataset.sample_id.duplicated().any() or dataset.membership.isna().any():
        raise RuntimeError("public_train_contract_mismatch")
    source = dataset.loc[
        dataset.sample_id.astype(str).isin(scored_ids),
        ["sample_id", "language", "content", "membership"],
    ].copy()
    source["sample_id"] = source.sample_id.astype(str)
    if len(source) != 10_000 or set(source.sample_id) != scored_ids:
        raise RuntimeError("cache_ids_not_exact_public_train_subset")

    parsers = _parsers()
    word_is_known = _known_word_checker()
    by_sample_tokens = {
        sample_id: group
        for sample_id, group in token_rows.groupby(token_rows.sample_id.astype(str), sort=False)
    }
    by_sample_alignment = {
        sample_id: group
        for sample_id, group in alignment.groupby(alignment.sample_id.astype(str), sort=False)
    }

    rows: list[dict[str, Any]] = []
    helper_ast_tree = 0
    helper_flake8 = 0
    caller_generic = 0
    for row in source.itertuples(index=False):
        caller_chars, caller_diag = caller_literal_character_weights(row.content)
        helper_chars, helper_diag = helper_language_aware_character_weights(
            row.content,
            row.language,
            parsers[row.language],
            word_is_known,
        )
        scores = _score_sample(
            by_sample_tokens[row.sample_id],
            by_sample_alignment[row.sample_id],
            row.content,
            caller_chars,
            helper_chars,
        )
        base_score, caller_windowed, caller_union, helper_windowed, helper_union = scores
        caller_generic += int(caller_diag["linter_backend"].startswith("generic"))
        helper_ast_tree += int(helper_diag["ast_backend"] == "tree_sitter")
        helper_flake8 += int(helper_diag["linter_backend"] == "flake8")
        rows.append(
            {
                "language": row.language,
                "membership": row.membership,
                "zsigmoid_window_mean": base_score,
                "sersem_caller_literal_windowed": caller_windowed,
                "sersem_caller_literal_union": caller_union,
                "sersem_helper_language_aware_windowed": helper_windowed,
                "sersem_helper_language_aware_union": helper_union,
            }
        )

    evaluation = pd.DataFrame(rows)
    evaluation["label"] = _labels(evaluation.membership)
    score_names = [
        "zsigmoid_window_mean",
        "sersem_caller_literal_windowed",
        "sersem_caller_literal_union",
        "sersem_helper_language_aware_windowed",
        "sersem_helper_language_aware_union",
    ]
    metrics = {name: _low_fpr_metrics(evaluation.label, evaluation[name]) for name in score_names}
    bootstrap = {
        name: _bootstrap(
            evaluation[["language", "label"]],
            evaluation[name],
            bootstrap_replicates,
            bootstrap_seed,
        )
        for name in score_names
    }
    conservative = {
        name: _conservative_tp_count(evaluation.label, evaluation[name], 0.01)
        for name in score_names
    }
    from sklearn.metrics import roc_auc_score

    per_language_auc = {
        name: {
            language: float(roc_auc_score(group.label, group[name]))
            for language, group in evaluation.groupby("language")
        }
        for name in score_names
    }

    summary = {
        "schema_version": 1,
        "protocol": "P1-03 SERSEM historical-cache reproduction v3",
        "author_repository": AUTHOR_REPOSITORY,
        "author_commit": AUTHOR_COMMIT,
        "dataset_revision": DATASET_REVISION,
        "model_revision": MODEL_REVISION,
        "input_protocol": (
            "historical Starter++ transformers=5.0.0/tokenizers=0.22.2 tokenization replay; "
            "max_context=768 whole/prefix-middle-suffix cache; not the author's single-sequence "
            "max_length=8192 hidden-state LUMIA runner"
        ),
        "caller_path_finding": (
            "released LumiaAttack._extract_activations calls generate_char_weight_mask(text) "
            "without language; literal weighting therefore uses regex AST fallback and generic linter"
        ),
        "tokenizer_differential_evidence": {
            "diagnostic_run": 34303764258,
            "public_probe_samples": 10_000,
            "token_id_mismatch_samples": 919,
            "offset_mismatch_samples": 919,
            "token_count_mismatch_samples": 918,
        },
        "historical_token_alignment": alignment_manifest,
        "sersem_runtime_versions": versions,
        "samples": int(len(evaluation)),
        "languages": sorted(evaluation.language.unique().tolist()),
        "diagnostics": {
            "caller_generic_linter_samples": int(caller_generic),
            "helper_tree_sitter_samples": int(helper_ast_tree),
            "helper_flake8_samples": int(helper_flake8),
        },
        "metrics": metrics,
        "bootstrap": bootstrap,
        "per_language_auc": per_language_auc,
        "conservative_tp_at_1pct_fpr": conservative,
    }
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _self_test() -> None:
    np = _np()
    text = "alphaIdentifierLong = 'a very long string value'  + x # TODO"
    caller, diagnostic = caller_literal_character_weights(text)
    assert len(caller) == len(text)
    assert diagnostic["ast_backend"].startswith("regex_fallback")
    weights = token_weights(text, [(0, 19), (20, 21), (22, 48)], caller)
    assert weights.shape == (3,)
    z = np.asarray([-2.0, 0.0, 2.0])
    assert 0 < _weighted_score(z, [0.1, 1.0, 10.0]) < 1
    assert _split_identifier("HTTPServer_longName") == ["H", "T", "T", "P", "Server", "long", "Name"]
    print("SERSEM_CACHE_EVAL_V3_SELF_TEST PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-parquet", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--alignment-parquet", type=Path)
    parser.add_argument("--alignment-manifest-json", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260909)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return 0
    required = (
        args.dataset_parquet,
        args.cache_dir,
        args.alignment_parquet,
        args.alignment_manifest_json,
        args.output_json,
    )
    if any(value is None for value in required):
        raise SystemExit("dataset/cache/alignment/output are required")
    try:
        summary = evaluate(
            args.dataset_parquet,
            args.cache_dir,
            args.alignment_parquet,
            args.alignment_manifest_json,
            args.output_json,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
    except Exception as exc:
        digest = hashlib.sha256(
            f"{type(exc).__name__}:{exc}".encode("utf-8", errors="replace")
        ).hexdigest()
        print(
            "SERSEM_CACHE_EVAL_V3_FAILURE "
            + json.dumps(
                {
                    "status": "failure",
                    "exception_type": type(exc).__name__,
                    "error_sha256": digest,
                },
                sort_keys=True,
            )
        )
        return 1
    print(
        "SERSEM_CACHE_EVAL_V3_LOCAL_RESULT "
        + json.dumps(
            {
                "status": "success",
                "samples": summary["samples"],
                "metrics": summary["metrics"],
                "conservative_tp_at_1pct_fpr": summary["conservative_tp_at_1pct_fpr"],
                "diagnostics": summary["diagnostics"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
