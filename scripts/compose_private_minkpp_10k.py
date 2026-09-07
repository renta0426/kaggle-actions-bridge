#!/usr/bin/env python3
"""Compose a private 10k Min-K++ extraction notebook from private Kaggle sources.

Security boundary:
- This program never reads GitHub or any research repository.
- Source notebooks are supplied as local paths by the protected runner after
  `kaggle kernels pull` and are expected to live under a temporary directory.
- No source cell body is printed, serialized to logs, or written anywhere
  except the final private Kaggle notebook requested by the caller.
- Public bridge content contains orchestration only, not private source bodies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import textwrap
from typing import Iterable

import nbformat


FORBIDDEN_SOURCE_FRAGMENTS = (
    "raw.githubusercontent.com",
    "api.github.com/repos/",
    "RESEARCH_REPO_READ_TOKEN",
    "GITHUB_PAT",
    "GH_TOKEN",
    "SSH_PRIVATE_KEY",
)


def _load_single_notebook(directory: Path) -> nbformat.NotebookNode:
    candidates = sorted(directory.glob("*.ipynb"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one notebook in {directory}, found {len(candidates)}")
    notebook = nbformat.read(candidates[0], as_version=4)
    nbformat.validate(notebook)
    return notebook


def _source(cell: nbformat.NotebookNode) -> str:
    value = cell.get("source", "")
    return value if isinstance(value, str) else "".join(value)


def _find_code(notebook: nbformat.NotebookNode, markers: Iterable[str]) -> str:
    markers = tuple(markers)
    matches = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = _source(cell)
        if all(marker in source for marker in markers):
            matches.append(source)
    if len(matches) != 1:
        raise RuntimeError(f"expected one private source cell for marker set, found {len(matches)}")
    return matches[0]


def _tail_from(source: str, marker: str) -> str:
    offset = source.find(marker)
    if offset < 0:
        raise RuntimeError("private parity marker missing")
    return source[offset:]


def _assert_no_forbidden_transport(source: str) -> None:
    for fragment in FORBIDDEN_SOURCE_FRAGMENTS:
        if fragment in source:
            raise RuntimeError("private notebook unexpectedly contains a forbidden transport reference")


def _wrap_private_run(source: str) -> str:
    required = (
        "def summarize_window",
        "window_rows = []",
        "probability_weighted_token_statistics_blocked",
        "legacy_z",
        "window_features = pd.DataFrame(window_rows)",
        "token_statistics = pd.DataFrame(token_rows)",
    )
    if not all(marker in source for marker in required):
        raise RuntimeError("private pilot run cell contract changed")
    patched = source
    replacements = {
        "MINK_FRACTIONS": "SCALE_FRACTIONS",
        "CONFIG.paper_vocab_block_size": "PAPER_VOCAB_BLOCK_SIZE",
        "CONFIG.paper_variance_floor": "PAPER_VARIANCE_FLOOR",
        "CONFIG.token_reduction_chunk": "CONFIG.vocab_chunk_tokens",
    }
    for old, new in replacements.items():
        patched = patched.replace(old, new)
    _assert_no_forbidden_transport(patched)
    return (
        "def run_private_paper_records(records):\n"
        + textwrap.indent(patched.rstrip() + "\n", "    ")
        + textwrap.indent(
            "return {\n"
            "    'window_features': window_features,\n"
            "    'token_statistics': token_statistics,\n"
            "    'target_tokens': int(target_tokens),\n"
            "    'real_dense_blocked_max_z_diff': float(real_dense_blocked_max_z_diff),\n"
            "    'real_dense_blocked_max_variance_diff': float(real_dense_blocked_max_var_diff),\n"
            "    'valid_target_tokens': int(diag_valid_tokens),\n"
            "    'variance_clamp_count': int(diag_variance_clamps),\n"
            "    'min_variance_before_clamp': (float(diag_min_variance) if math.isfinite(diag_min_variance) else None),\n"
            "    'max_abs_paper_z': float(diag_max_abs_z),\n"
            "    'run_seconds': float(run_seconds),\n"
            "    'peak_gpu_memory_bytes': int(peak_gpu_bytes),\n"
            "}\n",
            "    ",
        )
    )


def _setup_cell() -> str:
    return textwrap.dedent(
        r'''
        from pathlib import Path
        import gc
        import json
        import math
        import os
        import random
        import shutil
        import time

        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

        import numpy as np
        import pandas as pd
        import torch
        from sklearn.metrics import roc_auc_score, roc_curve
        from transformers import AutoModelForCausalLM, AutoTokenizer

        CONFIG = FeatureCacheV2Config(
            output_dir="/kaggle/working/stage1-minkpp-paper-10k-v1",
            retain_token_statistics=False,
        )
        OUTPUT = Path(CONFIG.output_dir)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        SCRATCH = Path("/tmp/stage1-minkpp-paper-10k-v1-parts")
        if SCRATCH.exists():
            shutil.rmtree(SCRATCH)
        SCRATCH.mkdir(parents=True, exist_ok=True)
        SCALE_FRACTIONS = (0.01, 0.05, 0.10, 0.20, 0.30)
        PAPER_VOCAB_BLOCK_SIZE = 8192
        PAPER_VARIANCE_FLOOR = 1e-12
        seed_everything(CONFIG.seed)
        assert CONFIG.train_samples_per_language == 2000
        assert CONFIG.samples_per_shard == 250
        assert CONFIG.max_length == 768
        assert CONFIG.max_batch_tokens == 12288
        print({
            "experiment_id": "stage1-minkpp-paper-10k-v1",
            "execution_stage": "gpu_extraction_only",
            "private_github_access": False,
            "rows": 10000,
            "paper_main_fraction": 0.10,
        })
        '''
    ).strip()


def _fidelity_cell() -> str:
    return textwrap.dedent(
        r'''
        fidelity_records = make_window_records(train_sample.iloc[:8], tokenizer, CONFIG.max_length)
        historical_window_features, _, _ = extract_window_features_optimized(
            fidelity_records, model, tokenizer, CONFIG, progress=False
        )
        private_result = run_private_paper_records(fidelity_records)
        private_window_features = private_result["window_features"]
        keys = ["sample_id", "position", "window_start"]
        left = historical_window_features.sort_values(keys).reset_index(drop=True)
        right = private_window_features.sort_values(keys).reset_index(drop=True)
        if not left[keys].equals(right[keys]):
            raise RuntimeError("historical/private window contract mismatch")
        fidelity_diffs = {
            "mean_logp": float(np.max(np.abs(
                left["score_loss_mean"].to_numpy(float)
                - right["mean_logp"].to_numpy(float)
            ))),
            "legacy_minkpp_10": float(np.max(np.abs(
                left["min_kpp_zselect_10"].to_numpy(float)
                - right["legacy_uniform_vocab_minkpp_10"].to_numpy(float)
            ))),
        }
        fidelity_max = max(fidelity_diffs.values())
        if fidelity_max > 1e-6:
            raise RuntimeError(f"formula-only historical fidelity gate failed: {fidelity_diffs}")
        fidelity = {
            "rows": 8,
            "windows": int(len(left)),
            "atol": 1e-6,
            "max_absolute_difference": fidelity_max,
            "per_feature": fidelity_diffs,
            "cpu_oracle_abs_error": float(oracle_abs_error),
            "synthetic_dense_blocked_max_z_diff": float(synthetic_z_diff),
            "synthetic_dense_blocked_max_variance_diff": float(synthetic_var_diff),
        }
        (OUTPUT / "fidelity.json").write_text(
            json.dumps(fidelity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print({"fidelity_gate": "PASS", "max_abs_diff": fidelity_max})
        del historical_window_features, private_window_features, private_result, fidelity_records, left, right
        gc.collect()
        torch.cuda.empty_cache()
        '''
    ).strip()


def _driver_cell() -> str:
    return textwrap.dedent(
        r'''
        def aggregate_private_windows(window_frame):
            identity = ["sample_id", "language"]
            grouped = window_frame.groupby(identity, sort=False)
            result = pd.concat(
                [
                    grouped["file_token_count"].max().rename("token_count"),
                    grouped.size().rename("window_count"),
                ],
                axis=1,
            ).reset_index()
            score_columns = [
                column for column in window_frame.columns
                if column in ("mean_logp", "best_local_64")
                or column.startswith("legacy_uniform_vocab_minkpp_")
                or column.startswith("paper_prob_weighted_minkpp_")
            ]
            for column in score_columns:
                stats = pd.concat(
                    [
                        grouped[column].mean().rename(f"{column}__mean"),
                        grouped[column].max().rename(f"{column}__max"),
                        grouped[column].std(ddof=0).fillna(0).rename(f"{column}__std"),
                    ],
                    axis=1,
                ).reset_index()
                result = result.merge(stats, on=identity, how="left", validate="one_to_one")
            return result


        def low_fpr(y, score):
            y = np.asarray(y, dtype=int)
            score = np.asarray(score, dtype=float)
            fpr, tpr, _ = roc_curve(y, score)
            return {
                "auc": float(roc_auc_score(y, score)),
                "pauc_01": float(roc_auc_score(y, score, max_fpr=0.01)),
                "tpr_at_0.01_fpr": float(tpr[fpr <= 0.01].max()),
            }


        shard_count = math.ceil(len(train_sample) / CONFIG.samples_per_shard)
        assert shard_count == 40
        diagnostics = {
            "target_tokens": 0,
            "valid_target_tokens": 0,
            "variance_clamp_count": 0,
            "min_variance_before_clamp": math.inf,
            "max_abs_paper_z": 0.0,
            "real_dense_blocked_max_z_diff": 0.0,
            "real_dense_blocked_max_variance_diff": 0.0,
            "run_seconds": 0.0,
            "peak_gpu_memory_bytes": 0,
        }
        extraction_started = time.perf_counter()
        for shard in range(shard_count):
            start = shard * CONFIG.samples_per_shard
            stop = min(len(train_sample), start + CONFIG.samples_per_shard)
            shard_frame = train_sample.iloc[start:stop].copy()
            records = make_window_records(shard_frame, tokenizer, CONFIG.max_length)
            result = run_private_paper_records(records)
            window_part = result["window_features"]
            token_part = result["token_statistics"]
            if set(window_part.sample_id) != set(shard_frame.sample_id):
                raise RuntimeError("shard window coverage mismatch")
            sample_part = aggregate_private_windows(window_part).merge(
                shard_frame[["sample_id", "membership", "label"]],
                on="sample_id", how="left", validate="one_to_one",
            )
            if len(sample_part) != len(shard_frame) or sample_part.label.isna().any():
                raise RuntimeError("shard sample coverage mismatch")
            sample_part.to_parquet(SCRATCH / f"sample.part{shard:03d}.parquet", index=False)
            window_part.to_parquet(SCRATCH / f"window.part{shard:03d}.parquet", index=False)
            diagnostics["target_tokens"] += result["target_tokens"]
            diagnostics["valid_target_tokens"] += result["valid_target_tokens"]
            diagnostics["variance_clamp_count"] += result["variance_clamp_count"]
            if result["min_variance_before_clamp"] is not None:
                diagnostics["min_variance_before_clamp"] = min(
                    diagnostics["min_variance_before_clamp"], result["min_variance_before_clamp"]
                )
            diagnostics["max_abs_paper_z"] = max(
                diagnostics["max_abs_paper_z"], result["max_abs_paper_z"]
            )
            diagnostics["real_dense_blocked_max_z_diff"] = max(
                diagnostics["real_dense_blocked_max_z_diff"],
                result["real_dense_blocked_max_z_diff"],
            )
            diagnostics["real_dense_blocked_max_variance_diff"] = max(
                diagnostics["real_dense_blocked_max_variance_diff"],
                result["real_dense_blocked_max_variance_diff"],
            )
            diagnostics["run_seconds"] += result["run_seconds"]
            diagnostics["peak_gpu_memory_bytes"] = max(
                diagnostics["peak_gpu_memory_bytes"], result["peak_gpu_memory_bytes"]
            )
            print({
                "shard": shard,
                "rows": len(sample_part),
                "windows": len(window_part),
                "target_tokens": result["target_tokens"],
            })
            del records, result, window_part, token_part, sample_part, shard_frame
            gc.collect()
            torch.cuda.empty_cache()

        sample_features = pd.concat(
            [pd.read_parquet(path) for path in sorted(SCRATCH.glob("sample.part*.parquet"))],
            ignore_index=True,
        ).sort_values("sample_id").reset_index(drop=True)
        window_features = pd.concat(
            [pd.read_parquet(path) for path in sorted(SCRATCH.glob("window.part*.parquet"))],
            ignore_index=True,
        ).sort_values(["sample_id", "window_start"]).reset_index(drop=True)
        assert len(sample_features) == 10000 and sample_features.sample_id.is_unique
        assert sample_features.groupby(["language", "label"]).size().eq(1000).all()
        if diagnostics["target_tokens"] != diagnostics["valid_target_tokens"]:
            raise RuntimeError("full 10k target-token accounting mismatch")
        if diagnostics["variance_clamp_count"] != 0:
            raise RuntimeError("paper variance clamp observed in scale run")
        if diagnostics["real_dense_blocked_max_z_diff"] > 5e-5:
            raise RuntimeError("real-logit paper Z parity failed")
        if diagnostics["real_dense_blocked_max_variance_diff"] > 5e-5:
            raise RuntimeError("real-logit paper variance parity failed")
        if not math.isfinite(diagnostics["min_variance_before_clamp"]):
            diagnostics["min_variance_before_clamp"] = None

        sample_features.to_parquet(OUTPUT / "sample_features.parquet", index=False)
        window_features.to_parquet(OUTPUT / "window_features.parquet", index=False)
        sample_features[[
            "sample_id", "language", "label", "token_count", "window_count",
            "legacy_uniform_vocab_minkpp_10__max",
            "paper_prob_weighted_minkpp_10__max",
        ]].to_csv(OUTPUT / "sample_score_extract.csv", index=False)

        score_metrics = {}
        score_columns = [
            column for column in sample_features.columns
            if column.startswith("legacy_uniform_vocab_minkpp_")
            or column.startswith("paper_prob_weighted_minkpp_")
        ]
        for column in score_columns:
            score_metrics[column] = low_fpr(sample_features.label, sample_features[column])
        same_fraction_spearman = {}
        for fraction in (1, 5, 10, 20, 30):
            suffix = f"{fraction:02d}"
            left = f"legacy_uniform_vocab_minkpp_{suffix}__max"
            right = f"paper_prob_weighted_minkpp_{suffix}__max"
            same_fraction_spearman[suffix] = float(
                sample_features[[left, right]].corr(method="spearman").iloc[0, 1]
            )

        metrics = {
            "warning": "10k GPU extraction scale gate; supervised historical-cache fusion is a separate CPU stage.",
            "rows": 10000,
            "member_rows": int(sample_features.label.sum()),
            "paper_main_fraction": 0.10,
            "standalone_low_fpr": score_metrics,
            "paper_legacy_spearman_max": same_fraction_spearman,
        }
        manifest = {
            "status": "complete_gpu_extraction",
            "experiment_id": "stage1-minkpp-paper-10k-v1",
            "rows": 10000,
            "windows": int(len(window_features)),
            "shards": 40,
            "window_contract": "historical_make_window_records",
            "same_forward_paper_and_legacy": True,
            "character_bounded_pilot_regions_reused": False,
            "token_statistics_persisted": False,
            "scratch_parts_persisted": False,
            "private_github_access": False,
            "diagnostics": diagnostics,
            "fidelity": fidelity,
            "wall_seconds": float(time.perf_counter() - extraction_started),
            "next_stage": "CPU fusion with frozen historical 10k cache; no GPU rerun",
        }
        (OUTPUT / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (OUTPUT / "extraction_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (OUTPUT / "REPORT.md").write_text(
            "# Stage 1 paper-faithful Min-K++ — fixed 10k GPU extraction\n\n"
            "Status: complete GPU extraction. Paper and legacy Min-K++ were computed "
            "from the same StarCoder2-3B forward stream on the exact historical "
            "10k cohort and token-level windows. Supervised historical-cache fusion "
            "is intentionally deferred to a CPU-only stage.\n",
            encoding="utf-8",
        )
        shutil.rmtree(SCRATCH)
        print({
            "status": manifest["status"],
            "rows": manifest["rows"],
            "windows": manifest["windows"],
            "target_tokens": diagnostics["target_tokens"],
            "variance_clamps": diagnostics["variance_clamp_count"],
            "paper10_auc": metrics["standalone_low_fpr"]["paper_prob_weighted_minkpp_10__max"]["auc"],
            "legacy10_auc": metrics["standalone_low_fpr"]["legacy_uniform_vocab_minkpp_10__max"]["auc"],
            "paper10_legacy10_spearman": same_fraction_spearman["10"],
        })
        '''
    ).strip()


def compose(pilot_dir: Path, starter_dir: Path, output_dir: Path, target: str, title: str) -> None:
    pilot = _load_single_notebook(pilot_dir)
    starter = _load_single_notebook(starter_dir)

    starter_source = _find_code(starter, ("def make_window_records", "def aggregate_windows"))
    v2_source = _find_code(starter, ("class FeatureCacheV2Config", "def extract_window_features_optimized"))
    starter_load = _find_code(starter, ("train_sample = sample_training_rows", "AutoModelForCausalLM.from_pretrained"))
    paper_source = _find_code(pilot, ("MINKPP_PAPER_SCHEMA_VERSION", "probability_weighted_token_statistics_blocked"))
    pilot_model = _find_code(pilot, ("oracle_expected = -2.336452981844921", "synthetic_dense_blocked_max_z_diff"))
    pilot_run = _find_code(pilot, ("def summarize_window", "window_rows = []", "paper_prob_weighted_minkpp_"))

    private_sources = (starter_source, v2_source, starter_load, paper_source, pilot_model, pilot_run)
    for source in private_sources:
        _assert_no_forbidden_transport(source)

    parity_tail = _tail_from(pilot_model, "# Exact CPU oracle frozen")
    parity_tail = parity_tail.replace("CONFIG.paper_vocab_block_size", "PAPER_VOCAB_BLOCK_SIZE")
    parity_tail = parity_tail.replace("CONFIG.paper_variance_floor", "PAPER_VARIANCE_FLOOR")
    wrapped_run = _wrap_private_run(pilot_run)

    notebook = nbformat.v4.new_notebook()
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python", "version": "3.12"}
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# Stage 1 — paper-faithful Min-K++ fixed 10k GPU extraction\n\n"
            "Private Kaggle composition. No competition submission is created."
        ),
        nbformat.v4.new_code_cell(
            "%pip install -q datasets transformers==5.0.0 accelerate pyarrow scikit-learn"
        ),
        nbformat.v4.new_code_cell(starter_source),
        nbformat.v4.new_code_cell(v2_source),
        nbformat.v4.new_code_cell(paper_source),
        nbformat.v4.new_code_cell(_setup_cell()),
        nbformat.v4.new_code_cell(starter_load),
        nbformat.v4.new_code_cell(parity_tail),
        nbformat.v4.new_code_cell(wrapped_run),
        nbformat.v4.new_code_cell(_fidelity_cell()),
        nbformat.v4.new_code_cell(_driver_cell()),
    ]
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    nbformat.validate(notebook)

    combined = "\n".join(_source(cell) for cell in notebook.cells if cell.cell_type == "code")
    _assert_no_forbidden_transport(combined)
    if "bounded_window_records(" in combined or "bounded_regions(" in combined:
        raise RuntimeError("500-row bounded windowing leaked into scale notebook")
    if "submission.csv" in combined or "competitions submit" in combined:
        raise RuntimeError("submission side effect leaked into extraction notebook")
    required = (
        "make_window_records(train_sample.iloc[:8]",
        "run_private_paper_records",
        "same_forward_paper_and_legacy",
        "stage1-minkpp-paper-10k-v1",
    )
    if not all(marker in combined for marker in required):
        raise RuntimeError("final private notebook contract incomplete")

    output_dir.mkdir(parents=True, exist_ok=True)
    notebook_name = target.split("/", 1)[1] + ".ipynb"
    nbformat.write(notebook, output_dir / notebook_name)
    metadata = {
        "id": target,
        "title": title,
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu", "membership-inference", "minkpp", "fixed-10k"],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "COMPOSE_PASS",
        "cells": len(notebook.cells),
        "private_source_cells": len(private_sources),
        "target": target,
        "source_bodies_logged": False,
        "private_github_access": False,
    }, sort_keys=True))


def self_test() -> None:
    fake = nbformat.v4.new_notebook()
    fake.cells = [
        nbformat.v4.new_code_cell("def make_window_records():\n    pass\ndef aggregate_windows():\n    pass"),
        nbformat.v4.new_code_cell("class FeatureCacheV2Config: pass\ndef extract_window_features_optimized():\n    pass"),
        nbformat.v4.new_code_cell("train_sample = sample_training_rows\nAutoModelForCausalLM.from_pretrained"),
    ]
    assert "make_window_records" in _find_code(fake, ("def make_window_records", "def aggregate_windows"))
    sample_run = """def summarize_window():\n    pass\nwindow_rows = []\nlegacy_z = None\nprobability_weighted_token_statistics_blocked\npaper_prob_weighted_minkpp_\nwindow_features = pd.DataFrame(window_rows)\ntoken_statistics = pd.DataFrame(token_rows)\n"""
    wrapped = _wrap_private_run(sample_run)
    assert wrapped.startswith("def run_private_paper_records(records):")
    assert "return {" in wrapped
    for generated in (_setup_cell(), _fidelity_cell(), _driver_cell(), wrapped):
        _assert_no_forbidden_transport(generated)
    print("PRIVATE_KAGGLE_COMPOSER_SELF_TEST PASS private_repo_access=0 source_bodies_logged=0")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--pilot-dir", type=Path)
    parser.add_argument("--starter-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target")
    parser.add_argument("--title")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    required = (args.pilot_dir, args.starter_dir, args.output_dir, args.target, args.title)
    if any(value is None for value in required):
        parser.error("composition requires pilot/starter/output/target/title")
    compose(args.pilot_dir, args.starter_dir, args.output_dir, args.target, args.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
