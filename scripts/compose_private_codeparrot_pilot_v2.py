#!/usr/bin/env python3
# Compose a private, label-blind CodeParrot T4 pilot from one private Kaggle scorer source.
# The public bridge contains orchestration only; private source bodies stay runner-local.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
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
EXPECTED_COHORT_SHA256 = "994276dc4ae9e5cf8cc17e66ddcb46ffa500b3236ec3d1ccbda10d2670466446"
MODEL_ID = "codeparrot/codeparrot"
MODEL_REVISION = "065248a99f051da363b1c2cbf05da943c8b6211b"
COHORT_KERNEL = "renta0426/codeparrot-fresh-cohort-v1"


def _load_single_notebook(directory: Path) -> nbformat.NotebookNode:
    candidates = sorted(directory.glob("*.ipynb"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one notebook in {directory}, found {len(candidates)}"
        )
    notebook = nbformat.read(candidates[0], as_version=4)
    nbformat.validate(notebook)
    return notebook


def _source(cell: nbformat.NotebookNode) -> str:
    value = cell.get("source", "")
    return value if isinstance(value, str) else "".join(value)


def _find_code(notebook: nbformat.NotebookNode, markers: Iterable[str]) -> str:
    markers = tuple(markers)
    matches: list[str] = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = _source(cell)
        if all(marker in source for marker in markers):
            matches.append(source)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one private source cell for marker set, found {len(matches)}"
        )
    return matches[0]


def _assert_no_forbidden_transport(source: str) -> None:
    for fragment in FORBIDDEN_SOURCE_FRAGMENTS:
        if fragment in source:
            raise RuntimeError(
                "private notebook unexpectedly contains a forbidden transport reference"
            )


def _setup_cell() -> str:
    return textwrap.dedent(
        r'''
        from hashlib import sha256
        from pathlib import Path
        import gc
        import json
        import math
        import os
        import time

        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

        import numpy as np
        import pandas as pd
        import torch
        import transformers
        from huggingface_hub import HfApi
        from transformers import AutoModelForCausalLM, AutoTokenizer

        EXPERIMENT_ID = "codeparrot-fresh-pilot-v1"
        MODEL_ID = "codeparrot/codeparrot"
        MODEL_REVISION = "065248a99f051da363b1c2cbf05da943c8b6211b"
        EXPECTED_COHORT_SHA256 = "994276dc4ae9e5cf8cc17e66ddcb46ffa500b3236ec3d1ccbda10d2670466446"
        OUTPUT = Path("/kaggle/working/codeparrot_fresh_pilot_v1")
        OUTPUT.mkdir(parents=True, exist_ok=True)

        CONFIG = FeatureCacheV2Config(
            output_dir=str(OUTPUT),
            max_length=768,
            max_batch_tokens=1536,
            vocab_chunk_tokens=32,
            rank_vocab_block_size=8192,
            retain_token_statistics=False,
        )
        SCALE_FRACTIONS = (0.10,)
        PAPER_VOCAB_BLOCK_SIZE = 8192
        PAPER_VARIANCE_FLOOR = 1e-12

        assert CONFIG.max_length == 768
        assert CONFIG.max_batch_tokens == 1536
        assert CONFIG.vocab_chunk_tokens == 32
        assert CONFIG.rank_vocab_block_size == 8192
        assert CONFIG.retain_token_statistics is False
        assert SCALE_FRACTIONS == (0.10,)
        print({
            "experiment_id": EXPERIMENT_ID,
            "rows": 200,
            "performance_metrics_allowed": False,
            "membership_join_performed": False,
            "private_github_access": False,
            "competition_submission": False,
            "scorer_source": "existing_private_minkpp_same_forward",
        })
        '''
    ).strip()


def _load_cell() -> str:
    return textwrap.dedent(
        r'''
        input_candidates = sorted(Path("/kaggle/input").rglob("prediction_input.parquet"))
        if len(input_candidates) != 1:
            raise RuntimeError(
                f"expected exactly one attached prediction_input.parquet, found {len(input_candidates)}"
            )
        cohort_path = input_candidates[0]
        actual_cohort_sha = sha256(cohort_path.read_bytes()).hexdigest()
        if actual_cohort_sha != EXPECTED_COHORT_SHA256:
            raise RuntimeError(f"cohort SHA-256 mismatch: {actual_cohort_sha}")

        cohort = pd.read_parquet(cohort_path)
        expected_columns = [
            "schema_version", "sample_id", "content_sha256", "language", "content"
        ]
        if cohort.columns.tolist() != expected_columns:
            raise RuntimeError(f"unexpected cohort columns: {cohort.columns.tolist()}")
        forbidden = {
            "membership", "label", "source_split", "source_position",
            "repo", "repo_name", "license", "path", "dataset_id",
            "prior_model_score", "repo_group_sha256",
        }
        if forbidden.intersection(cohort.columns):
            raise RuntimeError("label/provenance field leaked into GPU input")
        if len(cohort) != 4000 or cohort.sample_id.duplicated().any():
            raise RuntimeError("cohort row/id contract failed")
        cohort = cohort.sort_values("sample_id").reset_index(drop=True)
        if (
            cohort.schema_version.nunique() != 1
            or cohort.schema_version.iloc[0] != "codeparrot_prediction_input_v1"
        ):
            raise RuntimeError("prediction input schema version mismatch")
        if set(cohort.language) != {"Python"}:
            raise RuntimeError("pilot language contract failed")
        recomputed = cohort.content.map(
            lambda value: sha256(value.encode("utf-8")).hexdigest()
        )
        if not np.array_equal(recomputed.to_numpy(), cohort.content_sha256.to_numpy()):
            raise RuntimeError("cohort content SHA mismatch")
        if not np.array_equal(
            ("cp-" + cohort.content_sha256).to_numpy(),
            cohort.sample_id.to_numpy(),
        ):
            raise RuntimeError("cohort sample ID contract failed")

        pilot = cohort.iloc[:200].copy()
        if len(pilot) != 200 or pilot.sample_id.duplicated().any():
            raise RuntimeError("pilot label-free selection failed")
        '''
    ).strip()


def _model_cell() -> str:
    return textwrap.dedent(
        r'''
        if not torch.cuda.is_available():
            raise RuntimeError("CodeParrot pilot requires CUDA/T4")

        info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
        if str(info.sha) != MODEL_REVISION:
            raise RuntimeError(f"Hugging Face model revision mismatch: {info.sha}")
        files = {sibling.rfilename: sibling for sibling in info.siblings}
        if "pytorch_model.bin" not in files:
            raise RuntimeError("pinned CodeParrot revision no longer exposes pytorch_model.bin")
        if any(name.endswith(".safetensors") for name in files):
            raise RuntimeError(
                "pinned CodeParrot contract unexpectedly contains safetensors; "
                "do not silently change weight format"
            )
        weight_lfs = getattr(files["pytorch_model.bin"], "lfs", None)
        weight_sha256 = (
            getattr(weight_lfs, "sha256", None) if weight_lfs is not None else None
        )

        model_load_started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            use_fast=True,
            trust_remote_code=False,
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise RuntimeError("CodeParrot tokenizer has neither pad nor EOS token")
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            trust_remote_code=False,
            use_safetensors=False,
        ).to("cuda:0").eval()
        model_load_seconds = time.perf_counter() - model_load_started

        resolved_model_revision = getattr(model.config, "_commit_hash", None)
        resolved_tokenizer_revision = tokenizer.init_kwargs.get("_commit_hash")
        if resolved_model_revision and resolved_model_revision != MODEL_REVISION:
            raise RuntimeError(f"resolved model revision mismatch: {resolved_model_revision}")
        if resolved_tokenizer_revision and resolved_tokenizer_revision != MODEL_REVISION:
            raise RuntimeError(
                f"resolved tokenizer revision mismatch: {resolved_tokenizer_revision}"
            )
        if getattr(model.config, "model_type", None) != "gpt2":
            raise RuntimeError(f"unexpected CodeParrot model type: {model.config.model_type}")
        if not model.__class__.__module__.startswith("transformers.models.gpt2"):
            raise RuntimeError(
                f"unexpected CodeParrot implementation: {model.__class__.__module__}"
            )

        runtime_identity = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "resolved_model_revision": resolved_model_revision,
            "resolved_tokenizer_revision": resolved_tokenizer_revision,
            "model_class": model.__class__.__name__,
            "model_module": model.__class__.__module__,
            "model_type": model.config.model_type,
            "trust_remote_code": False,
            "use_safetensors": False,
            "pytorch_model_bin_lfs_sha256": weight_sha256,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
            "model_load_seconds": model_load_seconds,
        }
        print(json.dumps(runtime_identity, indent=2, sort_keys=True))
        '''
    ).strip()


def _oracle_cell() -> str:
    return textwrap.dedent(
        r'''
        oracle_logits = torch.tensor([[[3.0, 1.0, -2.0]]], device="cuda")
        oracle_target = torch.tensor([[1]], device="cuda")
        oracle = probability_weighted_token_statistics_blocked(
            oracle_logits, oracle_target, vocab_block_size=2
        )
        oracle_expected = -2.336452981844921
        oracle_abs_error = abs(
            float(oracle.standardized_log_prob.item()) - oracle_expected
        )
        if oracle_abs_error > 2e-5:
            raise RuntimeError(f"paper CPU-oracle gate failed: {oracle_abs_error}")

        generator = torch.Generator(device="cuda").manual_seed(2027)
        synthetic_logits = torch.randn(
            2, 5, 97, generator=generator, device="cuda"
        )
        synthetic_targets = torch.randint(
            0, 97, (2, 5), generator=generator, device="cuda"
        )
        paper_dense = probability_weighted_token_statistics_dense(
            synthetic_logits, synthetic_targets
        )
        paper_block = probability_weighted_token_statistics_blocked(
            synthetic_logits, synthetic_targets, vocab_block_size=17
        )
        synthetic_paper_z_diff = float(
            (
                paper_dense.standardized_log_prob
                - paper_block.standardized_log_prob
            ).abs().max().item()
        )
        synthetic_paper_var_diff = float(
            (
                paper_dense.probability_weighted_variance
                - paper_block.probability_weighted_variance
            ).abs().max().item()
        )
        if synthetic_paper_z_diff > 3e-5 or synthetic_paper_var_diff > 3e-5:
            raise RuntimeError("paper synthetic dense/block parity failed")

        values = synthetic_logits.float()
        correct = values.gather(
            -1, synthetic_targets.unsqueeze(-1)
        ).squeeze(-1)
        direct_logp = correct - torch.logsumexp(values, dim=-1)
        direct_z = (
            (correct - values.mean(dim=-1))
            / values.std(dim=-1, correction=0).clamp_min(1e-6)
        )
        direct_rank = 1 + (
            values > correct.unsqueeze(-1)
        ).sum(dim=-1)

        blocked = _chunk_statistics(
            synthetic_logits, synthetic_targets, rank_vocab_block_size=17
        )
        legacy_logp_diff = float(
            (direct_logp - blocked[..., 0]).abs().max().item()
        )
        legacy_z_diff = float(
            (direct_z - blocked[..., 1]).abs().max().item()
        )
        legacy_rank_match = bool(
            torch.equal(direct_rank, blocked[..., 2].to(torch.int64))
        )
        if (
            legacy_logp_diff > 1e-6
            or legacy_z_diff > 1e-6
            or not legacy_rank_match
        ):
            raise RuntimeError("legacy synthetic exact/blocked parity failed")

        del (
            oracle_logits, oracle_target, oracle, synthetic_logits,
            synthetic_targets, paper_dense, paper_block, values, correct,
            direct_logp, direct_z, direct_rank, blocked,
        )
        gc.collect()
        torch.cuda.empty_cache()
        '''
    ).strip()


def _pilot_cell() -> str:
    return textwrap.dedent(
        r'''
        def scorer_frame(frame):
            local = frame[["content", "language"]].copy().reset_index(drop=True)
            local.insert(
                0,
                "sample_id",
                [f"row-{index:06d}" for index in range(len(local))],
            )
            return local

        def aggregate_private_features(window_frame, sample_count):
            local = window_frame.copy()
            local["sample_index"] = (
                local["sample_id"].str.removeprefix("row-").astype(int)
            )
            rows = []
            for sample_index, group in local.groupby("sample_index", sort=True):
                rows.append({
                    "sample_index": int(sample_index),
                    "language": str(group.language.iloc[0]),
                    "token_count": int(group.file_token_count.max()),
                    "window_count": int(len(group)),
                    "loss_multiwindow": float(group.mean_logp.max()),
                    "legacy_minkpp10": float(
                        group.legacy_uniform_vocab_minkpp_10.max()
                    ),
                    "paper_minkpp10": float(
                        group.paper_prob_weighted_minkpp_10.max()
                    ),
                    "local_64": float(group.best_local_64.max()),
                    "neg_mean_log_rank": float(-group.mean_log_rank.mean()),
                })
            result = (
                pd.DataFrame(rows)
                .sort_values("sample_index")
                .reset_index(drop=True)
            )
            if (
                len(result) != sample_count
                or result.sample_index.tolist() != list(range(sample_count))
            ):
                raise RuntimeError("pilot sample aggregation coverage/order failure")
            numeric = result.drop(columns=["language"])
            if not np.isfinite(numeric.to_numpy(float)).all():
                raise RuntimeError("non-finite pilot feature")
            return result

        torch.cuda.reset_peak_memory_stats()
        run_started = time.perf_counter()
        private_input = scorer_frame(pilot)
        records = make_window_records(private_input, tokenizer, CONFIG.max_length)
        private_result = run_private_paper_records(records)
        window_features = private_result["window_features"].copy()
        expected_internal_ids = {f"row-{index:06d}" for index in range(200)}
        if set(window_features.sample_id) != expected_internal_ids:
            raise RuntimeError("private scorer internal-ID coverage failure")
        sample_features = aggregate_private_features(window_features, 200)
        run_seconds = time.perf_counter() - run_started

        first8_private = scorer_frame(pilot.iloc[:8])
        first8_records = make_window_records(
            first8_private, tokenizer, CONFIG.max_length
        )
        historical8, _, historical8_profile = extract_window_features_optimized(
            first8_records, model, tokenizer, CONFIG, progress=False
        )
        private8_result = run_private_paper_records(first8_records)
        private8_windows = private8_result["window_features"].copy()
        keys = ["sample_id", "position", "window_start"]
        left = historical8.sort_values(keys).reset_index(drop=True)
        right = private8_windows.sort_values(keys).reset_index(drop=True)
        if not left[keys].equals(right[keys]):
            raise RuntimeError("first-8 historical/private window alignment failed")

        first8_compare = {
            "mean_logp": float(np.max(np.abs(
                left["score_loss_mean"].to_numpy(float)
                - right["mean_logp"].to_numpy(float)
            ))),
            "legacy_minkpp10": float(np.max(np.abs(
                left["min_kpp_zselect_10"].to_numpy(float)
                - right["legacy_uniform_vocab_minkpp_10"].to_numpy(float)
            ))),
            "local_64": float(np.max(np.abs(
                left["best_local_64"].to_numpy(float)
                - right["best_local_64"].to_numpy(float)
            ))),
            "mean_log_rank": float(np.max(np.abs(
                left["mean_log_rank"].to_numpy(float)
                - right["mean_log_rank"].to_numpy(float)
            ))),
        }
        first8_max = max(first8_compare.values())
        if first8_max > 1e-6:
            raise RuntimeError(
                f"first-8 historical/private fidelity failed: {first8_compare}"
            )

        paper_real_z_diff = float(
            private_result["real_dense_blocked_max_z_diff"]
        )
        paper_real_var_diff = float(
            private_result["real_dense_blocked_max_variance_diff"]
        )
        if paper_real_z_diff > 5e-5 or paper_real_var_diff > 5e-5:
            raise RuntimeError("real-logit paper dense/block parity failed")
        if (
            int(private_result["target_tokens"])
            != int(private_result["valid_target_tokens"])
        ):
            raise RuntimeError("paper target-token accounting failed")

        predictions = pd.DataFrame({
            "sample_id": pilot.sample_id.to_numpy(),
            "content_sha256": pilot.content_sha256.to_numpy(),
            "language": pilot.language.to_numpy(),
            "token_count": sample_features.token_count.to_numpy(),
            "window_count": sample_features.window_count.to_numpy(),
            "loss_multiwindow": sample_features.loss_multiwindow.to_numpy(),
            "legacy_minkpp10": sample_features.legacy_minkpp10.to_numpy(),
            "paper_minkpp10": sample_features.paper_minkpp10.to_numpy(),
            "local_64": sample_features.local_64.to_numpy(),
            "neg_mean_log_rank": sample_features.neg_mean_log_rank.to_numpy(),
        })
        if len(predictions) != 200 or predictions.sample_id.duplicated().any():
            raise RuntimeError("pilot prediction coverage failure")
        if not np.isfinite(
            predictions.select_dtypes(include=[np.number]).to_numpy(float)
        ).all():
            raise RuntimeError("non-finite pilot predictions")
        predictions.to_parquet(
            OUTPUT / "pilot_predictions.parquet", index=False
        )

        fidelity = {
            "status": "pass",
            "performance_metrics_computed": False,
            "membership_join_performed": False,
            "original_sample_id_passed_to_scorer": False,
            "cpu_oracle_abs_error": float(oracle_abs_error),
            "synthetic_paper_dense_blocked_max_z_diff":
                synthetic_paper_z_diff,
            "synthetic_paper_dense_blocked_max_variance_diff":
                synthetic_paper_var_diff,
            "synthetic_legacy_exact_blocked_max_logp_diff":
                legacy_logp_diff,
            "synthetic_legacy_exact_blocked_max_z_diff":
                legacy_z_diff,
            "synthetic_legacy_rank_exact_match": legacy_rank_match,
            "real_paper_dense_blocked_max_z_diff": paper_real_z_diff,
            "real_paper_dense_blocked_max_variance_diff":
                paper_real_var_diff,
            "historical_same_forward_first8_max_abs_diff": first8_max,
            "historical_same_forward_first8_per_feature": first8_compare,
            "historical_first8_profile": historical8_profile,
            "variance_clamp_count":
                int(private_result["variance_clamp_count"]),
            "target_tokens": int(private_result["target_tokens"]),
            "valid_target_tokens":
                int(private_result["valid_target_tokens"]),
        }
        profile = {
            "status": "complete_label_blind_runtime_fidelity_pilot",
            "rows": 200,
            "windows": int(len(window_features)),
            "run_seconds": float(run_seconds),
            "private_runner_seconds":
                float(private_result["run_seconds"]),
            "model_load_seconds": float(model_load_seconds),
            "peak_gpu_memory_bytes": max(
                int(private_result["peak_gpu_memory_bytes"]),
                int(torch.cuda.max_memory_allocated()),
            ),
            "gpu": torch.cuda.get_device_name(0),
            "cohort_prediction_input_sha256": actual_cohort_sha,
            "private_scorer_source":
                "renta0426/poisoned-chalice-minkpp-paper-10k-v1",
        }
        manifest = {
            "schema_version": "codeparrot_fresh_pilot_manifest_v1",
            "experiment_id": EXPERIMENT_ID,
            "status": "sealed_label_blind_pilot_predictions",
            "rows": 200,
            "selection":
                "first_200_sample_ids_from_label_free_4000_prediction_input",
            "model": runtime_identity,
            "same_forward_paper_and_legacy": True,
            "window_contract":
                "whole_or_prefix_middle_suffix_768_from_existing_private_scorer",
            "max_length": 768,
            "max_batch_tokens": 1536,
            "sequence_chunk_tokens": 32,
            "rank_vocab_block_size": 8192,
            "paper_vocab_block_size": 8192,
            "paper_variance_floor": 1e-12,
            "features": [
                "loss_multiwindow", "legacy_minkpp10",
                "paper_minkpp10", "local_64", "neg_mean_log_rank",
            ],
            "performance_metrics_allowed": False,
            "performance_metrics_computed": False,
            "membership_join_performed": False,
            "evaluation_manifest_attached": False,
            "private_github_access": False,
            "competition_submission": False,
            "full_4000_scoring_authorized": False,
            "automatic_compute_retries": 0,
        }
        for name, payload in (
            ("pilot_fidelity.json", fidelity),
            ("pilot_profile.json", profile),
            ("pilot_manifest.json", manifest),
        ):
            (OUTPUT / name).write_text(
                json.dumps(payload, indent=2, sort_keys=True, default=str)
                + "\n",
                encoding="utf-8",
            )
        print(json.dumps({
            "status": manifest["status"],
            "rows": 200,
            "windows": profile["windows"],
            "fidelity": fidelity["status"],
            "performance_metrics_computed": False,
            "membership_join_performed": False,
            "full_4000_scoring_authorized": False,
        }, sort_keys=True))
        '''
    ).strip()


def compose(
    minkpp_dir: Path,
    output_dir: Path,
    target: str,
    title: str,
) -> None:
    minkpp_notebook = _load_single_notebook(minkpp_dir)

    starter_source = _find_code(
        minkpp_notebook,
        ("def make_window_records", "def dynamic_batches", "def summarize_tokens"),
    )
    v2_source = _find_code(
        minkpp_notebook,
        (
            "class FeatureCacheV2Config",
            "def _chunk_statistics",
            "def extract_window_features_optimized",
        ),
    )
    paper_source = _find_code(
        minkpp_notebook,
        (
            "MINKPP_PAPER_SCHEMA_VERSION",
            "probability_weighted_token_statistics_dense",
            "probability_weighted_token_statistics_blocked",
        ),
    )
    runner_source = _find_code(
        minkpp_notebook,
        (
            "def run_private_paper_records(records):",
            "window_features",
            "real_dense_blocked_max_z_diff",
        ),
    )

    private_sources = (
        starter_source,
        v2_source,
        paper_source,
        runner_source,
    )
    for source in private_sources:
        _assert_no_forbidden_transport(source)

    notebook = nbformat.v4.new_notebook()
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata.language_info = {"name": "python", "version": "3.12"}
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# CodeParrot fresh confirmation — 200-row label-blind T4 pilot\n\n"
            "Runtime/fidelity only. No membership labels, performance metric, "
            "competition submission, or full-4k authorization."
        ),
        nbformat.v4.new_code_cell(
            "%pip install -q 'transformers==5.0.0' accelerate pyarrow huggingface_hub"
        ),
        nbformat.v4.new_code_cell(starter_source),
        nbformat.v4.new_code_cell(v2_source),
        nbformat.v4.new_code_cell(paper_source),
        nbformat.v4.new_code_cell(runner_source),
        nbformat.v4.new_code_cell(_setup_cell()),
        nbformat.v4.new_code_cell(_load_cell()),
        nbformat.v4.new_code_cell(_model_cell()),
        nbformat.v4.new_code_cell(_oracle_cell()),
        nbformat.v4.new_code_cell(_pilot_cell()),
    ]
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []

    combined = "\n".join(
        _source(cell) for cell in notebook.cells if cell.cell_type == "code"
    )
    forbidden_final = FORBIDDEN_SOURCE_FRAGMENTS + (
        "competitions submit",
        "submission.csv",
    )
    if any(fragment in combined for fragment in forbidden_final):
        raise RuntimeError(
            "final private pilot contains forbidden transport/side effect"
        )
    required = (
        "make_window_records(private_input",
        "run_private_paper_records",
        "EXPECTED_COHORT_SHA256",
        "first_200_sample_ids_from_label_free_4000_prediction_input",
        "performance_metrics_computed",
        "membership_join_performed",
        "original_sample_id_passed_to_scorer",
        "full_4000_scoring_authorized",
        "synthetic_legacy_rank_exact_match",
        "historical_same_forward_first8_max_abs_diff",
    )
    if not all(marker in combined for marker in required):
        raise RuntimeError("final private pilot contract incomplete")
    nbformat.validate(notebook)

    output_dir.mkdir(parents=True, exist_ok=True)
    code_file = target.split("/", 1)[1] + ".ipynb"
    nbformat.write(notebook, output_dir / code_file)
    metadata = {
        "id": target,
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [
            "gpu", "codeparrot", "membership-inference", "fidelity-pilot"
        ],
        "dataset_sources": [],
        "kernel_sources": [COHORT_KERNEL],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "COMPOSE_PASS",
        "target": target,
        "private_source_cells": len(private_sources),
        "private_github_access": False,
        "source_bodies_logged": False,
        "stage2_private_source_required": False,
    }, sort_keys=True))


def _synthetic_notebook(path: Path) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_code_cell(
            "def make_window_records(*a, **k): pass\n"
            "def dynamic_batches(*a, **k): pass\n"
            "def summarize_tokens(*a, **k): pass"
        ),
        nbformat.v4.new_code_cell(
            "class FeatureCacheV2Config: pass\n"
            "def _chunk_statistics(*a, **k): pass\n"
            "def extract_window_features_optimized(*a, **k): pass"
        ),
        nbformat.v4.new_code_cell(
            "MINKPP_PAPER_SCHEMA_VERSION='x'\n"
            "def probability_weighted_token_statistics_dense(*a, **k): pass\n"
            "def probability_weighted_token_statistics_blocked(*a, **k): pass"
        ),
        nbformat.v4.new_code_cell(
            "def run_private_paper_records(records):\n"
            "    window_features=None\n"
            "    real_dense_blocked_max_z_diff=0\n"
            "    return {}"
        ),
    ]
    nbformat.write(notebook, path)


def self_test() -> None:
    with tempfile.TemporaryDirectory() as temp_value:
        root = Path(temp_value)
        minkpp_dir = root / "minkpp"
        output_dir = root / "out"
        minkpp_dir.mkdir()
        _synthetic_notebook(minkpp_dir / "source.ipynb")
        compose(
            minkpp_dir,
            output_dir,
            "renta0426/codeparrot-fresh-pilot-v1",
            "CodeParrot Fresh Pilot V1",
        )
        metadata = json.loads(
            (output_dir / "kernel-metadata.json").read_text()
        )
        assert metadata["kernel_sources"] == [COHORT_KERNEL]
        assert metadata["enable_gpu"] is True
        assert metadata["machine_shape"] == "NvidiaTeslaT4"
        assert len(list(output_dir.glob("*.ipynb"))) == 1
    print(
        "CODEPARROT_PILOT_V2_COMPOSER_SELF_TEST PASS "
        "private_repo_access=0 stage2_private_source_required=0"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minkpp-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target")
    parser.add_argument("--title")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    missing = [
        name
        for name, value in (
            ("--minkpp-dir", args.minkpp_dir),
            ("--output-dir", args.output_dir),
            ("--target", args.target),
            ("--title", args.title),
        )
        if value is None
    ]
    if missing:
        parser.error("required arguments missing: " + ", ".join(missing))
    compose(args.minkpp_dir, args.output_dir, args.target, args.title)


if __name__ == "__main__":
    main()
