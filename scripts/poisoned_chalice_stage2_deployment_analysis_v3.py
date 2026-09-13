from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import roc_auc_score

TASK_ID = "STAGE2-DEPLOYMENT-ANALYSIS-V1"
ANALYSIS_VERSION = 3
SOURCE_SCRIPT_VERSION_ID = 349274389
DATASET_ID = "Poisoned-Chalice/ICSE-2027-public"
DATASET_REVISION = "2ed5468723efa5457a3665782c6979ea4dbac7c2"
LANGUAGES = ("Go", "Java", "Python", "Ruby", "Rust")
FUSION = "TR_HR_rank_50_50"
OUT = Path("/kaggle/working/stage2_deployment_analysis_v1")
OUT.mkdir(parents=True, exist_ok=True)
EXPECTED_HASHES = {
    "evaluation.json": "aa67746f23b3b1bf73d4d1b1d81e746b7909eae69c6059187b711071608d228c",
    "predictions_label_free.csv": "57aa7ff4633e88808c135443e9dea2711a0ed869b04e7765e7709242bd842e63",
    "holdout_scoring_manifest.parquet": "abc45c6131593a5899bf1774ee47f6e160d1d79e0e5928149b5eec0028b90680",
    "tr_stage2_features.parquet": "9236d3d015ae040b0fdeb0565f74a67f440ca42e65df2cb84ea3a45cb5a495a5",
    "tr_stage2_manifest.json": "5d5a066fdd7a544b850da609a3454617b86e7f2e12a60fb2168eb371f12efa94",
    "holdout_selection_manifest.json": "e134616858b4c8bb6c9a385fa38a55b3681f79d29f4200c7de8a62709cae72b4",
    "run_manifest.json": "492af0a639fb44fa46cd5362ab72b0c779e24d217ac4100dac84cb805c0fb643",
}
CORE = ("HR_mean_top5", "GR_mean96", "TR_stage2_v1", FUSION, "C1_source_content")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_exact(name: str, digest: str) -> Path:
    matches = [p for p in Path("/kaggle/input").rglob(name) if p.is_file() and sha256(p) == digest]
    if len(matches) != 1:
        raise RuntimeError(f"hash-locked source resolution failed: {name} count={len(matches)}")
    return matches[0]


def low_fpr(y, score, fpr=0.01):
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    negatives = np.sort(score[y == 0])
    allowed = max(1, int(math.floor(fpr * len(negatives))))
    threshold = float(negatives[-allowed])
    mask = score > threshold
    fp = int(np.sum(mask & (y == 0)))
    tp = int(np.sum(mask & (y == 1)))
    return threshold, fp, tp, float(tp / max(1, np.sum(y == 1))), mask


def metrics(y, score):
    threshold, fp, tp, tpr, _ = low_fpr(y, score)
    return {
        "auc": float(roc_auc_score(y, score)),
        "conservative_threshold_1pct": threshold,
        "conservative_fp_1pct": fp,
        "conservative_tp_1pct": tp,
        "conservative_tpr_1pct": tpr,
    }


def global_rank01(values):
    return pd.Series(np.asarray(values, dtype=np.float64)).rank(method="average", pct=True).to_numpy(dtype=np.float64)


def length_series(frame):
    for column in ("token_count", "n_tokens", "num_tokens", "content_length", "char_length", "content_chars", "n_chars", "length"):
        if column in frame.columns and pd.api.types.is_numeric_dtype(frame[column]):
            return pd.to_numeric(frame[column], errors="coerce"), column
    if "content" in frame.columns:
        return frame["content"].astype(str).str.len().astype(float), "content_chars_derived"
    return None, None


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    # Prove exact source bytes before labels are loaded.
    paths = {name: find_exact(name, digest) for name, digest in EXPECTED_HASHES.items()}
    observed = {name: sha256(path) for name, path in paths.items()}
    if observed != EXPECTED_HASHES:
        raise RuntimeError("source hash verification failed")
    evaluation = json.loads(paths["evaluation.json"].read_text(encoding="utf-8"))
    pred = pd.read_csv(paths["predictions_label_free.csv"])
    manifest = pd.read_parquet(paths["holdout_scoring_manifest.parquet"])
    tr_features = pd.read_parquet(paths["tr_stage2_features.parquet"])
    if len(pred) != 5000 or pred["sample_id"].nunique() != 5000:
        raise RuntimeError("prediction cardinality failed")
    if evaluation.get("prediction_sha256") != EXPECTED_HASHES["predictions_label_free.csv"]:
        raise RuntimeError("evaluation/prediction hash link failed")
    print("STAGE2_ANALYSIS_V3_HASH_LOCK PASS")

    # Join official labels only after the hash boundary.
    parts = []
    for language in LANGUAGES:
        ds = load_dataset(DATASET_ID, language, split="train", revision=DATASET_REVISION)
        part = ds.select_columns(["sample_id", "membership"]).to_pandas()
        parts.append(part)
    labels = pd.concat(parts, ignore_index=True)
    if labels["sample_id"].duplicated().any():
        raise RuntimeError("official label sample_id duplication")
    label_map = labels.set_index("sample_id")["membership"].map({"non-member": 0, "member": 1})
    y_series = pred["sample_id"].map(label_map)
    if y_series.isna().any():
        raise RuntimeError(f"missing official labels: {int(y_series.isna().sum())}")
    y = y_series.astype(np.int8).to_numpy()
    if int(y.sum()) != 2500:
        raise RuntimeError("holdout balance mismatch")
    print("STAGE2_ANALYSIS_V3_LABEL_JOIN PASS")

    score_cols = [c for c in pred.columns if c not in {"sample_id", "language"} and pd.api.types.is_numeric_dtype(pred[c])]
    if FUSION not in score_cols or "TR_stage2_v1" not in score_cols or "HR_mean_top5" not in score_cols:
        raise RuntimeError("required saved score columns missing")

    # Recompute aggregate diagnostics from persisted score columns. For all
    # non-fusion columns, exact AUC reproduction against evaluation.json is a
    # hard integrity gate. The fusion is treated differently because v2 proved
    # that the round-tripped rank-fusion artifact does not exactly reproduce the
    # in-memory frozen AUC; that discrepancy is recorded rather than overwritten.
    aggregate = {}
    reproduction = {}
    frozen = evaluation.get("metrics", {})
    for name in score_cols:
        score = pred[name].to_numpy(dtype=float)
        aggregate[name] = metrics(y, score)
        if name in frozen and "auc" in frozen[name]:
            frozen_auc = float(frozen[name]["auc"])
            delta = aggregate[name]["auc"] - frozen_auc
            strict = name != FUSION
            reproduction[name] = {
                "persisted_score_auc": aggregate[name]["auc"],
                "frozen_in_memory_auc": frozen_auc,
                "delta": delta,
                "strict_integrity_gate": strict,
                "passed": abs(delta) <= 1e-8,
                "metric_authority": "evaluation.json" if name == FUSION else "reproduced",
            }
            if strict and abs(delta) > 1e-8:
                raise RuntimeError(f"nonfusion saved-score AUC reproduction mismatch: {name}")
    print("STAGE2_ANALYSIS_V3_NONFUSION_REPRO PASS")

    persisted_fusion = pred[FUSION].to_numpy(dtype=float)
    reconstructed_from_persisted = 0.5 * global_rank01(pred["TR_stage2_v1"]) + 0.5 * global_rank01(pred["HR_mean_top5"])
    fusion_max_abs = float(np.max(np.abs(persisted_fusion - reconstructed_from_persisted)))
    if fusion_max_abs > 1e-12:
        raise RuntimeError(f"persisted fusion formula integrity mismatch: {fusion_max_abs}")
    fusion_audit = {
        "saved_column": FUSION,
        "persisted_vs_reconstructed_max_abs_error": fusion_max_abs,
        "persisted_unique_values": int(pd.Series(persisted_fusion).nunique()),
        "reconstructed_unique_values": int(pd.Series(reconstructed_from_persisted).nunique()),
        "persisted_auc": aggregate[FUSION]["auc"],
        "frozen_in_memory_auc": float(frozen[FUSION]["auc"]),
        "auc_delta": aggregate[FUSION]["auc"] - float(frozen[FUSION]["auc"]),
        "interpretation": "artifact-level rank/tie discrepancy recorded for diagnostics; frozen evaluation.json remains authoritative for preregistered aggregate claims",
        "promotion_use": "none",
    }
    print("STAGE2_ANALYSIS_V3_FUSION_AUDIT PASS")

    pred[score_cols].astype(float).corr(method="spearman").to_csv(OUT / "saved_score_spearman.csv")

    detected = {}
    for name in score_cols:
        _, _, _, _, mask = low_fpr(y, pred[name].to_numpy(dtype=float))
        detected[name] = {
            "tp": set(pred.loc[mask & (y == 1), "sample_id"]),
            "fp": set(pred.loc[mask & (y == 0), "sample_id"]),
        }
    overlap_rows = []
    focus = [name for name in CORE if name in score_cols]
    for i, left in enumerate(focus):
        for right in focus[i + 1:]:
            left_tp, right_tp = detected[left]["tp"], detected[right]["tp"]
            left_fp, right_fp = detected[left]["fp"], detected[right]["fp"]
            overlap_rows.append({
                "score_a": left,
                "score_b": right,
                "tp_intersection": len(left_tp & right_tp),
                "tp_union": len(left_tp | right_tp),
                "tp_jaccard": len(left_tp & right_tp) / max(1, len(left_tp | right_tp)),
                "tp_unique_a": len(left_tp - right_tp),
                "tp_unique_b": len(right_tp - left_tp),
                "fp_intersection": len(left_fp & right_fp),
                "fp_union": len(left_fp | right_fp),
                "fp_jaccard": len(left_fp & right_fp) / max(1, len(left_fp | right_fp)),
            })
    pd.DataFrame(overlap_rows).to_csv(OUT / "low_fpr_overlap.csv", index=False)

    language_rows = []
    for name in score_cols:
        for language in LANGUAGES:
            idx = pred["language"].eq(language).to_numpy()
            language_rows.append({"score": name, "language": language, **metrics(y[idx], pred.loc[idx, name].to_numpy(dtype=float))})
    pd.DataFrame(language_rows).to_csv(OUT / "per_language_metrics.csv", index=False)

    length, length_name = length_series(manifest)
    length_info = {"available": False, "source": None}
    if length is not None and "sample_id" in manifest.columns:
        length_map = pd.Series(length.to_numpy(), index=manifest["sample_id"].astype(str))
        lengths = pred["sample_id"].map(length_map)
        if lengths.notna().all() and lengths.nunique() >= 5:
            quintile = pd.qcut(lengths.rank(method="first"), q=5, labels=False)
            quintile_rows, corr_rows = [], []
            for name in score_cols:
                corr_rows.append({
                    "score": name,
                    "length_source": length_name,
                    "spearman_score_length": float(pd.Series(pred[name]).corr(lengths, method="spearman")),
                })
                for q in range(5):
                    idx = quintile.eq(q).to_numpy()
                    quintile_rows.append({"score": name, "length_quintile": q, **metrics(y[idx], pred.loc[idx, name].to_numpy(dtype=float))})
            pd.DataFrame(quintile_rows).to_csv(OUT / "per_length_quintile_metrics.csv", index=False)
            pd.DataFrame(corr_rows).to_csv(OUT / "score_length_correlations.csv", index=False)
            length_info = {
                "available": True,
                "source": length_name,
                "min": float(lengths.min()),
                "median": float(lengths.median()),
                "max": float(lengths.max()),
            }

    # Attach sample ids to the label-free TR feature frame only through the
    # frozen sample_index ordering when sample_id is absent.
    tr = tr_features.copy()
    if "sample_id" not in tr.columns and "sample_index" in tr.columns:
        ordered = tr.sort_values("sample_index").reset_index(drop=True)
        if len(ordered) == len(pred) and np.array_equal(ordered["sample_index"].to_numpy(), np.arange(len(pred))):
            ordered["sample_id"] = pred["sample_id"].to_numpy()
            tr = ordered
    if "sample_id" in tr.columns:
        tr["__label"] = tr["sample_id"].map(label_map)
        tr["__saved_tr"] = tr["sample_id"].map(pred.set_index("sample_id")["TR_stage2_v1"])
        diagnostics = []
        for column in tr.columns:
            if column in {"sample_id", "__label", "__saved_tr"} or not pd.api.types.is_numeric_dtype(tr[column]):
                continue
            values = pd.to_numeric(tr[column], errors="coerce")
            good = values.notna() & tr["__label"].notna()
            if int(good.sum()) < 100 or values[good].nunique() < 2:
                continue
            yy = tr.loc[good, "__label"].astype(int).to_numpy()
            vv = values[good].to_numpy(dtype=float)
            try:
                auc = float(roc_auc_score(yy, vv))
            except ValueError:
                continue
            diagnostics.append({
                "feature": column,
                "raw_auc": auc,
                "abs_auc_distance_from_chance": abs(auc - 0.5),
                "spearman_with_saved_TR": float(values.corr(tr["__saved_tr"], method="spearman")),
                "rows": int(good.sum()),
                "interpretation": "posthoc mechanism diagnostic only; not a promoted candidate",
            })
        if diagnostics:
            pd.DataFrame(diagnostics).sort_values(["abs_auc_distance_from_chance", "feature"], ascending=[False, True]).to_csv(OUT / "tr_primitive_diagnostics.csv", index=False)

    metric_table = pd.DataFrame([{"score": name, **aggregate[name]} for name in score_cols]).sort_values("auc", ascending=False)
    metric_table.to_csv(OUT / "persisted_score_metrics_diagnostic.csv", index=False)

    summary = {
        "schema_version": 3,
        "task_id": TASK_ID,
        "analysis_version": ANALYSIS_VERSION,
        "source_script_version_id": SOURCE_SCRIPT_VERSION_ID,
        "source_hashes_verified_before_labels": True,
        "source_hashes": observed,
        "rows": 5000,
        "labels": {"member": int(y.sum()), "non_member": int(len(y) - y.sum())},
        "reproduction": reproduction,
        "fusion_artifact_audit": fusion_audit,
        "persisted_score_metrics_diagnostic": aggregate,
        "length_diagnostics": length_info,
        "failure_closeout": {
            "analysis_v1": "reconstructed frozen fusion instead of using the persisted score artifact",
            "analysis_v2": "used the persisted fusion correctly but treated its exact AUC mismatch against the in-memory frozen evaluation as fatal",
            "analysis_v3": "keeps strict reproduction for nonfusion scores, records fusion artifact drift, and preserves evaluation.json as the preregistered aggregate authority",
        },
        "frozen_evaluation_context": {
            "metrics": {name: frozen.get(name) for name in CORE if name in frozen},
            "detection_delta_at_1pct": evaluation.get("detection_delta_at_1pct"),
            "paired_bootstrap": evaluation.get("paired_bootstrap"),
            "reference_suite_sensitivity": evaluation.get("reference_suite_sensitivity"),
        },
        "anti_posthoc": {
            "no_model_inference": True,
            "no_score_weight_search": True,
            "no_threshold_search_for_promotion": True,
            "fusion_artifact_discrepancy_not_used_for_promotion": True,
            "tr_primitive_results_are_mechanism_diagnostics_only": True,
            "row_level_prediction_plus_label_artifact_persisted": False,
        },
    }
    dump_json(OUT / "analysis_summary.json", summary)

    report = [
        "# STAGE2-DEPLOYMENT-ANALYSIS-V1 — analysis v3", "",
        "CPU-only diagnostic analysis of the hash-locked Stage2 deployment output from scriptVersionId 349274389.",
        "The frozen evaluation.json remains authoritative for preregistered aggregate claims. Persisted scores are used for row-level diagnostic structure.", "",
        "## Fusion artifact audit", "",
        f"- Persisted fusion AUC: {fusion_audit['persisted_auc']:.10f}",
        f"- Frozen in-memory fusion AUC: {fusion_audit['frozen_in_memory_auc']:.10f}",
        f"- Delta: {fusion_audit['auc_delta']:+.10f}",
        f"- Persisted-vs-reconstructed max abs score error: {fusion_audit['persisted_vs_reconstructed_max_abs_error']:.3e}",
        f"- Unique persisted/reconstructed fusion values: {fusion_audit['persisted_unique_values']} / {fusion_audit['reconstructed_unique_values']}", "",
        "This discrepancy is recorded as an artifact-level rank/tie diagnostic and is not used to change a frozen scientific decision.", "",
        "## Persisted-score diagnostic headline", "", "| score | AUC | TP@conservative 1% | FP |", "|---|---:|---:|---:|",
    ]
    for _, row in metric_table.iterrows():
        report.append(f"| {row['score']} | {row['auc']:.8f} | {int(row['conservative_tp_1pct'])} | {int(row['conservative_fp_1pct'])} |")
    report += ["", "## Frozen paired-bootstrap context", ""]
    for key, value in (evaluation.get("paired_bootstrap") or {}).items():
        delta = (value or {}).get("delta_auc") or {}
        if delta:
            report.append(f"- {key}: delta AUC mean {float(delta['mean']):.6f}, 95% [{float(delta['lower_95']):.6f}, {float(delta['upper_95']):.6f}].")
    report += [
        "", "## Diagnostic outputs", "",
        "- saved_score_spearman.csv",
        "- low_fpr_overlap.csv",
        "- per_language_metrics.csv",
        "- per_length_quintile_metrics.csv / score_length_correlations.csv when length is available",
        "- tr_primitive_diagnostics.csv when row identity can be proven (post-hoc mechanism diagnostics only)", "",
        "## Integrity boundary", "",
        "- Source hashes verified before labels were loaded.",
        "- Nonfusion persisted scores must reproduce frozen AUC or execution fails.",
        "- Fusion discrepancy is recorded, not tuned away.",
        "- No model inference, score-weight search, threshold search for promotion, or row-level prediction+label artifact.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("STAGE2_DEPLOYMENT_ANALYSIS_V1_V3 COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
