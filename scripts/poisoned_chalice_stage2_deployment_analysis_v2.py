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
ANALYSIS_VERSION = 2
SOURCE_SCRIPT_VERSION_ID = 349274389
DATASET_ID = "Poisoned-Chalice/ICSE-2027-public"
DATASET_REVISION = "2ed5468723efa5457a3665782c6979ea4dbac7c2"
LANGUAGES = ("Go", "Java", "Python", "Ruby", "Rust")
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
CORE = ("HR_mean_top5", "GR_mean96", "TR_stage2_v1", "TR_HR_rank_50_50", "C1_source_content")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
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
    neg = np.sort(score[y == 0])
    allowed = max(1, int(math.floor(fpr * len(neg))))
    threshold = float(neg[-allowed])
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


def length_series(frame):
    for c in ("token_count", "n_tokens", "num_tokens", "content_length", "char_length", "content_chars", "n_chars", "length"):
        if c in frame.columns and pd.api.types.is_numeric_dtype(frame[c]):
            return pd.to_numeric(frame[c], errors="coerce"), c
    if "content" in frame.columns:
        return frame["content"].astype(str).str.len().astype(float), "content_chars_derived"
    return None, None


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    # Prove the exact Stage2 source output before loading any label.
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
    print("STAGE2_ANALYSIS_V2_HASH_LOCK PASS")

    # Labels are loaded only after the hash boundary above.
    parts = []
    for language in LANGUAGES:
        ds = load_dataset(DATASET_ID, language, split="train", revision=DATASET_REVISION)
        parts.append(ds.select_columns(["sample_id", "membership"]).to_pandas())
    labels = pd.concat(parts, ignore_index=True)
    if labels["sample_id"].duplicated().any():
        raise RuntimeError("official label sample_id duplication")
    label_map = labels.set_index("sample_id")["membership"].map({"non-member": 0, "member": 1})
    y_s = pred["sample_id"].map(label_map)
    if y_s.isna().any():
        raise RuntimeError(f"missing official labels: {int(y_s.isna().sum())}")
    y = y_s.astype(np.int8).to_numpy()
    if int(y.sum()) != 2500:
        raise RuntimeError("holdout balance mismatch")
    print("STAGE2_ANALYSIS_V2_LABEL_JOIN PASS")

    # Critical v2 fix: every candidate, including the frozen 50/50 fusion,
    # is evaluated DIRECTLY from its persisted prediction column. No fusion
    # reconstruction, weight search, or threshold tuning is performed here.
    score_cols = [c for c in pred.columns if c not in {"sample_id", "language"} and pd.api.types.is_numeric_dtype(pred[c])]
    if "TR_HR_rank_50_50" not in score_cols:
        raise RuntimeError("saved frozen fusion column missing")
    aggregate, reproduction = {}, {}
    frozen = evaluation.get("metrics", {})
    for name in score_cols:
        score = pred[name].to_numpy(dtype=float)
        aggregate[name] = metrics(y, score)
        if name in frozen and "auc" in frozen[name]:
            delta = aggregate[name]["auc"] - float(frozen[name]["auc"])
            reproduction[name] = {
                "saved_score_auc": aggregate[name]["auc"],
                "frozen_evaluation_auc": float(frozen[name]["auc"]),
                "delta": delta,
                "passed": abs(delta) <= 1e-8,
                "source": "persisted predictions_label_free.csv column",
            }
            if abs(delta) > 1e-8:
                raise RuntimeError(f"saved-score AUC reproduction mismatch: {name}")
    print(f"STAGE2_ANALYSIS_V2_SAVED_SCORE_REPRO PASS scores={len(reproduction)}")

    pred[score_cols].astype(float).corr(method="spearman").to_csv(OUT / "saved_score_spearman.csv")

    # Low-FPR overlap at each score's own conservative threshold.
    detected = {}
    for name in score_cols:
        _, _, _, _, mask = low_fpr(y, pred[name].to_numpy(dtype=float))
        detected[name] = {
            "tp": set(pred.loc[mask & (y == 1), "sample_id"]),
            "fp": set(pred.loc[mask & (y == 0), "sample_id"]),
        }
    overlap = []
    focus = [x for x in CORE if x in score_cols]
    for i, a in enumerate(focus):
        for b in focus[i + 1:]:
            ta, tb = detected[a]["tp"], detected[b]["tp"]
            fa, fb = detected[a]["fp"], detected[b]["fp"]
            overlap.append({
                "score_a": a, "score_b": b,
                "tp_intersection": len(ta & tb), "tp_union": len(ta | tb),
                "tp_jaccard": len(ta & tb) / max(1, len(ta | tb)),
                "tp_unique_a": len(ta - tb), "tp_unique_b": len(tb - ta),
                "fp_intersection": len(fa & fb), "fp_union": len(fa | fb),
                "fp_jaccard": len(fa & fb) / max(1, len(fa | fb)),
            })
    pd.DataFrame(overlap).to_csv(OUT / "low_fpr_overlap.csv", index=False)

    # Language diagnostics.
    rows = []
    for name in score_cols:
        for language in LANGUAGES:
            idx = pred["language"].eq(language).to_numpy()
            rows.append({"score": name, "language": language, **metrics(y[idx], pred.loc[idx, name].to_numpy(dtype=float))})
    pd.DataFrame(rows).to_csv(OUT / "per_language_metrics.csv", index=False)

    # Length diagnostics from the label-free scoring manifest when available.
    length, length_name = length_series(manifest)
    length_info = {"available": False, "source": None}
    if length is not None and "sample_id" in manifest.columns:
        lengths = pred["sample_id"].map(pd.Series(length.to_numpy(), index=manifest["sample_id"].astype(str)))
        if lengths.notna().all() and lengths.nunique() >= 5:
            q = pd.qcut(lengths.rank(method="first"), q=5, labels=False)
            qrows, crows = [], []
            for name in score_cols:
                crows.append({"score": name, "length_source": length_name, "spearman_score_length": float(pd.Series(pred[name]).corr(lengths, method="spearman"))})
                for quintile in range(5):
                    idx = q.eq(quintile).to_numpy()
                    qrows.append({"score": name, "length_quintile": quintile, **metrics(y[idx], pred.loc[idx, name].to_numpy(dtype=float))})
            pd.DataFrame(qrows).to_csv(OUT / "per_length_quintile_metrics.csv", index=False)
            pd.DataFrame(crows).to_csv(OUT / "score_length_correlations.csv", index=False)
            length_info = {"available": True, "source": length_name, "min": float(lengths.min()), "median": float(lengths.median()), "max": float(lengths.max())}

    # TR primitive results are explicitly post-hoc mechanism diagnostics only.
    if "sample_id" in tr_features.columns:
        tr = tr_features.copy()
        tr["__label"] = tr["sample_id"].map(label_map)
        tr["__saved_tr"] = tr["sample_id"].map(pred.set_index("sample_id")["TR_stage2_v1"])
        diag = []
        for c in tr.columns:
            if c in {"__label", "__saved_tr"} or not pd.api.types.is_numeric_dtype(tr[c]):
                continue
            values = pd.to_numeric(tr[c], errors="coerce")
            good = values.notna() & tr["__label"].notna()
            if good.sum() < 100 or values[good].nunique() < 2:
                continue
            yy = tr.loc[good, "__label"].astype(int).to_numpy()
            vv = values[good].to_numpy(dtype=float)
            try:
                auc = float(roc_auc_score(yy, vv))
            except ValueError:
                continue
            diag.append({
                "feature": c, "raw_auc": auc,
                "abs_auc_distance_from_chance": abs(auc - 0.5),
                "spearman_with_saved_TR": float(values.corr(tr["__saved_tr"], method="spearman")),
                "rows": int(good.sum()),
                "interpretation": "posthoc mechanism diagnostic only; not a promoted candidate",
            })
        if diag:
            pd.DataFrame(diag).sort_values(["abs_auc_distance_from_chance", "feature"], ascending=[False, True]).to_csv(OUT / "tr_primitive_diagnostics.csv", index=False)

    metric_table = pd.DataFrame([{"score": s, **aggregate[s]} for s in score_cols]).sort_values("auc", ascending=False)
    metric_table.to_csv(OUT / "saved_score_metrics.csv", index=False)
    summary = {
        "schema_version": 2,
        "task_id": TASK_ID,
        "analysis_version": ANALYSIS_VERSION,
        "source_script_version_id": SOURCE_SCRIPT_VERSION_ID,
        "source_hashes_verified_before_labels": True,
        "source_hashes": observed,
        "rows": 5000,
        "labels": {"member": int(y.sum()), "non_member": int(len(y) - y.sum())},
        "saved_score_reproduction": reproduction,
        "saved_score_metrics": aggregate,
        "length_diagnostics": length_info,
        "v1_failure_closeout": {
            "root_cause": "analysis-v1 reconstructed TR_HR_rank_50_50 instead of evaluating the persisted frozen fusion column",
            "scientific_impact": "none on the frozen Stage2 source run; analysis-only implementation error",
            "v2_fix": "evaluate every frozen candidate directly from predictions_label_free.csv",
        },
        "frozen_evaluation_context": {
            "detection_delta_at_1pct": evaluation.get("detection_delta_at_1pct"),
            "paired_bootstrap": evaluation.get("paired_bootstrap"),
            "reference_suite_sensitivity": evaluation.get("reference_suite_sensitivity"),
        },
        "anti_posthoc": {
            "no_model_inference": True,
            "no_score_weight_search": True,
            "no_threshold_search_for_promotion": True,
            "tr_primitive_results_are_mechanism_diagnostics_only": True,
            "row_level_prediction_plus_label_artifact_persisted": False,
        },
    }
    dump_json(OUT / "analysis_summary.json", summary)

    report = [
        "# STAGE2-DEPLOYMENT-ANALYSIS-V1 — analysis v2", "",
        "CPU-only diagnostic analysis of the hash-locked output from scriptVersionId 349274389.",
        "All frozen metrics use persisted score columns; the 50/50 fusion is not reconstructed.", "",
        "## Saved-score headline", "", "| score | AUC | TP@conservative 1% | FP |", "|---|---:|---:|---:|",
    ]
    for _, r in metric_table.iterrows():
        report.append(f"| {r['score']} | {r['auc']:.8f} | {int(r['conservative_tp_1pct'])} | {int(r['conservative_fp_1pct'])} |")
    report += ["", "## Frozen paired-bootstrap context", ""]
    for key, value in (evaluation.get("paired_bootstrap") or {}).items():
        d = (value or {}).get("delta_auc") or {}
        if d:
            report.append(f"- {key}: delta AUC mean {float(d['mean']):.6f}, 95% [{float(d['lower_95']):.6f}, {float(d['upper_95']):.6f}].")
    report += [
        "", "## Diagnostic outputs", "",
        "- saved_score_spearman.csv", "- low_fpr_overlap.csv", "- per_language_metrics.csv",
        "- per_length_quintile_metrics.csv / score_length_correlations.csv when a label-free length field is available",
        "- tr_primitive_diagnostics.csv (post-hoc mechanism diagnostics only)", "",
        "## Integrity boundary", "",
        "- Source hashes verified before labels were loaded.", "- No model inference.",
        "- No result-derived weight/threshold/feature/layer/sign search for promotion.",
        "- No row-level prediction+label artifact is persisted.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("STAGE2_DEPLOYMENT_ANALYSIS_V1_V2 COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
