#!/usr/bin/env python3
"""E10 builder v2: add a narrow frozen-bundle models compatibility shim."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cmi_flu_strategy_e10_prepare_v1 as base


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repository_root.expanduser().resolve()
    reference_dir = args.reference_dir.expanduser().resolve()
    base.validate_request(root)
    e10, synth, anchor, rank_transfer, study_similarity = base.load_exact_sources(root)
    v3, parent_runtime = base.build_parent_runtime(root, reference_dir)
    runtime = base.patch_runtime(
        v3, parent_runtime, e10, synth, anchor, rank_transfer, study_similarity
    )

    old = '''def load_e10_modules() -> tuple[object, object, object]:
    load_e05_module()
    def install(name: str, source: str) -> object:
'''
    new = '''def load_e10_modules() -> tuple[object, object, object]:
    load_e05_module()
    # E05-v6 predates models.summarize_metric_frame, imported by the exact
    # Phase-A rank/anchor helpers. Add only this pure aggregate helper. The
    # bundle's model classes/build/fit/preparation functions remain unchanged.
    from cmi_flu import models as _e10_models
    for _required in ("ModelSpec", "build_estimator", "fit_final_model", "prepare_model_frame"):
        if not hasattr(_e10_models, _required):
            raise BridgeContractError(f"e10_frozen_models_missing:{_required}")
    if not hasattr(_e10_models, "summarize_metric_frame"):
        def _e10_summarize_metric_frame(frame):
            if frame.empty:
                return {"count":0,"spearman_mean":None,"spearman_median":None,"spearman_min":None,"spearman_max":None,"spearman_std":None,"rmse_mean":None,"rmse_median":None,"rmse_max":None,"rmse_std":None}
            spearman = pd.to_numeric(frame["spearman"], errors="coerce").to_numpy(dtype=float)
            rmse = pd.to_numeric(frame["rmse"], errors="coerce").to_numpy(dtype=float)
            spearman = spearman[np.isfinite(spearman)]; rmse = rmse[np.isfinite(rmse)]
            return {
                "count":int(spearman.size),
                "spearman_mean":float(np.mean(spearman)) if spearman.size else None,
                "spearman_median":float(np.median(spearman)) if spearman.size else None,
                "spearman_min":float(np.min(spearman)) if spearman.size else None,
                "spearman_max":float(np.max(spearman)) if spearman.size else None,
                "spearman_std":float(np.std(spearman)) if spearman.size else None,
                "rmse_mean":float(np.mean(rmse)) if rmse.size else None,
                "rmse_median":float(np.median(rmse)) if rmse.size else None,
                "rmse_max":float(np.max(rmse)) if rmse.size else None,
                "rmse_std":float(np.std(rmse)) if rmse.size else None,
            }
        _e10_models.summarize_metric_frame = _e10_summarize_metric_frame
    def install(name: str, source: str) -> object:
'''
    if runtime.count(old) != 1:
        raise SystemExit("E10 v2 compatibility shim anchor changed")
    runtime = runtime.replace(old, new, 1)
    compile(runtime, "generated_e10_runtime_v2.py", "exec")

    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(out), "--self-test"], check=True)
    print(
        "CMI_FLU_E10_PREPARE_V2_PASS "
        f"science_commit={base.SCIENCE_COMMIT} e10_blob={base.E10_BLOB} "
        f"runtime_sha256={base.sha256(runtime.encode())} "
        "models_compatibility=summarize_metric_frame_only"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
