#!/usr/bin/env python3
"""E10 builder v2: add the one missing current-models helper to the frozen E05-v6 bundle."""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

BASE = "scripts/cmi_flu_strategy_e10_prepare.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def load_base(root: Path):
    path = root / BASE
    spec = importlib.util.spec_from_file_location("cmi_flu_e10_prepare_v1", path)
    if spec is None or spec.loader is None:
        raise SystemExit("E10 v1 builder unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIXED_LOADER = '''def load_e10_modules() -> tuple[object, object, object]:
    load_e05_module()
    # The frozen E05-v6 bundle predates summarize_metric_frame.  Exact current
    # rank_transfer/anchor_residual sources import it at module import time, so
    # provide the current helper verbatim as a compatibility surface.  This
    # affects only diagnostics in those modules; E10 itself never calls it.
    from cmi_flu import models as _e10_models
    if not hasattr(_e10_models, "summarize_metric_frame"):
        def _e10_summarize_metric_frame(frame):
            import numpy as _np
            import pandas as _pd
            if frame.empty:
                return {
                    "count": 0,
                    "spearman_mean": None,
                    "spearman_median": None,
                    "spearman_min": None,
                    "spearman_max": None,
                    "spearman_std": None,
                    "rmse_mean": None,
                    "rmse_median": None,
                    "rmse_max": None,
                    "rmse_std": None,
                }
            spearman = _pd.to_numeric(frame["spearman"], errors="coerce").to_numpy(dtype=float)
            rmse = _pd.to_numeric(frame["rmse"], errors="coerce").to_numpy(dtype=float)
            spearman = spearman[_np.isfinite(spearman)]
            rmse = rmse[_np.isfinite(rmse)]
            return {
                "count": int(spearman.size),
                "spearman_mean": float(_np.mean(spearman)) if spearman.size else None,
                "spearman_median": float(_np.median(spearman)) if spearman.size else None,
                "spearman_min": float(_np.min(spearman)) if spearman.size else None,
                "spearman_max": float(_np.max(spearman)) if spearman.size else None,
                "spearman_std": float(_np.std(spearman)) if spearman.size else None,
                "rmse_mean": float(_np.mean(rmse)) if rmse.size else None,
                "rmse_median": float(_np.median(rmse)) if rmse.size else None,
                "rmse_max": float(_np.max(rmse)) if rmse.size else None,
                "rmse_std": float(_np.std(rmse)) if rmse.size else None,
            }
        _e10_models.summarize_metric_frame = _e10_summarize_metric_frame
    def install(name: str, source: str) -> object:
        module = types.ModuleType(name)
        module.__file__ = f"<{name}>"; module.__package__ = "cmi_flu"; sys.modules[name] = module
        exec(compile(source, name.replace(".", "/") + ".py", "exec"), module.__dict__, module.__dict__)
        return module
    install("cmi_flu.rank_transfer", RANK_TRANSFER_SOURCE)
    install("cmi_flu.anchor_residual", ANCHOR_SOURCE)
    install("cmi_flu.study_similarity", STUDY_SIMILARITY_SOURCE)
    e10 = install("cmi_flu.strategy_e10", E10_SOURCE)
    synthetic = install("cmi_flu.strategy_e10_synthetic", E10_SYNTH_SOURCE)
    run = getattr(e10, "run_e10", None); run_synthetic = getattr(synthetic, "run_synthetic", None)
    if not callable(run) or not callable(run_synthetic):
        raise BridgeContractError("e10_entry_missing")
    return run, run_synthetic, e10'''


def main() -> int:
    args = parse_args()
    root = args.repository_root.expanduser().resolve()
    reference_dir = args.reference_dir.expanduser().resolve()
    out = args.output.expanduser().resolve()
    base = load_base(root)
    base.validate_request(root)
    e10, synth, anchor, rank_transfer, study_similarity = base.load_exact_sources(root)
    v3, parent_runtime = base.build_parent_runtime(root, reference_dir)
    runtime = base.patch_runtime(
        v3, parent_runtime, e10, synth, anchor, rank_transfer, study_similarity
    )
    runtime = v3.replace_top_level_function(runtime, "load_e10_modules", FIXED_LOADER)
    names = v3.top_level_functions(runtime)
    if names.count("load_e10_modules") != 1:
        raise SystemExit("E10 v2 loader binding mismatch")
    compile(runtime, "generated_e10_runtime_v2.py", "exec")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(runtime, encoding="utf-8")
    subprocess.run([sys.executable, str(out), "--self-test"], check=True)
    print(
        "CMI_FLU_E10_PREPARE_V2_PASS "
        f"science_commit={base.SCIENCE_COMMIT} e10_blob={base.E10_BLOB} "
        f"compatibility=summarize_metric_frame_only runtime_sha256={base.sha256(runtime.encode())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
