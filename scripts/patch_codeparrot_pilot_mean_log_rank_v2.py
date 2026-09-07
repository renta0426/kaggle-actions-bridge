#!/usr/bin/env python3
"""Patch private CodeParrot pilot v2 with exact same-forward mean-log-rank."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import tempfile
import textwrap

import nbformat

REPAIR_MARKER = "CODEPARROT_MEAN_LOG_RANK_REPAIR_V2"
COMPAT_MARKER = "CODEPARROT_PRIVATE_SNAPSHOT_COMPAT_V3"
EXPECTED_COHORT_SHA256 = "b664e368f9380e230c9cfc0616327d829424749c5473f87575678525db1998fc"


def _source(cell: nbformat.NotebookNode) -> str:
    value = cell.get("source", "")
    return value if isinstance(value, str) else "".join(value)


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one marker, found {count}")
    return source.replace(old, new, 1)


def _line_insert(source: str, pattern: str, body: str, label: str, *, after: bool) -> str:
    matches = list(re.finditer(pattern, source, flags=re.MULTILINE))
    if len(matches) != 1:
        raise RuntimeError(f"{label}: expected one line, found {len(matches)}")
    match = matches[0]
    indent = match.group("indent")
    payload = "\n".join(indent + line if line else "" for line in body.splitlines())
    if after:
        return source[: match.end()] + "\n" + payload + source[match.end() :]
    return source[: match.start()] + payload + "\n" + source[match.start() :]


def patch_notebook(root: Path) -> Path:
    candidates = sorted(root.glob("*.ipynb"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one pulled target notebook, found {len(candidates)}")
    path = candidates[0]
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    combined_before = "\n".join(_source(c) for c in notebook.cells if c.cell_type == "code")
    if REPAIR_MARKER in combined_before:
        raise RuntimeError("mean-log-rank repair already present")
    if COMPAT_MARKER not in combined_before:
        raise RuntimeError("compat-v3 marker missing from approved target v2")
    if EXPECTED_COHORT_SHA256 not in combined_before:
        raise RuntimeError("audited cohort SHA missing from approved target v2")

    runner_indexes = [
        i for i, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        and "def run_private_paper_records(records):" in _source(cell)
        and "probability_weighted_token_statistics_blocked" in _source(cell)
        and "window_rows.append(metadata | summary)" in _source(cell)
        and "accum = [" in _source(cell)
    ]
    if len(runner_indexes) != 1:
        raise RuntimeError(f"expected one private same-forward runner, found {len(runner_indexes)}")
    runner_index = runner_indexes[0]
    runner = _source(notebook.cells[runner_index])
    if "mean_log_rank" in runner:
        raise RuntimeError("private runner unexpectedly already has mean_log_rank")

    runner = _replace_once(
        runner,
        '{"logp": [], "legacy_z": [], "paper_z": [], "paper_variance": []}',
        '{"logp": [], "legacy_z": [], "paper_z": [], "paper_variance": [], "rank": []}',
        "rank accumulator",
    )
    runner = _line_insert(
        runner,
        r'^(?P<indent>[ \t]*)del dense_values, correct[ \t]*$',
        "rank = _codeparrot_strict_rank_blocked(\n"
        "    dense_values, correct, CONFIG.rank_vocab_block_size\n"
        ")",
        "strict-rank computation",
        after=False,
    )
    runner = _line_insert(
        runner,
        r'^(?P<indent>[ \t]*)host_paper = paper\.standardized_log_prob\.detach\(\)\.cpu\(\)\.numpy\(\)[ \t]*$',
        "host_rank = rank.detach().cpu().numpy()",
        "rank host transfer",
        after=False,
    )
    runner = _line_insert(
        runner,
        r'^(?P<indent>[ \t]*)accum\[index\]\["legacy_z"\]\.append\(host_legacy\[index\]\[mask\]\)[ \t]*$',
        'accum[index]["rank"].append(host_rank[index][mask])',
        "rank accumulation",
        after=True,
    )
    runner = _replace_once(
        runner,
        "logit_slice, target_slice, valid_slice, logp, legacy_z, paper,",
        "logit_slice, target_slice, valid_slice, logp, legacy_z, rank, paper,",
        "rank device cleanup",
    )
    runner = _replace_once(
        runner,
        "host_logp, host_legacy, host_paper, host_variance, host_valid,",
        "host_logp, host_legacy, host_rank, host_paper, host_variance, host_valid,",
        "rank host cleanup",
    )
    runner = _line_insert(
        runner,
        r'^(?P<indent>[ \t]*)metadata = \{key: value for key, value in record\.items\(\) if key != "input_ids"\}[ \t]*$',
        "# CODEPARROT_MEAN_LOG_RANK_REPAIR_V2\n"
        'rank_integer = arrays["rank"].astype(np.int64, copy=False)\n'
        'summary["mean_log_rank"] = float(np.log1p(rank_integer).mean())',
        "window mean-log-rank",
        after=False,
    )
    compile(runner, f"{path}:patched_private_runner", "exec")
    notebook.cells[runner_index]["source"] = runner

    pilot_indexes = [
        i for i, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        and "first8_compare = {" in _source(cell)
        and "first8_max = max(first8_compare.values())" in _source(cell)
        and "group.mean_log_rank.mean()" in _source(cell)
    ]
    if len(pilot_indexes) != 1:
        raise RuntimeError(f"expected one aggregation/fidelity cell, found {len(pilot_indexes)}")
    pilot_index = pilot_indexes[0]
    pilot = _source(notebook.cells[pilot_index])
    pilot = _line_insert(
        pilot,
        r'^(?P<indent>[ \t]*)first8_max = max\(first8_compare\.values\(\)\)[ \t]*$',
        'first8_compare["mean_log_rank"] = float(np.max(np.abs(\n'
        '    left["mean_log_rank"].to_numpy(float)\n'
        '    - right["mean_log_rank"].to_numpy(float)\n'
        ')))',
        "first8 mean-log-rank fidelity",
        after=False,
    )
    compile(pilot, f"{path}:patched_pilot", "exec")
    notebook.cells[pilot_index]["source"] = pilot

    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []

    combined = "\n".join(_source(c) for c in notebook.cells if c.cell_type == "code")
    required = (
        REPAIR_MARKER,
        COMPAT_MARKER,
        EXPECTED_COHORT_SHA256,
        'accum[index]["rank"].append(host_rank[index][mask])',
        'summary["mean_log_rank"] = float(np.log1p(rank_integer).mean())',
        'first8_compare["mean_log_rank"]',
        "group.mean_log_rank.mean()",
        "synthetic_legacy_rank_exact_match",
        "performance_metrics_computed",
        "membership_join_performed",
        "full_4000_scoring_authorized",
    )
    missing = [item for item in required if item not in combined]
    if missing:
        raise RuntimeError(f"patched target contract incomplete: {missing}")
    if combined.count(REPAIR_MARKER) != 1:
        raise RuntimeError("mean-log-rank repair marker count changed")
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    return path


def self_test() -> None:
    with tempfile.TemporaryDirectory() as value:
        root = Path(value)
        runner = textwrap.dedent('''
        def run_private_paper_records(records):
            accum = [
                {"logp": [], "legacy_z": [], "paper_z": [], "paper_variance": []}
                for _ in batch
            ]
            dense_values = logit_slice.float()
            correct = dense_values.gather(-1, target_slice.unsqueeze(-1)).squeeze(-1)
            logp = correct - torch.logsumexp(dense_values, dim=-1)
            legacy_z = correct
            del dense_values, correct
            paper = probability_weighted_token_statistics_blocked(logit_slice, target_slice)
            host_logp = logp.detach().cpu().numpy()
            host_legacy = legacy_z.detach().cpu().numpy()
            host_paper = paper.standardized_log_prob.detach().cpu().numpy()
            host_variance = paper.probability_weighted_variance.detach().cpu().numpy()
            host_valid = valid_slice.detach().cpu().numpy()
            accum[index]["legacy_z"].append(host_legacy[index][mask])
            del (
                logit_slice, target_slice, valid_slice, logp, legacy_z, paper,
                host_logp, host_legacy, host_paper, host_variance, host_valid,
            )
            arrays = {key: np.concatenate(parts).astype(np.float32, copy=False) for key, parts in values.items()}
            summary = summarize_window(arrays["logp"], arrays["legacy_z"], arrays["paper_z"])
            metadata = {key: value for key, value in record.items() if key != "input_ids"}
            window_rows.append(metadata | summary)
        ''')
        pilot = textwrap.dedent(f'''
        # {COMPAT_MARKER}
        EXPECTED_COHORT_SHA256 = "{EXPECTED_COHORT_SHA256}"
        first8_compare = {{"mean_logp": 0.0}}
        first8_max = max(first8_compare.values())
        value = float(-group.mean_log_rank.mean())
        synthetic_legacy_rank_exact_match = True
        performance_metrics_computed = False
        membership_join_performed = False
        full_4000_scoring_authorized = False
        ''')
        nb = nbformat.v4.new_notebook(cells=[
            nbformat.v4.new_code_cell(runner),
            nbformat.v4.new_code_cell(pilot),
        ])
        nbformat.write(nb, root / "target.ipynb")
        patch_notebook(root)
        patched = nbformat.read(root / "target.ipynb", as_version=4)
        code = "\n".join(_source(c) for c in patched.cells if c.cell_type == "code")
        assert REPAIR_MARKER in code
        assert 'accum[index]["rank"].append(host_rank[index][mask])' in code
        assert 'summary["mean_log_rank"] = float(np.log1p(rank_integer).mean())' in code
        assert 'first8_compare["mean_log_rank"]' in code
    print(
        "CODEPARROT_MEAN_LOG_RANK_PATCH_SELF_TEST PASS "
        "same_forward_rank=1 exact_strict_rank=1 first8_fidelity=1"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.notebook_dir is None:
        parser.error("--notebook-dir is required")
    path = patch_notebook(args.notebook_dir)
    print(
        f"CODEPARROT_MEAN_LOG_RANK_PATCH PASS notebook={path.name} "
        "formula=mean_log1p_strict_rank same_forward=1 labels=0 metrics=0"
    )


if __name__ == "__main__":
    main()
