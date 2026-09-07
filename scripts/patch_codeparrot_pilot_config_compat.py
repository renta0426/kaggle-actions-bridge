#!/usr/bin/env python3
"""Patch the composed CodeParrot pilot for the frozen private scorer snapshot.

The private `poisoned-chalice-minkpp-paper-10k-v1` notebook embeds an older
`FeatureCacheV2Config` / `_chunk_statistics` API than current research main.
This patch is public orchestration only: it never reads GitHub credentials or
private source outside the already composed runner-local notebook.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import textwrap

import nbformat


COMPAT_MARKER = "CODEPARROT_PRIVATE_SNAPSHOT_COMPAT_V1"


def _source(cell: nbformat.NotebookNode) -> str:
    value = cell.get("source", "")
    return value if isinstance(value, str) else "".join(value)


def _compat_cell() -> str:
    return textwrap.dedent(
        r'''
        # CODEPARROT_PRIVATE_SNAPSHOT_COMPAT_V1
        # The frozen private scorer snapshot predates rank-vocabulary blocking in
        # FeatureCacheV2Config/_chunk_statistics.  Preserve the exact formulas while
        # accepting both old and new private snapshots.
        import inspect as _codeparrot_inspect

        _CODEPARROT_ORIGINAL_CHUNK_STATISTICS = _chunk_statistics
        _CODEPARROT_CHUNK_PARAMETERS = tuple(
            _codeparrot_inspect.signature(_CODEPARROT_ORIGINAL_CHUNK_STATISTICS).parameters
        )

        def _codeparrot_strict_rank_blocked(chunk, correct, block_size):
            rank = torch.ones_like(correct, dtype=torch.int64)
            threshold = correct.unsqueeze(-1)
            for start in range(0, chunk.shape[-1], block_size):
                stop = min(start + block_size, chunk.shape[-1])
                rank += (chunk[..., start:stop] > threshold).sum(dim=-1)
            return rank

        def _chunk_statistics(logits, targets, rank_vocab_block_size=8192):
            if "rank_vocab_block_size" in _CODEPARROT_CHUNK_PARAMETERS:
                return _CODEPARROT_ORIGINAL_CHUNK_STATISTICS(
                    logits, targets, rank_vocab_block_size
                )
            chunk = logits.float()
            correct = chunk.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            log_norm = torch.logsumexp(chunk, dim=-1)
            logp = correct - log_norm
            z = (
                (correct - chunk.mean(dim=-1))
                / chunk.std(dim=-1, correction=0).clamp_min(1e-6)
            )
            rank = _codeparrot_strict_rank_blocked(
                chunk, correct, rank_vocab_block_size
            )
            top2 = chunk.topk(k=2, dim=-1).values
            best_other = torch.where(
                top2[..., 0] == correct, top2[..., 1], top2[..., 0]
            )
            margin = correct - best_other
            probabilities = torch.softmax(chunk, dim=-1)
            entropy = log_norm - (probabilities * chunk).sum(dim=-1)
            return torch.stack(
                (logp, z, rank.float(), margin, entropy), dim=-1
            )
        '''
    ).strip()


def patch_notebook(root: Path) -> Path:
    candidates = sorted(root.glob("*.ipynb"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one composed notebook, found {len(candidates)}"
        )
    path = candidates[0]
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)

    code_sources = [_source(cell) for cell in notebook.cells if cell.cell_type == "code"]
    if any(COMPAT_MARKER in source for source in code_sources):
        raise RuntimeError("compatibility patch already present")

    v2_indexes = [
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        and "class FeatureCacheV2Config" in _source(cell)
        and "def _chunk_statistics" in _source(cell)
    ]
    if len(v2_indexes) != 1:
        raise RuntimeError(
            f"expected one private FeatureCacheV2 source cell, found {len(v2_indexes)}"
        )

    setup_indexes = [
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        and "CONFIG = FeatureCacheV2Config(" in _source(cell)
        and "rank_vocab_block_size=8192," in _source(cell)
        and "SCALE_FRACTIONS = (0.10,)" in _source(cell)
    ]
    if len(setup_indexes) != 1:
        raise RuntimeError(
            f"expected one CodeParrot setup cell needing compatibility, found {len(setup_indexes)}"
        )

    setup_index = setup_indexes[0]
    setup = _source(notebook.cells[setup_index])
    setup = setup.replace(
        "            rank_vocab_block_size=8192,\n",
        "",
        1,
    )
    anchor = "        SCALE_FRACTIONS = (0.10,)\n"
    replacement = (
        "        # Frozen private snapshot may not declare this dataclass field.\n"
        "        object.__setattr__(CONFIG, \"rank_vocab_block_size\", 8192)\n"
        + anchor
    )
    if anchor not in setup:
        raise RuntimeError("setup compatibility insertion anchor missing")
    setup = setup.replace(anchor, replacement, 1)
    notebook.cells[setup_index]["source"] = setup

    insert_at = v2_indexes[0] + 1
    notebook.cells.insert(insert_at, nbformat.v4.new_code_cell(_compat_cell()))

    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
            compile(_source(cell), str(path), "exec")

    combined = "\n".join(
        _source(cell) for cell in notebook.cells if cell.cell_type == "code"
    )
    if COMPAT_MARKER not in combined:
        raise RuntimeError("compatibility marker missing after patch")
    patched_setup = _source(notebook.cells[setup_index + (1 if insert_at <= setup_index else 0)])
    if "rank_vocab_block_size=8192," in patched_setup:
        raise RuntimeError("constructor-only incompatible rank argument remains")
    if 'object.__setattr__(CONFIG, "rank_vocab_block_size", 8192)' not in patched_setup:
        raise RuntimeError("frozen rank compatibility assignment missing")
    if "def _chunk_statistics(logits, targets, rank_vocab_block_size=8192):" not in combined:
        raise RuntimeError("chunk-statistics compatibility wrapper missing")

    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    return path


def self_test() -> None:
    with tempfile.TemporaryDirectory() as temp_value:
        root = Path(temp_value)
        notebook = nbformat.v4.new_notebook()
        notebook.cells = [
            nbformat.v4.new_code_cell(
                "class FeatureCacheV2Config: pass\n"
                "def _chunk_statistics(logits, targets): return None\n"
            ),
            nbformat.v4.new_code_cell(
                "CONFIG = FeatureCacheV2Config(\n"
                "            output_dir='x',\n"
                "            rank_vocab_block_size=8192,\n"
                ")\n"
                "        SCALE_FRACTIONS = (0.10,)\n"
            ),
        ]
        # The synthetic setup must be valid Python for the post-patch compile gate.
        notebook.cells[1].source = textwrap.dedent(
            '''
            CONFIG = FeatureCacheV2Config(
                output_dir='x',
                rank_vocab_block_size=8192,
            )
            SCALE_FRACTIONS = (0.10,)
            '''
        )
        nbformat.write(notebook, root / "pilot.ipynb")
        patch_notebook(root)
        patched = nbformat.read(root / "pilot.ipynb", as_version=4)
        combined = "\n".join(
            _source(cell) for cell in patched.cells if cell.cell_type == "code"
        )
        assert COMPAT_MARKER in combined
        assert 'object.__setattr__(CONFIG, "rank_vocab_block_size", 8192)' in combined
        assert "def _chunk_statistics(logits, targets, rank_vocab_block_size=8192):" in combined
    print("CODEPARROT_CONFIG_COMPAT_SELF_TEST PASS old_private_api_supported=1")


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
    patched = patch_notebook(args.notebook_dir)
    print(
        f"CODEPARROT_CONFIG_COMPAT_PATCH PASS notebook={patched.name} "
        "old_private_api_supported=1 formulas_changed=0"
    )


if __name__ == "__main__":
    main()
