"""Competition-Data-free fixture for the E04b material bridge."""
from __future__ import annotations

from types import SimpleNamespace

from .strategy_e04_synthetic import make_e04_tables


def make_e04b_tables():
    tables = make_e04_tables()
    # Recreate the E08 ontology caveat on every Challenge flow population.
    # E04b must change only named strict ASC rows back to PBMC; non-ASC rows
    # deliberately remain PBMCs so no auxiliary measurement view is added.
    challenge = tables["challenge_flow"].copy(deep=True)
    challenge["material"] = "PBMCs"
    tables["challenge_flow"] = challenge
    return tables


def run_synthetic(config):
    from .strategy_e04b import run_e04b

    tables = make_e04b_tables()
    result = run_e04b(config, SimpleNamespace(tables=tables), expected_counts=None)
    return result, tables
