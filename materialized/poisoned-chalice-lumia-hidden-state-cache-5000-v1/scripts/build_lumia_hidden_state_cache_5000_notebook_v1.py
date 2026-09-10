"""Build the frozen P1-03a 5,000-row LUMIA hidden-state cache Notebook.

The proven 1k builder is treated as an immutable template.  This builder verifies
its exact Git blob identity and applies only the predeclared scale/storage changes:
5,000 rows, 500 rows per language x label cell, 250-row shards (20 total), a
fresh target/output path, and a longer child timeout.  Model/extraction semantics
and dependency pins remain byte-derived from the proven builder.
"""
from __future__ import annotations

from hashlib import sha1
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "scripts/build_lumia_hidden_state_cache_1000_notebook_v1.py"
EXPECTED_BASE_BLOB = "6ff76a25f235fb90ce9548c41ef23ba85593c787"


def git_blob_sha(data: bytes) -> str:
    return sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


raw = BASE.read_bytes()
if git_blob_sha(raw) != EXPECTED_BASE_BLOB:
    raise RuntimeError("proven 1k cache builder blob identity changed")
source = raw.decode("utf-8")

# Mechanical target/name expansion first.
source = source.replace("1,000", "5,000")
source = source.replace("1000", "5000")

# The cache core remains unchanged; override only this experiment's frozen
# cohort/storage configuration at instantiation time.
old_config = "CONFIG = LumiaHiddenStateCacheConfig()"
new_config = (
    "CONFIG = LumiaHiddenStateCacheConfig("
    "rows=5000, rows_per_language_label=500, shard_size=250, "
    "output_dir=\"/kaggle/working/lumia_hidden_state_cache_5000_v1\")"
)
if source.count(old_config) != 1:
    raise RuntimeError("base builder CONFIG instantiation changed")
source = source.replace(old_config, new_config)

# Remaining row-cell metadata is not covered by the 1000->5000 replacement.
old_manifest_cell = '"rows_per_language_label": 100,'
if source.count(old_manifest_cell) != 1:
    raise RuntimeError("base builder rows_per_language_label manifest marker changed")
source = source.replace(old_manifest_cell, '"rows_per_language_label": 500,')
old_shard_size = '"shard_size": 50,'
if source.count(old_shard_size) != 1:
    raise RuntimeError("base builder shard_size manifest marker changed")
source = source.replace(old_shard_size, '"shard_size": 250,')

# Five times the observed scoring load needs a larger fail-closed child timeout.
if source.count("timeout=3300") != 1:
    raise RuntimeError("base builder child timeout marker changed")
source = source.replace("timeout=3300", "timeout=9000")

# The generated runner still expects twenty shards because the storage-only
# shard size was predeclared as 250 rows.
required = [
    'TARGET = "renta0426/poisoned-chalice-lumia-hidden-state-cache-5000-v1"',
    'rows=5000, rows_per_language_label=500, shard_size=250',
    'output_dir="/kaggle/working/lumia_hidden_state_cache_5000_v1"',
    'assert cache["rows"] == 5000',
    'assert len(cache["shards"]) == 20',
    '"rows_per_language_label": 500',
    '"shard_size": 250',
    'range(20)',
    'timeout=9000',
    'LUMIA_HIDDEN_STATE_CACHE_5000_V1 COMPLETE rows=5000 shards=20',
]
for marker in required:
    if marker not in source:
        raise RuntimeError(f"5k transformed builder marker missing: {marker}")
for forbidden in (
    'TARGET = "renta0426/poisoned-chalice-lumia-hidden-state-cache-1000-v1"',
    'LUMIA_HIDDEN_STATE_CACHE_1000_V1 COMPLETE',
    'rows=1000, rows_per_language_label=100',
):
    if forbidden in source:
        raise RuntimeError(f"legacy executable marker survived 5k transform: {forbidden}")

namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "__main__"}
exec(compile(source, str(Path(__file__).resolve()), "exec"), namespace)
