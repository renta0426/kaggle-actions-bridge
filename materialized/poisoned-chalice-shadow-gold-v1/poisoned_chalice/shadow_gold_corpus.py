"""Deterministic five-language code-like corpus for the Gold shadow benchmark.

The corpus is generated locally from a frozen seed and contains no competition
rows, external repository contents, pre-existing membership labels, sample IDs,
or provenance-derived target information.  It exists specifically to make the
architecture-only shadow experiment fully controlled by construction.
"""

from __future__ import annotations

import hashlib
from typing import Callable

import pandas as pd


SHADOW_GOLD_CORPUS_VERSION = "shadow-gold-code-like-corpus-v1"
SHADOW_GOLD_LANGUAGES = ("Go", "Java", "Python", "Ruby", "Rust")


def _digest_words(seed: int, language: str, family: int, variant: int) -> list[int]:
    payload = f"{seed}\0{language}\0{family}\0{variant}".encode("utf-8")
    raw = hashlib.blake2b(payload, digest_size=32).digest()
    return [int.from_bytes(raw[index : index + 4], "big") for index in range(0, 32, 4)]


def _names(family: int, variant: int) -> tuple[str, str, str, str]:
    suffix = f"{family:04d}_{variant}"
    return (
        f"fold_{suffix}",
        f"state_{suffix}",
        f"value_{suffix}",
        f"limit_{suffix}",
    )


def _python(family: int, variant: int, words: list[int]) -> str:
    fn, state, value, limit = _names(family, variant)
    a, b, c, d = [word % 97 + 3 for word in words[:4]]
    return f'''def {fn}(items):
    {state} = {a}
    {limit} = {b + 8}
    for index, {value} in enumerate(items):
        mixed = ({value} * {c} + index + {d}) % ({limit} + {a})
        if mixed % 3 == {variant % 3}:
            {state} ^= mixed + {b}
        elif mixed & 1:
            {state} += mixed * {c}
        else:
            {state} -= mixed // {max(2, d % 11)}
    tail = [{state} + offset * {a} for offset in range({4 + variant})]
    return sum(tail) ^ ({state} << 1)
'''


def _rust(family: int, variant: int, words: list[int]) -> str:
    fn, state, value, limit = _names(family, variant)
    a, b, c, d = [word % 89 + 5 for word in words[:4]]
    return f'''fn {fn}(items: &[i64]) -> i64 {{
    let mut {state}: i64 = {a};
    let {limit}: i64 = {b + 9};
    for (index, {value}) in items.iter().copied().enumerate() {{
        let mixed = ({value} * {c} + index as i64 + {d}) % ({limit} + {a});
        if mixed % 3 == {variant % 3} {{
            {state} ^= mixed + {b};
        }} else if mixed & 1 == 1 {{
            {state} += mixed * {c};
        }} else {{
            {state} -= mixed / {max(2, d % 11)};
        }}
    }}
    (0..{4 + variant}).map(|offset| {state} + offset * {a}).sum::<i64>() ^ ({state} << 1)
}}
'''


def _go(family: int, variant: int, words: list[int]) -> str:
    fn, state, value, limit = _names(family, variant)
    a, b, c, d = [word % 83 + 7 for word in words[:4]]
    return f'''func {fn}(items []int64) int64 {{
    var {state} int64 = {a}
    const {limit} int64 = {b + 11}
    for index, {value} := range items {{
        mixed := ({value}*{c} + int64(index) + {d}) % ({limit} + {a})
        if mixed%3 == {variant % 3} {{
            {state} ^= mixed + {b}
        }} else if mixed&1 == 1 {{
            {state} += mixed * {c}
        }} else {{
            {state} -= mixed / {max(2, d % 11)}
        }}
    }}
    total := int64(0)
    for offset := int64(0); offset < {4 + variant}; offset++ {{
        total += {state} + offset*{a}
    }}
    return total ^ ({state} << 1)
}}
'''


def _java(family: int, variant: int, words: list[int]) -> str:
    fn, state, value, limit = _names(family, variant)
    a, b, c, d = [word % 79 + 11 for word in words[:4]]
    return f'''static long {fn}(long[] items) {{
    long {state} = {a}L;
    final long {limit} = {b + 13}L;
    for (int index = 0; index < items.length; index++) {{
        long {value} = items[index];
        long mixed = ({value} * {c}L + index + {d}L) % ({limit} + {a}L);
        if (mixed % 3L == {variant % 3}L) {{
            {state} ^= mixed + {b}L;
        }} else if ((mixed & 1L) == 1L) {{
            {state} += mixed * {c}L;
        }} else {{
            {state} -= mixed / {max(2, d % 11)}L;
        }}
    }}
    long total = 0L;
    for (int offset = 0; offset < {4 + variant}; offset++) {{
        total += {state} + offset * {a}L;
    }}
    return total ^ ({state} << 1);
}}
'''


def _ruby(family: int, variant: int, words: list[int]) -> str:
    fn, state, value, limit = _names(family, variant)
    a, b, c, d = [word % 73 + 13 for word in words[:4]]
    return f'''def {fn}(items)
  {state} = {a}
  {limit} = {b + 15}
  items.each_with_index do |{value}, index|
    mixed = ({value} * {c} + index + {d}) % ({limit} + {a})
    if mixed % 3 == {variant % 3}
      {state} ^= mixed + {b}
    elsif mixed.odd?
      {state} += mixed * {c}
    else
      {state} -= mixed / {max(2, d % 11)}
    end
  end
  tail = (0...{4 + variant}).map {{ |offset| {state} + offset * {a} }}
  tail.sum ^ ({state} << 1)
end
'''


_RENDERERS: dict[str, Callable[[int, int, list[int]], str]] = {
    "Go": _go,
    "Java": _java,
    "Python": _python,
    "Ruby": _ruby,
    "Rust": _rust,
}


def build_shadow_gold_corpus(
    *,
    rows_per_language: int = 2_048,
    family_size: int = 4,
    seed: int = 2027,
) -> pd.DataFrame:
    """Return the exact unlabeled source frame used by the Gold benchmark."""

    if rows_per_language <= 0:
        raise ValueError("rows_per_language must be positive")
    if family_size <= 0 or rows_per_language % family_size:
        raise ValueError("family_size must be positive and divide rows_per_language")
    rows: list[dict[str, object]] = []
    families = rows_per_language // family_size
    for language in SHADOW_GOLD_LANGUAGES:
        renderer = _RENDERERS[language]
        for family in range(families):
            family_id = f"gold-{language.casefold()}-{family:04d}"
            for variant in range(family_size):
                words = _digest_words(seed, language, family, variant)
                content = renderer(family, variant, words)
                rows.append(
                    {
                        "content": content,
                        "language": language,
                        "synthetic_family_id": family_id,
                    }
                )
    frame = pd.DataFrame(rows, columns=["content", "language", "synthetic_family_id"])
    if len(frame) != rows_per_language * len(SHADOW_GOLD_LANGUAGES):
        raise RuntimeError("Gold corpus row count invariant failed")
    if frame.content.duplicated().any():
        raise RuntimeError("Gold corpus contains duplicate content")
    if frame.synthetic_family_id.isna().any():
        raise RuntimeError("Gold corpus family ID invariant failed")
    return frame


def shadow_gold_corpus_sha256(frame: pd.DataFrame) -> str:
    required = ["content", "language", "synthetic_family_id"]
    if list(frame.columns) != required:
        raise ValueError(f"Gold corpus columns must equal {required}")
    digest = hashlib.sha256()
    for row in frame.itertuples(index=False, name=None):
        for value in row:
            digest.update(str(value).encode("utf-8"))
            digest.update(b"\0")
        digest.update(b"\n")
    return digest.hexdigest()


__all__ = [
    "SHADOW_GOLD_CORPUS_VERSION",
    "SHADOW_GOLD_LANGUAGES",
    "build_shadow_gold_corpus",
    "shadow_gold_corpus_sha256",
]
