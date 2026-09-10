"""SERSEM released-helper fidelity helpers.

Pinned author source: Serdark4ra/SERSEM-Poisoned-Chalice-Competition-2026
commit 413f56040e5b4805bcf15ed794dec56bc4e16b41.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any
import re
import numpy as np

W_BOILERPLATE = 0.1
W_STANDARD = 1.0
W_IDENTIFIER_LONG = 3.0
W_STRING = 5.0
W_FORMATTING_ERROR = 5.0
W_COMMENT = 10.0
W_MULTILINGUAL = 10.0
W_PSYCHOLOGICAL = 10.0

REGEX_COMMENT_FB = re.compile(r"(//[^\n]*|#[^\n]*|/\*.*?\*/)", re.DOTALL)
REGEX_STRING_FB = re.compile(r"(\".*?\"|'.*?')", re.DOTALL)
REGEX_IDENTIFIER_FB = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")
REGEX_PSYCH = re.compile(r"(TODO|FIXME|HACK|Note to self)", re.IGNORECASE)
ERRATIC_SPACING = re.compile(r"([^ ]  +[^ ])")
MIXED_INDENT = re.compile(r"^(\t+ +| +\t+)", re.MULTILINE)

@dataclass(frozen=True)
class AuthorWeightDiagnostics:
    ast_backend: str
    ast_success: bool
    enchant_available: bool
    linter_backend: str
    lint_error_count: int
    psychological_matches: int

def split_identifier(ident: str) -> list[str]:
    words: list[str] = []
    for part in ident.split("_"):
        subparts = re.findall(r"[A-Za-z][a-z0-9]*|[A-Z]+(?=[A-Z][a-z0-9]|\b)", part)
        if subparts:
            words.extend(subparts)
        elif part:
            words.append(part)
    return words

def _byte_to_char(source: bytes, byte_index: int) -> int:
    return len(source[:int(byte_index)].decode("utf-8", errors="replace"))

def regex_ast_weights(text: str) -> np.ndarray:
    weights = np.full(len(text), W_STANDARD, dtype=np.float64)
    for match in REGEX_COMMENT_FB.finditer(text):
        weights[match.start():match.end()] = W_COMMENT
    for match in REGEX_STRING_FB.finditer(text):
        if match.end() - match.start() > 15:
            weights[match.start():match.end()] = W_STRING
    for match in REGEX_IDENTIFIER_FB.finditer(text):
        if match.end() - match.start() > 10:
            view = weights[match.start():match.end()]
            view[view == W_STANDARD] = W_IDENTIFIER_LONG
    return weights

def ast_character_weights(text: str, *, parser: Any | None, word_is_known: Callable[[str], bool] | None = None) -> tuple[np.ndarray, bool]:
    if parser is None:
        return regex_ast_weights(text), False
    weights = np.full(len(text), W_STANDARD, dtype=np.float64)
    source = text.encode("utf-8")
    try:
        tree = parser.parse(source)
        def visit(node: Any) -> None:
            node_type = str(node.type)
            start = _byte_to_char(source, int(node.start_byte))
            stop = _byte_to_char(source, int(node.end_byte))
            if "comment" in node_type:
                weights[start:stop] = W_COMMENT
            elif "string" in node_type:
                if stop - start > 15:
                    weights[start:stop] = W_STRING
            elif "identifier" in node_type:
                ident = text[start:stop]
                multilingual = False
                if word_is_known is not None:
                    words = split_identifier(ident)
                    unknown_count = sum(1 for word in words if len(word) >= 4 and not word_is_known(word))
                    multilingual = unknown_count > 0 and len(words) > 0 and unknown_count / len(words) >= 0.5
                view = weights[start:stop]
                mask = view == W_STANDARD
                if multilingual:
                    view[mask] = W_MULTILINGUAL
                elif stop - start > 10:
                    view[mask] = W_IDENTIFIER_LONG
            for child in node.children:
                visit(child)
        visit(tree.root_node)
        return weights, True
    except Exception:
        return regex_ast_weights(text), False

def generic_formatting_errors(text: str) -> list[tuple[int, int]]:
    errors: list[tuple[int, int]] = []
    for index, line in enumerate(text.splitlines()):
        line_num = index + 1
        if MIXED_INDENT.match(line):
            match = MIXED_INDENT.search(line)
            assert match is not None
            errors.append((line_num, match.start() + 1))
        for match in ERRATIC_SPACING.finditer(line):
            if "//" not in line[:match.start()] and "#" not in line[:match.start()]:
                errors.append((line_num, match.start() + 2))
    return errors

def parse_flake8_errors(stdout: str) -> list[tuple[int, int]]:
    errors: list[tuple[int, int]] = []
    for line in stdout.splitlines():
        match = re.match(r"^.+?:(\d+):(\d+): (.+)", line)
        if match:
            errors.append((int(match.group(1)), int(match.group(2))))
    return errors

def choose_linter_errors(text: str, language: str, flake8_errors: Sequence[tuple[int, int]] | None = None) -> tuple[list[tuple[int, int]], str]:
    if language == "Python" and flake8_errors:
        return list(flake8_errors), "flake8"
    return generic_formatting_errors(text), "generic_fallback"

def apply_linter_and_psychological_weights(text: str, base_weights: Sequence[float], errors: Iterable[tuple[int, int]]) -> tuple[np.ndarray, int]:
    weights = np.asarray(base_weights, dtype=np.float64).copy()
    if weights.shape != (len(text),):
        raise ValueError("base_weights length must equal text length")
    psychological = 0
    for match in REGEX_PSYCH.finditer(text):
        psychological += 1
        weights[match.start():match.end()] = W_PSYCHOLOGICAL
    lines = text.splitlines(keepends=True)
    for line_num, col_num in errors:
        if line_num <= len(lines):
            offset = sum(len(lines[index]) for index in range(line_num - 1)) + (col_num - 1)
            start = max(0, offset - 2)
            stop = min(len(text), offset + 3)
            view = weights[start:stop]
            view[view < W_FORMATTING_ERROR] = W_FORMATTING_ERROR
    return weights, psychological

def build_author_character_weights(text: str, language: str, *, parser: Any | None, word_is_known: Callable[[str], bool] | None = None, flake8_errors: Sequence[tuple[int, int]] | None = None) -> tuple[np.ndarray, AuthorWeightDiagnostics]:
    ast_weights, ast_success = ast_character_weights(text, parser=parser, word_is_known=word_is_known)
    errors, linter_backend = choose_linter_errors(text, language, flake8_errors)
    weights, psychological = apply_linter_and_psychological_weights(text, ast_weights, errors)
    return weights, AuthorWeightDiagnostics("tree_sitter" if ast_success else "regex_fallback", ast_success, word_is_known is not None, linter_backend, len(errors), psychological)

def token_weights_from_character_weights(text: str, offset_mapping: Sequence[tuple[int, int]], character_weights: Sequence[float]) -> np.ndarray:
    char_weights = np.asarray(character_weights, dtype=np.float64)
    if char_weights.shape != (len(text),):
        raise ValueError("character_weights length must equal text length")
    output = np.full(len(offset_mapping), W_BOILERPLATE, dtype=np.float64)
    for index, (raw_start, raw_stop) in enumerate(offset_mapping):
        start, stop = int(raw_start), int(raw_stop)
        if start < 0 or stop < start or stop > len(text):
            raise ValueError(f"invalid token offsets {(start, stop)}")
        structural_weight = float(np.max(char_weights[start:stop])) if start != stop else W_STANDARD
        if structural_weight <= W_STANDARD:
            token_text = text[start:stop].strip()
            if len(token_text) > 2 and REGEX_IDENTIFIER_FB.match(token_text):
                output[index] = W_STANDARD
            else:
                output[index] = W_BOILERPLATE
        else:
            output[index] = structural_weight
    return output

def sigmoid_z(correct_z: Sequence[float]) -> np.ndarray:
    z = np.asarray(correct_z, dtype=np.float64)
    out = np.empty_like(z)
    nonnegative = z >= 0
    out[nonnegative] = 1.0 / (1.0 + np.exp(-z[nonnegative]))
    exp_z = np.exp(z[~nonnegative])
    out[~nonnegative] = exp_z / (1.0 + exp_z)
    return out

def weighted_output_score(correct_z: Sequence[float], token_weights: Sequence[float]) -> float:
    values = sigmoid_z(correct_z)
    weights = np.asarray(token_weights, dtype=np.float64)
    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError("correct_z and token_weights must be aligned one-dimensional arrays")
    if values.size == 0:
        return float("nan")
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError("SERSEM inputs must be finite")
    if (weights < 0).any() or float(weights.sum()) <= 0:
        raise ValueError("SERSEM weights must have positive total mass")
    return float(np.average(values, weights=weights))
