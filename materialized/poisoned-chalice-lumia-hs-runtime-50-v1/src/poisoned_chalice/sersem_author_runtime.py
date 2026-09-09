"""Runtime adapter for the released SERSEM structural weighting helpers.

This module reproduces the dependency/runtime behavior of the public SERSEM
replication package at commit 413f56040e5b4805bcf15ed794dec56bc4e16b41
without importing that repository at runtime.  The pure weighting rules live in
``sersem_author_faithful.py``; this adapter supplies the released split
Tree-sitter parsers, ``en_US`` PyEnchant dictionary, and unfiltered flake8 call.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Any, Callable
import shutil
import subprocess
import tempfile

from poisoned_chalice.sersem_author_faithful import parse_flake8_errors


AUTHOR_RUNTIME_PINS = {
    "transformers": "4.52.0",
    "tokenizers": "0.21.0",
    "tree-sitter": "0.25.2",
    "tree-sitter-go": "0.25.0",
    "tree-sitter-java": "0.23.5",
    "tree-sitter-python": "0.25.0",
    "tree-sitter-ruby": "0.23.1",
    "tree-sitter-rust": "0.24.0",
    "pyenchant": "3.3.0",
    "flake8": "7.3.0",
}
LANGUAGES = ("Go", "Java", "Python", "Ruby", "Rust")


@dataclass(frozen=True)
class AuthorRuntime:
    parsers: dict[str, Any | None]
    word_is_known: Callable[[str], bool] | None
    ast_available: bool
    enchant_available: bool
    flake8_available: bool
    installed_versions: dict[str, str | None]
    version_mismatches: dict[str, dict[str, str | None]]
    ast_backend: str
    enchant_backend: str
    flake8_backend: str

    def flake8_errors(self, text: str, language: str) -> list[tuple[int, int]] | None:
        """Return released flake8 diagnostics for Python; None for other languages."""
        if language != "Python":
            return None
        return run_flake8_errors(text)


def installed_author_runtime_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in AUTHOR_RUNTIME_PINS:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def author_runtime_version_mismatches(
    versions: dict[str, str | None] | None = None,
) -> dict[str, dict[str, str | None]]:
    observed = installed_author_runtime_versions() if versions is None else versions
    mismatches: dict[str, dict[str, str | None]] = {}
    for distribution, expected in AUTHOR_RUNTIME_PINS.items():
        actual = observed.get(distribution)
        if actual != expected:
            mismatches[distribution] = {"expected": expected, "actual": actual}
    return mismatches


def _load_released_parsers() -> tuple[dict[str, Any | None], bool, str]:
    """Mirror ASTExtractor.py's all-or-nothing import boundary."""
    try:
        import tree_sitter
        import tree_sitter_go as tsgo
        import tree_sitter_java as tsjava
        import tree_sitter_python as tspython
        import tree_sitter_ruby as tsruby
        import tree_sitter_rust as tsrust
    except ImportError:
        return (
            {language: None for language in LANGUAGES},
            False,
            "regex_fallback_missing_split_tree_sitter_runtime",
        )

    modules = {
        "Go": tsgo,
        "Java": tsjava,
        "Python": tspython,
        "Ruby": tsruby,
        "Rust": tsrust,
    }

    # The released code catches ImportError around the import block but does not
    # suppress parser-construction failures. Preserve that behavior here.
    parsers: dict[str, Any | None] = {}
    for language, module in modules.items():
        tree_language = tree_sitter.Language(module.language())
        parsers[language] = tree_sitter.Parser(tree_language)
    return parsers, True, "released_split_tree_sitter_grammars"


def _load_released_dictionary() -> tuple[Callable[[str], bool] | None, bool, str]:
    """Mirror ASTExtractor.py's broad PyEnchant fallback."""
    try:
        import enchant

        dictionary = enchant.Dict("en_US")
    except Exception:
        return None, False, "disabled_pyenchant_en_us_unavailable"

    def word_is_known(word: str) -> bool:
        return bool(dictionary.check(word))

    return word_is_known, True, "pyenchant_en_US"


def run_flake8_errors(
    text: str,
    *,
    executable: str | None = None,
) -> list[tuple[int, int]]:
    """Run the released unfiltered ``flake8 <tmp.py>`` command.

    Any execution failure returns an empty list, exactly like the released
    ``LinterExtractor.get_flake8_errors`` helper. Empty diagnostics therefore
    trigger the generic formatting fallback in ``build_author_character_weights``.
    """
    binary = executable if executable is not None else shutil.which("flake8")
    if not binary:
        return []
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as handle:
            handle.write(text)
            handle.flush()
            result = subprocess.run(
                [binary, handle.name],
                capture_output=True,
                text=True,
            )
    except Exception:
        return []
    return parse_flake8_errors(result.stdout)


def load_author_runtime(
    *,
    require_exact_versions: bool = False,
    require_full_components: bool = False,
) -> AuthorRuntime:
    versions = installed_author_runtime_versions()
    mismatches = author_runtime_version_mismatches(versions)
    if require_exact_versions and mismatches:
        raise RuntimeError(f"SERSEM author runtime version mismatch: {mismatches}")

    parsers, ast_available, ast_backend = _load_released_parsers()
    word_is_known, enchant_available, enchant_backend = _load_released_dictionary()
    flake8_available = shutil.which("flake8") is not None
    flake8_backend = (
        "released_unfiltered_flake8"
        if flake8_available
        else "generic_fallback_no_flake8"
    )

    if require_full_components:
        missing = []
        if not ast_available:
            missing.append("split_tree_sitter")
        if not enchant_available:
            missing.append("pyenchant_en_US")
        if not flake8_available:
            missing.append("flake8")
        if missing:
            raise RuntimeError(f"SERSEM author runtime components unavailable: {missing}")

    return AuthorRuntime(
        parsers=parsers,
        word_is_known=word_is_known,
        ast_available=ast_available,
        enchant_available=enchant_available,
        flake8_available=flake8_available,
        installed_versions=versions,
        version_mismatches=mismatches,
        ast_backend=ast_backend,
        enchant_backend=enchant_backend,
        flake8_backend=flake8_backend,
    )
