"""CI-only import stubs for scientific third-party packages.

This module is loaded only by credential-free/protected static smoke processes via
an explicit PYTHONPATH. It lets those processes resolve cmi_flu's local import
closure without installing NumPy/pandas/SciPy/scikit-learn. Kaggle execution does
not include this path and therefore uses the real Kaggle scientific stack.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import sys
import types

_PREFIXES = ("numpy", "pandas", "scipy", "sklearn")


class _DummyMeta(type):
    def __getattr__(cls, name):
        return cls

    def __iter__(cls):
        return iter(())


class _Dummy(metaclass=_DummyMeta):
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return _Dummy

    def __call__(self, *args, **kwargs):
        return _Dummy()

    def __iter__(self):
        return iter(())

    def __or__(self, other):
        return self

    def __ror__(self, other):
        return self


class _Loader(importlib.abc.Loader):
    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__path__ = []
        module.__all__ = []
        module.__getattr__ = lambda name: _Dummy
        return module

    def exec_module(self, module):
        return None


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == _PREFIXES or any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in _PREFIXES):
            return importlib.util.spec_from_loader(fullname, _Loader(), is_package=True)
        return None


# Append after the standard finders so a real installed package wins. On the
# GitHub runner these scientific packages are absent; on accidental use in a
# richer environment this file therefore does not shadow them.
sys.meta_path.append(_Finder())
