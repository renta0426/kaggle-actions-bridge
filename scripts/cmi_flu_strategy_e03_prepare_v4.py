#!/usr/bin/env python3
"""E03 preparer repair: YAML-aware relay validation plus proven bridge B2.1 package pin."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import yaml

BASE = Path(__file__).with_name("cmi_flu_strategy_e03_prepare_v2.py")
PROVEN_B21_PACKAGE_SHA256 = "48ff4ed2eadec8b059b8d37fb677af1f249b6486bccfbd55dca2274cfc6f3dc3"


def load_base():
    spec = importlib.util.spec_from_file_location("cmi_flu_strategy_e03_prepare_v2_proven_b21", BASE)
    if spec is None or spec.loader is None:
        raise SystemExit("unable to import E03 v2 preparer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_base()
    module.FROZEN_B21_PACKAGE_SHA256 = PROVEN_B21_PACKAGE_SHA256

    def require_blob(data: bytes, expected: str, label: str) -> str:
        found = hashlib.sha1(
            b"blob " + str(len(data)).encode("ascii") + b"\0" + data
        ).hexdigest()
        if found != expected:
            raise SystemExit(
                f"{label} relay blob mismatch: expected={expected} found={found}"
            )
        text = data.decode("utf-8")
        if label == "baseline_b021_robust.yaml":
            parsed = yaml.safe_load(text)
            if not isinstance(parsed, dict):
                raise SystemExit("E03 relayed config is not a YAML mapping")
            if parsed.get("baseline") != "b021_taskwise_robust":
                raise SystemExit("E03 relayed config baseline changed")
            if (parsed.get("selection") or {}).get("policy") != "robust_v1":
                raise SystemExit("E03 relayed config selection policy changed")
        else:
            compile(text, label, "exec")
        return text

    module.require_blob = require_blob
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
