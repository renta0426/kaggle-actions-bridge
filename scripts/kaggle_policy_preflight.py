#!/usr/bin/env python3
"""Credential-free availability check for the fixed Kaggle policy pages.

This module intentionally owns the canonical public-policy URLs so individual
experiment workflows do not duplicate or guess them. Competition-specific
Rules/Evaluation/Data are checked separately through the authenticated Kaggle
API immediately before a write.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import time
import urllib.request
from urllib.parse import urlparse

POLICY_PAGES = (
    ("terms", "https://www.kaggle.com/terms"),
    ("aup", "https://www.kaggle.com/aup"),
    ("community_guidelines", "https://www.kaggle.com/community-guidelines"),
    ("api_docs", "https://www.kaggle.com/docs/api"),
    ("notebook_docs", "https://www.kaggle.com/docs/notebooks"),
)
USER_AGENT = "kaggle-actions-bridge/1"
MAX_BYTES = 2_000_000
MIN_BYTES = 256


def _fetch(name: str, url: str, attempts: int = 3) -> dict[str, object]:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "www.kaggle.com":
        raise RuntimeError(f"unexpected policy origin for {name}")
    context = ssl.create_default_context()
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=30, context=context) as response:
                final = urlparse(response.geturl())
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}")
                if final.scheme != "https" or final.hostname not in {"www.kaggle.com", "kaggle.com"}:
                    raise RuntimeError("unexpected redirect origin")
                data = response.read(MAX_BYTES + 1)
            if not MIN_BYTES <= len(data) <= MAX_BYTES:
                raise RuntimeError(f"unexpected byte count: {len(data)}")
            return {
                "name": name,
                "url": url,
                "final_path": final.path,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        except Exception as exc:  # bounded read-only retry only
            last = exc
            if attempt + 1 < attempts:
                time.sleep(attempt + 1)
    raise RuntimeError(f"official Kaggle policy page unavailable: {name}: {last}")


def run() -> list[dict[str, object]]:
    return [_fetch(name, url) for name, url in POLICY_PAGES]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run()
    if args.json:
        print(json.dumps({"status": "pass", "pages": result}, sort_keys=True))
    else:
        compact = {item["name"]: {"bytes": item["bytes"], "sha256": item["sha256"]} for item in result}
        print("KAGGLE_POLICY_PREFLIGHT PASS " + json.dumps(compact, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
