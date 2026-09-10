#!/usr/bin/env python3
"""Fetch only the exact public historical bridge sources needed by P1-03 fresh cache."""
from __future__ import annotations

import argparse
from pathlib import Path
import ssl
import urllib.request

COMMIT = "ac70f67bda548c8d0e2f29fe25f430cae9b2ff54"
ROOT = "materialized/poisoned-chalice-lumia-hidden-state-cache-5000-v1"
FILES = {
    "scripts/build_lumia_hidden_state_cache_1000_notebook_v1.py": 131072,
    "scripts/build_lumia_hidden_state_cache_5000_notebook_v1.py": 32768,
    "configs/p1_03_lumia_hidden_state_scale_5000_v1_20260910.json": 32768,
    "src/poisoned_chalice/sersem_author_faithful.py": 65536,
    "src/poisoned_chalice/sersem_author_runtime.py": 65536,
    "src/poisoned_chalice/lumia_hidden_state_pilot.py": 131072,
    "src/poisoned_chalice/lumia_hidden_state_cache.py": 131072,
}


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    ctx=ssl.create_default_context()
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,req,fp,code,msg,headers,newurl): return None
    opener=urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx),NoRedirect())
    for rel,maximum in FILES.items():
        url=f"https://raw.githubusercontent.com/renta0426/kaggle-actions-bridge/{COMMIT}/{ROOT}/{rel}"
        req=urllib.request.Request(url,headers={"User-Agent":"p1-03-fresh-cache-history/1"})
        with opener.open(req,timeout=30) as response:
            data=response.read(maximum+1)
        if not data or len(data)>maximum:
            raise SystemExit(f"historical byte budget failed:{rel}")
        path=out/rel; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
        if rel.endswith(".py"): compile(data,rel,"exec")
    print(f"P1_03_FRESH_HISTORY_FETCH PASS commit={COMMIT} files={len(FILES)}")
    return 0

if __name__ == "__main__": raise SystemExit(main())
