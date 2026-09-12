#!/usr/bin/env python3
"""Run V3-06 full-path CI against the final science-main-bound runtime."""
from __future__ import annotations

import cmi_flu_v3_v06_ci as base
import cmi_flu_v3_v06_prepare_v4 as final_prepare

base.prepare = final_prepare

if __name__ == "__main__":
    raise SystemExit(base.main())
