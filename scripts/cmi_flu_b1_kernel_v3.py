#!/usr/bin/env python3
"""Request-003 entrypoint for the bridge-contained CMI-Flu B1 implementation."""

from __future__ import annotations

import cmi_flu_b1_kernel_impl as implementation

REQUEST_ID = "20260902-cmi-flu-b1-003"
CMI_CONTRACT_COMMIT = "6362cd8596cd996d13f66615eb1833ad26af1b39"
CMI_STANDALONE_RUNNER_BLOB = "7e28ed606f63883f9c2e230dfea1382389dce23b"

implementation.REQUEST_ID = REQUEST_ID
implementation.CMI_COMMIT = CMI_CONTRACT_COMMIT
implementation.SOURCE_BLOBS = {
    **implementation.SOURCE_BLOBS,
    "run_b1_kaggle_standalone.py": CMI_STANDALONE_RUNNER_BLOB,
}


if __name__ == "__main__":
    raise SystemExit(implementation.main())
