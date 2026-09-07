# CodeParrot pilot config-compat compile-gate incident

Date: 2026-09-07

## Observed failure

Protected repair run `34118051478` passed live admission against exact target version 1 / ERROR / GPU, pulled the private Min-K++ source, and composed the repaired notebook. The compatibility patch then failed before `kaggle kernels push`:

`SyntaxError: invalid syntax` while calling ordinary Python `compile()` on every code cell in the composed Jupyter notebook.

## Root cause

The compatibility patch introduced an over-broad validation gate. It compiled every existing notebook code cell with CPython. A private Kaggle/Jupyter source cell can be valid in the notebook execution environment while not being valid as a standalone ordinary-Python compilation unit. The patch only modifies the setup cell and inserts one compatibility cell; compiling unrelated inherited private cells is neither necessary nor a valid notebook-integrity check.

## Write / compute impact

- version-2 `kaggle kernels push`: not reached
- target remains version 1
- new GPU execution: not started
- membership metrics / holdout evaluation: not computed
- automatic retries: 0

## Repair rule

Only syntax-check source introduced or modified by the compatibility patch. Continue to run `nbformat.validate()` over the full notebook structure, but do not reinterpret unrelated inherited Jupyter cells as standalone CPython files.

The same repair also refreshes the frozen cohort artifact SHA from the stale bridge constant to the already-audited `prediction_input.parquet` SHA-256 `b664e368f9380e230c9cfc0616327d829424749c5473f87575678525db1998fc`, otherwise the next Kaggle execution would fail before scoring.
