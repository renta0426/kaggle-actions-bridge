"""E06b runtime compatibility patch for the two-head Ridge fit.

The first science CI run caught a local variable typo in the initial E06b
implementation before any Kaggle execution.  This adapter replaces only the
fit helper and installs it into the base module globals used by the already
registered E06b evaluator.  The scientific design is unchanged.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .contracts import DataContractError, require_finite
from .datasets import HAIModelDataset
from .models import ModelSpec, build_estimator, prepare_model_frame
from . import strategy_e06b as _base


def _fit_two_head_ridge(
    day28_train: pd.DataFrame,
    day365_train: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    *,
    day28: HAIModelDataset,
    day365: HAIModelDataset,
    spec: ModelSpec,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    if spec.family != "ridge" or spec.name != _base.MODEL_NAME:
        raise DataContractError("E06b two-head control requires ridge_exact_a100")
    alpha = float(spec.params.get("alpha", np.nan))
    if not np.isfinite(alpha) or alpha != 100.0:
        raise DataContractError("E06b two-head control requires Ridge alpha=100")

    features = _base._require_compatible_feature_space(day28, day365, spec)
    d28_x = day28_train[features].copy()
    d365_x = day365_train[features].copy()
    pred_x = prediction_frame[features].copy()
    stacked = pd.concat([d28_x, d365_x], ignore_index=True)

    model_stack, numeric, categorical = prepare_model_frame(
        stacked,
        excluded_columns=(),
    )
    model_pred, pred_numeric, pred_categorical = prepare_model_frame(
        pred_x,
        excluded_columns=(),
    )
    if numeric != pred_numeric or categorical != pred_categorical:
        raise DataContractError("E06b train/prediction feature typing differs")

    pipeline = build_estimator(
        spec,
        numeric_columns=numeric,
        categorical_columns=categorical,
    )
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = np.asarray(preprocessor.fit_transform(model_stack), dtype=float)
    transformed_pred = np.asarray(preprocessor.transform(model_pred), dtype=float)
    require_finite(transformed, name="E06b transformed training features")
    require_finite(transformed_pred, name="E06b transformed prediction features")

    d28_target = pd.to_numeric(
        day28_train[day28.target_column], errors="coerce"
    ).to_numpy(dtype=float)
    d365_target = pd.to_numeric(
        day365_train[day365.target_column], errors="coerce"
    ).to_numpy(dtype=float)
    target = np.concatenate([d28_target, d365_target])
    require_finite(target, name="E06b two-head target")

    day_code = np.concatenate(
        [
            np.full(len(day28_train), _base.DAY_CODE[_base.AUXILIARY_DAY], dtype=float),
            np.full(len(day365_train), _base.DAY_CODE[_base.PRIMARY_DAY], dtype=float),
        ]
    )
    design = np.column_stack(
        [
            transformed,
            transformed * day_code[:, None],
            day_code,
        ]
    )
    prediction_code = np.full(
        len(prediction_frame), _base.DAY_CODE[_base.PRIMARY_DAY], dtype=float
    )
    prediction_design = np.column_stack(
        [
            transformed_pred,
            transformed_pred * prediction_code[:, None],
            prediction_code,
        ]
    )
    require_finite(design, name="E06b two-head design")
    require_finite(prediction_design, name="E06b two-head prediction design")

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(design, target)
    prediction = np.asarray(model.predict(prediction_design), dtype=float).reshape(-1)
    require_finite(prediction, name="E06b two-head prediction")

    return prediction, {
        "alpha": alpha,
        "day_code_d28": _base.DAY_CODE[_base.AUXILIARY_DAY],
        "day_code_d365": _base.DAY_CODE[_base.PRIMARY_DAY],
        "d28_training_rows": int(len(day28_train)),
        "d365_training_rows": int(len(day365_train)),
        "d28_training_studies": int(day28_train["study_group"].astype(str).nunique()),
        "d365_training_studies": int(day365_train["study_group"].astype(str).nunique()),
        "raw_feature_count": int(len(features)),
        "transformed_feature_count": int(transformed.shape[1]),
        "two_head_design_feature_count": int(design.shape[1]),
        "observed_d28_used_as_d365_feature": False,
        "compatibility_patch": "ci_caught_day28_variable_typo_only",
    }


# Base evaluator resolves this helper from its module globals at runtime.
_base._fit_two_head_ridge = _fit_two_head_ridge


def run_strategy_e06b(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    result = dict(_base.run_strategy_e06b(*args, **kwargs))
    contract = dict(result.get("model_contract") or {})
    contract["two_head_fit_adapter"] = "strategy_e06b_v2_ci_typo_fix"
    result["model_contract"] = contract
    return _base._json_safe(result)


EXPERIMENT = _base.EXPERIMENT
TASK = _base.TASK
PRIMARY_DAY = _base.PRIMARY_DAY
AUXILIARY_DAY = _base.AUXILIARY_DAY
MODEL_NAME = _base.MODEL_NAME
BASE_CONDITIONS = _base.BASE_CONDITIONS

__all__ = [
    "EXPERIMENT",
    "TASK",
    "PRIMARY_DAY",
    "AUXILIARY_DAY",
    "MODEL_NAME",
    "BASE_CONDITIONS",
    "_fit_two_head_ridge",
    "run_strategy_e06b",
]
