"""Evaluation metrics for forecasting and model comparison."""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _safe_sign(values: np.ndarray) -> np.ndarray:
    out = np.sign(values)
    out[np.isclose(values, 0.0)] = 0.0
    return out


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Percentage of times the predicted and realized signs match."""
    signs_true = _safe_sign(y_true)
    signs_pred = _safe_sign(y_pred)
    return float(np.mean(signs_true == signs_pred))


def sign_hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Sign hit rate excluding zero-return observations."""
    signs_true = _safe_sign(y_true)
    signs_pred = _safe_sign(y_pred)
    mask = signs_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(signs_true[mask] == signs_pred[mask]))


def compute_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute core out-of-sample predictive metrics."""
    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "Directional Accuracy": directional_accuracy(y_true, y_pred),
        "Sign Hit Rate": sign_hit_rate(y_true, y_pred),
        "Mean Predicted Return": float(np.mean(y_pred)),
        "Mean Realized Return": float(np.mean(y_true)),
        "Bias (Pred-Realized)": float(np.mean(y_pred - y_true)),
    }


def build_benchmark_table(
    y_true: np.ndarray,
    predictions: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Create side-by-side benchmark comparison table."""
    rows = []
    for model_name, preds in predictions.items():
        metrics = compute_forecast_metrics(y_true, preds)
        metrics["Model"] = model_name
        rows.append(metrics)

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    table = table[
        [
            "Model",
            "MSE",
            "MAE",
            "RMSE",
            "Directional Accuracy",
            "Sign Hit Rate",
            "Mean Predicted Return",
            "Mean Realized Return",
            "Bias (Pred-Realized)",
        ]
    ]

    table = table.sort_values("RMSE", ascending=True).reset_index(drop=True)
    return table
