"""Baseline forecasting models for benchmark comparison."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def _target_proxy_column(target_type: str) -> str:
    if target_type == "log_return":
        return "log_return"
    if target_type == "simple_return":
        return "simple_return"
    raise ValueError(f"Unsupported target_type: {target_type}")


def naive_last_return_predictor(
    aligned_frame: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
    target_type: str,
) -> np.ndarray:
    """Predict next return as the most recent observed return."""
    source_col = _target_proxy_column(target_type)
    preds = aligned_frame.loc[pd.Index(dates), source_col].values
    return preds.astype(float)


def moving_average_predictor(
    aligned_frame: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
    target_type: str,
    window: int = 5,
) -> np.ndarray:
    """Forecast with rolling mean of recent returns."""
    if window < 1:
        raise ValueError("window must be >= 1")

    source_col = _target_proxy_column(target_type)
    rolling_mean = (
        aligned_frame[source_col].rolling(window=window, min_periods=1).mean().rename("ma")
    )
    preds = rolling_mean.loc[pd.Index(dates)].values
    return preds.astype(float)


def linear_regression_predictor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, LinearRegression]:
    """Fit linear regression on flattened sequences and predict test samples."""
    model = LinearRegression()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    model.fit(X_train_flat, y_train)
    preds = model.predict(X_test_flat)
    return preds.astype(float), model


def linear_regression_walk_forward(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    retrain_every: int = 1,
) -> np.ndarray:
    """Expanding-window walk-forward linear regression predictions."""
    if retrain_every < 1:
        raise ValueError("retrain_every must be >= 1")

    X_hist = X_train.reshape(X_train.shape[0], -1).copy()
    y_hist = y_train.copy()
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    model = LinearRegression()
    model.fit(X_hist, y_hist)

    preds = np.zeros(len(X_test_flat), dtype=float)
    for i, x_row in enumerate(X_test_flat):
        preds[i] = float(model.predict(x_row.reshape(1, -1))[0])

        X_hist = np.vstack([X_hist, x_row])
        y_hist = np.append(y_hist, y_test[i])

        if (i + 1) % retrain_every == 0:
            model.fit(X_hist, y_hist)

    return preds


def arima_predictor(
    train_target: np.ndarray,
    test_target: np.ndarray,
    order: Tuple[int, int, int] = (1, 0, 0),
) -> Tuple[Optional[np.ndarray], str]:
    """Optional ARIMA baseline with guarded failure handling."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception:
        return None, "statsmodels is unavailable; ARIMA baseline skipped."

    try:
        if len(train_target) < 40:
            return None, "Training set too short for stable ARIMA fit."

        model = ARIMA(train_target, order=order)
        fitted = model.fit()
        preds = fitted.forecast(steps=len(test_target))
        return np.asarray(preds, dtype=float), "ARIMA baseline fitted successfully."
    except Exception as exc:  # noqa: BLE001
        return None, f"ARIMA failed: {exc}"
