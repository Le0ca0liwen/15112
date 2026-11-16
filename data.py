"""Data acquisition and preprocessing utilities for financial forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN_MAP: Dict[str, str] = {
    "log_return": "target_log_return",
    "simple_return": "target_simple_return",
}

AVAILABLE_FEATURES: List[str] = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "simple_return",
    "log_return",
    "rolling_mean_5",
    "rolling_mean_20",
    "rolling_vol_10",
    "rolling_vol_20",
    "momentum_5",
    "momentum_10",
    "rsi_14",
    "ma_fast_10",
    "ma_slow_30",
    "ma_ratio_10_30",
    "price_to_ma20",
    "benchmark_log_return",
]


@dataclass
class SplitConfig:
    """Configuration for chronological train/validation/test splits."""

    train_ratio: float = 0.7
    val_ratio: float = 0.15

    @property
    def test_ratio(self) -> float:
        return 1.0 - self.train_ratio - self.val_ratio

    def validate(self) -> None:
        if self.train_ratio <= 0 or self.val_ratio <= 0:
            raise ValueError("train_ratio and val_ratio must be positive.")
        if self.test_ratio <= 0:
            raise ValueError("train_ratio + val_ratio must be less than 1.")


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""

    rsi_window: int = 14
    momentum_fast: int = 5
    momentum_slow: int = 10
    rolling_short: int = 5
    rolling_long: int = 20
    vol_short: int = 10
    vol_long: int = 20
    ma_fast: int = 10
    ma_slow: int = 30


def _flatten_download_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def download_price_data(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance."""
    if not ticker or not ticker.strip():
        raise ValueError("Ticker cannot be empty.")

    df = yf.download(
        tickers=ticker.strip().upper(),
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df.empty:
        raise ValueError(
            f"No data returned for {ticker} in range {start_date} to {end_date}."
        )

    df = _flatten_download_columns(df)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    out = df[required].copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    avg_gain = gains.rolling(window=window, min_periods=window).mean()
    avg_loss = losses.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def engineer_features(
    price_df: pd.DataFrame,
    feature_config: FeatureConfig | None = None,
    include_benchmark_return: bool = False,
    benchmark_ticker: str = "^GSPC",
) -> pd.DataFrame:
    """Create a multivariate feature matrix from OHLCV data."""
    feature_config = feature_config or FeatureConfig()

    df = price_df.copy()
    df["simple_return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    df["rolling_mean_5"] = df["Close"].rolling(feature_config.rolling_short).mean()
    df["rolling_mean_20"] = df["Close"].rolling(feature_config.rolling_long).mean()
    df["rolling_vol_10"] = df["log_return"].rolling(feature_config.vol_short).std()
    df["rolling_vol_20"] = df["log_return"].rolling(feature_config.vol_long).std()

    df["momentum_5"] = df["Close"] / df["Close"].shift(feature_config.momentum_fast) - 1.0
    df["momentum_10"] = df["Close"] / df["Close"].shift(feature_config.momentum_slow) - 1.0

    df["rsi_14"] = _compute_rsi(df["Close"], feature_config.rsi_window)

    df["ma_fast_10"] = df["Close"].rolling(feature_config.ma_fast).mean()
    df["ma_slow_30"] = df["Close"].rolling(feature_config.ma_slow).mean()
    df["ma_ratio_10_30"] = (
        df["ma_fast_10"] / (df["ma_slow_30"] + 1e-12) - 1.0
    )
    df["price_to_ma20"] = df["Close"] / (df["rolling_mean_20"] + 1e-12) - 1.0

    if include_benchmark_return:
        benchmark_df = download_price_data(
            benchmark_ticker,
            start_date=df.index.min().strftime("%Y-%m-%d"),
            end_date=(df.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        benchmark_log_ret = np.log(
            benchmark_df["Close"] / benchmark_df["Close"].shift(1)
        ).rename("benchmark_log_return")
        df = df.join(benchmark_log_ret, how="left")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill()

    # Target columns are shifted so features at t predict return from t to t+1.
    df["target_log_return"] = df["log_return"].shift(-1)
    df["target_simple_return"] = df["simple_return"].shift(-1)

    df = df.dropna()
    if len(df) < 120:
        raise ValueError(
            "Insufficient rows after feature engineering; expand the date range."
        )

    return df


def build_model_frame(
    feature_df: pd.DataFrame,
    selected_features: Sequence[str],
    target_type: str = "log_return",
) -> Tuple[pd.DataFrame, List[str], str]:
    """Select final feature/target columns used for modeling."""
    if target_type not in TARGET_COLUMN_MAP:
        raise ValueError(f"Unsupported target_type: {target_type}")

    target_col = TARGET_COLUMN_MAP[target_type]
    feature_cols = list(dict.fromkeys(selected_features))

    if not feature_cols:
        raise ValueError("At least one feature must be selected.")

    missing = [col for col in feature_cols if col not in feature_df.columns]
    if missing:
        raise ValueError(f"Selected features not present in frame: {missing}")

    required_cols = feature_cols + [
        target_col,
        "Close",
        "simple_return",
        "log_return",
    ]
    required_cols = list(dict.fromkeys(required_cols))

    model_df = feature_df[required_cols].copy()
    model_df = model_df.loc[:, ~model_df.columns.duplicated()].copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(model_df) < 120:
        raise ValueError(
            "Model frame too short after filtering; reduce feature windows or widen date range."
        )

    return model_df, feature_cols, target_col


def _build_sequences(
    features: np.ndarray,
    target: np.ndarray,
    dates: np.ndarray,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    d_list: List[pd.Timestamp] = []

    for i in range(sequence_length - 1, len(features)):
        X_list.append(features[i - sequence_length + 1 : i + 1])
        y_list.append(float(target[i]))
        d_list.append(dates[i])

    if not X_list:
        raise ValueError(
            "No training samples created. Lower sequence length or add more history."
        )

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    d = np.asarray(d_list)
    return X, y, d


def make_train_val_test_sequences(
    model_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    sequence_length: int,
    split_config: SplitConfig | None = None,
) -> Dict[str, object]:
    """Create leakage-safe sequences and chronological split masks."""
    split_config = split_config or SplitConfig()
    split_config.validate()

    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2.")

    n_rows = len(model_df)
    min_required = max(3 * sequence_length, 150)
    if n_rows < min_required:
        raise ValueError(
            f"Need at least {min_required} rows for reliable splits; got {n_rows}."
        )

    train_end = int(n_rows * split_config.train_ratio)
    val_end = train_end + int(n_rows * split_config.val_ratio)

    if train_end <= sequence_length + 5:
        raise ValueError("Training segment too short for selected sequence length.")
    if val_end <= train_end + 5:
        raise ValueError("Validation segment too short; increase date range.")
    if n_rows - val_end <= 5:
        raise ValueError("Test segment too short; adjust split ratios.")

    scaler = StandardScaler()
    train_feature_block = model_df.iloc[:train_end][list(feature_cols)].values
    scaler.fit(train_feature_block)

    scaled_features = scaler.transform(model_df[list(feature_cols)].values)
    target = model_df[target_col].values
    dates = model_df.index.to_numpy()

    X_all, y_all, d_all = _build_sequences(scaled_features, target, dates, sequence_length)

    train_boundary_date = model_df.index[train_end - 1]
    val_boundary_date = model_df.index[val_end - 1]

    train_mask = d_all <= train_boundary_date
    val_mask = (d_all > train_boundary_date) & (d_all <= val_boundary_date)
    test_mask = d_all > val_boundary_date

    if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError(
            "One split has zero sequence samples. Adjust sequence length or split ratios."
        )

    aligned_frame = model_df.loc[pd.Index(d_all)].copy()

    return {
        "X_train": X_all[train_mask],
        "y_train": y_all[train_mask],
        "dates_train": d_all[train_mask],
        "X_val": X_all[val_mask],
        "y_val": y_all[val_mask],
        "dates_val": d_all[val_mask],
        "X_test": X_all[test_mask],
        "y_test": y_all[test_mask],
        "dates_test": d_all[test_mask],
        "feature_scaler": scaler,
        "aligned_frame": aligned_frame,
        "feature_cols": list(feature_cols),
        "target_col": target_col,
    }


def returns_to_simple(returns: np.ndarray, target_type: str) -> np.ndarray:
    """Convert modeled returns to simple-return space for portfolio analytics."""
    if target_type == "log_return":
        return np.expm1(returns)
    if target_type == "simple_return":
        return returns
    raise ValueError(f"Unsupported target_type: {target_type}")


def reconstruct_price_paths(
    start_price: float,
    predicted_returns: np.ndarray,
    realized_returns: np.ndarray,
    target_type: str,
) -> pd.DataFrame:
    """Reconstruct cumulative price paths from return forecasts and realizations."""
    pred_simple = returns_to_simple(np.asarray(predicted_returns, dtype=float), target_type)
    real_simple = returns_to_simple(np.asarray(realized_returns, dtype=float), target_type)

    pred_prices = [start_price]
    real_prices = [start_price]

    for ret in pred_simple:
        pred_prices.append(pred_prices[-1] * (1.0 + ret))
    for ret in real_simple:
        real_prices.append(real_prices[-1] * (1.0 + ret))

    return pd.DataFrame(
        {
            "step": np.arange(len(pred_prices)),
            "predicted_price": pred_prices,
            "realized_price": real_prices,
        }
    )


def build_walk_forward_splits(
    n_samples: int,
    initial_train_size: int,
    step_size: int = 1,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build expanding-window rolling-origin split indices."""
    if n_samples <= initial_train_size:
        raise ValueError("n_samples must exceed initial_train_size.")
    if initial_train_size < 10:
        raise ValueError("initial_train_size too small for walk-forward.")
    if step_size < 1:
        raise ValueError("step_size must be >= 1.")

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for idx in range(initial_train_size, n_samples, step_size):
        train_idx = np.arange(0, idx)
        test_idx = np.arange(idx, min(idx + step_size, n_samples))
        splits.append((train_idx, test_idx))

    return splits
