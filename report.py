"""Markdown report generation for the forecasting dashboard."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def _fmt_pct(value: float) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "N/A"
    return f"{value * 100:.2f}%"


def _fmt_num(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "N/A"
    return f"{value:.{digits}f}"


def _benchmark_interpretation(benchmark_table: pd.DataFrame) -> str:
    if benchmark_table.empty:
        return "No benchmark table was available for interpretation."

    best_row = benchmark_table.iloc[0]
    best_model = str(best_row["Model"])

    transformer_rows = benchmark_table[benchmark_table["Model"] == "Transformer"]
    if transformer_rows.empty:
        return (
            f"The top model by RMSE was **{best_model}**. Transformer results were not found in the table."
        )

    transformer_rmse = float(transformer_rows.iloc[0]["RMSE"])
    best_rmse = float(best_row["RMSE"])

    if best_model == "Transformer":
        return (
            "In this run, the Transformer ranked first by RMSE. This should be interpreted as "
            "sample-specific evidence rather than a claim of persistent alpha."
        )

    rmse_gap = transformer_rmse - best_rmse
    return (
        f"In this run, **{best_model}** outperformed the Transformer on RMSE. "
        f"The RMSE gap was {rmse_gap:.6f}, highlighting that simple baselines can remain competitive."
    )


def generate_project_report_markdown(
    ticker: str,
    start_date: str,
    end_date: str,
    target_type: str,
    selected_features: Iterable[str],
    benchmark_table: pd.DataFrame,
    backtest_metrics: Dict[str, float],
    transformer_config: Dict[str, float | int | str],
    walk_forward_enabled: bool,
) -> str:
    """Generate a concise, academically styled markdown report."""
    feature_list: List[str] = list(selected_features)

    target_desc = (
        "next-day log return" if target_type == "log_return" else "next-day simple return"
    )

    benchmark_interpretation = _benchmark_interpretation(benchmark_table)

    return f"""# Transformer-Based Financial Forecasting and Backtesting Report

## 1. Objective
This project studies whether a Transformer encoder can forecast {target_desc} for **{ticker}** using market-derived features, and whether those forecasts translate into useful out-of-sample trading behavior once transaction costs are included.

## 2. Data
- Source: Yahoo Finance via `yfinance`
- Asset: `{ticker}`
- Date range: `{start_date}` to `{end_date}`
- Frequency: Daily
- Feature set size: {len(feature_list)}
- Features used: {", ".join(feature_list)}

Data limitations include survivorship and corporate-action handling dependence on the vendor feed, as well as the limited explanatory power of purely market-derived daily features.

## 3. Methodology
- Target construction: model predicts {target_desc}.
- Preprocessing: chronological split, missing-value handling, and feature scaling fit only on the training segment.
- Split design: train/validation/test chronological partitions with no random shuffling.
- Transformer architecture: configurable encoder with positional encoding, multi-head self-attention, residual connections, layer normalization, feed-forward layers, dropout, and dense output head.
- Benchmarks: naive last-return, moving-average return, linear regression baseline, and optional ARIMA.
- Evaluation: out-of-sample regression metrics (MSE, MAE, RMSE) and directional metrics (directional accuracy/sign hit rate).
- Backtest: threshold-based signal generation with transaction costs, configurable long/short or long/cash exposure, and risk-adjusted performance metrics.
- Walk-forward mode: {"enabled" if walk_forward_enabled else "disabled"}.

Transformer settings snapshot:
`{transformer_config}`

## 4. Results
### Predictive Results
{benchmark_interpretation}

### Backtest Results
- Cumulative return: {_fmt_pct(backtest_metrics.get("Cumulative Return"))}
- Annualized return: {_fmt_pct(backtest_metrics.get("Annualized Return"))}
- Annualized volatility: {_fmt_pct(backtest_metrics.get("Annualized Volatility"))}
- Sharpe ratio: {_fmt_num(backtest_metrics.get("Sharpe Ratio", float("nan")))}
- Max drawdown: {_fmt_pct(backtest_metrics.get("Max Drawdown"))}
- Hit rate: {_fmt_pct(backtest_metrics.get("Hit Rate"))}
- Number of trades: {_fmt_num(backtest_metrics.get("Number of Trades", float("nan")), 0)}

Buy-and-hold is reported in parallel to avoid evaluating model strategies in isolation.

## 5. Interpretation
The project emphasizes methodological defensibility over model hype. Performance should be interpreted as evidence under one sample period and design choice, not as proof of persistent tradable edge.

## 6. Limitations
- Financial returns are nonstationary and have low signal-to-noise ratios.
- Overfitting risk remains material for expressive models, including Transformers.
- Results can change materially across market regimes and ticker selection.
- Backtest outcomes are sensitive to threshold choices, costs, and position sizing.
- Model quality depends on benchmark set; simpler models can outperform in many windows.
- Daily OHLCV-derived features capture only a subset of information relevant for returns.

## 7. Future Work
- Add broader factor, macro, and cross-asset features.
- Expand to cross-sectional or portfolio-level modeling.
- Introduce regime detection and volatility-state conditioning.
- Couple forecasts with explicit portfolio optimization and risk budgeting.
- Evaluate probabilistic forecasts and calibration quality, not only point forecasts.
"""


def default_limitations_markdown() -> str:
    """Reusable limitations text for dashboard display."""
    return """
### Limitations
- Financial return prediction has inherently low signal-to-noise ratio.
- Model performance can degrade under regime shifts and structural breaks.
- Out-of-sample success in one ticker or period does not guarantee robustness elsewhere.
- Transaction costs, slippage, and execution assumptions can materially alter live outcomes.
- Benchmark selection materially affects interpretation of model quality.
- Price/volume-only features have limited explanatory depth for many market moves.
"""
