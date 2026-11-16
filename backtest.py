"""Backtesting utilities for return-forecast-driven trading strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd

from data import returns_to_simple


@dataclass
class BacktestConfig:
    """Backtest strategy configuration."""

    mode: Literal["long_short", "long_cash"] = "long_short"
    positive_threshold: float = 0.001
    negative_threshold: float = -0.001
    transaction_cost_bps: float = 5.0
    position_size: float = 1.0
    annualization_factor: int = 252


def _generate_positions(predictions: np.ndarray, config: BacktestConfig) -> np.ndarray:
    positions = np.zeros(len(predictions), dtype=float)

    if config.mode == "long_short":
        positions[predictions > config.positive_threshold] = 1.0
        positions[predictions < config.negative_threshold] = -1.0
    elif config.mode == "long_cash":
        positions[predictions > config.positive_threshold] = 1.0
    else:
        raise ValueError(f"Unsupported mode: {config.mode}")

    return positions


def _annualized_return(equity_curve: np.ndarray, annualization_factor: int) -> float:
    if len(equity_curve) == 0:
        return float("nan")
    total_return_multiplier = float(equity_curve[-1])
    if total_return_multiplier <= 0:
        return float("nan")
    years = len(equity_curve) / annualization_factor
    if years <= 0:
        return float("nan")
    return total_return_multiplier ** (1 / years) - 1.0


def _max_drawdown(equity_curve: np.ndarray) -> float:
    if len(equity_curve) == 0:
        return float("nan")
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve / peaks - 1.0
    return float(np.min(drawdowns))


def run_backtest(
    dates: np.ndarray,
    predicted_returns: np.ndarray,
    realized_returns: np.ndarray,
    target_type: str,
    config: BacktestConfig,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run a threshold-based strategy backtest on out-of-sample predictions."""
    preds = np.asarray(predicted_returns, dtype=float)
    realized = np.asarray(realized_returns, dtype=float)

    if len(preds) != len(realized):
        raise ValueError("predicted_returns and realized_returns must have same length.")

    realized_simple = returns_to_simple(realized, target_type)
    positions = _generate_positions(preds, config)

    gross_strategy_ret = config.position_size * positions * realized_simple

    prev_positions = np.roll(positions, 1)
    prev_positions[0] = 0.0
    turnover = np.abs(positions - prev_positions)

    cost_rate = config.transaction_cost_bps / 10000.0
    transaction_cost = turnover * cost_rate * config.position_size

    net_strategy_ret = gross_strategy_ret - transaction_cost

    equity_curve = np.cumprod(1.0 + net_strategy_ret)
    buy_hold_curve = np.cumprod(1.0 + realized_simple)

    strategy_drawdown = equity_curve / np.maximum.accumulate(equity_curve) - 1.0
    buy_hold_drawdown = buy_hold_curve / np.maximum.accumulate(buy_hold_curve) - 1.0

    active_mask = positions != 0.0
    if active_mask.sum() > 0:
        hit_rate = float(np.mean((positions[active_mask] * realized_simple[active_mask]) > 0))
    else:
        hit_rate = float("nan")

    ann_return = _annualized_return(equity_curve, config.annualization_factor)
    ann_vol = float(np.std(net_strategy_ret, ddof=1) * np.sqrt(config.annualization_factor))
    sharpe = float(ann_return / ann_vol) if np.isfinite(ann_vol) and ann_vol > 1e-12 else float("nan")

    bh_ann_return = _annualized_return(buy_hold_curve, config.annualization_factor)
    bh_ann_vol = float(np.std(realized_simple, ddof=1) * np.sqrt(config.annualization_factor))
    bh_sharpe = (
        float(bh_ann_return / bh_ann_vol)
        if np.isfinite(bh_ann_vol) and bh_ann_vol > 1e-12
        else float("nan")
    )

    backtest_df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "predicted_return": preds,
            "realized_return": realized,
            "realized_simple_return": realized_simple,
            "position": positions,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "strategy_return": net_strategy_ret,
            "buy_hold_return": realized_simple,
            "equity_curve": equity_curve,
            "buy_hold_curve": buy_hold_curve,
            "strategy_drawdown": strategy_drawdown,
            "buy_hold_drawdown": buy_hold_drawdown,
        }
    )

    backtest_df["rolling_sharpe_63"] = (
        backtest_df["strategy_return"].rolling(63).mean()
        / (backtest_df["strategy_return"].rolling(63).std() + 1e-12)
        * np.sqrt(config.annualization_factor)
    )

    metrics = {
        "Cumulative Return": float(equity_curve[-1] - 1.0),
        "Annualized Return": ann_return,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": _max_drawdown(equity_curve),
        "Hit Rate": hit_rate,
        "Number of Trades": float(np.sum(turnover > 0.0)),
        "Average Turnover": float(np.mean(turnover)),
        "Buy & Hold Cumulative Return": float(buy_hold_curve[-1] - 1.0),
        "Buy & Hold Annualized Return": bh_ann_return,
        "Buy & Hold Annualized Volatility": bh_ann_vol,
        "Buy & Hold Sharpe Ratio": bh_sharpe,
        "Buy & Hold Max Drawdown": _max_drawdown(buy_hold_curve),
    }

    return backtest_df, metrics
