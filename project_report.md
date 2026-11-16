# Transformer-Based Financial Time-Series Forecasting and Backtesting Project Report

## 1. Objective
The objective is to build and evaluate a methodologically defensible forecasting and backtesting pipeline for daily equity returns. The central question is whether a Transformer-based sequence model provides incremental predictive or trading value relative to simpler baselines under out-of-sample evaluation.

## 2. Data
- **Source**: Yahoo Finance (`yfinance`)
- **Frequency**: Daily
- **Primary inputs**: OHLCV market data
- **Engineered features**: simple/log returns, rolling means, rolling volatility, momentum, RSI, moving-average levels and ratios, optional benchmark-index return
- **Coverage**: user-configurable ticker and date range in the Streamlit dashboard

### Data Limitations
The project depends on vendor-provided historical data quality and only uses market-derived variables, which limits explanatory coverage of fundamental, macro, and alternative information.

## 3. Methodology
### Target Construction
The default prediction target is **next-day log return**. A next-day simple return option is also available.

### Preprocessing
- Missing values are handled via forward-fill and then dropped where necessary.
- Train/validation/test splits are chronological.
- Feature scaling is fit on the training split only to reduce leakage.
- Sequence windows are created so features at time _t_ predict return from _t_ to _t+1_.

### Transformer Architecture
The model is a configurable Transformer encoder with:
- Input projection for multivariate features
- Positional encoding (sinusoidal or learned)
- 1-3 encoder blocks
- Multi-head self-attention
- Feed-forward sublayers
- Residual connections
- Layer normalization
- Dropout
- Dense output head

### Benchmark Models
- Naive last-return predictor
- Moving-average return predictor
- Linear regression baseline on flattened windows
- Optional ARIMA baseline

### Evaluation and Backtest
- Out-of-sample predictive metrics: MSE, MAE, RMSE, directional accuracy, sign hit rate
- Strategy: threshold-based long/short or long/cash decisions from predicted returns
- Backtest assumptions: out-of-sample predictions only, transaction costs, configurable position size
- Portfolio metrics: cumulative return, annualized return/volatility, Sharpe ratio, max drawdown, hit rate, number of trades, turnover

## 4. Results
Results are intentionally reported inside the dashboard for the selected ticker and period. The benchmark table is treated as the primary model-comparison output. The project does not assume Transformer superiority; if simpler methods perform better, that outcome is reported directly.

## 5. Interpretation
This project is designed to demonstrate rigorous computational finance workflow rather than optimized short-term alpha claims. The key analytic value is in comparing model classes under consistent out-of-sample conditions and separating predictive fit from risk-adjusted trading performance.

## 6. Limitations
- Financial return series are nonstationary and regime-dependent.
- Signal-to-noise ratio is low, making unstable performance expected.
- Flexible models can overfit despite validation controls.
- Backtest outcomes are sensitive to strategy thresholds and costs.
- Model ranking is benchmark-dependent and sample-period-dependent.
- Price-only and technical features have limited explanatory power for many return drivers.
- Implementation assumptions (liquidity, execution quality, slippage) may differ from live trading conditions.

## 7. Future Work
- Add macroeconomic and factor data.
- Extend to cross-sectional multi-asset and portfolio-level prediction.
- Add regime detection/state switching and volatility-conditioned models.
- Use probabilistic forecasting and calibration diagnostics.
- Integrate portfolio optimization with explicit risk budgeting constraints.

---
This report is intentionally conservative in interpretation and is intended for academic and portfolio presentation use in computational finance contexts.
