# Transformer-Based Financial Time-Series Forecasting and Backtesting Dashboard

## At a Glance
- Builds an end-to-end quantitative finance workflow for daily equity return forecasting and strategy evaluation.
- Uses a TensorFlow/Keras Transformer encoder (not LSTM) to predict next-day returns (default: log returns).
- Benchmarks against naive, moving-average, linear regression, and optional ARIMA models.
- Evaluates models using out-of-sample error and directional metrics, not isolated in-sample fit.
- Runs a transaction-cost-aware backtest with equity curve, drawdown, and risk statistics.
- Demonstrates that rigorous validation and benchmark discipline matter more than model complexity.

## One-Paragraph Summary
This project is a modular Python dashboard for quantitative financial forecasting and evaluation. It uses a Transformer-based sequence model on engineered market features (OHLCV and technical signals), compares performance against simple and statistical baselines, and maps out-of-sample predictions into a threshold-based backtest with transaction costs. The objective is methodological rigor: leakage prevention, chronological splits, benchmark-centered interpretation, and realistic limitations, rather than exaggerated claims of predictive edge.

## Motivation
Financial forecasting is noisy, unstable, and easy to overstate. In computational finance, strong modeling claims should survive comparison against simple baselines and should be evaluated through out-of-sample testing and risk-aware backtesting. This project is designed to reflect that standard directly.

## Features
- Transformer forecasting model with configurable architecture and training hyperparameters.
- Multivariate feature engineering: OHLCV, returns, rolling moments, momentum, RSI, and moving-average ratios.
- Optional benchmark index return feature (e.g., S&P 500 log return).
- Chronological train/validation/test design with training-only feature scaling.
- Optional walk-forward linear baseline mode for rolling-origin comparison.
- Benchmark table with MSE, MAE, RMSE, directional accuracy, and sign hit rate.
- Backtesting module with long/short or long/cash mode, transaction costs, and position sizing.
- Streamlit + Plotly dashboard with downloadable benchmark/backtest CSVs and markdown findings.
- Built-in report and explicit limitations section.

## Methodology
- **Data source**: Yahoo Finance daily OHLCV via `yfinance`.
- **Target**: next-day return (`log_return` default, `simple_return` optional).
- **Feature engineering**: market and technical features computed from historical information only.
- **Preprocessing**: missing-value handling, leakage-safe scaling (fit on train only), chronological split.
- **Model**: Transformer encoder with input projection, positional encoding, multi-head attention, feed-forward layers, residual connections, normalization, dropout, and dense output.
- **Benchmarks**: naive last-return, moving-average return, linear regression on flattened windows, optional ARIMA.
- **Backtest logic**: threshold-based signal generation from out-of-sample predictions with transaction costs and risk/performance metrics.

## Results Summary
Results are intentionally reported in-model and period specific via the dashboard benchmark table and backtest panels. In some runs, the Transformer ranks best; in others, simpler baselines are competitive or superior. This is expected in low signal-to-noise financial settings and is treated as an analytical result, not a failure.

## Repository Structure
```text
.
©À©¤©¤ app.py
©À©¤©¤ data.py
©À©¤©¤ model.py
©À©¤©¤ baselines.py
©À©¤©¤ metrics.py
©À©¤©¤ backtest.py
©À©¤©¤ report.py
©À©¤©¤ requirements.txt
©À©¤©¤ README.md
©À©¤©¤ project_report.md
©À©¤©¤ resume_bullets.md
©À©¤©¤ project_summary_for_applications.md
©À©¤©¤ screenshot_guide.md
©¸©¤©¤ screenshots/
    ©¸©¤©¤ README.md
```

## Installation and Usage
1. Create and activate a virtual environment:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch the dashboard:
```bash
streamlit run app.py
```

## Limitations
- Financial return data is nonstationary and regime dependent.
- Signal-to-noise ratio is low, so unstable predictive performance is common.
- Backtest outcomes depend on threshold choices, costs, and implementation assumptions.
- Daily market-derived features have limited explanatory depth.
- Results are benchmark-sensitive and sample-period-sensitive.

## Future Work
- Add macro, factor, and alternative data features.
- Extend to cross-sectional multi-asset models and portfolio construction.
- Add regime detection and state-dependent modeling.
- Evaluate probabilistic forecasts and calibration.
- Integrate portfolio optimization under explicit risk constraints.

## Key Takeaways (What I Learned)
- Financial data is noisy enough that simple baselines can remain difficult to beat.
- Leakage prevention (especially scaling and split design) materially affects conclusions.
- Better forecast metrics do not automatically imply better trading performance.
- Backtesting with transaction costs and drawdown analysis is essential for credibility.
- Model complexity should be justified by robust out-of-sample evidence.

## How to Discuss This Project
- **Why Transformer**: It captures temporal interactions across multivariate sequences through self-attention while remaining configurable and interpretable at an architectural level.
- **What baselines were used**: Naive last-return, moving-average return, linear regression, and optional ARIMA to prevent isolated model claims.
- **How backtest works**: Out-of-sample return predictions are converted into threshold-based positions with transaction costs and evaluated using return and risk metrics.
- **Main limitations**: Nonstationarity, low signal-to-noise ratio, overfitting risk, sensitivity to market regime and assumptions.
- **Relevance to computational finance**: The project integrates modeling, statistical validation, market interpretation, and risk-aware implementation in one reproducible workflow.
