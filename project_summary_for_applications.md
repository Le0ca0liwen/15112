# Project Summary for Applications

## 100-Word Summary
I built a Python-based quantitative finance dashboard for daily equity return forecasting and strategy evaluation. The project uses a TensorFlow Transformer encoder to predict next-day returns (default log returns) from engineered market features including OHLCV, momentum, volatility, RSI, and moving-average signals. I evaluated the model against naive, moving-average, linear regression, and optional ARIMA benchmarks using out-of-sample RMSE, MAE, and directional metrics. I then connected forecasts to a transaction-cost-aware backtest with configurable long/short rules, position sizing, and risk statistics (Sharpe, volatility, drawdown). The goal was methodological rigor: leakage prevention, benchmark discipline, and realistic interpretation rather than overclaiming predictive edge.

## 150-Word Summary
This project is an end-to-end computational finance workflow for financial time-series forecasting, model comparison, and backtesting. I replaced an earlier LSTM prototype with a configurable Transformer encoder in TensorFlow/Keras and focused on predicting next-day returns rather than raw price levels. The feature pipeline includes OHLCV data, return transformations, rolling moments, momentum, RSI, moving-average features, and optional benchmark-index returns from Yahoo Finance data.

To keep evaluation defensible, I used chronological train/validation/test splits, scaling fit only on training data, and out-of-sample metrics. The Transformer is benchmarked against naive last-return, moving-average, linear regression, and optional ARIMA baselines. I also implemented a threshold-based, transaction-cost-aware backtest with long/short or long/cash modes and risk metrics such as annualized return, Sharpe ratio, and max drawdown.

The Streamlit + Plotly dashboard presents benchmark tables, forecast charts, equity curves, drawdown behavior, and explicit limitations for realistic financial interpretation.

## 250-Word Summary
I developed a portfolio-ready quantitative finance application that integrates machine learning, statistical benchmarking, and risk-aware strategy evaluation in a single reproducible codebase. The project centers on forecasting next-day equity returns using a Transformer encoder (TensorFlow/Keras) with configurable sequence length, attention heads, model dimension, feed-forward size, block depth, dropout, and optimization settings.

Rather than relying on raw prices alone, I implemented a modular feature-engineering pipeline based on daily OHLCV data and derived signals such as log/simple returns, rolling averages, rolling volatility, momentum, RSI, and moving-average relationships. The default target is next-day log return to align modeling with return dynamics instead of level prediction.

For methodological rigor, the pipeline enforces chronological train/validation/test splitting, training-only scaling, and out-of-sample evaluation. I compared the Transformer against naive last-return, moving-average, linear regression, and optional ARIMA baselines using MSE, MAE, RMSE, directional accuracy, and sign hit rate. This benchmark-first framing is central to the project and prevents overinterpreting model complexity.

I also built a transaction-cost-aware backtesting module where predictions are converted into configurable long/short or long/cash positions using thresholds. The backtest reports cumulative and annualized returns, volatility, Sharpe ratio, max drawdown, hit rate, turnover, and trade count, alongside buy-and-hold results.

The final Streamlit + Plotly dashboard includes benchmark tables, forecast diagnostics, equity and drawdown charts, downloadable outputs, and an explicit limitations section. The project is designed to demonstrate computational finance maturity: careful validation, honest interpretation, and clear communication of uncertainty.
