# Resume Bullets

## Short Bullet Options (3)
- Built a Transformer-based equity return forecasting dashboard in Python with benchmark comparison and transaction-cost-aware backtesting.
- Developed a modular quantitative finance pipeline using TensorFlow, pandas, scikit-learn, Streamlit, and Plotly for out-of-sample model evaluation.
- Implemented leakage-safe preprocessing and risk-focused backtest analytics, including Sharpe ratio and drawdown tracking.

## Medium Bullet Options (3)
- Designed and implemented a Transformer-based time-series forecasting system for next-day equity returns, integrating engineered technical features, chronological train/validation/test splits, and training-only scaling to reduce data leakage risk.
- Built a benchmark-driven evaluation framework comparing Transformer forecasts against naive, moving-average, linear regression, and optional ARIMA baselines using RMSE, MAE, directional accuracy, and sign hit rate.
- Created a Streamlit analytics dashboard with Plotly visualizations and a configurable backtesting engine (long/short or long/cash, thresholds, transaction costs, position sizing) to connect predictive metrics with risk-adjusted strategy outcomes.

## Polished Project-Description Paragraphs (2)
This project implements a full quantitative finance research workflow for daily equity return forecasting and strategy evaluation. I replaced an earlier LSTM prototype with a configurable Transformer encoder, engineered a multivariate feature set from market data, and enforced leakage-aware preprocessing with chronological out-of-sample testing. I then compared model behavior against naive and statistical baselines to avoid isolated machine-learning claims.

I also built a transaction-cost-aware backtesting module and a Streamlit + Plotly dashboard to present forecast quality, benchmark tables, equity curves, and drawdowns in one interface. The project emphasizes realistic interpretation: in noisy financial data, simple baselines can be competitive, predictive fit does not guarantee trading value, and limitations must be stated explicitly.
