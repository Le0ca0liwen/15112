"""Streamlit dashboard for Transformer-based financial forecasting and backtesting."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest import BacktestConfig, run_backtest
from baselines import (
    arima_predictor,
    linear_regression_predictor,
    linear_regression_walk_forward,
    moving_average_predictor,
    naive_last_return_predictor,
)
from data import (
    AVAILABLE_FEATURES,
    SplitConfig,
    build_model_frame,
    download_price_data,
    engineer_features,
    make_train_val_test_sequences,
    reconstruct_price_paths,
)
from metrics import build_benchmark_table, compute_forecast_metrics
from model import (
    TransformerConfig,
    build_transformer_model,
    model_summary_string,
    predict_transformer,
    set_random_seed,
    train_transformer,
)
from report import default_limitations_markdown, generate_project_report_markdown


DEFAULT_FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "log_return",
    "rolling_vol_20",
    "momentum_10",
    "rsi_14",
    "ma_ratio_10_30",
]


def _format_metric(value: float, pct: bool = False) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "N/A"
    return f"{value * 100:.2f}%" if pct else f"{value:.4f}"


def _loss_figure(history_dict: Dict[str, List[float]]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=history_dict.get("loss", []),
            mode="lines",
            name="Train Loss",
            line=dict(color="#1f77b4"),
        )
    )
    fig.add_trace(
        go.Scatter(
            y=history_dict.get("val_loss", []),
            mode="lines",
            name="Validation Loss",
            line=dict(color="#d62728"),
        )
    )
    fig.update_layout(
        title="Training vs Validation Loss",
        xaxis_title="Epoch",
        yaxis_title="MSE Loss",
        template="plotly_white",
        height=360,
    )
    return fig


def _forecast_figure(forecast_df: pd.DataFrame, model_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["actual_return"],
            mode="lines",
            name="Actual Return",
            line=dict(color="#222222"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df[model_name],
            mode="lines",
            name=f"{model_name} Prediction",
            line=dict(color="#1f77b4"),
        )
    )
    fig.update_layout(
        title=f"Out-of-Sample Returns: Actual vs {model_name}",
        xaxis_title="Date",
        yaxis_title="Return",
        template="plotly_white",
        height=380,
    )
    return fig


def _equity_figure(backtest_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=backtest_df["date"],
            y=backtest_df["equity_curve"],
            mode="lines",
            name="Strategy Equity",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=backtest_df["date"],
            y=backtest_df["buy_hold_curve"],
            mode="lines",
            name="Buy & Hold Equity",
            line=dict(color="#ff7f0e", width=2),
        )
    )
    fig.update_layout(
        title="Cumulative Equity Curve",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        template="plotly_white",
        height=380,
    )
    return fig


def _drawdown_figure(backtest_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=backtest_df["date"],
            y=backtest_df["strategy_drawdown"],
            mode="lines",
            name="Strategy Drawdown",
            line=dict(color="#d62728"),
            fill="tozeroy",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=backtest_df["date"],
            y=backtest_df["buy_hold_drawdown"],
            mode="lines",
            name="Buy & Hold Drawdown",
            line=dict(color="#7f7f7f"),
            fill="tozeroy",
            opacity=0.5,
        )
    )
    fig.update_layout(
        title="Drawdown Comparison",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        template="plotly_white",
        height=360,
    )
    return fig


def _rolling_sharpe_figure(backtest_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=backtest_df["date"],
            y=backtest_df["rolling_sharpe_63"],
            mode="lines",
            name="63D Rolling Sharpe",
            line=dict(color="#2ca02c"),
        )
    )
    fig.update_layout(
        title="Rolling Sharpe (63 Trading Days)",
        xaxis_title="Date",
        yaxis_title="Sharpe",
        template="plotly_white",
        height=320,
    )
    return fig


def _price_reconstruction_figure(price_path_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=price_path_df["step"],
            y=price_path_df["realized_price"],
            mode="lines",
            name="Reconstructed Realized Price",
            line=dict(color="#111111"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=price_path_df["step"],
            y=price_path_df["predicted_price"],
            mode="lines",
            name="Reconstructed Predicted Price",
            line=dict(color="#1f77b4"),
        )
    )
    fig.update_layout(
        title="Price Path Reconstructed from Predicted vs Realized Returns",
        xaxis_title="Out-of-Sample Step",
        yaxis_title="Price",
        template="plotly_white",
        height=360,
    )
    return fig


def _run_pipeline(
    ticker: str,
    start_date: date,
    end_date: date,
    selected_features: List[str],
    include_benchmark: bool,
    benchmark_ticker: str,
    target_type: str,
    sequence_length: int,
    train_ratio: float,
    val_ratio: float,
    d_model: int,
    num_heads: int,
    ff_dim: int,
    num_blocks: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    use_learned_positional_encoding: bool,
    moving_average_window: int,
    include_arima: bool,
    walk_forward_linear: bool,
    linear_retrain_every: int,
    seed: int,
) -> Dict[str, object]:
    set_random_seed(seed)

    raw_df = download_price_data(
        ticker=ticker,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
    )

    feature_df = engineer_features(
        raw_df,
        include_benchmark_return=include_benchmark,
        benchmark_ticker=benchmark_ticker,
    )

    usable_features = [f for f in selected_features if f in AVAILABLE_FEATURES]
    if include_benchmark and "benchmark_log_return" not in usable_features:
        usable_features.append("benchmark_log_return")
    if not include_benchmark:
        usable_features = [f for f in usable_features if f != "benchmark_log_return"]

    model_df, feature_cols, target_col = build_model_frame(
        feature_df=feature_df,
        selected_features=usable_features,
        target_type=target_type,
    )

    split_data = make_train_val_test_sequences(
        model_df=model_df,
        feature_cols=feature_cols,
        target_col=target_col,
        sequence_length=sequence_length,
        split_config=SplitConfig(train_ratio=train_ratio, val_ratio=val_ratio),
    )

    transformer_config = TransformerConfig(
        sequence_length=sequence_length,
        num_features=len(feature_cols),
        d_model=d_model,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_transformer_blocks=num_blocks,
        dropout=dropout,
        learning_rate=learning_rate,
        use_learned_positional_encoding=use_learned_positional_encoding,
    )

    model = build_transformer_model(transformer_config)
    history = train_transformer(
        model,
        X_train=split_data["X_train"],
        y_train=split_data["y_train"],
        X_val=split_data["X_val"],
        y_val=split_data["y_val"],
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    y_test = split_data["y_test"]
    test_dates = split_data["dates_test"]
    aligned_frame = split_data["aligned_frame"]

    transformer_preds = predict_transformer(model, split_data["X_test"])
    naive_preds = naive_last_return_predictor(aligned_frame, test_dates, target_type)
    moving_avg_preds = moving_average_predictor(
        aligned_frame,
        test_dates,
        target_type,
        window=moving_average_window,
    )

    X_train_lr = np.concatenate([split_data["X_train"], split_data["X_val"]], axis=0)
    y_train_lr = np.concatenate([split_data["y_train"], split_data["y_val"]], axis=0)

    if walk_forward_linear:
        linear_preds = linear_regression_walk_forward(
            X_train=X_train_lr,
            y_train=y_train_lr,
            X_test=split_data["X_test"],
            y_test=y_test,
            retrain_every=linear_retrain_every,
        )
    else:
        linear_preds, _ = linear_regression_predictor(
            X_train=X_train_lr,
            y_train=y_train_lr,
            X_test=split_data["X_test"],
        )

    predictions: Dict[str, np.ndarray] = {
        "Transformer": transformer_preds,
        "Naive Last Return": naive_preds,
        "Moving Average": moving_avg_preds,
        "Linear Regression": linear_preds,
    }

    arima_status = "ARIMA not requested."
    if include_arima:
        arima_preds, arima_status = arima_predictor(
            train_target=y_train_lr,
            test_target=y_test,
            order=(1, 0, 0),
        )
        if arima_preds is not None:
            predictions["ARIMA(1,0,0)"] = arima_preds

    benchmark_table = build_benchmark_table(y_true=y_test, predictions=predictions)

    forecast_df = pd.DataFrame({"date": pd.to_datetime(test_dates), "actual_return": y_test})
    for model_name, preds in predictions.items():
        forecast_df[model_name] = preds

    first_test_date = pd.to_datetime(test_dates[0])
    start_price = float(aligned_frame.loc[first_test_date, "Close"])
    price_path_df = reconstruct_price_paths(
        start_price=start_price,
        predicted_returns=transformer_preds,
        realized_returns=y_test,
        target_type=target_type,
    )

    return {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "target_type": target_type,
        "feature_cols": feature_cols,
        "model_df_rows": len(model_df),
        "split_data": split_data,
        "history": history.history,
        "model_summary": model_summary_string(model),
        "predictions": predictions,
        "y_test": y_test,
        "test_dates": test_dates,
        "benchmark_table": benchmark_table,
        "forecast_df": forecast_df,
        "price_path_df": price_path_df,
        "transformer_metrics": compute_forecast_metrics(y_test, transformer_preds),
        "arima_status": arima_status,
        "transformer_config": transformer_config,
        "run_settings": {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": 1.0 - train_ratio - val_ratio,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "walk_forward_linear": walk_forward_linear,
            "moving_average_window": moving_average_window,
            "include_arima": include_arima,
            "seed": seed,
        },
    }


def main() -> None:
    st.set_page_config(
        page_title="Transformer Quant Finance Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Transformer-Based Financial Time-Series Forecasting and Backtesting Dashboard")
    st.caption(
        "Methodologically focused project: return forecasting, benchmark comparison, "
        "and transaction-cost-aware out-of-sample backtesting."
    )

    with st.sidebar:
        st.header("Data and Settings")
        ticker = st.text_input("Ticker", value="AAPL").strip().upper()

        default_end = date.today()
        default_start = default_end - timedelta(days=4 * 365)
        start_date = st.date_input("Start Date", value=default_start)
        end_date = st.date_input("End Date", value=default_end)

        include_benchmark = st.checkbox("Include benchmark return feature", value=True)
        benchmark_ticker = st.text_input("Benchmark ticker", value="^GSPC").strip().upper()

        selected_features = st.multiselect(
            "Feature selection",
            options=AVAILABLE_FEATURES,
            default=DEFAULT_FEATURES,
        )

        target_type = st.selectbox(
            "Target type",
            options=["log_return", "simple_return"],
            index=0,
            help="Default is next-day log return.",
        )

        sequence_length = st.slider("Sequence length", min_value=10, max_value=120, value=30)

        train_ratio = st.slider("Train proportion", min_value=0.50, max_value=0.85, value=0.70)
        val_ratio = st.slider("Validation proportion", min_value=0.05, max_value=0.30, value=0.15)

        walk_forward_linear = st.checkbox(
            "Walk-forward linear baseline", value=True, help="Expanding-window retraining for linear baseline."
        )
        linear_retrain_every = st.slider("Linear retrain frequency (days)", 1, 20, 5)

        st.header("Transformer Hyperparameters")
        d_model = st.slider("d_model", min_value=32, max_value=256, value=64, step=32)
        num_heads = st.selectbox("num_heads", options=[1, 2, 4, 8], index=2)
        ff_dim = st.slider("ff_dim", min_value=64, max_value=512, value=128, step=32)
        num_blocks = st.slider("Transformer blocks", min_value=1, max_value=3, value=2)
        dropout = st.slider("Dropout", min_value=0.0, max_value=0.5, value=0.1, step=0.05)
        batch_size = st.selectbox("Batch size", options=[16, 32, 64, 128], index=1)
        epochs = st.slider("Epochs", min_value=10, max_value=200, value=60, step=5)
        learning_rate = st.select_slider(
            "Learning rate", options=[1e-4, 3e-4, 5e-4, 1e-3, 2e-3], value=1e-3
        )
        use_learned_positional_encoding = st.checkbox(
            "Use learned positional encoding", value=False
        )

        st.header("Benchmarks")
        moving_average_window = st.slider("Moving-average window", 2, 30, 5)
        include_arima = st.checkbox("Include ARIMA baseline", value=False)

        st.header("Backtest Settings")
        backtest_mode = st.selectbox(
            "Strategy mode", options=["long_short", "long_cash"], index=0
        )
        positive_threshold = st.number_input(
            "Long threshold", min_value=-0.10, max_value=0.10, value=0.001, step=0.001
        )
        negative_threshold = st.number_input(
            "Short threshold", min_value=-0.10, max_value=0.10, value=-0.001, step=0.001
        )
        transaction_cost_bps = st.number_input(
            "Transaction cost (bps)", min_value=0.0, max_value=100.0, value=5.0, step=0.5
        )
        position_size = st.number_input(
            "Position size", min_value=0.0, max_value=5.0, value=1.0, step=0.1
        )

        seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)

        run_button = st.button("Run Full Pipeline", type="primary")

    if run_button:
        if train_ratio + val_ratio >= 0.95:
            st.error("Please leave at least 5% for test data.")
        elif end_date <= start_date:
            st.error("End date must be after start date.")
        elif not selected_features:
            st.error("Select at least one feature.")
        elif d_model % num_heads != 0:
            st.error("d_model must be divisible by num_heads.")
        else:
            try:
                with st.spinner("Running data pipeline, training Transformer, and benchmarking models..."):
                    results = _run_pipeline(
                        ticker=ticker,
                        start_date=start_date,
                        end_date=end_date,
                        selected_features=selected_features,
                        include_benchmark=include_benchmark,
                        benchmark_ticker=benchmark_ticker,
                        target_type=target_type,
                        sequence_length=sequence_length,
                        train_ratio=train_ratio,
                        val_ratio=val_ratio,
                        d_model=d_model,
                        num_heads=num_heads,
                        ff_dim=ff_dim,
                        num_blocks=num_blocks,
                        dropout=dropout,
                        batch_size=batch_size,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        use_learned_positional_encoding=use_learned_positional_encoding,
                        moving_average_window=moving_average_window,
                        include_arima=include_arima,
                        walk_forward_linear=walk_forward_linear,
                        linear_retrain_every=linear_retrain_every,
                        seed=int(seed),
                    )
                st.session_state["results"] = results
                st.success("Pipeline completed successfully.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Pipeline failed: {exc}")

    results = st.session_state.get("results")

    if results is None:
        st.info("Configure settings in the sidebar and click 'Run Full Pipeline'.")
        return

    st.header("1. Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ticker", results["ticker"])
    c2.metric("Target", results["target_type"])
    c3.metric("Feature Count", len(results["feature_cols"]))
    c4.metric("Model Rows", int(results["model_df_rows"]))

    st.markdown(
        "This dashboard evaluates one-step-ahead daily return forecasts using a Transformer and "
        "explicit benchmark models, then maps forecasts into an out-of-sample trading simulation."
    )

    st.header("2. Data and Settings")
    run_settings = results["run_settings"]
    config_df = pd.DataFrame(
        {
            "Setting": [
                "Date Range",
                "Features",
                "Sequence Length",
                "Train / Val / Test",
                "Transformer (d_model, heads, ff_dim, blocks)",
                "Dropout / Batch / Epochs / LR",
                "Walk-forward linear baseline",
            ],
            "Value": [
                f"{results['start_date']} to {results['end_date']}",
                ", ".join(results["feature_cols"]),
                results["transformer_config"].sequence_length,
                f"{run_settings['train_ratio']:.2f} / {run_settings['val_ratio']:.2f} / {run_settings['test_ratio']:.2f}",
                f"{results['transformer_config'].d_model}, {results['transformer_config'].num_heads}, "
                f"{results['transformer_config'].ff_dim}, {results['transformer_config'].num_transformer_blocks}",
                f"{results['transformer_config'].dropout}, {run_settings['batch_size']}, {run_settings['epochs']}, {run_settings['learning_rate']}",
                str(run_settings["walk_forward_linear"]),
            ],
        }
    )
    st.dataframe(config_df, use_container_width=True, hide_index=True)

    st.header("3. Model Training")
    st.plotly_chart(_loss_figure(results["history"]), use_container_width=True)

    with st.expander("Transformer architecture summary"):
        st.code(results["model_summary"], language="text")

    st.header("4. Forecast Evaluation")

    benchmark_table = results["benchmark_table"].copy()
    st.subheader("Benchmark Comparison Table")
    st.dataframe(
        benchmark_table.style.format(
            {
                "MSE": "{:.6f}",
                "MAE": "{:.6f}",
                "RMSE": "{:.6f}",
                "Directional Accuracy": "{:.2%}",
                "Sign Hit Rate": "{:.2%}",
                "Mean Predicted Return": "{:.5f}",
                "Mean Realized Return": "{:.5f}",
                "Bias (Pred-Realized)": "{:.5f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="Download benchmark table (CSV)",
        data=benchmark_table.to_csv(index=False),
        file_name="benchmark_comparison.csv",
        mime="text/csv",
    )

    st.caption(results.get("arima_status", ""))

    model_options = list(results["predictions"].keys())
    eval_model = st.selectbox("Model to visualize against actual returns", options=model_options)
    st.plotly_chart(
        _forecast_figure(results["forecast_df"], eval_model), use_container_width=True
    )

    transformer_metrics = results["transformer_metrics"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Transformer RMSE", _format_metric(transformer_metrics["RMSE"]))
    m2.metric(
        "Transformer Directional Accuracy",
        _format_metric(transformer_metrics["Directional Accuracy"], pct=True),
    )
    m3.metric("Transformer MAE", _format_metric(transformer_metrics["MAE"]))
    m4.metric(
        "Transformer Sign Hit Rate",
        _format_metric(transformer_metrics["Sign Hit Rate"], pct=True),
    )

    st.plotly_chart(_price_reconstruction_figure(results["price_path_df"]), use_container_width=True)

    st.header("5. Backtest Results")
    signal_model = st.selectbox("Signal model for backtest", options=model_options, index=0)

    backtest_config = BacktestConfig(
        mode=backtest_mode,
        positive_threshold=float(positive_threshold),
        negative_threshold=float(negative_threshold),
        transaction_cost_bps=float(transaction_cost_bps),
        position_size=float(position_size),
    )

    backtest_df, backtest_metrics = run_backtest(
        dates=results["test_dates"],
        predicted_returns=results["predictions"][signal_model],
        realized_returns=results["y_test"],
        target_type=results["target_type"],
        config=backtest_config,
    )

    metrics_table = pd.DataFrame(
        {"Metric": list(backtest_metrics.keys()), "Value": list(backtest_metrics.values())}
    )
    st.dataframe(
        metrics_table.style.format({"Value": "{:.6f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.plotly_chart(_equity_figure(backtest_df), use_container_width=True)
    st.plotly_chart(_drawdown_figure(backtest_df), use_container_width=True)
    st.plotly_chart(_rolling_sharpe_figure(backtest_df), use_container_width=True)

    st.download_button(
        label="Download backtest time series (CSV)",
        data=backtest_df.to_csv(index=False),
        file_name="backtest_results.csv",
        mime="text/csv",
    )

    st.caption(
        f"Thresholds used: long > {positive_threshold:.4f}, short < {negative_threshold:.4f}. "
        f"Mode: {backtest_mode}."
    )

    st.header("6. Project Report")
    report_md = generate_project_report_markdown(
        ticker=results["ticker"],
        start_date=str(results["start_date"]),
        end_date=str(results["end_date"]),
        target_type=results["target_type"],
        selected_features=results["feature_cols"],
        benchmark_table=benchmark_table,
        backtest_metrics=backtest_metrics,
        transformer_config={
            "d_model": results["transformer_config"].d_model,
            "num_heads": results["transformer_config"].num_heads,
            "ff_dim": results["transformer_config"].ff_dim,
            "num_transformer_blocks": results["transformer_config"].num_transformer_blocks,
            "dropout": results["transformer_config"].dropout,
            "learning_rate": results["transformer_config"].learning_rate,
        },
        walk_forward_enabled=run_settings["walk_forward_linear"],
    )
    st.markdown(report_md)

    st.markdown(default_limitations_markdown())

    st.download_button(
        label="Download project findings (Markdown)",
        data=report_md,
        file_name="project_findings.md",
        mime="text/markdown",
    )

    st.header("Why This Matters for Computational Finance")
    st.markdown(
        "- Combines sequence modeling with explicit benchmark-based statistical evaluation.\n"
        "- Separates predictive metrics from strategy outcomes and risk metrics.\n"
        "- Demonstrates leakage prevention, out-of-sample testing, and transaction-cost-aware backtesting.\n"
        "- Supports interview-level discussion on model choice, assumptions, and limitations."
    )


if __name__ == "__main__":
    main()
