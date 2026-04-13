import os
import json
import boto3
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from ta import add_all_ta_features
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import mlflow
import mlflow.xgboost
import warnings
warnings.filterwarnings("ignore")

# ── SageMaker passes hyperparameters as environment variables ─────────────
TICKER   = os.environ.get("TICKER",   "AAPL")
PERIOD   = os.environ.get("PERIOD",   "5y")
ROLE_ARN = os.environ.get("ROLE_ARN", "")

# SageMaker writes model artifacts here — do not change this path
MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

print(f"Training StockSense AI for ticker: {TICKER}")
print(f"History period: {PERIOD}")

# ── Fetch and engineer features ────────────────────────────────────────────
def fetch_and_engineer(ticker, period):
    print(f"Fetching {period} of {ticker} data...")
    raw = yf.download(ticker, period=period,
                      interval="1d", auto_adjust=True)
    raw.columns = raw.columns.droplevel(1)
    raw.columns = [c.lower() for c in raw.columns]
    raw = add_all_ta_features(
        raw, open="open", high="high",
        low="low", close="close", volume="volume",
        fillna=True
    )

    close  = raw["close"]
    volume = raw["volume"]

    eng = pd.DataFrame(index=raw.index)
    eng["return_1d"]       = close.pct_change()
    eng["return_5d"]       = close.pct_change(5)
    eng["return_10d"]      = close.pct_change(10)
    eng["return_20d"]      = close.pct_change(20)
    eng["price_vs_sma20"]  = close / raw["trend_sma_fast"] - 1
    eng["price_vs_sma50"]  = close / raw["trend_sma_slow"] - 1
    eng["vol_ratio"]       = volume / volume.rolling(20).mean()
    eng["rsi_trend"]       = raw["momentum_rsi"] - raw["momentum_rsi"].shift(5)
    eng["macd_hist_trend"] = raw["trend_macd_diff"] - raw["trend_macd_diff"].shift(3)
    eng["volatility_20d"]  = eng["return_1d"].rolling(20).std()

    for lag in [1, 2, 3, 5]:
        eng[f"rsi_lag{lag}"]    = raw["momentum_rsi"].shift(lag)
        eng[f"macd_lag{lag}"]   = raw["trend_macd"].shift(lag)
        eng[f"ret_lag{lag}"]    = eng["return_1d"].shift(lag)
        eng[f"volume_lag{lag}"] = eng["vol_ratio"].shift(lag)

    df = pd.concat([raw, eng], axis=1).copy()
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Three targets
    future_vol         = eng["return_1d"].shift(-1).rolling(5).std()
    df["target_vol"]   = (future_vol > future_vol.rolling(60).mean()).astype(int)
    df["target_price"] = (close.shift(-5) > close).astype(int)
    rsi                = raw["momentum_rsi"]
    df["target_rsi"]   = ((rsi.shift(-5) > 50) & (rsi <= 50)).astype(int)
    df["close"]        = close

    return df

# ── Train one model ────────────────────────────────────────────────────────
def train_one(df, target_col, model_name):
    exclude      = ["open", "high", "low", "close", "volume",
                    "target_vol", "target_price", "target_rsi"]
    feature_cols = [c for c in df.columns if c not in exclude]
    data         = df[feature_cols + [target_col, "close"]].dropna()

    X = data[feature_cols]
    y = data[target_col]

    split           = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split],  X.iloc[split:]
    y_train, y_test = y.iloc[:split],  y.iloc[split:]

    scaler      = StandardScaler()
    X_train_s   = scaler.fit_transform(X_train)
    X_test_s    = scaler.transform(X_test)

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos if pos > 0 else 1

    params = {
        "n_estimators"     : 500,
        "max_depth"        : 3,
        "learning_rate"    : 0.02,
        "subsample"        : 0.7,
        "colsample_bytree" : 0.7,
        "scale_pos_weight" : spw,
        "random_state"     : 42,
        "eval_metric"      : "logloss",
        "verbosity"        : 0,
    }

    model = XGBClassifier(**params)
    model.fit(X_train_s, y_train,
              eval_set=[(X_test_s, y_test)],
              verbose=False)

    y_pred   = model.predict(X_test_s)
    accuracy = float((y_pred == y_test).mean())
    baseline = float(y_test.mean())
    lift     = accuracy - baseline

    print(f"\n── {model_name} ──────────────────────────────────")
    print(classification_report(y_test, y_pred,
                                target_names=["NO", "YES"], zero_division=0))
    print(f"Accuracy: {accuracy*100:.1f}%  Baseline: {baseline*100:.1f}%  "
          f"Lift: {lift*100:+.1f}%")

    return model, scaler, feature_cols, {
        "accuracy": accuracy,
        "baseline": baseline,
        "lift"    : lift,
        "params"  : params,
    }

# ── Main training flow ─────────────────────────────────────────────────────
if __name__ == "__main__":

    # Fetch data once — reuse for all three models
    df = fetch_and_engineer(TICKER, PERIOD)

    # Configure MLflow — logs to S3 via environment variable if set
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)

    mlflow.set_experiment(f"stocksense-{TICKER.lower()}")

    all_metrics = {}

    # Train all three models inside a single MLflow run
    with mlflow.start_run(run_name=f"{TICKER}-{PERIOD}") as run:

        # Log run-level params
        mlflow.log_param("ticker", TICKER)
        mlflow.log_param("period", PERIOD)

        for target_col, model_name in [
            ("target_vol",   "volatility_model"),
            ("target_price", "price_model"),
            ("target_rsi",   "rsi_model"),
        ]:
            model, scaler, feature_cols, metrics = train_one(
                df, target_col, model_name
            )

            # Log metrics to MLflow
            mlflow.log_metric(f"{model_name}_accuracy", metrics["accuracy"])
            mlflow.log_metric(f"{model_name}_baseline", metrics["baseline"])
            mlflow.log_metric(f"{model_name}_lift",     metrics["lift"])

            # Log model to MLflow
            mlflow.xgboost.log_model(model, artifact_path=model_name)

            # Save model + scaler + feature list to SageMaker model dir
            model_path = os.path.join(MODEL_DIR, model_name)
            os.makedirs(model_path, exist_ok=True)

            model.save_model(os.path.join(model_path, "model.json"))
            joblib.dump(scaler,       os.path.join(model_path, "scaler.pkl"))
            joblib.dump(feature_cols, os.path.join(model_path, "features.pkl"))

            all_metrics[model_name] = metrics
            print(f"Saved {model_name} to {model_path}")

        # Save combined metrics summary
        summary = {
            "ticker"  : TICKER,
            "period"  : PERIOD,
            "models"  : all_metrics,
            "run_id"  : run.info.run_id,
        }
        summary_path = os.path.join(MODEL_DIR, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        mlflow.log_artifact(summary_path)
        print(f"\nAll models saved. MLflow run ID: {run.info.run_id}")
        print(json.dumps(
            {k: {m: f"{v*100:.1f}%" for m, v in metrics.items()
                 if isinstance(v, float)}
             for k, metrics in all_metrics.items()},
            indent=2
        ))