import boto3
import json
import joblib
import os
import tarfile
import tempfile
import numpy as np
import pandas as pd
import yfinance as yf
from ta import add_all_ta_features
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from datetime import datetime
import mlflow
import mlflow.xgboost
import warnings
warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────
ROLE_ARN  = "arn:aws:iam::448049787062:role/StockSenseAIRole"
BUCKET    = "stocksense-ai-448049787062"
REGION    = "us-east-1"
TICKER    = "AAPL"
PERIOD    = "5y"

s3        = boto3.client("s3", region_name=REGION)
sm_client = boto3.client("sagemaker", region_name=REGION)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
job_name  = f"stocksense-{TICKER.lower()}-{timestamp}"

print(f"StockSense AI — Local Training + S3 Upload")
print(f"Ticker  : {TICKER}  |  Period : {PERIOD}")
print(f"Job name: {job_name}")
print(f"Bucket  : s3://{BUCKET}")

# ── Feature engineering ────────────────────────────────────────────────────
def fetch_and_engineer(ticker, period):
    print(f"\nFetching {period} of {ticker} data...")
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
        "scale_pos_weight" : float(spw),
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
    print(f"Accuracy: {accuracy*100:.1f}%  "
          f"Baseline: {baseline*100:.1f}%  "
          f"Lift: {lift*100:+.1f}%")

    return model, scaler, feature_cols, {
        "accuracy": accuracy,
        "baseline": baseline,
        "lift"    : lift,
    }

# ── Upload file to S3 ──────────────────────────────────────────────────────
def upload_to_s3(local_path, s3_key):
    s3.upload_file(local_path, BUCKET, s3_key)
    print(f"Uploaded → s3://{BUCKET}/{s3_key}")

# ── Register model in SageMaker Model Registry ────────────────────────────
def register_model(model_s3_uri, model_name, metrics):
    try:
        # Create model package group if it doesn't exist
        try:
            sm_client.create_model_package_group(
                ModelPackageGroupName        = model_name,
                ModelPackageGroupDescription = f"StockSense AI {model_name}"
            )
            print(f"Created model package group: {model_name}")
        except sm_client.exceptions.ClientError:
            pass  # group already exists

        # Register model package
        response = sm_client.create_model_package(
            ModelPackageGroupName    = model_name,
            ModelPackageDescription  = (
                f"Ticker: {TICKER} | "
                f"Accuracy: {metrics['accuracy']*100:.1f}% | "
                f"Lift: {metrics['lift']*100:+.1f}%"
            ),
            InferenceSpecification   = {
                "Containers": [{
                    "Image"          : "683313688378.dkr.ecr.us-east-1.amazonaws.com"
                                       "/sagemaker-scikit-learn:1.2-1-cpu-py3",
                    "ModelDataUrl"   : model_s3_uri,
                }],
                "SupportedContentTypes"    : ["text/csv"],
                "SupportedResponseMIMETypes": ["text/csv"],
            },
            ModelApprovalStatus      = "Approved",
            CustomerMetadataProperties = {
                "accuracy": f"{metrics['accuracy']*100:.1f}",
                "baseline": f"{metrics['baseline']*100:.1f}",
                "lift"    : f"{metrics['lift']*100:.1f}",
                "ticker"  : TICKER,
            }
        )
        print(f"Registered in Model Registry: {response['ModelPackageArn']}")
        return response["ModelPackageArn"]
    except Exception as e:
        print(f"Model registry note: {e}")
        return None

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Configure MLflow — store experiments locally
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(f"stocksense-{TICKER.lower()}")

    # Fetch data
    df = fetch_and_engineer(TICKER, PERIOD)

    all_metrics = {}
    model_arns  = {}

    with mlflow.start_run(run_name=f"{TICKER}-{timestamp}") as run:
        mlflow.log_param("ticker",   TICKER)
        mlflow.log_param("period",   PERIOD)
        mlflow.log_param("job_name", job_name)

        for target_col, model_name in [
            ("target_vol",   f"stocksense-{TICKER.lower()}-volatility"),
            ("target_price", f"stocksense-{TICKER.lower()}-price"),
            ("target_rsi",   f"stocksense-{TICKER.lower()}-rsi"),
        ]:
            # Train
            model, scaler, feature_cols, metrics = train_one(
                df, target_col, model_name
            )

            # Log to MLflow
            mlflow.log_metric(f"{model_name}_accuracy", metrics["accuracy"])
            mlflow.log_metric(f"{model_name}_baseline", metrics["baseline"])
            mlflow.log_metric(f"{model_name}_lift",     metrics["lift"])
            mlflow.xgboost.log_model(model, artifact_path=model_name)

            # Save artifacts to temp dir then tar.gz them
            with tempfile.TemporaryDirectory() as tmpdir:
                model.save_model(os.path.join(tmpdir, "model.json"))
                joblib.dump(scaler,
                            os.path.join(tmpdir, "scaler.pkl"))
                joblib.dump(feature_cols,
                            os.path.join(tmpdir, "features.pkl"))
                joblib.dump(metrics,
                            os.path.join(tmpdir, "metrics.pkl"))

                # Create model.tar.gz (SageMaker standard format)
                tar_path = os.path.join(tmpdir, "model.tar.gz")
                with tarfile.open(tar_path, "w:gz") as tar:
                    for fname in ["model.json", "scaler.pkl",
                                  "features.pkl", "metrics.pkl"]:
                        tar.add(os.path.join(tmpdir, fname),
                                arcname=fname)

                # Upload to S3
                s3_key = (f"model-artifacts/{job_name}"
                          f"/{model_name}/model.tar.gz")
                upload_to_s3(tar_path, s3_key)
                model_s3_uri = f"s3://{BUCKET}/{s3_key}"

            # Register in SageMaker Model Registry
            arn = register_model(model_s3_uri, model_name, metrics)
            if arn:
                model_arns[model_name] = arn

            all_metrics[model_name] = metrics

        # Save and upload summary
        summary = {
            "ticker"    : TICKER,
            "period"    : PERIOD,
            "job_name"  : job_name,
            "timestamp" : timestamp,
            "run_id"    : run.info.run_id,
            "models"    : all_metrics,
            "model_arns": model_arns,
        }

        summary_path = f"summary-{timestamp}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        upload_to_s3(summary_path, f"summaries/{summary_path}")
        mlflow.log_artifact(summary_path)

        print(f"\n{'='*55}")
        print(f"Training complete — {TICKER}")
        print(f"MLflow run ID : {run.info.run_id}")
        print(f"S3 artifacts  : s3://{BUCKET}/model-artifacts/{job_name}/")
        print(f"{'='*55}")
        print("\nModel summary:")
        for name, m in all_metrics.items():
            print(f"  {name:45s} "
                  f"acc={m['accuracy']*100:.1f}%  "
                  f"lift={m['lift']*100:+.1f}%")

    print(f"\nView MLflow UI:")
    print(f"  mlflow ui --backend-store-uri mlruns")
    print(f"  Then open http://localhost:5000")

    print(f"\nView models in SageMaker Registry:")
    print(f"  https://us-east-1.console.aws.amazon.com/sagemaker/home"
          f"?region=us-east-1#/model-registry")