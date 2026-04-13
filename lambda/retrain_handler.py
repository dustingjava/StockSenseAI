import boto3
import json
import os
import io
import tarfile
import tempfile
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────
BUCKET    = os.environ["BUCKET"]
ROLE_ARN  = os.environ["ROLE_ARN"]
REGION    = os.environ.get("REGION", "us-east-1")
TICKERS   = os.environ.get("TICKERS", "CRM,PLTR,NVDA").split(",")
THRESHOLD = float(os.environ.get("ACCURACY_THRESHOLD", "0.55"))
SNS_ARN   = os.environ.get("SNS_ARN", "")

s3  = boto3.client("s3",  region_name=REGION)
sns = boto3.client("sns", region_name=REGION)

# ── Manual StandardScaler ──────────────────────────────────────────────────
class StandardScaler:
    def fit_transform(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_  = np.std(X,  axis=0) + 1e-8
        return (X - self.mean_) / self.std_
    def transform(self, X):
        return (X - self.mean_) / self.std_

# ── Read data from S3 ──────────────────────────────────────────────────────
def load_from_s3(ticker):
    print(f"Loading {ticker} from S3...")
    s3_key  = f"training-data/{ticker}/latest.csv"
    obj     = s3.get_object(Bucket=BUCKET, Key=s3_key)
    df      = pd.read_csv(
        io.BytesIO(obj["Body"].read()),
        index_col=0,
        parse_dates=True
    )
    print(f"Loaded {len(df)} rows for {ticker}")
    return df

# ── Train one model ────────────────────────────────────────────────────────
def train_one(df, target_col):
    exclude      = ["open", "high", "low", "close", "volume",
                    "target_vol", "target_price", "target_rsi"]
    feature_cols = [c for c in df.columns if c not in exclude]
    data         = df[feature_cols + [target_col, "close"]].dropna()

    X = data[feature_cols].values
    y = data[target_col].values

    split           = int(len(data) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler      = StandardScaler()
    X_train_s   = scaler.fit_transform(X_train)
    X_test_s    = scaler.transform(X_test)

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = float(neg / pos) if pos > 0 else 1.0

    # Use XGBoost native API directly — no sklearn wrapper
    import xgboost as xgb

    dtrain = xgb.DMatrix(X_train_s, label=y_train)
    dtest  = xgb.DMatrix(X_test_s,  label=y_test)

    params = {
        "max_depth"        : 3,
        "learning_rate"    : 0.02,
        "subsample"        : 0.7,
        "colsample_bytree" : 0.7,
        "scale_pos_weight" : spw,
        "eval_metric"      : "logloss",
        "objective"        : "binary:logistic",
        "seed"             : 42,
        "verbosity"        : 0,
        "nthread"          : 1,
    }

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round = 300,
        evals           = [(dtest, "test")],
        verbose_eval    = False,
    )

    y_prob   = booster.predict(dtest)
    y_pred   = (y_prob > 0.5).astype(int)
    accuracy = float((y_pred == y_test).mean())
    baseline = float(y_test.mean())
    lift     = accuracy - baseline

    print(f"  accuracy={accuracy*100:.1f}%  "
          f"baseline={baseline*100:.1f}%  "
          f"lift={lift*100:+.1f}%")

    return booster, scaler, feature_cols, {
        "accuracy": accuracy,
        "baseline": baseline,
        "lift"    : lift,
    }

# ── Upload model to S3 ─────────────────────────────────────────────────────
def upload_model(booster, scaler, feature_cols,
                 metrics, ticker, model_name, timestamp):
    with tempfile.TemporaryDirectory() as tmpdir:
        booster.save_model(os.path.join(tmpdir, "model.json"))
        joblib.dump(scaler,       os.path.join(tmpdir, "scaler.pkl"))
        joblib.dump(feature_cols, os.path.join(tmpdir, "features.pkl"))
        joblib.dump(metrics,      os.path.join(tmpdir, "metrics.pkl"))

        tar_path = os.path.join(tmpdir, "model.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            for fname in ["model.json", "scaler.pkl",
                          "features.pkl", "metrics.pkl"]:
                tar.add(os.path.join(tmpdir, fname), arcname=fname)

        s3_key = (f"model-artifacts/lambda-retrain-{timestamp}"
                  f"/{ticker}/{model_name}/model.tar.gz")
        s3.upload_file(tar_path, BUCKET, s3_key)
        print(f"  Uploaded → s3://{BUCKET}/{s3_key}")
        return f"s3://{BUCKET}/{s3_key}"

# ── Main handler ───────────────────────────────────────────────────────────
def lambda_handler(event, context):
    timestamp    = datetime.now().strftime("%Y%m%d-%H%M%S")
    results      = {}
    drift_alerts = []
    errors       = []

    print(f"Starting retrain — {timestamp}")
    print(f"Tickers: {TICKERS}")

    for ticker in TICKERS:
        print(f"\nProcessing {ticker}...")
        try:
            df             = load_from_s3(ticker)
            ticker_results = {}

            for target_col, model_name in [
                ("target_vol",   "volatility"),
                ("target_price", "price"),
                ("target_rsi",   "rsi"),
            ]:
                print(f"  Training {model_name}...")
                booster, scaler, feature_cols, metrics = train_one(
                    df, target_col
                )
                s3_uri = upload_model(
                    booster, scaler, feature_cols,
                    metrics, ticker, model_name, timestamp
                )

                if metrics["accuracy"] < THRESHOLD:
                    drift_alerts.append({
                        "ticker"  : ticker,
                        "model"   : model_name,
                        "accuracy": metrics["accuracy"],
                    })

                ticker_results[model_name] = {
                    **metrics,
                    "s3_uri": s3_uri
                }

            results[ticker] = ticker_results

        except Exception as e:
            error_msg = f"Error processing {ticker}: {str(e)}"
            print(error_msg)
            errors.append(error_msg)

    # Send alerts
    if drift_alerts and SNS_ARN:
        lines = [f"StockSense AI — Drift Alert\nTimestamp: {timestamp}\n"]
        for a in drift_alerts:
            lines.append(f"  {a['ticker']}/{a['model']}: "
                         f"{a['accuracy']*100:.1f}%")
        sns.publish(
            TopicArn = SNS_ARN,
            Subject  = "StockSense AI — Drift Alert",
            Message  = "\n".join(lines)
        )

    if errors and SNS_ARN:
        sns.publish(
            TopicArn = SNS_ARN,
            Subject  = "StockSense AI — Training Errors",
            Message  = "\n".join(errors)
        )

    summary = {
        "timestamp"   : timestamp,
        "tickers"     : TICKERS,
        "drift_alerts": len(drift_alerts),
        "errors"      : len(errors),
        "results"     : {
            ticker: {
                model: f"{m['accuracy']*100:.1f}%"
                for model, m in ticker_results.items()
            }
            for ticker, ticker_results in results.items()
        },
    }

    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False) as f:
        json.dump(summary, f, indent=2)
        tmp_path = f.name

    s3.upload_file(
        tmp_path, BUCKET,
        f"retrain-summaries/retrain-{timestamp}.json"
    )

    print(f"\nDone:")
    print(json.dumps(summary, indent=2))
    return {"statusCode": 200, "body": json.dumps(summary)}