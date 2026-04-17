import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import boto3
import joblib
import tempfile
import os
import json
import tarfile
import io
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from utils import StandardScaler
from bedrock_explainer import explain_signal
from xgboost import XGBClassifier
import shap
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSense AI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 StockSense AI")
st.caption("Powered by XGBoost + AWS Lambda daily retraining pipeline")

# ── Config ─────────────────────────────────────────────────────────────────
BUCKET = os.environ.get("S3_BUCKET", "stocksense-ai-448049787062")
REGION = "us-east-1"

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    ticker   = st.text_input("Ticker symbol", value="CRM").upper()
    conf_thr = st.slider("Confidence threshold", 0.50, 0.90, 0.60, 0.05)
    st.caption("Only show signals above this confidence level")
    run_btn  = st.button("Run Analysis", type="primary",
                         use_container_width=True)
    st.divider()
    st.caption("Models retrained daily at 4:30pm ET via AWS Lambda")

# ── S3 helper ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_s3_client():
    return boto3.client("s3", region_name=REGION)

# ── Find latest model in S3 ────────────────────────────────────────────────
def find_latest_model_prefix(ticker):
    """Find the most recent Lambda retrain folder for this ticker."""
    s3 = get_s3_client()
    response = s3.list_objects_v2(
        Bucket=BUCKET,
        Prefix=f"model-artifacts/lambda-retrain-"
    )
    if "Contents" not in response:
        return None

    # Get all unique retrain timestamps
    prefixes = set()
    for obj in response["Contents"]:
        parts = obj["Key"].split("/")
        if len(parts) >= 2:
            prefixes.add(parts[1])

    if not prefixes:
        return None

    # Get latest timestamp
    latest = sorted(prefixes)[-1]
    return f"model-artifacts/{latest}/{ticker}"

# ── Load model from S3 ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_from_s3(ticker, model_name):
    """Download and load a trained model from S3."""
    s3      = get_s3_client()
    prefix  = find_latest_model_prefix(ticker)

    if not prefix:
        return None, None, None, None

    s3_key  = f"{prefix}/{model_name}/model.tar.gz"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = os.path.join(tmpdir, "model.tar.gz")
            s3.download_file(BUCKET, s3_key, tar_path)

            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(tmpdir)

            import xgboost as xgb
            booster = xgb.Booster()
            booster.load_model(os.path.join(tmpdir, "model.json"))
            scaler       = joblib.load(os.path.join(tmpdir, "scaler.pkl"))
            feature_cols = joblib.load(os.path.join(tmpdir, "features.pkl"))
            metrics      = joblib.load(os.path.join(tmpdir, "metrics.pkl"))

            return booster, scaler, feature_cols, metrics

    except Exception as e:
        st.warning(f"Could not load {model_name} model for {ticker}: {e}")
        return None, None, None, None

# ── Fetch and engineer features ────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_and_engineer(ticker):
    raw = yf.download(ticker, period="2y",
                      interval="1d", auto_adjust=True)
    if raw.empty:
        return None

    raw.columns = raw.columns.droplevel(1)
    raw.columns = [c.lower() for c in raw.columns]

    close  = raw["close"]
    high   = raw["high"]
    low    = raw["low"]
    volume = raw["volume"]

    df = pd.DataFrame(index=raw.index)
    df["close"]  = close
    df["volume"] = volume
    df["high"]   = high
    df["low"]    = low
    df["open"]   = raw["open"]

    # Indicators
    df["momentum_rsi"]          = RSIIndicator(close).rsi()
    stoch                       = StochasticOscillator(high, low, close)
    df["momentum_stoch"]        = stoch.stoch()
    df["momentum_stoch_signal"] = stoch.stoch_signal()
    df["momentum_wr"]           = WilliamsRIndicator(high, low, close).williams_r()
    df["momentum_roc"]          = ROCIndicator(close).roc()
    macd                        = MACD(close)
    df["trend_macd"]            = macd.macd()
    df["trend_macd_signal"]     = macd.macd_signal()
    df["trend_macd_diff"]       = macd.macd_diff()
    df["trend_sma_fast"]        = SMAIndicator(close, window=20).sma_indicator()
    df["trend_sma_slow"]        = SMAIndicator(close, window=50).sma_indicator()
    df["trend_ema_fast"]        = EMAIndicator(close, window=12).ema_indicator()
    df["trend_ema_slow"]        = EMAIndicator(close, window=26).ema_indicator()
    df["trend_adx"]             = ADXIndicator(high, low, close).adx()
    bb                          = BollingerBands(close)
    df["volatility_bbh"]        = bb.bollinger_hband()
    df["volatility_bbl"]        = bb.bollinger_lband()
    df["volatility_bbm"]        = bb.bollinger_mavg()
    df["volatility_bbw"]        = bb.bollinger_wband()
    df["volatility_atr"]        = AverageTrueRange(high, low, close).average_true_range()
    df["volume_obv"]            = OnBalanceVolumeIndicator(close, volume).on_balance_volume()

    # Engineered features
    df["return_1d"]       = close.pct_change()
    df["return_5d"]       = close.pct_change(5)
    df["return_10d"]      = close.pct_change(10)
    df["return_20d"]      = close.pct_change(20)
    df["price_vs_sma20"]  = close / df["trend_sma_fast"] - 1
    df["price_vs_sma50"]  = close / df["trend_sma_slow"] - 1
    df["vol_ratio"]       = volume / volume.rolling(20).mean()
    df["rsi_trend"]       = df["momentum_rsi"] - df["momentum_rsi"].shift(5)
    df["macd_hist_trend"] = df["trend_macd_diff"] - df["trend_macd_diff"].shift(3)
    df["volatility_20d"]  = df["return_1d"].rolling(20).std()

    for lag in [1, 2, 3, 5]:
        df[f"rsi_lag{lag}"]    = df["momentum_rsi"].shift(lag)
        df[f"macd_lag{lag}"]   = df["trend_macd"].shift(lag)
        df[f"ret_lag{lag}"]    = df["return_1d"].shift(lag)
        df[f"volume_lag{lag}"] = df["vol_ratio"].shift(lag)

    return df

# ── Predict using loaded booster ───────────────────────────────────────────
def predict_today(df, booster, scaler, feature_cols):
    """Run prediction on the most recent day."""
    import xgboost as xgb
    available = [c for c in feature_cols if c in df.columns]
    latest    = df[available].dropna().iloc[[-1]]
    scaled    = scaler.transform(latest.values)
    dmatrix   = xgb.DMatrix(scaled)
    prob      = float(booster.predict(dmatrix)[0])
    return prob, latest.index[0]

# ── Signal box helper ──────────────────────────────────────────────────────
def signal_box(label, prob, conf_thr, pos_label, neg_label):
    confidence = max(prob, 1 - prob)
    is_conf    = confidence >= conf_thr
    is_pos     = prob >= 0.5
    if not is_conf:
        st.warning(f"⚪ **{label}** — Uncertain "
                   f"({confidence*100:.1f}% < {conf_thr*100:.0f}% threshold)")
    elif is_pos:
        st.error(f"🔴 **{label}** → {pos_label} — {prob*100:.1f}% confidence")
    else:
        st.success(f"🟢 **{label}** → {neg_label} — "
                   f"{(1-prob)*100:.1f}% confidence")

def get_indicator_snapshot(df):
    latest = df.dropna().iloc[-1]
    return {
        "rsi"           : f"{latest.get('momentum_rsi', 0):.1f}",
        "macd_signal"   : "bullish" if latest.get("trend_macd", 0) > latest.get("trend_macd_signal", 0) else "bearish",
        "bb_width"      : f"{latest.get('volatility_bbw', 0):.3f}",
        "vol_ratio"     : f"{latest.get('vol_ratio', 1):.2f}",
        "price_vs_sma20": f"{latest.get('price_vs_sma20', 0)*100:.1f}",
    }

def get_shap_factors(booster, scaler, feature_cols, df):
    try:
        import shap
        import xgboost as xgb
        available   = [c for c in feature_cols if c in df.columns]
        latest      = df[available].dropna().iloc[[-1]]
        scaled      = scaler.transform(latest.values)
        explainer   = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(scaled)[0]
        factors = []
        for i, col in enumerate(available):
            factors.append({
                "indicator": col,
                "impact"   : float(shap_values[i]),
                "direction": "pushes HIGH vol" if shap_values[i] > 0 else "pushes LOW vol"
            })
        return sorted(factors, key=lambda x: abs(x["impact"]), reverse=True)[:5]
    except Exception:
        return []
# ═════════════════════════════════════════════════════════════════════════════
# Main app
# ═════════════════════════════════════════════════════════════════════════════
if run_btn:

    # ── Load models from S3 ────────────────────────────────────────────────
    with st.spinner(f"Loading {ticker} models from S3..."):
        booster_vol,   scaler_vol,   features_vol,   metrics_vol   = load_model_from_s3(ticker, "volatility")
        booster_price, scaler_price, features_price, metrics_price = load_model_from_s3(ticker, "price")
        booster_rsi,   scaler_rsi,   features_rsi,   metrics_rsi   = load_model_from_s3(ticker, "rsi")

    # Check if models exist for this ticker
    if booster_vol is None:
        st.error(f"No trained models found for **{ticker}** in S3.")
        st.info(f"Models available for: CRM, PLTR, NVDA — "
                f"or run `fetch_and_upload.py` locally to add {ticker}.")
        st.stop()

    # ── Fetch recent data for charts and prediction ────────────────────────
    with st.spinner(f"Fetching {ticker} market data..."):
        df = fetch_and_engineer(ticker)

    if df is None or df.empty:
        st.error(f"Could not fetch data for '{ticker}'.")
        st.stop()

    # ── Today's predictions ────────────────────────────────────────────────
    prob_vol,   pred_date = predict_today(df, booster_vol,
                                          scaler_vol,   features_vol)
    prob_price, _         = predict_today(df, booster_price,
                                          scaler_price, features_price)
    prob_rsi,   _         = predict_today(df, booster_rsi,
                                          scaler_rsi,   features_rsi)

    last_date  = pred_date.strftime("%b %d, %Y")
    signals    = [prob_vol >= 0.5, prob_price >= 0.5, prob_rsi >= 0.5]
    agreement  = sum(signals)

    st.divider()

    # ── Metric cards ───────────────────────────────────────────────────────
    st.subheader(f"Today's Signals — {ticker}  ({last_date})")

    if metrics_vol:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Volatility accuracy",
                  f"{metrics_vol['accuracy']*100:.1f}%",
                  f"+{metrics_vol['lift']*100:.1f}% lift")
        m2.metric("Price accuracy",
                  f"{metrics_price['accuracy']*100:.1f}%",
                  f"+{metrics_price['lift']*100:.1f}% lift")
        m3.metric("RSI accuracy",
                  f"{metrics_rsi['accuracy']*100:.1f}%",
                  f"+{metrics_rsi['lift']*100:.1f}% lift")
        m4.metric("Models trained",
                  "Daily",
                  "AWS Lambda")

    # ── Signal boxes ───────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        signal_box("Volatility (5d)", prob_vol,
                   conf_thr, "HIGH volatility", "LOW volatility")
    with col2:
        signal_box("Price direction (5d)", prob_price,
                   conf_thr, "Price UP", "Price DOWN")
    with col3:
        signal_box("RSI momentum (5d)", prob_rsi,
                   conf_thr, "RSI bullish", "RSI bearish")

    # ── Ensemble agreement ─────────────────────────────────────────────────
    st.divider()
    if agreement == 3:
        st.success("✅ **Strong ensemble agreement** — all three models agree.")
    elif agreement == 2:
        st.info("📊 **Partial agreement** — two of three models agree.")
    else:
        st.warning("⚠️ **No agreement** — models conflict. Stay cautious.")


    # ── LLM Explanation ────────────────────────────────────────────────────
    st.divider()
    st.subheader("What does this mean? (AI Explanation)")

    with st.spinner("Generating plain English explanation via Amazon Bedrock..."):
        indicators   = get_indicator_snapshot(df)
        shap_factors = get_shap_factors(
            booster_vol, scaler_vol, features_vol, df
        )
        explanation  = explain_signal(
            ticker     = ticker,
            prob_vol   = prob_vol,
            prob_price = prob_price,
            prob_rsi   = prob_rsi,
            indicators = indicators,
            shap_factors = shap_factors
        )

    st.info(explanation)

    # Show which indicators drove the prediction
    if shap_factors:
        st.caption("Top factors driving the volatility prediction:")
        cols = st.columns(len(shap_factors[:5]))
        for i, factor in enumerate(shap_factors[:5]):
            with cols[i]:
                color = "🔴" if factor["impact"] > 0 else "🟢"
                st.metric(
                    label = factor["indicator"].replace("_", " "),
                    value = f"{abs(factor['impact']):.3f}",
                    delta = "↑ HIGH vol" if factor["impact"] > 0 else "↓ LOW vol"
                )

    # ── Tabs ───────────────────────────────────────────────────────────────
    st.divider()
    tab1, tab2 = st.tabs(["Price & Indicators", "Model Info"])

    with tab1:
        st.subheader("Technical indicators — last 90 days")
        last_90 = df.tail(90)

        fig = plt.figure(figsize=(13, 12))
        gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.5)

        # Price + Bollinger Bands
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(last_90.index, last_90["close"],
                 color="black", linewidth=1, label="Close")
        ax1.plot(last_90.index, last_90["volatility_bbh"],
                 color="red",   linestyle="--",
                 linewidth=0.8, label="BB Upper")
        ax1.plot(last_90.index, last_90["volatility_bbl"],
                 color="green", linestyle="--",
                 linewidth=0.8, label="BB Lower")
        ax1.fill_between(last_90.index,
                         last_90["volatility_bbh"],
                         last_90["volatility_bbl"],
                         alpha=0.05, color="gray")
        ax1.set_title("Price + Bollinger Bands")
        ax1.legend(fontsize=8)

        # RSI
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(last_90.index, last_90["momentum_rsi"],
                 color="purple", linewidth=1)
        ax2.axhline(70, color="red",   linestyle="--",
                    linewidth=0.8, label="Overbought (70)")
        ax2.axhline(50, color="gray",  linestyle="--",
                    linewidth=0.6, label="Midline (50)")
        ax2.axhline(30, color="green", linestyle="--",
                    linewidth=0.8, label="Oversold (30)")
        ax2.fill_between(last_90.index,
                         last_90["momentum_rsi"], 70,
                         where=(last_90["momentum_rsi"] >= 70),
                         alpha=0.3, color="red")
        ax2.fill_between(last_90.index,
                         last_90["momentum_rsi"], 30,
                         where=(last_90["momentum_rsi"] <= 30),
                         alpha=0.3, color="green")
        ax2.set_ylim(0, 100)
        ax2.set_title("RSI (14)")
        ax2.legend(fontsize=8)

        # MACD
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(last_90.index, last_90["trend_macd"],
                 color="blue", linewidth=1, label="MACD")
        ax3.plot(last_90.index, last_90["trend_macd_signal"],
                 color="red",  linewidth=1, label="Signal")
        ax3.bar(last_90.index, last_90["trend_macd_diff"],
                color=["green" if v >= 0 else "red"
                       for v in last_90["trend_macd_diff"]],
                alpha=0.5)
        ax3.axhline(0, color="black", linewidth=0.5)
        ax3.set_title("MACD")
        ax3.legend(fontsize=8)

        # BB Width
        ax4 = fig.add_subplot(gs[3])
        ax4.plot(last_90.index, last_90["volatility_bbw"],
                 color="darkorange", linewidth=1)
        ax4.set_title("Bollinger Band Width (volatility)")

        st.pyplot(fig)
        plt.close()

    with tab2:
        st.subheader("Model information")
        st.markdown(f"""
        **Tickers with trained models:** CRM · PLTR · NVDA

        **Retraining schedule:** Every weekday at 4:30pm ET via AWS Lambda

        **Data pipeline:**
        - GitHub Actions fetches fresh data at 4:00pm ET
        - Uploads feature-engineered CSV to S3
        - Lambda reads CSV and trains 9 models
        - Models saved as `model.tar.gz` in S3

        **Model details:**
        - Algorithm: XGBoost (native API — no sklearn)
        - Features: 30+ technical indicators + engineered features
        - Training window: 5 years of daily OHLCV data
        - Train/test split: 80/20 time-series split
        """)

        if metrics_vol:
            st.subheader("Latest model metrics")
            metrics_data = {
                "Model"    : ["Volatility", "Price Direction", "RSI Momentum"],
                "Accuracy" : [f"{metrics_vol['accuracy']*100:.1f}%",
                              f"{metrics_price['accuracy']*100:.1f}%",
                              f"{metrics_rsi['accuracy']*100:.1f}%"],
                "Baseline" : [f"{metrics_vol['baseline']*100:.1f}%",
                              f"{metrics_price['baseline']*100:.1f}%",
                              f"{metrics_rsi['baseline']*100:.1f}%"],
                "Lift"     : [f"+{metrics_vol['lift']*100:.1f}%",
                              f"+{metrics_price['lift']*100:.1f}%",
                              f"+{metrics_rsi['lift']*100:.1f}%"],
            }
            st.dataframe(pd.DataFrame(metrics_data),
                         use_container_width=True,
                         hide_index=True)

else:
    st.info("Enter a ticker in the sidebar and click **Run Analysis**.")
    st.markdown("""
    **Available tickers:** CRM · PLTR · NVDA

    **How it works:**
    - Models are retrained daily at 4:30pm ET via AWS Lambda
    - Dashboard loads pre-trained models from S3 — no retraining needed
    - Predictions are instant — sub-second response

    **What each signal means:**
    - 🔴 **High volatility** — expect large price swings in next 5 days
    - 🟢 **Low volatility** — expect calm price action in next 5 days
    - 🟢 **Price UP** — model predicts price higher in 5 days
    - 🟢 **RSI bullish** — momentum turning positive
    """)

