import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ta import add_all_ta_features
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
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
st.caption("Three-model ensemble: volatility regime · price direction · RSI momentum")

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    ticker   = st.text_input("Ticker symbol", value="AAPL").upper()
    period   = st.selectbox("Training history", ["3y", "5y", "7y"], index=1)
    conf_thr = st.slider("Confidence threshold", 0.50, 0.90, 0.60, 0.05)
    st.caption("Only show signals above this confidence level")
    run_btn  = st.button("Run Analysis", type="primary", use_container_width=True)

# ── Data fetching & feature engineering ───────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_and_engineer(ticker, period):
    raw = yf.download(ticker, period=period, interval="1d", auto_adjust=True)
    if raw.empty:
        return None
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

    # ── Three targets ──────────────────────────────────────────────────────
    # 1. Volatility regime — will next 5 days be more volatile than usual?
    future_vol        = eng["return_1d"].shift(-1).rolling(5).std()
    df["target_vol"]  = (future_vol > future_vol.rolling(60).mean()).astype(int)

    # 2. Price direction — will price be higher in 5 days?
    df["target_price"] = (close.shift(-5) > close).astype(int)

    # 3. RSI momentum — will RSI cross above 50 in 5 days? (momentum turning bullish)
    rsi = raw["momentum_rsi"]
    df["target_rsi"] = (
            (rsi.shift(-5) > 50) & (rsi <= 50)
    ).astype(int)

    return df

# ── Model training ────────────────────────────────────────────────────────
def train_model(df, target_col):
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

    model = XGBClassifier(
        n_estimators=500, max_depth=3, learning_rate=0.02,
        subsample=0.7,    colsample_bytree=0.7,
        scale_pos_weight=spw,
        random_state=42,  eval_metric="logloss", verbosity=0
    )
    model.fit(X_train_s, y_train,
              eval_set=[(X_test_s, y_test)], verbose=False)

    y_pred      = model.predict(X_test_s)
    y_prob      = model.predict_proba(X_test_s)[:, 1]
    accuracy    = (y_pred == y_test).mean()
    baseline    = y_test.mean()
    test_prices = data["close"].loc[X_test.index]

    importance_df = pd.DataFrame({
        "feature"   : feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).head(10)

    return {
        "model"        : model,
        "scaler"       : scaler,
        "feature_cols" : feature_cols,
        "X_train"      : X_train,
        "X_test"       : X_test,
        "y_test"       : y_test,
        "y_pred"       : y_pred,
        "y_prob"       : y_prob,
        "accuracy"     : accuracy,
        "baseline"     : baseline,
        "lift"         : accuracy - baseline,
        "test_prices"  : test_prices,
        "importance"   : importance_df,
        "data"         : data,
        "X_train_s"    : X_train_s,
        "X_test_s"     : X_test_s,
    }

# ── SHAP explanation for today ─────────────────────────────────────────────
def explain_today(res, df, label_positive, label_negative):
    feature_cols   = res["feature_cols"]
    latest_row     = df[feature_cols].dropna().iloc[[-1]]
    latest_scaled  = res["scaler"].transform(latest_row)
    prob           = res["model"].predict_proba(latest_scaled)[0][1]

    explainer   = shap.TreeExplainer(res["model"])
    shap_vals   = explainer.shap_values(latest_scaled)[0]

    explanation = pd.DataFrame({
        "indicator": feature_cols,
        "impact"   : shap_vals
    }).sort_values("impact", key=abs, ascending=False).head(6)

    lines = []
    for _, row in explanation.iterrows():
        arrow     = "▲" if row["impact"] > 0 else "▼"
        direction = label_positive if row["impact"] > 0 else label_negative
        lines.append(f"{arrow} **{row['indicator']}** → {direction}  "
                     f"*(impact: {row['impact']:+.3f})*")

    return prob, "\n\n".join(lines)

# ── Signal badge helper ────────────────────────────────────────────────────
def signal_box(label, prob, conf_thr, pos_label, neg_label):
    confidence = max(prob, 1 - prob)
    is_conf    = confidence >= conf_thr
    is_pos     = prob >= 0.5

    if not is_conf:
        st.warning(f"⚪ **{label}** — Uncertain  "
                   f"({confidence*100:.1f}% < {conf_thr*100:.0f}% threshold)")
    elif is_pos:
        st.error(f"🔴 **{label}** → {pos_label}  —  {prob*100:.1f}% confidence")
    else:
        st.success(f"🟢 **{label}** → {neg_label}  —  {(1-prob)*100:.1f}% confidence")

# ── Accuracy badge colour ──────────────────────────────────────────────────
def acc_colour(lift):
    if lift > 0.15:  return "🟢"
    if lift > 0.05:  return "🟡"
    return "🔴"

# ═════════════════════════════════════════════════════════════════════════════
# Main app
# ═════════════════════════════════════════════════════════════════════════════
if run_btn:

    # ── Fetch data ─────────────────────────────────────────────────────────
    with st.spinner(f"Fetching {ticker} data..."):
        df = fetch_and_engineer(ticker, period)

    if df is None or df.empty:
        st.error(f"Could not fetch data for '{ticker}'. Check the ticker symbol.")
        st.stop()

    # ── Train all three models ─────────────────────────────────────────────
    with st.spinner("Training three models — this takes ~30 seconds..."):
        prog = st.progress(0, text="Training volatility model...")
        res_vol   = train_model(df, "target_vol")
        prog.progress(33, text="Training price direction model...")
        res_price = train_model(df, "target_price")
        prog.progress(66, text="Training RSI momentum model...")
        res_rsi   = train_model(df, "target_rsi")
        prog.progress(100, text="Generating SHAP explanations...")

    # ── SHAP explanations for today ────────────────────────────────────────
    prob_vol,   shap_vol   = explain_today(res_vol,   df,
                                           "pushes HIGH volatility",
                                           "pushes LOW volatility")
    prob_price, shap_price = explain_today(res_price, df,
                                           "pushes price UP",
                                           "pushes price DOWN")
    prob_rsi,   shap_rsi   = explain_today(res_rsi,   df,
                                           "RSI turning BULLISH",
                                           "RSI staying below 50")

    # Ensemble agreement — all three pointing same direction?
    signals      = [prob_vol >= 0.5, prob_price >= 0.5, prob_rsi >= 0.5]
    agreement    = sum(signals)
    last_date    = df.index[-1].strftime("%b %d, %Y")

    st.divider()

    # ── Today's signals ────────────────────────────────────────────────────
    st.subheader(f"Today's Signals — {ticker}  ({last_date})")

    col1, col2, col3 = st.columns(3)
    with col1:
        signal_box("Volatility (5d)",  prob_vol,
                   conf_thr, "HIGH volatility", "LOW volatility")
    with col2:
        signal_box("Price direction (5d)", prob_price,
                   conf_thr, "Price UP", "Price DOWN")
    with col3:
        signal_box("RSI momentum (5d)", prob_rsi,
                   conf_thr, "RSI turning bullish", "RSI staying bearish")

    # Ensemble summary
    st.divider()
    if agreement == 3:
        st.success("✅ **Strong ensemble agreement** — all three models agree. "
                   "Highest confidence signal.")
    elif agreement == 2:
        st.info("📊 **Partial agreement** — two of three models agree. "
                "Moderate confidence.")
    else:
        st.warning("⚠️ **No agreement** — models conflict. "
                   "Stay cautious, avoid acting on this signal.")

    # ── Model performance summary ──────────────────────────────────────────
    st.divider()
    st.subheader("Model Performance")

    mc1, mc2, mc3 = st.columns(3)
    for col, res, name in [
        (mc1, res_vol,   "Volatility model"),
        (mc2, res_price, "Price model"),
        (mc3, res_rsi,   "RSI model"),
    ]:
        with col:
            badge = acc_colour(res["lift"])
            st.metric(f"{badge} {name}",
                      f"{res['accuracy']*100:.1f}% accuracy",
                      f"+{res['lift']*100:.1f}% lift over baseline")

    # ── Tabs ───────────────────────────────────────────────────────────────
    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs([
        "Price & Signals",
        "Technical Indicators",
        "Why this prediction? (SHAP)",
        "Model Deep Dive",
    ])

    # ── Tab 1: Price chart with signals ───────────────────────────────────
    with tab1:
        st.subheader("Price chart — last 180 days with model signals")
        last_180      = df.tail(180)
        last_180_start = last_180.index[0]

        fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
        fig.suptitle(f"{ticker} — Model Signals on Price",
                     fontsize=13, fontweight="bold")

        for ax, res, title, pos_col, neg_col in [
            (axes[0], res_vol,
             "Volatility regime", "red",   "green"),
            (axes[1], res_price,
             "Price direction",   "green", "red"),
            (axes[2], res_rsi,
             "RSI momentum",      "blue",  "gray"),
        ]:
            test_index = res["X_test"].index
            y_prob_s   = pd.Series(res["y_prob"], index=test_index)
            t_prices   = res["test_prices"]

            pos_signals = t_prices[
                (y_prob_s >= conf_thr) &
                (y_prob_s.index >= last_180_start)
                ]
            neg_signals = t_prices[
                (y_prob_s <= 1 - conf_thr) &
                (y_prob_s.index >= last_180_start)
                ]

            ax.plot(last_180.index, last_180["close"],
                    color="black", linewidth=1, label="Close price")
            ax.scatter(pos_signals.index, pos_signals.values,
                       color=pos_col, s=45, zorder=5,
                       label="Positive signal")
            ax.scatter(neg_signals.index, neg_signals.values,
                       color=neg_col, s=45, zorder=5,
                       label="Negative signal")
            ax.set_title(title, fontsize=11)
            ax.set_ylabel("Price ($)")
            ax.legend(fontsize=8)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tab 2: Technical indicators ───────────────────────────────────────
    with tab2:
        st.subheader("Technical indicators — last 90 days")
        last_90 = df.tail(90)

        fig2 = plt.figure(figsize=(13, 12))
        gs   = gridspec.GridSpec(4, 1, figure=fig2, hspace=0.5)

        # Price + Bollinger Bands
        ax1 = fig2.add_subplot(gs[0])
        ax1.plot(last_90.index, last_90["close"],
                 color="black", linewidth=1, label="Close")
        ax1.plot(last_90.index, last_90["volatility_bbh"],
                 color="red",   linestyle="--", linewidth=0.8, label="BB Upper")
        ax1.plot(last_90.index, last_90["volatility_bbl"],
                 color="green", linestyle="--", linewidth=0.8, label="BB Lower")
        ax1.fill_between(last_90.index,
                         last_90["volatility_bbh"],
                         last_90["volatility_bbl"],
                         alpha=0.05, color="gray")
        ax1.set_title("Price + Bollinger Bands")
        ax1.legend(fontsize=8)

        # RSI
        ax2 = fig2.add_subplot(gs[1])
        ax2.plot(last_90.index, last_90["momentum_rsi"],
                 color="purple", linewidth=1)
        ax2.axhline(70, color="red",   linestyle="--", linewidth=0.8,
                    label="Overbought (70)")
        ax2.axhline(50, color="gray",  linestyle="--", linewidth=0.6,
                    label="Midline (50)")
        ax2.axhline(30, color="green", linestyle="--", linewidth=0.8,
                    label="Oversold (30)")
        ax2.fill_between(last_90.index, last_90["momentum_rsi"], 70,
                         where=(last_90["momentum_rsi"] >= 70),
                         alpha=0.3, color="red")
        ax2.fill_between(last_90.index, last_90["momentum_rsi"], 30,
                         where=(last_90["momentum_rsi"] <= 30),
                         alpha=0.3, color="green")
        ax2.set_ylim(0, 100)
        ax2.set_title("RSI (14)")
        ax2.legend(fontsize=8)

        # MACD
        ax3 = fig2.add_subplot(gs[2])
        ax3.plot(last_90.index, last_90["trend_macd"],
                 color="blue", linewidth=1, label="MACD")
        ax3.plot(last_90.index, last_90["trend_macd_signal"],
                 color="red",  linewidth=1, label="Signal")
        ax3.bar(last_90.index, last_90["trend_macd_diff"],
                color=["green" if v >= 0 else "red"
                       for v in last_90["trend_macd_diff"]],
                alpha=0.5, label="Histogram")
        ax3.axhline(0, color="black", linewidth=0.5)
        ax3.set_title("MACD")
        ax3.legend(fontsize=8)

        # Bollinger Band Width
        ax4 = fig2.add_subplot(gs[3])
        ax4.plot(last_90.index, last_90["volatility_bbw"],
                 color="darkorange", linewidth=1)
        ax4.set_title("Bollinger Band Width (volatility proxy)")

        st.pyplot(fig2)
        plt.close()

    # ── Tab 3: SHAP — why this prediction ─────────────────────────────────
    with tab3:
        st.subheader("Why is the model making today's prediction?")
        st.caption("SHAP values show exactly which indicators pushed the model "
                   "towards each signal. ▲ = pushes positive, ▼ = pushes negative.")

        s1, s2, s3 = st.columns(3)

        with s1:
            direction = "HIGH volatility" if prob_vol >= 0.5 else "LOW volatility"
            conf      = max(prob_vol, 1 - prob_vol)
            st.markdown(f"### Volatility model\n"
                        f"**Predicts:** {direction}  \n"
                        f"**Confidence:** {conf*100:.1f}%")
            st.markdown(shap_vol)

        with s2:
            direction = "Price UP" if prob_price >= 0.5 else "Price DOWN"
            conf      = max(prob_price, 1 - prob_price)
            st.markdown(f"### Price model\n"
                        f"**Predicts:** {direction}  \n"
                        f"**Confidence:** {conf*100:.1f}%")
            st.markdown(shap_price)

        with s3:
            direction = "RSI bullish" if prob_rsi >= 0.5 else "RSI bearish"
            conf      = max(prob_rsi, 1 - prob_rsi)
            st.markdown(f"### RSI model\n"
                        f"**Predicts:** {direction}  \n"
                        f"**Confidence:** {conf*100:.1f}%")
            st.markdown(shap_rsi)

        st.divider()

        # SHAP bar charts for all three models
        st.subheader("Feature impact charts")
        fc1, fc2, fc3 = st.columns(3)

        for col, res, title in [
            (fc1, res_vol,   "Volatility model"),
            (fc2, res_price, "Price model"),
            (fc3, res_rsi,   "RSI model"),
        ]:
            with col:
                explainer  = shap.TreeExplainer(res["model"])
                shap_vals  = explainer.shap_values(res["X_test_s"])
                mean_shap  = pd.DataFrame({
                    "feature": res["feature_cols"],
                    "impact" : np.abs(shap_vals).mean(axis=0)
                }).sort_values("impact", ascending=False).head(8)

                fig_s, ax_s = plt.subplots(figsize=(5, 4))
                ax_s.barh(mean_shap["feature"], mean_shap["impact"],
                          color="steelblue")
                ax_s.set_title(f"{title}\nMean |SHAP| impact", fontsize=10)
                ax_s.invert_yaxis()
                ax_s.set_xlabel("Mean absolute SHAP value")
                plt.tight_layout()
                st.pyplot(fig_s)
                plt.close()

    # ── Tab 4: Model deep dive ─────────────────────────────────────────────
    with tab4:
        st.subheader("Accuracy, confidence distribution and feature importance")

        for res, name, target_label in [
            (res_vol,   "Volatility model",      ["LOW vol", "HIGH vol"]),
            (res_price, "Price direction model",  ["DOWN",    "UP"]),
            (res_rsi,   "RSI momentum model",     ["Bearish", "Bullish"]),
        ]:
            with st.expander(f"{name}  —  "
                             f"{res['accuracy']*100:.1f}% accuracy  |  "
                             f"+{res['lift']*100:.1f}% lift", expanded=False):

                d1, d2, d3 = st.columns(3)
                d1.metric("Accuracy",  f"{res['accuracy']*100:.1f}%")
                d2.metric("Baseline",  f"{res['baseline']*100:.1f}%")
                d3.metric("Lift",      f"+{res['lift']*100:.1f}%")

                fig_d, axes_d = plt.subplots(1, 3, figsize=(14, 4))

                # Confidence histogram
                axes_d[0].hist(res["y_prob"], bins=25,
                               color="steelblue", alpha=0.7, edgecolor="white")
                axes_d[0].axvline(conf_thr,     color="red",
                                  linestyle="--", label=f">{conf_thr*100:.0f}%")
                axes_d[0].axvline(1 - conf_thr, color="green",
                                  linestyle="--", label=f"<{(1-conf_thr)*100:.0f}%")
                axes_d[0].set_title("Confidence distribution")
                axes_d[0].set_xlabel("Probability of positive class")
                axes_d[0].legend(fontsize=8)

                # Feature importance
                imp = res["importance"]
                axes_d[1].barh(imp["feature"], imp["importance"],
                               color="coral")
                axes_d[1].set_title("Top 10 feature importance")
                axes_d[1].invert_yaxis()

                # Prediction vs actual on price
                t_prices  = res["test_prices"].tail(120)
                y_prob_s  = pd.Series(res["y_prob"],
                                      index=res["X_test"].index).tail(120)
                y_test_s  = res["y_test"].tail(120)
                y_pred_s  = pd.Series(res["y_pred"],
                                      index=res["X_test"].index).tail(120)
                correct   = t_prices[y_pred_s.values == y_test_s.values]
                wrong     = t_prices[y_pred_s.values != y_test_s.values]
                axes_d[2].plot(t_prices.index, t_prices.values,
                               color="black", linewidth=0.8)
                axes_d[2].scatter(correct.index, correct.values,
                                  color="green", s=15, alpha=0.7,
                                  label="Correct")
                axes_d[2].scatter(wrong.index,   wrong.values,
                                  color="red",   s=15, alpha=0.7,
                                  label="Wrong")
                axes_d[2].set_title("Correct vs wrong predictions")
                axes_d[2].legend(fontsize=8)

                plt.tight_layout()
                st.pyplot(fig_d)
                plt.close()

else:
    st.info("Enter a ticker in the sidebar and click **Run Analysis** to start.")
    st.markdown("""
    **What this app does:**
    - Trains **three separate XGBoost models** — volatility, price direction, RSI momentum
    - Shows **ensemble agreement** — are all three models pointing the same way?
    - Uses **SHAP values** to explain exactly why each model made its prediction
    - Visualizes all technical indicators and model signals on a price chart

    **Try these tickers:** AAPL · TSLA · NVDA · MSFT · SPY · AMZN
    """)