import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from ta import add_all_ta_features
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ── Fetch 5 years of data ──────────────────────────────────────────────────
print("Fetching 5 years of data...")
raw = yf.download("AAPL", period="5y", interval="1d", auto_adjust=True)
raw.columns = raw.columns.droplevel(1)
raw.columns = [c.lower() for c in raw.columns]
raw = add_all_ta_features(
    raw, open="open", high="high",
    low="low", close="close", volume="volume",
    fillna=True
)

close  = raw["close"]
volume = raw["volume"]

# ── Engineer features ──────────────────────────────────────────────────────
engineered = pd.DataFrame(index=raw.index)

engineered["return_1d"]       = close.pct_change()
engineered["return_5d"]       = close.pct_change(5)
engineered["return_10d"]      = close.pct_change(10)
engineered["return_20d"]      = close.pct_change(20)
engineered["price_vs_sma20"]  = close / raw["trend_sma_fast"] - 1
engineered["price_vs_sma50"]  = close / raw["trend_sma_slow"] - 1
engineered["vol_ratio"]       = volume / volume.rolling(20).mean()
engineered["rsi_trend"]       = raw["momentum_rsi"] - raw["momentum_rsi"].shift(5)
engineered["macd_hist_trend"] = raw["trend_macd_diff"] - raw["trend_macd_diff"].shift(3)
engineered["volatility_20d"]  = engineered["return_1d"].rolling(20).std()

for lag in [1, 2, 3, 5]:
    engineered[f"rsi_lag{lag}"]    = raw["momentum_rsi"].shift(lag)
    engineered[f"macd_lag{lag}"]   = raw["trend_macd"].shift(lag)
    engineered[f"ret_lag{lag}"]    = engineered["return_1d"].shift(lag)
    engineered[f"volume_lag{lag}"] = engineered["vol_ratio"].shift(lag)

indicators = raw[[
    "momentum_rsi",
    "momentum_stoch",
    "momentum_wr",
    "trend_macd",
    "trend_macd_signal",
    "trend_macd_diff",
    "trend_adx",
    "volatility_bbw",
    "volatility_atr",
    "volume_obv",
]]

df = pd.concat([indicators, engineered], axis=1).copy()
df["close"] = close

# ── THREE better-framed targets ───────────────────────────────────────────
# Target 1: Will volatility be HIGH in next 5 days?
# Technical indicators are great at predicting volatility regimes
future_vol = engineered["return_1d"].shift(-1).rolling(5).std()
df["target_vol"] = (future_vol > future_vol.rolling(60).mean()).astype(int)

# Target 2: Will price be >2% higher in 20 days?
# Longer horizon = much stronger signal from indicators
df["target_20d"] = (close.shift(-20) > close * 1.02).astype(int)

# Target 3: Is RSI about to cross 30 (oversold bounce signal)?
# Very specific, learnable pattern
rsi = raw["momentum_rsi"]
df["target_bounce"] = (
        (rsi < 35) &                          # RSI currently low
        (rsi.shift(-3) > rsi)                 # RSI rising in 3 days
).astype(int)

print("Three prediction problems compared:")
print(f"  High volatility in 5d  : {df['target_vol'].mean()*100:.1f}% of days are positive")
print(f"  Price +2% in 20 days   : {df['target_20d'].mean()*100:.1f}% of days are positive")
print(f"  RSI bounce signal      : {df['target_bounce'].mean()*100:.1f}% of days are positive")

# ── We'll train on all three and compare ──────────────────────────────────
feature_cols = [c for c in df.columns
                if c not in ["close", "target_vol", "target_20d", "target_bounce"]]

results = {}

for target_name, target_col in [
    ("Volatility regime (5d)",  "target_vol"),
    ("Price +2% in 20 days",    "target_20d"),
    ("RSI oversold bounce",     "target_bounce"),
]:
    data = df[feature_cols + [target_col, "close"]].dropna()
    X    = data[feature_cols]
    y    = data[target_col]

    split      = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler         = StandardScaler()
    X_train_s      = scaler.fit_transform(X_train)
    X_test_s       = scaler.transform(X_test)

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos if pos > 0 else 1

    model = XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.02,
        subsample=0.7,
        colsample_bytree=0.7,
        scale_pos_weight=spw,
        random_state=42,
        eval_metric="logloss",
        verbosity=0
    )
    model.fit(X_train_s, y_train,
              eval_set=[(X_test_s, y_test)],
              verbose=False)

    y_pred   = model.predict(X_test_s)
    y_prob   = model.predict_proba(X_test_s)[:, 1]
    accuracy = (y_pred == y_test).mean()
    baseline = y_test.mean()

    results[target_name] = {
        "accuracy" : accuracy,
        "baseline" : baseline,
        "lift"     : accuracy - baseline,
        "y_test"   : y_test,
        "y_pred"   : y_pred,
        "y_prob"   : y_prob,
        "prices"   : data["close"].loc[X_test.index],
        "model"    : model,
        "scaler"   : scaler,
    }

    print(f"\n── {target_name} ──────────────────────────")
    print(classification_report(y_test, y_pred,
                                target_names=["NO", "YES"], zero_division=0))
    print(f"Accuracy : {accuracy*100:.1f}%  |  Baseline : {baseline*100:.1f}%  |  Lift : {(accuracy-baseline)*100:+.1f}%")

# ── Visualize all three side by side ──────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(14, 14))

for i, (target_name, res) in enumerate(results.items()):
    # Price chart with signals
    ax_left  = axes[i, 0]
    ax_right = axes[i, 1]

    prices   = res["prices"]
    y_pred   = res["y_pred"]
    y_test   = res["y_test"]
    y_prob   = res["y_prob"]

    ax_left.plot(prices.index, prices.values,
                 color="black", linewidth=0.8, label="Price")
    pred_s = pd.Series(y_pred, index=prices.index)
    test_s = y_test.loc[prices.index]

    correct  = prices[(pred_s == 1) & (pred_s.values == test_s.values)]
    wrong    = prices[(pred_s == 1) & (pred_s.values != test_s.values)]
    ax_left.scatter(correct.index, correct.values,
                    color="green", s=25, zorder=5, label="Correct signal")
    ax_left.scatter(wrong.index,   wrong.values,
                    color="red",   s=25, zorder=5, label="Wrong signal")
    ax_left.set_title(f"{target_name}\nAcc: {res['accuracy']*100:.1f}%  Baseline: {res['baseline']*100:.1f}%  Lift: {res['lift']*100:+.1f}%")
    ax_left.legend(fontsize=7)
    ax_left.set_ylabel("Price ($)")

    # Confidence histogram
    ax_right.hist(y_prob, bins=25, color="steelblue", alpha=0.7, edgecolor="white")
    ax_right.axvline(0.6, color="green", linestyle="--", label="60% threshold")
    ax_right.axvline(0.4, color="red",   linestyle="--", label="40% threshold")
    ax_right.set_title(f"Confidence Distribution — {target_name}")
    ax_right.set_xlabel("Probability of YES")
    ax_right.set_ylabel("Days")
    ax_right.legend(fontsize=7)

plt.suptitle("AAPL — Three Better-Framed Prediction Problems",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("aapl_three_targets.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved to aapl_three_targets.png")