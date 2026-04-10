import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("aapl_features.csv", index_col=0, parse_dates=True)

# ── Fetch more data — 5 years instead of 2 ────────────────────────────────
# More data = better model. Let's re-fetch with longer history
import yfinance as yf
from ta import add_all_ta_features

print("Fetching 5 years of data...")
df = yf.download("AAPL", period="5y", interval="1d", auto_adjust=True)
df.columns = df.columns.droplevel(1)
df.columns = [c.lower() for c in df.columns]
df = add_all_ta_features(
    df, open="open", high="high",
    low="low", close="close", volume="volume",
    fillna=True   # fill NaN values instead of dropping them
)

# ── Create target ──────────────────────────────────────────────────────────
df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

# Check class balance — should be roughly 50/50
print(f"\nClass balance:")
print(f"  UP days   : {df['target'].sum()} ({df['target'].mean()*100:.1f}%)")
print(f"  DOWN days : {(1-df['target']).sum()} ({(1-df['target'].mean())*100:.1f}%)")

# ── Select features ────────────────────────────────────────────────────────
feature_cols = [
    "momentum_rsi",
    "momentum_stoch",
    "momentum_stoch_signal",
    "momentum_wr",
    "momentum_roc",
    "momentum_macd",
    "trend_macd",
    "trend_macd_signal",
    "trend_macd_diff",
    "trend_sma_fast",
    "trend_sma_slow",
    "trend_ema_fast",
    "trend_ema_slow",
    "trend_adx",
    "volatility_bbh",
    "volatility_bbl",
    "volatility_bbm",
    "volatility_bbw",
    "volatility_atr",
    "volume_obv",
    "volume_vwap",
]

# Keep only columns that exist
feature_cols = [c for c in feature_cols if c in df.columns]
print(f"\nUsing {len(feature_cols)} features")

# ── Prepare data ───────────────────────────────────────────────────────────
data = df[feature_cols + ["target", "close"]].dropna()
X = data[feature_cols]
y = data["target"]

# ── Time series split ──────────────────────────────────────────────────────
split = int(len(data) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"Training on : {len(X_train)} days")
print(f"Testing on  : {len(X_test)} days")
print(f"Test period : {X_test.index[0].date()} → {X_test.index[-1].date()}")

# ── Scale ──────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Train Random Forest ────────────────────────────────────────────────────
# class_weight="balanced" fixes the class imbalance problem
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=4,
    min_samples_leaf=20,
    class_weight="balanced",   # KEY FIX — forces model to predict both classes
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\n── Model Performance ──────────────────────────────")
# zero_division=0 suppresses the warning cleanly
print(classification_report(
    y_test, y_pred,
    target_names=["DOWN", "UP"],
    zero_division=0
))

accuracy = (y_pred == y_test).mean()
print(f"Overall accuracy : {accuracy*100:.1f}%")
print(f"Baseline (always predict UP) : {y_test.mean()*100:.1f}%")
print(f"Our model beats baseline by  : {(accuracy - y_test.mean())*100:+.1f}%")

# ── Feature importance ─────────────────────────────────────────────────────
importance_df = pd.DataFrame({
    "feature"   : feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False).head(10)

print("\n── Top 10 Most Important Indicators ───────────────")
print(importance_df.to_string(index=False))

# ── Visualize ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Prediction vs Actual
ax1 = axes[0, 0]
test_prices = data["close"].loc[X_test.index]
ax1.plot(test_prices.index, test_prices.values,
         color="black", linewidth=1, label="Actual Price")
y_pred_series = pd.Series(y_pred, index=X_test.index)
y_test_aligned = y_test.loc[X_test.index]
correct   = test_prices[y_pred_series.values == y_test_aligned.values]
incorrect = test_prices[y_pred_series.values != y_test_aligned.values]
ax1.scatter(correct.index,   correct.values,
            color="green", s=20, alpha=0.6, label="Correct")
ax1.scatter(incorrect.index, incorrect.values,
            color="red",   s=20, alpha=0.6, label="Wrong")
ax1.set_title("Predictions vs Actual Price")
ax1.legend(fontsize=8)
ax1.set_ylabel("Price ($)")

# 2. Confusion matrix
ax2 = axes[0, 1]
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2,
            xticklabels=["DOWN", "UP"],
            yticklabels=["DOWN", "UP"])
ax2.set_title("Confusion Matrix")
ax2.set_ylabel("Actual")
ax2.set_xlabel("Predicted")

# 3. Feature importance
ax3 = axes[1, 0]
ax3.barh(importance_df["feature"], importance_df["importance"], color="steelblue")
ax3.set_title("Top 10 Most Important Indicators")
ax3.set_xlabel("Importance Score")
ax3.invert_yaxis()

# 4. Prediction confidence
ax4 = axes[1, 1]
ax4.plot(X_test.index, y_prob, color="purple", linewidth=0.8)
ax4.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="50% line")
ax4.fill_between(X_test.index, y_prob, 0.5,
                 where=(y_prob >= 0.5), alpha=0.3, color="green", label="Predicts UP")
ax4.fill_between(X_test.index, y_prob, 0.5,
                 where=(y_prob < 0.5),  alpha=0.3, color="red",   label="Predicts DOWN")
ax4.set_title("Model Confidence Over Time")
ax4.set_ylabel("Probability of UP")
ax4.legend(fontsize=8)

plt.suptitle("AAPL Price Direction Prediction — Random Forest (5yr)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("aapl_prediction.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved to aapl_prediction.png")