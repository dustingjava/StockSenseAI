import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("aapl_features.csv", index_col=0, parse_dates=True)

# ── Pick features for anomaly detection ────────────────────────────────────
# We use price change, volume, RSI, BB width and MACD
# These capture both price AND momentum anomalies
features = [
    "close",
    "volume",
    "momentum_rsi",
    "volatility_bbw",
    "trend_macd",
    "trend_macd_diff",
    "volatility_atr"   # Average True Range = how much price moves per day
]

# Drop rows where any feature is NaN (happens at the start due to rolling windows)
data = df[features].dropna()

# ── Scale the features ─────────────────────────────────────────────────────
# ML models work better when all features are on the same scale
# Same concept as normalizing metrics in AIOps
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# ── Train Isolation Forest ─────────────────────────────────────────────────
# contamination = how much of the data we expect to be anomalous (5%)
# This is like setting your alert threshold in AIOps
iso = IsolationForest(
    contamination=0.05,   # flag top 5% most unusual days
    random_state=42,
    n_estimators=100
)

# fit_predict returns: 1 = normal, -1 = anomaly
data["anomaly"] = iso.fit_predict(data_scaled)
data["anomaly_score"] = iso.score_samples(data_scaled)  # lower = more anomalous

# ── Print summary ──────────────────────────────────────────────────────────
total = len(data)
anomalies = len(data[data["anomaly"] == -1])
print(f"Total trading days analysed : {total}")
print(f"Anomalies detected          : {anomalies} ({anomalies/total*100:.1f}%)")
print(f"\nTop 10 most anomalous days:")
print(
    data[data["anomaly"] == -1]
    .sort_values("anomaly_score")
    [["close", "volume", "momentum_rsi", "anomaly_score"]]
    .head(10)
    .to_string()
)

# ── Visualize anomalies on price chart ─────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Price chart with anomaly markers
ax1.plot(data.index, data["close"], color="black", linewidth=1, label="Close Price")
anomaly_days = data[data["anomaly"] == -1]
ax1.scatter(
    anomaly_days.index,
    anomaly_days["close"],
    color="red",
    s=50,
    zorder=5,
    label=f"Anomaly ({len(anomaly_days)} days)"
)
ax1.set_title("AAPL Price — Anomalies Detected by Isolation Forest")
ax1.set_ylabel("Price ($)")
ax1.legend()

# Anomaly score over time (lower = more anomalous)
ax2.plot(data.index, data["anomaly_score"], color="purple", linewidth=0.8)
ax2.axhline(
    data[data["anomaly"] == -1]["anomaly_score"].max(),
    color="red",
    linestyle="--",
    linewidth=0.8,
    label="Anomaly threshold"
)
ax2.fill_between(
    data.index,
    data["anomaly_score"],
    data[data["anomaly"] == -1]["anomaly_score"].max(),
    where=(data["anomaly"] == -1),
    color="red",
    alpha=0.3
)
ax2.set_title("Anomaly Score (lower = more unusual)")
ax2.set_ylabel("Score")
ax2.legend()

plt.tight_layout()
plt.savefig("aapl_anomalies.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved to aapl_anomalies.png")