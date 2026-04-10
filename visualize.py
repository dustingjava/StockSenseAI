import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Load the data we created
df = pd.read_csv("aapl_features.csv", index_col=0, parse_dates=True)

# Keep only last 6 months for clarity
df = df.tail(180)

# Create a figure with 5 subplots stacked vertically
fig = gridspec.GridSpec(5, 1, figure=plt.figure(figsize=(14, 16)), hspace=0.4)

# --- Plot 1: Candlestick-style Price + Bollinger Bands + SMAs ---
ax1 = plt.subplot(fig[0])
ax1.plot(df.index, df["close"], label="Close Price", color="black", linewidth=1.2)
ax1.plot(df.index, df["volatility_bbh"], label="BB Upper", color="red", linestyle="--", linewidth=0.8)
ax1.plot(df.index, df["volatility_bbl"], label="BB Lower", color="green", linestyle="--", linewidth=0.8)
ax1.plot(df.index, df["trend_sma_fast"], label="SMA 20", color="blue", linewidth=0.8)
ax1.plot(df.index, df["trend_sma_slow"], label="SMA 50", color="orange", linewidth=0.8)
ax1.fill_between(df.index, df["volatility_bbh"], df["volatility_bbl"], alpha=0.05, color="gray")
ax1.set_title("AAPL Price + Bollinger Bands + SMAs")
ax1.legend(loc="upper left", fontsize=8)
ax1.set_ylabel("Price ($)")

# --- Plot 2: Volume + OBV ---
ax2 = plt.subplot(fig[1])
ax2.bar(df.index, df["volume"], color="steelblue", alpha=0.5, label="Volume")
ax2.set_title("Volume")
ax2.set_ylabel("Volume")

# --- Plot 3: RSI ---
ax3 = plt.subplot(fig[2])
ax3.plot(df.index, df["momentum_rsi"], color="purple", linewidth=1)
ax3.axhline(70, color="red", linestyle="--", linewidth=0.8, label="Overbought (70)")
ax3.axhline(30, color="green", linestyle="--", linewidth=0.8, label="Oversold (30)")
ax3.fill_between(df.index, df["momentum_rsi"], 70,
                 where=(df["momentum_rsi"] >= 70), alpha=0.3, color="red")
ax3.fill_between(df.index, df["momentum_rsi"], 30,
                 where=(df["momentum_rsi"] <= 30), alpha=0.3, color="green")
ax3.set_title("RSI (14)")
ax3.set_ylabel("RSI")
ax3.set_ylim(0, 100)
ax3.legend(loc="upper left", fontsize=8)

# --- Plot 4: MACD ---
ax4 = plt.subplot(fig[3])
ax4.plot(df.index, df["trend_macd"], label="MACD", color="blue", linewidth=1)
ax4.plot(df.index, df["trend_macd_signal"], label="Signal", color="red", linewidth=1)
ax4.bar(df.index, df["trend_macd_diff"],
        color=["green" if v >= 0 else "red" for v in df["trend_macd_diff"]],
        alpha=0.5, label="Histogram")
ax4.axhline(0, color="black", linewidth=0.5)
ax4.set_title("MACD")
ax4.legend(loc="upper left", fontsize=8)

# --- Plot 5: Bollinger Band Width (volatility indicator) ---
ax5 = plt.subplot(fig[4])
ax5.plot(df.index, df["volatility_bbw"], color="darkorange", linewidth=1)
ax5.set_title("Bollinger Band Width (volatility)")
ax5.set_ylabel("BB Width")

plt.suptitle("AAPL Technical Analysis Dashboard", fontsize=14, fontweight="bold", y=1.01)
plt.savefig("aapl_technical_analysis.png", bbox_inches="tight", dpi=150)
plt.show()
print("Chart saved to aapl_technical_analysis.png")