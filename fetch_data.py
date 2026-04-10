import yfinance as yf
import pandas as pd
from ta import add_all_ta_features

# Download 2 years of Apple data
df = yf.download("AAPL", period="2y", interval="1d", auto_adjust=True)

# Fix: flatten the multi-level column structure yfinance now returns
df.columns = df.columns.droplevel(1)

# Rename to lowercase so 'ta' library can find them
df.columns = [c.lower() for c in df.columns]

# Add every technical indicator automatically
df = add_all_ta_features(
    df, open="open", high="high",
    low="low", close="close", volume="volume"
)

print(f"Shape: {df.shape}")
print(f"Columns: {len(df.columns)} features created")
df.to_csv("aapl_features.csv")
print("Saved to aapl_features.csv")