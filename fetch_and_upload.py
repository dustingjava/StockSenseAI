import boto3
import pandas as pd
import yfinance as yf
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from datetime import datetime
import tempfile
import os

# ── Config ─────────────────────────────────────────────────────────────────
BUCKET  = "stocksense-ai-448049787062"
REGION  = "us-east-1"
TICKERS = ["CRM", "PLTR", "NVDA"]
PERIOD  = "5y"

s3 = boto3.client("s3", region_name=REGION)

def fetch_and_engineer(ticker, period):
    print(f"Fetching {ticker}...")
    raw = yf.download(ticker, period=period,
                      interval="1d", auto_adjust=True)
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

    # Momentum
    df["momentum_rsi"] = RSIIndicator(close).rsi()
    df["momentum_roc"] = ROCIndicator(close).roc()

    # Trend
    macd                        = MACD(close)
    df["trend_macd"]            = macd.macd()
    df["trend_macd_signal"]     = macd.macd_signal()
    df["trend_macd_diff"]       = macd.macd_diff()
    df["trend_sma_fast"]        = SMAIndicator(close, window=20).sma_indicator()
    df["trend_sma_slow"]        = SMAIndicator(close, window=50).sma_indicator()
    df["trend_ema_fast"]        = EMAIndicator(close, window=12).ema_indicator()
    df["trend_ema_slow"]        = EMAIndicator(close, window=26).ema_indicator()
    df["trend_adx"]             = ADXIndicator(high, low, close).adx()

    # Volatility
    bb                          = BollingerBands(close)
    df["volatility_bbh"]        = bb.bollinger_hband()
    df["volatility_bbl"]        = bb.bollinger_lband()
    df["volatility_bbm"]        = bb.bollinger_mavg()
    df["volatility_bbw"]        = bb.bollinger_wband()
    df["volatility_atr"]        = AverageTrueRange(high, low, close).average_true_range()

    # Volume
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

    # Targets
    future_vol         = df["return_1d"].shift(-1).rolling(5).std()
    df["target_vol"]   = (future_vol > future_vol.rolling(60).mean()).astype(int)
    df["target_price"] = (close.shift(-5) > close).astype(int)
    rsi                = df["momentum_rsi"]
    df["target_rsi"]   = ((rsi.shift(-5) > 50) & (rsi <= 50)).astype(int)

    return df

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    uploaded  = []

    for ticker in TICKERS:
        try:
            df      = fetch_and_engineer(ticker, PERIOD)
            s3_key  = f"training-data/{ticker}/latest.csv"

            with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".csv", delete=False) as f:
                df.to_csv(f.name)
                tmp_path = f.name

            s3.upload_file(tmp_path, BUCKET, s3_key)
            os.unlink(tmp_path)

            print(f"Uploaded {ticker} → s3://{BUCKET}/{s3_key}")
            uploaded.append(ticker)

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    print(f"\nDone — uploaded {len(uploaded)}/{len(TICKERS)} tickers")
    print(f"Lambda will now train models automatically")