import boto3
import json
import os

REGION = "us-east-1"

def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name=REGION)

def explain_signal(
        ticker,
        prob_vol,
        prob_price,
        prob_rsi,
        indicators: dict,
        shap_factors: list
) -> str:
    """
    Call Claude Haiku on Bedrock to generate a plain English
    explanation of the model's prediction.
    """

    # Build volatility direction
    vol_direction   = "HIGH volatility" if prob_vol   >= 0.5 else "LOW volatility"
    price_direction = "UP"             if prob_price >= 0.5 else "DOWN"
    rsi_direction   = "BULLISH"        if prob_rsi   >= 0.5 else "BEARISH"

    vol_conf   = max(prob_vol,   1 - prob_vol)   * 100
    price_conf = max(prob_price, 1 - prob_price) * 100
    rsi_conf   = max(prob_rsi,   1 - prob_rsi)   * 100

    # Format top SHAP factors
    shap_text = "\n".join([
        f"- {f['indicator']}: {f['direction']} "
        f"(impact: {f['impact']:+.3f})"
        for f in shap_factors[:5]
    ])

    prompt = f"""You are a quantitative analyst explaining stock market predictions to a trader.

Ticker: {ticker}
Model predictions for next 5 days:
- Volatility: {vol_direction} ({vol_conf:.0f}% confidence)
- Price direction: {price_direction} ({price_conf:.0f}% confidence)
- RSI momentum: {rsi_direction} ({rsi_conf:.0f}% confidence)

Current technical indicators:
- RSI(14): {indicators.get('rsi', 'N/A')}
- MACD signal: {indicators.get('macd_signal', 'N/A')}
- Bollinger Band Width: {indicators.get('bb_width', 'N/A')}
- Volume vs 20-day average: {indicators.get('vol_ratio', 'N/A')}x
- Price vs SMA20: {indicators.get('price_vs_sma20', 'N/A')}%

Top factors driving the volatility prediction:
{shap_text}

In 3-4 sentences, explain what these signals mean for {ticker} 
this week in plain English. Be specific about which indicators 
are most important and what they suggest. Avoid jargon. 
Do not use bullet points — write as a paragraph."""

    try:
        client   = get_bedrock_client()
        response = client.invoke_model(
            modelId     = "anthropic.claude-3-haiku-20240307-v1:0",
            contentType = "application/json",
            accept      = "application/json",
            body        = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens"       : 300,
                "messages"         : [{
                    "role"   : "user",
                    "content": prompt
                }]
            })
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]

    except Exception as e:
        error_msg = str(e)
        if "ResourceNotFoundException" in error_msg:
            return ("Bedrock model access pending — "
                    "please complete the Anthropic use case form "
                    "in AWS Console → Bedrock → Model catalog → Claude 3 Haiku.")
        elif "AccessDeniedException" in error_msg:
            return "Bedrock access denied — check IAM permissions."
        else:
            return f"Could not generate explanation: {error_msg}"

def explain_signal_fallback(ticker, prob_vol, prob_price,
                                    prob_rsi, indicators):
    """Rule-based explanation when Bedrock is unavailable."""
    vol_dir   = "HIGH volatility" if prob_vol   >= 0.5 else "LOW volatility"
    price_dir = "upward"          if prob_price >= 0.5 else "downward"
    rsi_dir   = "bullish"         if prob_rsi   >= 0.5 else "bearish"
    rsi_val   = float(indicators.get("rsi", 50))
    vol_ratio = float(indicators.get("vol_ratio", 1))

    rsi_comment = (
        "RSI is in oversold territory suggesting a potential bounce"
        if rsi_val < 30 else
        "RSI is in overbought territory suggesting possible pullback"
        if rsi_val > 70 else
        f"RSI at {rsi_val:.0f} is in neutral territory"
    )

    vol_comment = (
        f"Volume is running {vol_ratio:.1f}x above average "
        "indicating strong institutional activity"
        if vol_ratio > 1.5 else
        f"Volume is running {vol_ratio:.1f}x the 20-day average"
    )

    return (
        f"StockSense models predict {vol_dir} for {ticker} "
        f"over the next 5 days with price expected to move {price_dir} "
        f"and RSI momentum turning {rsi_dir}. "
        f"{rsi_comment}. {vol_comment}."
    )