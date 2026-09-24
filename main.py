"""Real technical-analysis engine.

Computes REAL indicators (SMA, EMA, RSI, MACD, Bollinger Bands) from uploaded
OHLC CSV data and emits honest RULE-BASED signals from documented rules.

This is NOT an AI predictor. It does not forecast prices. Signals are
deterministic functions of past price data (technical analysis heuristics).
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import numpy as np
import io
import math

app = FastAPI(title="Algorithmic Trading Bot — Technical Analysis Engine")

REQUIRED_COLS = ["date", "open", "high", "low", "close", "volume"]
MIN_ROWS = 30  # warmup for the 20-period indicators

RULES_DOC = [
    {
        "id": "RSI-14",
        "rule": "RSI(14) < 30 -> BUY (oversold); RSI(14) > 70 -> SELL (overbought); else HOLD",
        "note": "Wilder's RSI. Momentum heuristic, not a prediction.",
    },
    {
        "id": "MACD-X",
        "rule": "MACD line crosses above signal line -> BUY; crosses below -> SELL; else HOLD",
        "note": "MACD(12,26,9). Trend-momentum heuristic, not a prediction.",
    },
    {
        "id": "SMA-20",
        "rule": "close > SMA(20) -> BUY (uptrend); close < SMA(20) -> SELL (downtrend); else HOLD",
        "note": "Simple trend-following heuristic.",
    },
    {
        "id": "BB-20",
        "rule": "close < lower Bollinger band(20,2) -> BUY (stretched down); "
               "close > upper band -> SELL (stretched up); else HOLD",
        "note": "Mean-reversion heuristic.",
    },
    {
        "id": "VOTE",
        "rule": "Final signal = majority vote of the 4 rules above (BUY=+1, SELL=-1, HOLD=0).",
        "note": "Equal-weighted rule vote. Educational only — not financial advice.",
    },
]


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicator columns. All formulas are standard textbook definitions."""
    close = df["close"].astype(float)
    out = df.copy()

    # Simple moving average (20)
    out["sma20"] = close.rolling(window=20).mean()

    # Exponential moving averages (12, 26)
    out["ema12"] = close.ewm(span=12, adjust=False).mean()
    out["ema26"] = close.ewm(span=26, adjust=False).mean()

    # RSI(14), Wilder's smoothing
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out["rsi14"] = 100.0 - (100.0 / (1.0 + rs))
    out["rsi14"] = out["rsi14"].fillna(100.0 * (avg_gain > 0).astype(float))

    # MACD(12, 26, 9)
    out["macd"] = out["ema12"] - out["ema26"]
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # Bollinger Bands (20, 2 sigma)
    mid = close.rolling(window=20).mean()
    sd = close.rolling(window=20).std(ddof=1)
    out["bb_mid"] = mid
    out["bb_upper"] = mid + 2.0 * sd
    out["bb_lower"] = mid - 2.0 * sd
    return out


def load_ohlc(file: UploadFile) -> pd.DataFrame:
    try:
        raw = file.file.read()
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")
    if len(df) < MIN_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {MIN_ROWS} rows for indicator warmup, got {len(df)}",
        )
    return df


def _num(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def generate_signals(df: pd.DataFrame) -> dict:
    ind = compute_indicators(df)
    last = ind.iloc[-1]
    prev = ind.iloc[-2]
    votes = []

    rsi = float(last["rsi14"])
    if rsi < 30:
        votes.append({"rule": "RSI-14", "signal": "BUY", "reason": f"RSI(14)={rsi:.2f} < 30 (oversold)"})
    elif rsi > 70:
        votes.append({"rule": "RSI-14", "signal": "SELL", "reason": f"RSI(14)={rsi:.2f} > 70 (overbought)"})
    else:
        votes.append({"rule": "RSI-14", "signal": "HOLD", "reason": f"RSI(14)={rsi:.2f} neutral"})

    if prev["macd"] <= prev["macd_signal"] and last["macd"] > last["macd_signal"]:
        votes.append({"rule": "MACD-X", "signal": "BUY", "reason": "MACD crossed above signal line"})
    elif prev["macd"] >= prev["macd_signal"] and last["macd"] < last["macd_signal"]:
        votes.append({"rule": "MACD-X", "signal": "SELL", "reason": "MACD crossed below signal line"})
    else:
        votes.append({"rule": "MACD-X", "signal": "HOLD", "reason": "No MACD crossover on last bar"})

    if last["close"] > last["sma20"]:
        votes.append({"rule": "SMA-20", "signal": "BUY", "reason": f"close {last['close']:.2f} > SMA20 {last['sma20']:.2f}"})
    elif last["close"] < last["sma20"]:
        votes.append({"rule": "SMA-20", "signal": "SELL", "reason": f"close {last['close']:.2f} < SMA20 {last['sma20']:.2f}"})
    else:
        votes.append({"rule": "SMA-20", "signal": "HOLD", "reason": "close == SMA20"})

    if last["close"] < last["bb_lower"]:
        votes.append({"rule": "BB-20", "signal": "BUY", "reason": "close below lower Bollinger band"})
    elif last["close"] > last["bb_upper"]:
        votes.append({"rule": "BB-20", "signal": "SELL", "reason": "close above upper Bollinger band"})
    else:
        votes.append({"rule": "BB-20", "signal": "HOLD", "reason": "price inside Bollinger bands"})

    score = sum(1 for v in votes if v["signal"] == "BUY") - sum(1 for v in votes if v["signal"] == "SELL")
    final = "BUY" if score > 0 else "SELL" if score < 0 else "HOLD"
    return {
        "final_signal": final,
        "vote_score": score,
        "rule_votes": votes,
        "disclaimer": "Rule-based technical heuristics on past data. NOT a prediction, NOT financial advice.",
    }


@app.get("/health")
def health():
    return {"status": "ok", "engine": "technical-analysis", "ai_prediction": False}


@app.get("/rules")
def rules():
    return {"rules": RULES_DOC}


@app.post("/indicators")
def indicators(file: UploadFile = File(...), tail: int = 5):
    df = load_ohlc(file)
    ind = compute_indicators(df)
    cols = ["date", "close", "sma20", "ema12", "ema26", "rsi14",
            "macd", "macd_signal", "macd_hist", "bb_mid", "bb_upper", "bb_lower"]
    frame = ind[cols].tail(max(1, tail))
    rows = [{c: (_num(v) if c != "date" else str(v)) for c, v in row.items()}
            for _, row in frame.iterrows()]
    return {"rows": len(df), "indicators_tail": rows}


@app.post("/signals")
def signals(file: UploadFile = File(...)):
    df = load_ohlc(file)
    result = generate_signals(df)
    result["rows"] = len(df)
    result["last_close"] = float(df["close"].iloc[-1])
    return result
