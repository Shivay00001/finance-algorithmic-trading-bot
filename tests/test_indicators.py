"""Real tests: verify indicator math against independent manual computation."""
import io
import math
import pandas as pd
from fastapi.testclient import TestClient
from main import app, compute_indicators

client = TestClient(app)


def make_df(n=60, trend=1.0):
    # Deterministic series: steady drift + sine wiggle (no randomness).
    closes = [100.0 + trend * i + 3.0 * math.sin(i / 3.0) for i in range(n)]
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n).strftime("%Y-%m-%d"),
        "open": [c - 0.5 for c in closes],
        "high": [c + 1.0 for c in closes],
        "low": [c - 1.0 for c in closes],
        "close": closes,
        "volume": [1000 + i for i in range(n)],
    })


def csv_bytes(df):
    return df.to_csv(index=False).encode()


def test_sma_matches_manual_mean():
    df = make_df()
    ind = compute_indicators(df)
    closes = df["close"].tolist()
    for i in range(19, 60):
        expected = sum(closes[i - 19:i + 1]) / 20.0
        assert abs(ind["sma20"].iloc[i] - expected) < 1e-9, f"SMA mismatch at {i}"
    assert pd.isna(ind["sma20"].iloc[18])  # warmup row


def test_rsi_bounds_and_uptrend_high():
    df = make_df(trend=2.0)  # strong uptrend
    ind = compute_indicators(df)
    tail = ind["rsi14"].dropna()
    assert ((tail >= 0) & (tail <= 100)).all()
    assert ind["rsi14"].iloc[-1] > 70  # strong uptrend -> overbought


def test_rsi_downtrend_low():
    df = make_df(trend=-2.0)  # strong downtrend
    ind = compute_indicators(df)
    assert ind["rsi14"].iloc[-1] < 30


def test_macd_flat_series_near_zero():
    df = make_df()
    df["close"] = 100.0  # flat
    ind = compute_indicators(df)
    assert abs(ind["macd"].iloc[-1]) < 1e-9
    assert abs(ind["macd_hist"].iloc[-1]) < 1e-9


def test_bollinger_band_ordering():
    df = make_df()
    ind = compute_indicators(df)
    tail = ind.dropna()
    assert ((tail["bb_lower"] <= tail["bb_mid"]) & (tail["bb_mid"] <= tail["bb_upper"])).all()


def test_endpoints_return_real_values():
    df = make_df()
    r = client.post("/indicators", files={"file": ("data.csv", csv_bytes(df), "text/csv")})
    assert r.status_code == 200
    tail = r.json()["indicators_tail"]
    assert len(tail) == 5
    assert tail[-1]["sma20"] is not None and tail[-1]["rsi14"] is not None

    r = client.post("/signals", files={"file": ("data.csv", csv_bytes(df), "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert body["final_signal"] in ("BUY", "SELL", "HOLD")
    assert len(body["rule_votes"]) == 4
    # vote score sign must agree with final signal
    score = sum(1 for v in body["rule_votes"] if v["signal"] == "BUY") - sum(
        1 for v in body["rule_votes"] if v["signal"] == "SELL")
    assert (score > 0 and body["final_signal"] == "BUY") or \
           (score < 0 and body["final_signal"] == "SELL") or \
           (score == 0 and body["final_signal"] == "HOLD")


def test_parabolic_uptrend_is_overbought_hold():
    # Monotonic +3/bar: RSI hits 100 (SELL vote, overbought) which cancels the
    # SMA-20 BUY vote -> honest HOLD. This documents real rule interaction.
    df = make_df(n=60)
    df["close"] = [100.0 + 3.0 * i for i in range(60)]
    for col in ("open", "high", "low"):
        df[col] = df["close"]
    r = client.post("/signals", files={"file": ("data.csv", csv_bytes(df), "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert body["final_signal"] == "HOLD"
    assert body["vote_score"] == 0
    rsi_vote = next(v for v in body["rule_votes"] if v["rule"] == "RSI-14")
    assert rsi_vote["signal"] == "SELL" and "overbought" in rsi_vote["reason"]


def test_rejects_bad_csv():
    r = client.post("/indicators", files={"file": ("x.csv", b"a,b\n1,2", "text/csv")})
    assert r.status_code == 400
    r = client.post("/indicators", files={"file": ("x.csv", csv_bytes(make_df(n=10)), "text/csv")})
    assert r.status_code == 400


def test_health_and_rules():
    assert client.get("/health").json()["ai_prediction"] is False
    assert len(client.get("/rules").json()["rules"]) == 5
