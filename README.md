# Algorithmic Trading Bot — Technical Analysis Engine

**What it does:** Upload an OHLC CSV and get real technical indicators
(SMA-20, EMA-12/26, RSI-14 Wilder, MACD 12/26/9, Bollinger Bands 20/2)
plus deterministic **rule-based** trading signals.

**What it does NOT do:** It is not AI. It does not predict prices. Signals are
documented heuristics over past data (see `GET /rules`). Not financial advice.

## Run

```bash
pip install -r requirements.txt
uvicorn main:app --port 8000
```

## API

- `GET /health`, `GET /rules` — documented signal rules
- `POST /indicators` — multipart CSV upload (`date,open,high,low,close,volume`),
  returns last N rows of indicator values
- `POST /signals` — same upload, returns per-rule votes + majority final signal

```bash
curl -F "csv=@sample_data.csv" http://localhost:8000/signals
```

## Tests

```bash
pytest -q
```
