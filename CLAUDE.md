# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FAANG Pulse AI predicts whether FAANG stocks (Meta, Apple, Amazon, Netflix, Google) are **trending up, trending down, or not trending** using an Isolation Forest anomaly detection model. It is built for exploratory research — not trading signals.

## Commands

### Run the App
```powershell
.venv\Scripts\activate
python app.py
# Launches Gradio UI at http://127.0.0.1:7860
```

### Install Dependencies
```powershell
uv venv .venv --python "C:\Program Files\Python312\python.exe"
.venv\Scripts\activate
uv pip install -e .
uv lock
```

### Sync Dependencies to requirements.txt
```powershell
uv pip compile pyproject.toml -o requirements.txt
```

There is no test suite or linting configuration.

## Architecture

Three modules with clear separation of concerns:

**`app.py`** — Gradio UI and public API surface. Defines all UI components (stock dropdown, date picker, risk tolerance slider, prediction output, 30-day price plot, anomaly score histogram). Exposes two public API endpoints: `run_trend_prediction` and `get_prices_on_date`. Wires user actions to the ML engine and data fetcher.

**`isolation_forest_engine.py`** — ML inference engine (`isfEngine` class). Loads the pre-trained model from `optimized_isf_pipeline.skops` at startup. Computes four features from a 30-day price window: `slope` (linear trend of log prices), `zcr` (zero-crossing rate of daily differences), `volatility` (std of log returns), `trend_strength` (|slope| / volatility). Runs `decision_function()` and compares the score against the user-selected risk threshold. Also renders the anomaly score histogram with the test-set distribution from `support/normal_vs_anomaly_isf_scores.csv`.

**`finance.py`** — `StockDataFetcher` class wrapping yfinance. Fetches 30 working days of adjusted close prices with retry logic, holiday handling, and a 15-day backward search fallback when a requested date has no data.

### Decision Flow
```
User selects stock + date + risk tolerance
  → StockDataFetcher.get_30_day_window()  (Yahoo Finance OHLCV)
  → isfEngine.generate_input_features()   (slope, zcr, volatility, trend_strength)
  → model.decision_function()             (Isolation Forest score)
  → score <= threshold → TREND_UP / TREND_DOWN (based on slope sign)
  → score >  threshold → NO_TREND
```

### Key Files
| File | Role |
|------|------|
| `app.py` | Entry point, Gradio UI, API definitions |
| `isolation_forest_engine.py` | Feature engineering, model inference, histogram rendering |
| `finance.py` | Yahoo Finance data access layer |
| `optimized_isf_pipeline.skops` | Pre-trained Isolation Forest + StandardScaler (Git LFS) |
| `support/normal_vs_anomaly_isf_scores.csv` | Test-set scores for histogram |
| `pyproject.toml` | Project metadata; requires Python ≥ 3.13 |
| `.github/workflows/sync-to-hf.yml` | Auto-syncs `main` branch to Hugging Face Spaces on push |

## Model Details

- **Type**: Isolation Forest (anomaly detection — trending = anomalous price behavior)
- **Serialization**: skops format; loaded with `skops.io.load()` at startup
- **Optimal threshold**: 0.2554 contamination factor derived via Youden's J statistic
- **Best hyperparameters**: `max_features=0.8`, `n_estimators=200`, `max_samples=512`
- **Training split**: 2013–2017 (train), 2018–2021 (validation), 2024–2025 (test)

## Deployment

The app is hosted on Hugging Face Spaces. Every push to `main` triggers the GitHub Actions workflow (`.github/workflows/sync-to-hf.yml`) which syncs the repo to HF Spaces automatically. The model file (`optimized_isf_pipeline.skops`) is stored via Git LFS.
