# Cross-Sectional Alpha Generation via Dual-Attention Transformer

> Cross-sectional alpha generation using a Dual-Attention Transformer on S&P 500 equities — monthly rebalancing, 20 price/volume features, long-only top-decile portfolio with **Sharpe 1.50 out-of-sample (2025)**.

---

## Overview

This project implements an end-to-end quantitative equity research pipeline that trains a **Dual-Attention Transformer (DAT)** to rank S&P 500 stocks by predicted next-month return. The model combines two attention mechanisms:

- **Temporal attention** — processes each stock's 24-month feature history independently
- **Cross-sectional attention** — lets all ~484 stocks attend to each other within the same month, capturing relative positioning signals that purely time-series models miss

Two portfolio strategies are evaluated out-of-sample on 2025 data the model never saw during training: a **long/short decile** strategy and a **long-only top-decile** strategy.

---

## Results (Out-of-Sample 2025)

| Strategy | Ann. Return | Ann. Vol | Sharpe | Max DD | Win Rate |
|---|---|---|---|---|---|
| L/S Gross (no tcost) | +7.91% | 15.69% | 0.505 | -6.13% | 63.6% |
| L/S Net (20bps) | +5.95% | 16.13% | 0.369 | -6.25% | 63.6% |
| **Long Leg Net (20bps)** | **+19.42%** | **12.93%** | **1.502** | **-6.18%** | **63.6%** |
| SPY Buy-and-Hold | ~+15.00% | ~15.00% | ~1.00 | — | — |

- **Annual transaction cost drag:** ~1.96% (20bps one-way, ~20% monthly turnover)
- **Test period:** January 2025 – November 2025 (11 months, fully out-of-sample)
- **Universe:** 484 S&P 500 large-cap equities
- **Key finding:** The long leg (top decile) significantly outperforms SPY on a risk-adjusted basis. The short book destroys value in 2025's trending market, dragging combined Sharpe from 1.50 → 0.37. Long-only is the recommended deployment mode.

---

## Architecture

```
Input: (B, SEQ_LEN=24, N_FEATURES=20)
         │
         ▼
  ┌─────────────────┐
  │  Input Proj     │  LayerNorm → Linear → LayerNorm
  └────────┬────────┘
           │  + CLS token prepended
           │  + Learned positional embeddings
           ▼
  ┌─────────────────┐
  │ Temporal Encoder│  2× Pre-LN TransformerEncoderLayer
  │  (per stock)    │  processes 24-month time series
  └────────┬────────┘
           │  CLS pooling → (B, D_MODEL)
           │
           ├──────────────────────────────┐
           │                              │ temporal residual
           ▼                              │
  ┌─────────────────┐                     │
  │  Cross-Sectional│  2× Pre-LN MHA      │
  │     Blocks      │  all ~484 stocks    │
  │  (per month)    │  attend together    │
  └────────┬────────┘                     │
           │         ◄────────────────────┘
           ▼
  ┌─────────────────┐
  │    MLP Head     │  LayerNorm → Linear → GELU → Linear
  └────────┬────────┘
           ▼
    Output: (B,) predicted returns
```

**Model size:** 139,433 parameters — deliberately lightweight for the dataset size.

---

## Pipeline

```
Data Acquisition          Feature Engineering       Model Training
──────────────────        ───────────────────       ──────────────
Wikipedia S&P 500    →    20 price/volume      →    DualAttention
ticker scrape             features computed          Transformer
                          cross-sectionally          (PyTorch)
yFinance OHLCV            rank-normalised to         AdamW +
download (batched)        [-1, +1] per month         CosineAnnealingLR
                                                      Early stopping
                                                      (patience=8,
                                                       best @ epoch 16)
                               ▼
Portfolio Evaluation           Inference
────────────────────           ─────────
Two strategies evaluated:  ←   Monthly predictions
                               on 2025 test set
1. Long/Short decile           (484 stocks × 11 months)
   top vs bottom decile
   gross + net of 20bps

2. Long-Only decile
   top decile only
   gross + net of 20bps

Sharpe / drawdown /
win rate analysis
```

---

## License

MIT License — free to use, modify, and distribute with attribution.