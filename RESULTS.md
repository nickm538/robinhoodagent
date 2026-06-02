# Live validation — real data, captured 2026-06-02

This is a genuine run of the engine on **live market data**, not a mock. Prices,
fundamentals, sectors and market caps were pulled from **FinancialDatasets.AI**
at runtime (2026-06-02) for 11 liquid US equities plus SPY/RSP benchmarks, and
fed through the exact production scoring/risk/backtest code.

> **Read this honestly.** The 11-name set below is a small, **hand-picked** demo
> universe of well-known large/mega-caps. That introduces **selection bias** — it
> is *not* a full-universe scan and the backtest numbers should not be read as an
> expected forward return. In real deployment the agent scans the full liquid
> universe via the screener funnel. Also, in this restricted sandbox only **2 of
> the 5 pillars** (Momentum, Quant) had live feeds; the other three (Catalyst,
> Smart-Money, Sentiment) need Mboum/Alpha-Vantage/web egress that was blocked
> here, so they neutralised. With all five pillars live, rankings differ.

## 1) Multi-factor scan (regime-aware)

```
Regime: neutral (SPX > 200dma, breadth -8.0%) -> exposure 85%
Universe scanned: 11 | scored: 11 | eligible: 2 | target book: 2
(gates relaxed for the 2-pillar sandbox: RH_MIN_PILLARS=2, RH_MIN_CONVICTION=58)
```

| # | Ticker | Sector | Composite | Weight | $ (of $100k) | ATR stop | Take-profit | Drivers |
|---|--------|--------|----------:|-------:|-------------:|---------:|------------:|---------|
| 1 | NVDA | Information Technology | 64 | 10.0% | 10,000 | 205.35 | 269.99 | quant=92, momentum=66 |
| 2 | ANET | Information Technology | 63 | 10.0% | 10,000 | 152.25 | 214.91 | momentum=79, quant=74 |

Note the discipline: with only two names clearing the gates, the book is
**20% invested / 80% cash** — the per-name 10% cap holds and the agent refuses to
over-concentrate. (The earlier 42.5%/name output was a real bug in the allocator
caught during validation and fixed — caps are now hard ceilings, excess → cash.)

The regime engine correctly read a **narrow** tape: SPX above its 200dma but
equal-weight (RSP) lagging cap-weight (SPY) by ~8% over 3 months → `neutral`, not
`risk_on_trend`, so momentum was not over-weighted.

## 2) Walk-forward backtest vs SPY (momentum sleeve, point-in-time)

Monthly rebalance, 2023-07 → 2026-06, real adjusted prices, costs modelled
(1bp commission + 5bp slippage on turnover). This validates the **price/momentum
engine**, which is fully point-in-time; fundamentals/sentiment are deliberately
**not** replayed (snapshot APIs are current-only → would inject look-ahead bias).

| Metric | Strategy | SPY |
|---|---:|---:|
| Total return | **112.5%** | 70.7% |
| CAGR | **29.5%** | 20.1% |
| Sharpe | **1.30** | — |
| Sortino | **2.85** | — |
| Max drawdown | **-15.3%** | — |
| Annual alpha | **10.7%** | — |
| Beta | 0.92 | — |
| Return vs benchmark | **1.59×** | — |

**Caveat (important):** because the backtest universe is the same 11 curated
large-caps, the result reflects those specific names' momentum over a strong
2023–2026 tape — it overstates what a blind full-universe strategy would have
earned. It demonstrates the *mechanics and edge of the momentum sleeve*, not a
guaranteed forward 1.6×. The full system's true test is **forward paper trading**.

## 3) Autonomous loop (paper)

A single autonomous cycle on the live snapshot rebalanced and **filled the paper
account** on live prices:

```
NVDA  44.5712 sh @ 224.47  ($10,000)
ANET  58.5892 sh @ 170.77  ($10,000)
cash $79,990   equity $99,990   (–$10 = modelled slippage)
```

State persists to `state/`; subsequent cycles manage stops/take-profits every
tick and rebalance on the weekly cadence.

## Reproduce

```bash
RH_MIN_PILLARS=2 RH_MIN_CONVICTION=58 \
PYTHONPATH=src python -m rh_agent.cli scan --snapshot data_cache/live_snapshot_20260602.json --md
PYTHONPATH=src python -m rh_agent.cli backtest --snapshot data_cache/live_snapshot_20260602.json --start 2023-07-01
```

*(The snapshot file lives under `data_cache/` which is gitignored; regenerate it
from captured price files with `scripts/assemble_snapshot.py`, or point the agent
at live providers with your API keys.)*
