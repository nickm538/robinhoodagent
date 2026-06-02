"""Walk-forward backtester for the price/momentum sleeve.

At each month-end it ranks the universe on point-in-time momentum (12-1, 6-1,
3-1) computed only from data available up to that date, selects the top names,
weights them inverse-to-volatility (capped), holds to the next month-end, and
charges commission + slippage on turnover. Equity is compared to the benchmark.

HONEST SCOPE: this validates the momentum/technical engine, which is fully
point-in-time from price history. Fundamental/sentiment/insider factors are
*current-snapshot* from the data APIs and are deliberately NOT replayed here —
doing so would inject look-ahead bias. Treat live multi-factor results as the
forward test; treat this as the historical proof of the price engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_setup import get_logger
from . import metrics

log = get_logger("backtest")


@dataclass
class BacktestResult:
    equity: pd.Series
    benchmark: pd.Series
    stats: dict = field(default_factory=dict)
    holdings: dict = field(default_factory=dict)   # date -> {ticker: weight}


class Backtester:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.bt = cfg.get("backtest", {})
        self.liq = cfg.get("universe.liquidity", {})
        self.n = int(cfg.get("portfolio.target_positions", 15))
        self.max_w = float(cfg.get("portfolio.max_position_weight", 0.10))

    @staticmethod
    def _close(df: pd.DataFrame) -> pd.Series:
        return df["adj_close"] if "adj_close" in df.columns else df["close"]

    def _signal(self, c: pd.Series, asof) -> dict | None:
        c = c[c.index <= asof].dropna()
        if len(c) < 252:
            return None
        try:
            m12 = c.iloc[-21] / c.iloc[-252] - 1
            m6 = c.iloc[-21] / c.iloc[-126] - 1
            m3 = c.iloc[-21] / c.iloc[-63] - 1
        except Exception:
            return None
        vol = c.pct_change().dropna().iloc[-63:].std() * np.sqrt(252)
        composite = 0.5 * m12 + 0.3 * m6 + 0.2 * m3
        return {"composite": float(composite), "vol": float(vol or 0.3),
                "price": float(c.iloc[-1])}

    def run(self, prices: dict[str, pd.DataFrame], benchmark: pd.DataFrame,
            start: str | None = None, end: str | None = None) -> BacktestResult:
        bench_c = self._close(benchmark).sort_index()
        start = pd.to_datetime(start or self.bt.get("start", "2019-01-01"))
        end = pd.to_datetime(end) if end and end != "auto" else bench_c.index[-1]
        # month-end rebalance grid
        dates = bench_c.resample("ME").last().index
        dates = [d for d in dates if start <= d <= end]
        closes = {t: self._close(df).sort_index() for t, df in prices.items()}

        cost_bps = (self.bt.get("commission_bps", 1.0) + self.bt.get("slippage_bps", 5.0)) / 1e4
        equity = float(self.bt.get("initial_equity", 100_000))
        eq_curve = {dates[0]: equity}
        prev_w: dict[str, float] = {}
        holdings: dict = {}

        for i in range(len(dates) - 1):
            d, nxt = dates[i], dates[i + 1]
            sigs = {}
            for t, c in closes.items():
                if c.empty or float(c[c.index <= d].iloc[-1] if len(c[c.index <= d]) else 0) < \
                        self.liq.get("min_price", 5):
                    continue
                s = self._signal(c, d)
                if s:
                    sigs[t] = s
            if len(sigs) < 5:
                eq_curve[nxt] = equity
                continue
            top = sorted(sigs, key=lambda t: sigs[t]["composite"], reverse=True)[: self.n]
            inv = {t: 1.0 / max(sigs[t]["vol"], 0.08) for t in top}
            s = sum(inv.values())
            w = {t: inv[t] / s for t in top}
            w = self._cap(w, self.max_w)
            holdings[str(d.date())] = {t: round(wi, 4) for t, wi in w.items()}

            # period return per name d -> nxt
            port_ret = 0.0
            for t, wi in w.items():
                c = closes[t]
                p0 = c[c.index <= d]
                p1 = c[c.index <= nxt]
                if len(p0) and len(p1) and p0.iloc[-1] > 0:
                    port_ret += wi * (p1.iloc[-1] / p0.iloc[-1] - 1)
            turnover = sum(abs(w.get(t, 0) - prev_w.get(t, 0)) for t in set(w) | set(prev_w))
            equity *= (1 + port_ret - turnover * cost_bps)
            eq_curve[nxt] = equity
            prev_w = w

        eq = pd.Series(eq_curve).sort_index()
        bench_eq = bench_c.reindex(eq.index, method="ffill")
        bench_eq = bench_eq / bench_eq.iloc[0] * float(self.bt.get("initial_equity", 100_000))
        stats = metrics.summarize(eq, bench_eq, periods=12)
        log.info("backtest: total %.1f%% vs bench %.1f%% | CAGR %.1f%% vs %.1f%% | Sharpe %.2f | maxDD %.1f%%",
                 100 * stats["total_return"], 100 * stats.get("benchmark_total_return", 0),
                 100 * stats["cagr"], 100 * stats.get("benchmark_cagr", 0),
                 stats["sharpe"], 100 * stats["max_drawdown"])
        return BacktestResult(equity=eq, benchmark=bench_eq, stats=stats, holdings=holdings)

    @staticmethod
    def _cap(w: dict, cap: float) -> dict:
        w = dict(w)
        for _ in range(20):
            over = {t: v for t, v in w.items() if v > cap + 1e-9}
            if not over:
                break
            excess = sum(v - cap for v in over.values())
            for t in over:
                w[t] = cap
            under = [t for t in w if w[t] < cap - 1e-9]
            base = sum(w[t] for t in under)
            if base <= 0:
                break
            for t in under:
                w[t] += excess * w[t] / base
        return w
