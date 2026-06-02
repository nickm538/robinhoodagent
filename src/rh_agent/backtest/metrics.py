"""Performance metrics for an equity curve vs a benchmark."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    yrs = (equity.index[-1] - equity.index[0]).days / 365.25
    if yrs <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / yrs) - 1)


def sharpe(equity: pd.Series, rf: float = 0.0, periods: int = 12) -> float:
    r = _returns(equity)
    if r.std(ddof=0) == 0 or len(r) < 2:
        return 0.0
    return float((r.mean() - rf / periods) / r.std(ddof=0) * np.sqrt(periods))


def sortino(equity: pd.Series, periods: int = 12) -> float:
    r = _returns(equity)
    downside = r[r < 0]
    if len(downside) == 0 or downside.std(ddof=0) == 0:
        return 0.0
    return float(r.mean() / downside.std(ddof=0) * np.sqrt(periods))


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())


def alpha_beta(port: pd.Series, bench: pd.Series, periods: int = 12) -> tuple[float, float]:
    rp, rb = _returns(port), _returns(bench)
    df = pd.concat([rp, rb], axis=1).dropna()
    if len(df) < 3 or df.iloc[:, 1].var() == 0:
        return 0.0, 0.0
    beta = float(np.cov(df.iloc[:, 0], df.iloc[:, 1])[0, 1] / df.iloc[:, 1].var())
    alpha = float((df.iloc[:, 0].mean() - beta * df.iloc[:, 1].mean()) * periods)
    return alpha, beta


def summarize(equity: pd.Series, bench: pd.Series | None, periods: int = 12) -> dict:
    out = {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 1 else 0.0,
        "cagr": cagr(equity),
        "sharpe": sharpe(equity, periods=periods),
        "sortino": sortino(equity, periods=periods),
        "max_drawdown": max_drawdown(equity),
        "periods": int(len(equity)),
    }
    if bench is not None and len(bench) > 1:
        b = bench.reindex(equity.index).ffill()
        out["benchmark_total_return"] = float(b.iloc[-1] / b.iloc[0] - 1)
        out["benchmark_cagr"] = cagr(b)
        out["excess_cagr"] = out["cagr"] - out["benchmark_cagr"]
        a, beta = alpha_beta(equity, b, periods)
        out["alpha_annual"] = a
        out["beta"] = beta
        out["return_multiple_vs_benchmark"] = (
            out["total_return"] / out["benchmark_total_return"]
            if out["benchmark_total_return"] not in (0.0,) else None)
    return out
