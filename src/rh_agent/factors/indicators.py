"""Pure-python technical indicators computed from an OHLCV DataFrame.

Computing locally (vs hitting a metrics API per indicator) is faster, has no
rate limits, works offline on cached prices, and is fully reproducible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> float | None:
    if len(close) < period + 1:
        return None
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    val = 100 - 100 / (1 + rs.iloc[-1])
    return float(val) if pd.notna(val) else 100.0


def macd_hist(close: pd.Series, fast=12, slow=26, signal=9) -> float | None:
    if len(close) < slow + signal:
        return None
    macd = _ema(close, fast) - _ema(close, slow)
    sig = _ema(macd, signal)
    return float((macd - sig).iloc[-1])


def atr(df: pd.DataFrame, period: int = 14) -> float | None:
    if not {"high", "low", "close"}.issubset(df.columns) or len(df) < period + 1:
        return None
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1])


def adx(df: pd.DataFrame, period: int = 14) -> float | None:
    if not {"high", "low", "close"}.issubset(df.columns) or len(df) < 2 * period:
        return None
    h, l, c = df["high"], df["low"], df["close"]
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_ = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    val = dx.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    return float(val) if pd.notna(val) else None


def obv_trend(df: pd.DataFrame, lookback: int = 20) -> float | None:
    if "volume" not in df.columns or len(df) < lookback + 1:
        return None
    sign = np.sign(df["close"].diff().fillna(0))
    obv = (sign * df["volume"]).cumsum()
    recent = obv.iloc[-lookback:]
    if recent.std() == 0:
        return 0.0
    # slope normalised by volume scale -> dimensionless trend
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent.values, 1)[0]
    return float(slope / (df["volume"].iloc[-lookback:].mean() + 1e-9))


def sma(close: pd.Series, period: int) -> float | None:
    if len(close) < period:
        return None
    return float(close.iloc[-period:].mean())


def compute_indicators(df: pd.DataFrame) -> dict:
    """Compute the standard indicator bundle used by the technicals factor."""
    if df is None or len(df) < 30:
        return {}
    close = df["close"]
    out: dict = {}
    out["rsi"] = rsi(close)
    out["macd_hist"] = macd_hist(close)
    out["atr"] = atr(df)
    a = atr(df)
    px = float(close.iloc[-1])
    out["atr_pct"] = (a / px) if (a and px) else None
    out["adx"] = adx(df)
    out["obv_trend"] = obv_trend(df)
    out["sma50"] = sma(close, 50)
    out["sma200"] = sma(close, 200)
    out["price"] = px
    # annualised volatility from daily returns (last ~63 trading days)
    rets = close.pct_change().dropna()
    if len(rets) >= 20:
        out["volatility"] = float(rets.iloc[-63:].std() * np.sqrt(252))
    if len(close) >= 252:
        out["high_52w"] = float(close.iloc[-252:].max())
        out["low_52w"] = float(close.iloc[-252:].min())
    return out
