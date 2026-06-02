"""Cross-sectional normalisation — the core noise-control step.

Raw factor values are winsorised (outliers clipped) then rank-mapped to a
0..100 percentile across the universe. Rank mapping is robust to fat tails and
to differing factor scales, and missing values are neutralised at 50 rather
than guessed — so a name is never rewarded or punished for absent data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NEUTRAL = 50.0


def winsorize(s: pd.Series, sigma: float) -> pd.Series:
    if s.std(ddof=0) == 0 or len(s) < 3:
        return s
    mu, sd = s.mean(), s.std(ddof=0)
    return s.clip(mu - sigma * sd, mu + sigma * sd)


def cross_sectional_scores(raw: dict[str, float | None], *, winsor_sigma: float = 3.0,
                           min_coverage: float = 0.5) -> dict[str, float]:
    """Map {ticker: raw_value|None} -> {ticker: 0..100}. Higher raw = better."""
    tickers = list(raw.keys())
    present = {t: float(v) for t, v in raw.items()
               if v is not None and np.isfinite(v)}
    if not tickers:
        return {}
    coverage = len(present) / len(tickers)
    if len(present) < 3 or coverage < min_coverage:
        return {t: NEUTRAL for t in tickers}
    s = winsorize(pd.Series(present), winsor_sigma)
    pct = s.rank(pct=True, method="average") * 100.0
    out = {t: float(pct[t]) for t in present}
    for t in tickers:
        out.setdefault(t, NEUTRAL)  # missing -> neutral
    return out


def weighted_blend(scores: dict[str, float], weights: dict[str, float]) -> tuple[float, float]:
    """Weighted average over the factors that actually have a (non-neutral-by-
    absence) score. Returns (blended_score, coverage_fraction).

    Factors present in *weights* but missing from *scores* are dropped and the
    remaining weights are renormalised, so an analyst with one dark factor is
    not silently dragged toward 50.
    """
    num, wsum, have = 0.0, 0.0, 0
    for f, w in weights.items():
        if f in scores:
            num += scores[f] * w
            wsum += w
            have += 1
    if wsum == 0:
        return NEUTRAL, 0.0
    return num / wsum, have / max(len(weights), 1)
