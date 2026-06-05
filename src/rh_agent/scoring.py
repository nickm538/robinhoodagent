"""Scoring orchestrator: TickerData -> raw factors -> cross-sectional
normalisation -> panel verdicts. This is where the universe is ranked.
"""
from __future__ import annotations

from .analysts.panel import Panel
from .config import Config
from .factors.library import FACTORS, compute_raw_factors
from .factors.normalize import cross_sectional_scores
from .logging_setup import get_logger
from .models import TickerData, Verdict
from .regime import RegimeResult

log = get_logger("scoring")


class Scorer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.panel = Panel(cfg)
        self.norm = cfg.get("normalize", {})

    def score(self, data: list[TickerData], regime: RegimeResult) -> list[Verdict]:
        tickers = [td.ticker for td in data]
        if not tickers:
            return []

        # 1) raw factor values per ticker
        raw_per_ticker = {td.ticker: compute_raw_factors(td) for td in data}

        # 2) build per-factor raw maps over the full universe (None where absent)
        raw_by_factor: dict[str, dict] = {}
        presence: dict[str, set] = {}
        for fname in FACTORS:
            col, have = {}, set()
            for t in tickers:
                v = raw_per_ticker[t].get(fname)
                col[t] = v
                if v is not None:
                    have.add(t)
            raw_by_factor[fname] = col
            presence[fname] = have

        # 3) cross-sectional normalisation (winsorise + rank -> 0..100)
        ws = self.norm.get("winsorize_sigma", 3.0)
        mc = self.norm.get("min_factor_coverage", 0.5)
        normalized_by_factor = {
            f: cross_sectional_scores(raw_by_factor[f], winsor_sigma=ws, min_coverage=mc)
            for f in FACTORS
        }

        # 4) panel verdicts + flags
        verdicts = []
        td_map = {td.ticker: td for td in data}
        for t in tickers:
            v = self.panel.evaluate(t, normalized_by_factor, presence, regime)
            self._add_flags(v, td_map[t])
            verdicts.append(v)

        verdicts.sort(key=lambda x: x.composite, reverse=True)
        return verdicts

    def _add_flags(self, v: Verdict, td: TickerData) -> None:
        d = td.earnings.get("days_to_next")
        if isinstance(d, (int, float)) and d < 5:
            v.flags.append(f"earnings_in_{int(d)}d")
        atr_pct = td.technicals.get("atr_pct")
        if atr_pct and atr_pct > 0.06:
            v.flags.append("high_volatility")
        if td.market_cap and td.market_cap < 2e9:
            v.flags.append("smallcap")

    def eligible(self, verdicts: list[Verdict]) -> list[Verdict]:
        """Apply conviction floor + multi-pillar confirmation.

        The two gates can be overridden via env (RH_MIN_CONVICTION,
        RH_MIN_PILLARS) — useful when fewer data pillars are wired in a given
        environment, so the multi-pillar requirement scales to what is live.
        """
        import os
        min_conf = float(os.getenv("RH_MIN_CONVICTION",
                                   self.cfg.get("portfolio.min_conviction_score", 60.0)))
        min_pillars = int(float(os.getenv("RH_MIN_PILLARS",
                                          self.norm.get("min_pillars_passing", 3))))
        out = [v for v in verdicts
               if v.composite >= min_conf and v.pillars_passing >= min_pillars]
        log.info("eligible: %d/%d names clear conviction>=%.0f & pillars>=%d",
                 len(out), len(verdicts), min_conf, min_pillars)
        return out
