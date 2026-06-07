"""The Panel of Five.

Each persona (momentum_trader, quant, catalyst_trader, smart_money,
sentiment_analyst) blends its assigned factors into a 0..100 conviction. The
Chief PM then blends the five persona scores using the *regime* weights to
produce the composite. Requiring several personas to agree (min_pillars) is
what turns five noisy opinions into one robust decision.
"""
from __future__ import annotations

from ..config import Config
from ..factors.normalize import weighted_blend
from ..models import Verdict
from ..regime import RegimeResult


class Panel:
    def __init__(self, cfg: Config):
        self.personas: dict = cfg.get("analysts", {})
        self.norm = cfg.get("normalize", {})

    def evaluate(self, ticker: str, normalized_by_factor: dict, presence: dict,
                 regime: RegimeResult) -> Verdict:
        analyst_scores: dict[str, float] = {}
        coverages: dict[str, float] = {}
        for persona, spec in self.personas.items():
            wts = spec.get("factors", {})
            scores = {f: normalized_by_factor[f][ticker]
                      for f in wts if ticker in presence.get(f, set())}
            blend, cov = weighted_blend(scores, wts)
            analyst_scores[persona] = blend
            coverages[persona] = cov

        composite, _ = weighted_blend(analyst_scores, regime.weights)

        thr = self.norm.get("pillar_pass_threshold", 55.0)
        pillars = sum(1 for p, s in analyst_scores.items()
                      if s >= thr and coverages.get(p, 0) > 0)

        ranked = sorted(analyst_scores.items(), key=lambda kv: kv[1], reverse=True)
        rationale = " | ".join(f"{p}={s:.0f}" for p, s in ranked)
        return Verdict(ticker=ticker, composite=float(composite),
                       analyst_scores={p: round(s, 1) for p, s in analyst_scores.items()},
                       pillars_passing=pillars, rationale=rationale)
