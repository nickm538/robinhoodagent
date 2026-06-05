"""Portfolio construction: turn ranked verdicts into target positions with
volatility-aware sizing, per-name and per-sector caps, regime exposure, and
ATR-based protective stops.
"""
from __future__ import annotations

from .config import Config
from .logging_setup import get_logger
from .models import TargetPosition, TickerData, Verdict
from .regime import RegimeResult
from .risk import annualized_vol, atr_stop, take_profit

log = get_logger("portfolio")


class PortfolioBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.p = cfg.get("portfolio", {})

    def build(self, eligible: list[Verdict], td_map: dict[str, TickerData],
              regime: RegimeResult, equity: float) -> list[TargetPosition]:
        if not eligible or equity <= 0:
            return []
        n = int(self.p.get("target_positions", 15))
        max_w = float(self.p.get("max_position_weight", 0.10))
        min_w = float(self.p.get("min_position_weight", 0.02))
        max_sec = float(self.p.get("max_sector_weight", 0.35))
        tilt = float(self.p.get("conviction_tilt", 0.5))
        scaled = self._autoscale_params(equity)     # grow the book with the account
        if scaled:
            n, max_w, max_sec = scaled

        # ---- 1) select top names honouring an approximate sector cap ----
        selected: list[Verdict] = []
        sector_budget: dict[str, float] = {}
        per_name_cap_count = max(1, int(round(max_sec / max_w)))
        for v in eligible:
            sec = td_map[v.ticker].sector
            if sector_budget.get(sec, 0) >= per_name_cap_count:
                continue
            selected.append(v)
            sector_budget[sec] = sector_budget.get(sec, 0) + 1
            if len(selected) >= n:
                break
        if not selected:
            return []

        # ---- 2) raw inverse-vol and conviction weights ----
        inv_vol, conv = {}, {}
        for v in selected:
            td = td_map[v.ticker]
            vol = max(annualized_vol(td), 0.08)
            inv_vol[v.ticker] = 1.0 / vol
            conv[v.ticker] = v.composite / 100.0
        inv_sum, conv_sum = sum(inv_vol.values()), sum(conv.values())
        weights = {t: (1 - tilt) * inv_vol[t] / inv_sum + tilt * conv[t] / conv_sum
                   for t in inv_vol}

        # ---- 3) normalise raw blend to sum 1, then scale to regime exposure ----
        tot = sum(weights.values())
        if tot <= 0:
            return []
        weights = {t: w / tot for t, w in weights.items()}
        invest = min(regime.exposure, 1.0 - float(self.p.get("cash_floor", 0.02)))
        weights = {t: w * invest for t, w in weights.items()}
        # ---- 4) enforce per-name and per-sector caps as HARD ceilings.
        #         Excess that cannot be redistributed below the caps stays in
        #         cash (we never re-inflate past a cap). ----
        weights = self._cap(weights, max_w)
        weights = self._sector_cap(weights, td_map, max_sec)
        # ---- 5) drop dust positions (their weight becomes cash) ----
        weights = {t: w for t, w in weights.items() if w >= min_w * 0.5}

        # ---- 6) materialise target positions w/ stops ----
        rc = self.p.get("risk_controls", {})
        out: list[TargetPosition] = []
        for v in selected:
            t = v.ticker
            if t not in weights:
                continue
            td = td_map[t]
            px = td.price
            if not px:
                continue
            atr = td.technicals.get("atr")
            dollars = weights[t] * equity
            tp = TargetPosition(
                ticker=t, weight=round(weights[t], 4), score=round(v.composite, 1),
                dollars=round(dollars, 2), shares=round(dollars / px, 4),
                sector=td.sector,
                stop_price=atr_stop(px, atr, rc.get("stop_loss_atr_mult", 2.5),
                                    rc.get("hard_stop_pct", 0.18)),
                take_profit=take_profit(px, atr, rc.get("take_profit_atr_mult", 6.0)),
                rationale=v.rationale,
            )
            out.append(tp)
        out.sort(key=lambda x: x.weight, reverse=True)
        log.info("portfolio: %d positions, %.0f%% invested (%s regime)",
                 len(out), 100 * sum(x.weight for x in out), regime.name)
        return out

    def _autoscale_params(self, equity: float) -> "tuple[int, float, float] | None":
        """Scale the book to the account size: pick the highest tier whose
        min-equity <= current equity and return (positions, max_name_weight,
        max_sector_weight). Returns None when disabled or misconfigured, so the
        static portfolio.* caps are used instead."""
        a = self.p.get("autoscale", {}) or {}
        if not a.get("enabled"):
            return None
        try:
            tiers = [t for t in (a.get("tiers") or [])
                     if isinstance(t, (list, tuple)) and len(t) >= 4]
            usable = [t for t in tiers if equity >= float(t[0])]
            if not usable:
                return None
            t = max(usable, key=lambda x: float(x[0]))
            n, mw, msec = int(t[1]), float(t[2]), float(t[3])
        except (TypeError, ValueError):
            log.warning("autoscale: malformed tiers — using static portfolio caps")
            return None
        log.info("autoscale: equity $%.0f -> %d positions, %.0f%% name cap, %.0f%% sector cap",
                 equity, n, 100 * mw, 100 * msec)
        return n, mw, msec

    @staticmethod
    def _cap(weights: dict, cap: float) -> dict:
        w = dict(weights)
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
                w[t] += excess * (w[t] / base)
        return w

    def _sector_cap(self, weights: dict, td_map: dict, max_sec: float) -> dict:
        """Scale any sector whose total exceeds the cap back down to the cap.
        Freed weight becomes cash (we do not pile it into other names)."""
        w = dict(weights)
        sec_tot: dict[str, float] = {}
        for t, v in w.items():
            sec_tot[td_map[t].sector] = sec_tot.get(td_map[t].sector, 0.0) + v
        for s, tot in sec_tot.items():
            if tot > max_sec + 1e-9:
                scale = max_sec / tot
                for t in [t for t in w if td_map[t].sector == s]:
                    w[t] *= scale
        return w
