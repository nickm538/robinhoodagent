"""Portfolio construction: turn ranked verdicts into target positions with
volatility-aware sizing, per-name and per-sector caps, regime exposure, and
ATR-based protective stops.
"""
from __future__ import annotations

from .config import Config
from .debug_log import write_debug_log
from .logging_setup import get_logger
from .models import TargetPosition, TickerData, Verdict
from .regime import RegimeResult
from .risk import annualized_vol, atr_stop, risk_capped_weight, take_profit

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
        # ---- 4b) cap each name by stop-distance risk budget ----
        rc = self.p.get("risk_controls", {})
        risk_pct = float(rc.get("per_trade_risk_pct", 0.0) or 0.0)
        risk_capped: list[str] = []
        if risk_pct > 0:
            atr_mult = float(rc.get("stop_loss_atr_mult", 2.5))
            hard_pct = float(rc.get("hard_stop_pct", 0.18))
            capped = {}
            for t, w in weights.items():
                td = td_map[t]
                px = td.price
                if not px:
                    capped[t] = w
                    continue
                stop = atr_stop(px, td.technicals.get("atr"), atr_mult, hard_pct)
                capped[t] = risk_capped_weight(px, stop, w, risk_pct)
                if capped[t] + 1e-9 < w:
                    risk_capped.append(t)
            weights = capped
        post_risk_weights = dict(weights)
        # ---- 5) drop dust positions (their weight becomes cash) ----
        weights = {t: w for t, w in weights.items() if w >= min_w * 0.5}
        dust_dropped = sorted(set(post_risk_weights) - set(weights))

        # ---- 6) materialise target positions w/ stops ----
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
        # region agent log
        write_debug_log(
            hypothesis_id="E",
            location="portfolio.py:114",
            message="portfolio sizing summary",
            data={
                "eligible_count": len(eligible),
                "selected_count": len(selected),
                "target_count": len(out),
                "risk_pct": risk_pct,
                "risk_capped_count": len(risk_capped),
                "risk_capped": risk_capped[:8],
                "dust_dropped_count": len(dust_dropped),
                "dust_dropped": dust_dropped[:8],
                "invested_weight": round(sum(x.weight for x in out), 4),
                "min_weight": round(min((x.weight for x in out), default=0.0), 4),
                "max_weight": round(max((x.weight for x in out), default=0.0), 4),
            },
        )
        # endregion
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
        # Single pass: group tickers by sector while accumulating sector totals,
        # so over-cap sectors are scaled without re-scanning the whole weight map.
        sec_members: dict[str, list[str]] = {}
        sec_tot: dict[str, float] = {}
        for t, v in w.items():
            sec = td_map[t].sector
            sec_members.setdefault(sec, []).append(t)
            sec_tot[sec] = sec_tot.get(sec, 0.0) + v
        for s, tot in sec_tot.items():
            if tot > max_sec + 1e-9:
                scale = max_sec / tot
                for t in sec_members[s]:
                    w[t] *= scale
        return w
