"""TradingAgent — the orchestrator. Scans the universe, scores it through the
panel, constructs the target book, reconciles against the broker, and (only
when explicitly armed) executes via the Robinhood Agentic MCP.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd

from .config import Config
from .data.market_data import MarketData
from .execution import build_orders, execute_orders
from .logging_setup import get_logger
from .models import Account, Order, Position, TargetPosition, TickerData, Verdict
from .portfolio import PortfolioBuilder
from .providers import build_providers, snapshot_priorities
from .regime import RegimeResult, detect_regime
from .scoring import Scorer

log = get_logger("agent")


@dataclass
class ScanResult:
    regime: RegimeResult
    verdicts: list[Verdict]
    eligible: list[Verdict]
    targets: list[TargetPosition]
    equity: float
    universe_size: int
    scored_size: int
    td_map: dict = field(default_factory=dict)
    ai_market_read: str = ""


@dataclass
class RunResult:
    scan: ScanResult
    account: Account
    orders: list[Order]
    post_account: Account | None = None
    fills: list[dict] = field(default_factory=list)
    executed: bool = False
    mode: str = "paper"
    # Why-no-trade transparency: per-name decisions build_orders suppressed/shaped
    # (hold-within-band, min-notional skips, caps). Feeds the activity ledger.
    order_notes: list = field(default_factory=list)


class TradingAgent:
    def __init__(self, cfg: Config, snapshot_path: str | None = None):
        self.cfg = cfg
        self.providers = build_providers(cfg, snapshot_path)
        if snapshot_path:
            cfg.raw["providers"] = {**(cfg.get("providers") or {}), **snapshot_priorities()}
        self.md = MarketData(cfg, self.providers)
        self.scorer = Scorer(cfg)
        self.builder = PortfolioBuilder(cfg)
        from .analysts.ai_analyst import AIAnalyst
        self.ai = AIAnalyst(cfg)
        from .journal import Journal
        self.journal = Journal(cfg)
        self._quote_cache: dict[str, float | None] = {}

    # ---- helpers ----
    def default_equity(self) -> float:
        return float(os.getenv("PAPER_EQUITY", self.cfg.get("backtest.initial_equity", 100_000)))

    def price_fn(self, ticker: str, *, for_risk: bool = False) -> float | None:
        if not for_risk and ticker in self._quote_cache:
            return self._quote_cache[ticker]
        if for_risk:
            q = self.md.get_quote_for_risk(ticker)
        else:
            q = self.md.get_quote(ticker)
        px = q.price if q else None
        if not for_risk:
            self._quote_cache[ticker] = px
        return px

    def clear_price_cache(self) -> None:
        """Drop cached quotes so the next price_fn refetches. The daemon calls this
        every tick so stop-loss checks never evaluate against a stale, frozen price."""
        self._quote_cache.clear()

    def universe(self, limit: int | None = None, watchlist: list[str] | None = None,
                 include_tickers: list[str] | None = None) -> list[str]:
        # explicit --tickers (even if empty) overrides config universe.watchlist;
        # only fall back to config when no watchlist was passed at all.
        wl = watchlist if watchlist is not None else (self.cfg.get("universe.watchlist") or [])
        if wl:
            tickers = [t.strip().upper() for t in wl if t and t.strip()]
        elif "snapshot" in self.providers:
            tickers = self.providers["snapshot"].list_universe()
        else:
            from .data.universe import build_universe
            tickers = build_universe(self.md, self.cfg)
        if include_tickers:
            seen = set(tickers)
            for t in include_tickers:
                tk = t.strip().upper() if isinstance(t, str) else ""
                if tk and tk not in seen:
                    tickers.append(tk)
                    seen.add(tk)
        return tickers[:limit] if limit else tickers

    def _gather(self, tickers: list[str], deep: bool = True) -> list[TickerData]:
        """Build TickerData for the universe. Live providers are I/O-bound, so we
        fan out across a thread pool (each provider rate-limits itself); the
        in-memory snapshot path stays single-threaded."""
        def build(t: str) -> TickerData | None:
            try:
                td = self.md.build(t, deep=deep)
                return td if td.price else None
            except Exception as e:
                log.debug("build %s failed: %s", t, e)
                return None

        workers = int(self.cfg.get("data.max_workers", 8))
        if workers > 1 and "snapshot" not in self.providers and len(tickers) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(build, tickers))
        else:
            results = [build(t) for t in tickers]
        out = [r for r in results if r is not None]
        log.info("gathered %d/%d priced names", len(out), len(tickers))
        return out

    # ---- scan & score ----
    def scan(self, equity: float | None = None, limit: int | None = None,
             tickers: list[str] | None = None, include_tickers: list[str] | None = None,
             exclude_tickers: set[str] | None = None) -> ScanResult:
        if equity is None:                 # 0.0 is a valid (empty-account) sizing -> keep it
            equity = self.default_equity()
        names = self.universe(limit, watchlist=tickers, include_tickers=include_tickers)
        full_n = len(names)
        regime = detect_regime(self.md, self.cfg)

        # Two-stage funnel: on a broad universe, cheaply light-rank everything on
        # fast signals (momentum/quality from prices+fundamentals), then run the
        # full deep scan + AI analyst on only the top survivors. Keeps big scans
        # tractable on small hardware without losing the wide hunt.
        u = self.cfg.get("universe", {}) or {}
        threshold = int(u.get("two_stage_threshold", 40))
        top_k = int(u.get("deep_top_k", 30))
        if full_n > threshold and "snapshot" not in self.providers:
            log.info("two-stage scan: light-screening %d names -> deep top %d", full_n, top_k)
            light = self._gather(names, deep=False)
            light_v = self.scorer.score(light, regime)
            names = [v.ticker for v in light_v[:top_k]]
            if include_tickers:
                seen = set(names)
                for t in include_tickers:
                    tk = t.strip().upper() if isinstance(t, str) else ""
                    if tk and tk not in seen:
                        names.append(tk)
                        seen.add(tk)

        log.info("deep-scoring %d tickers", len(names))
        data = self._gather(names, deep=True)
        # A deep gather (+ AI) can take many minutes; quotes fetched at the start
        # would otherwise all flag stale_quote and zero out eligibility.
        freshness = self.cfg.get("data.freshness", {}) or {}
        max_quote_age = float(freshness.get("quote_max_age_seconds", 120))
        refresh = getattr(self.md, "refresh_quotes", None)
        if data and "snapshot" not in self.providers and refresh:
            refresh([td.ticker for td in data], max_age_seconds=max_quote_age)
            for td in data:
                q = self.md.get_quote(td.ticker)
                if q:
                    td.quote = q
        for td in data:
            if td.quote:
                self._quote_cache[td.ticker] = td.quote.price
        verdicts = self.scorer.score(data, regime)
        self._apply_intraday_hunter_boost(verdicts, {td.ticker: td for td in data})
        td_map = {td.ticker: td for td in data}
        ai_read = self._apply_ai_overlay(verdicts, td_map, regime)
        eligible = self.scorer.eligible(verdicts)
        if exclude_tickers:
            # Names under a re-entry cooldown must not consume a book slot — drop
            # them BEFORE construction so the next-best name backfills the book.
            before = len(eligible)
            eligible = [v for v in eligible if v.ticker not in exclude_tickers]
            if len(eligible) != before:
                log.info("scan: excluded %d cooled-down name(s) from the book",
                         before - len(eligible))
        targets = self.builder.build(eligible, td_map, regime, equity)
        return ScanResult(regime=regime, verdicts=verdicts, eligible=eligible, targets=targets,
                          equity=equity, universe_size=full_n, scored_size=len(data),
                          td_map=td_map, ai_market_read=ai_read)

    def _apply_intraday_hunter_boost(self, verdicts: list[Verdict],
                                     td_map: dict[str, TickerData]) -> None:
        """Lift today's runners in the cross-section so snipes beat sleepy mega-caps."""
        intraday = self.cfg.get("universe.intraday", {}) or {}
        if not intraday.get("enabled", False):
            return
        boost_max = float(intraday.get("composite_boost_max", 15.0))
        min_day = float(intraday.get("min_positive_day_change_pct", 0.5))
        for v in verdicts:
            td = td_map.get(v.ticker)
            if not td or not td.quote:
                continue
            dc = float(td.quote.day_change_pct or 0.0)
            if dc < min_day:
                continue
            rel = 1.0
            if td.prices is not None and len(td.prices) and "volume" in td.prices.columns:
                avg = float(td.prices["volume"].iloc[-22:-1].mean() or 0)
                latest = float(td.quote.volume or td.prices["volume"].iloc[-1] or 0)
                if avg > 0:
                    rel = min(latest / avg, 6.0)
            bump = min(boost_max, dc * 1.2 + rel * 1.5)
            if bump > 0:
                v.composite = round(min(100.0, v.composite + bump), 1)
                v.rationale += f" | snipe+{bump:.0f}"
        verdicts.sort(key=lambda x: x.composite, reverse=True)

    def _apply_ai_overlay(self, verdicts, td_map, regime) -> str:
        """Blend the Claude AI analyst's view into the composite. No-op if the
        analyst is disabled (no key/SDK) or the call fails."""
        if not getattr(self, "ai", None) or not self.ai.enabled or not verdicts:
            return ""
        top = verdicts[: self.ai.max_candidates]
        cands = []
        for v in top:
            td = td_map.get(v.ticker)
            if not td:
                continue
            f = td.fundamentals
            cands.append({
                "ticker": v.ticker, "sector": td.sector,
                "quant_composite": round(v.composite, 1),
                "quant_pillars": v.analyst_scores,
                "flags": list(v.flags),
                "data_quality": self._data_quality_context(td),
                "price_pattern": self._historical_pattern_context(td),
                "market_relationship": self._market_relationship_context(td),
                "options_flow": {k: td.options.get(k) for k in
                                 ("put_call_ratio", "iv_rank", "call_volume", "put_volume")
                                 if td.options.get(k) is not None},
                "short_interest": {k: td.short_interest.get(k) for k in
                                   ("short_pct_float", "days_to_cover", "short_shares")
                                   if td.short_interest.get(k) is not None},
                "smart_money": {
                    "institutional_net_change_pct": td.institutional.get("net_change_pct"),
                    "insider_transactions": len(td.insider or []),
                },
                "fundamentals": {k: round(f[k], 3) for k in
                                 ("roe", "net_margin", "revenue_growth", "earnings_growth",
                                  "pe_ratio", "debt_to_equity") if isinstance(f.get(k), (int, float))},
                "news_sentiment": td.news_sentiment.get("score"),
                "days_to_earnings": td.earnings.get("days_to_next"),
                "headlines": self.md.headlines(v.ticker, 5),
            })
        ctx = f"Regime: {regime.describe()}. Recent market headlines: {self.md.market_news(8)}"
        perf = self.journal.performance_summary()
        if perf:
            ctx += f"\n{perf}"
        res = self.ai.assess(ctx, cands)
        if not res.views:
            return res.market_read or ""
        w = self.ai.weight
        for v in verdicts:
            av = res.views.get(v.ticker)
            if not av:
                continue
            v.analyst_scores["ai_analyst"] = round(av["score"], 1)
            v.composite = round(max(0.0, min(100.0, (1 - w) * v.composite + w * av["score"])), 1)
            v.rationale += f" | AI {av['stance']}: {av['rationale']}"
            if av["stance"] == "bearish" and av["score"] < 40:
                v.flags.append("ai_caution")
        verdicts.sort(key=lambda x: x.composite, reverse=True)
        log.info("AI overlay: blended %d names (weight %.2f) | %s",
                 len(res.views), w, res.market_read[:120])
        return res.market_read or ""

    def _data_quality_context(self, td: TickerData) -> dict:
        quote_age = None
        if td.quote is not None:
            try:
                from .models import utcnow
                quote_age = round((utcnow() - td.quote.asof).total_seconds(), 1)
            except Exception:
                quote_age = None
        price_last = None
        if td.prices is not None and len(td.prices):
            try:
                price_last = str(pd.to_datetime(td.prices.index[-1]).date())
            except Exception:
                price_last = None
        return {
            "quote_source": getattr(td.quote, "source", "") if td.quote else "",
            "quote_age_seconds": quote_age,
            "price_history_last_date": price_last,
            "has_options": bool(td.options),
            "has_short_interest": bool(td.short_interest),
            "has_news_sentiment": bool(td.news_sentiment),
            "has_institutional": bool(td.institutional),
        }

    def _historical_pattern_context(self, td: TickerData) -> dict:
        if td.prices is None or "close" not in td.prices or len(td.prices) < 30:
            return {}
        close = td.prices["adj_close"] if "adj_close" in td.prices else td.prices["close"]
        close = close.dropna()
        if len(close) < 30:
            return {}
        out: dict = {}
        for days in (5, 21, 63, 126, 252):
            if len(close) > days and close.iloc[-days - 1] > 0:
                out[f"return_{days}d"] = round(float(close.iloc[-1] / close.iloc[-days - 1] - 1), 4)
        rolling_high = close.iloc[-63:].max() if len(close) >= 63 else close.max()
        if rolling_high:
            out["drawdown_from_63d_high"] = round(float(close.iloc[-1] / rolling_high - 1), 4)
        rets = close.pct_change().dropna()
        if len(rets) >= 20:
            out["realized_vol_63d"] = round(float(rets.iloc[-63:].std() * (252 ** 0.5)), 4)
        return out

    def _market_relationship_context(self, td: TickerData) -> dict:
        if td.prices is None or "close" not in td.prices or len(td.prices) < 63:
            return {}
        try:
            spy = self.md.get_index_prices("SPY")
        except Exception:
            spy = None
        if spy is None or "close" not in spy or len(spy) < 63:
            return {}
        own = (td.prices["adj_close"] if "adj_close" in td.prices else td.prices["close"]).pct_change()
        bench = (spy["adj_close"] if "adj_close" in spy else spy["close"]).pct_change()
        aligned = pd.concat([own.rename("own"), bench.rename("spy")], axis=1).dropna().iloc[-126:]
        if len(aligned) < 30:
            return {}
        corr = float(aligned["own"].corr(aligned["spy"]))
        var = float(aligned["spy"].var())
        beta = float(aligned["own"].cov(aligned["spy"]) / var) if var else None
        return {
            "spy_correlation": round(corr, 3),
            "spy_beta": round(beta, 3) if beta is not None else None,
            "note": "Correlation is descriptive only; require a plausible causal channel.",
        }

    # ---- broker ----
    def make_broker(self):
        from .broker.errors import LiveBrokerUnavailable
        from .broker.paper import PaperBroker
        if self.cfg.live_trading_armed:
            acct = os.getenv("ROBINHOOD_ACCOUNT_NUMBER")
            url = self.cfg.robinhood_url()
            # 1) durable SDK/OAuth path (preferred — auto-refreshes tokens)
            try:
                from .broker.oauth import FileTokenStorage
                if FileTokenStorage().has_tokens():
                    from .broker.robinhood_sdk import RobinhoodSDKBroker
                    log.warning("LIVE broker active (Robinhood SDK/OAuth)")
                    return RobinhoodSDKBroker(url, account_number=acct)
            except Exception as e:
                log.error("Robinhood SDK broker unavailable: %s", e)
            # 2) static-token path (ROBINHOOD_MCP_TOKEN)
            tok = self.cfg.robinhood_token()
            if tok:
                try:
                    from .broker.robinhood_mcp import RobinhoodMCPBroker
                    log.warning("LIVE broker active (Robinhood static-token MCP)")
                    return RobinhoodMCPBroker(url, tok, account_number=acct)
                except Exception as e:
                    log.error("Robinhood token broker unavailable: %s", e)
            raise LiveBrokerUnavailable(
                "Live trading is armed but Robinhood auth is unavailable. "
                "Run `rh-agent auth` or set ROBINHOOD_MCP_TOKEN — will NOT fall back to paper."
            )
        return PaperBroker(self.price_fn, starting_cash=self.default_equity(),
                           slippage_bps=self.cfg.get("backtest.slippage_bps", 5.0))

    # ---- full run ----
    def run(self, execute: bool = False, tickers: list[str] | None = None, *,
            allow_buys: bool = True, exclude_tickers: set[str] | None = None) -> RunResult:
        broker = self.make_broker()
        account = broker.get_account()
        if broker.supports_live and not account.reliable:
            log.error("live account snapshot unreliable — skipping cycle entirely (no orders)")
            empty = ScanResult(regime=RegimeResult("halted", {}, 0.0), verdicts=[], eligible=[],
                               targets=[], equity=0.0, universe_size=0, scored_size=0)
            return RunResult(scan=empty, account=account, orders=[], fills=[], executed=False,
                             mode="live")
        equity = account.equity if (account.equity and account.equity > 0) else self.default_equity()
        held_tickers = [p.ticker for p in account.positions]
        scan = self.scan(equity=equity, tickers=tickers, include_tickers=held_tickers)
        return self.reconcile_and_execute(
            scan, execute=execute, allow_buys=allow_buys, exclude_tickers=exclude_tickers,
            broker=broker, account=account, equity=equity)

    def reconcile_and_execute(self, scan: ScanResult, *, execute: bool = False,
                              allow_buys: bool = True, exclude_tickers: set[str] | None = None,
                              exclude_buy_tickers: set[str] | None = None,
                              broker=None, account: Account | None = None,
                              equity: float | None = None) -> RunResult:
        """Turn a (possibly pre-computed, off-thread) scan into orders and execute.

        Split out from run() so the daemon can compute the slow scan in a
        background worker while risk management keeps ticking, then reconcile
        against a FRESH account snapshot here on the main thread. All broker I/O
        lives in this method — the background scan never touches the broker.
        """
        if broker is None:
            broker = self.make_broker()
        if account is None:
            account = broker.get_account()
            if broker.supports_live and not account.reliable:
                log.error("live account snapshot unreliable — skipping reconcile (no orders)")
                live_now = self.cfg.live_trading_armed and broker.supports_live
                return RunResult(scan=scan, account=account, orders=[], fills=[],
                                 executed=False, mode="live" if live_now else "paper")
        if equity is None:
            equity = account.equity if (account.equity and account.equity > 0) else self.default_equity()
        scan = self._apply_hold_discipline(scan, account, equity)
        self._refresh_execution_quotes(scan, account)
        order_notes: list = []
        orders = build_orders(account, scan.targets, self.cfg, self.price_fn,
                              allow_buys=allow_buys, exclude_tickers=exclude_tickers,
                              exclude_buys=exclude_buy_tickers, explain=order_notes)

        live = self.cfg.live_trading_armed and broker.supports_live
        # dry_run gates only the LIVE brokerage. The paper broker always
        # simulates fills when we intend to execute (it has no real account).
        fills: list[dict] = []
        post_account: Account | None = None
        if execute:
            if live and not self.cfg.live_trading_armed:
                log.error("live trading disarmed — refusing order placement")
            else:
                fills, post_account = execute_orders(
                    broker, orders, self.cfg, account=account,
                    get_account=broker.get_account,
                )
        mode = "live" if live else "paper"
        if execute and not live:
            log.info("executed in PAPER mode (simulated fills on live prices)")
        if not orders and order_notes:
            log.info("no orders this cycle — decisions: %s",
                     "; ".join(f"{n['ticker']}:{n['action']}" for n in order_notes[:10]))
        return RunResult(scan=scan, account=account, post_account=post_account,
                         orders=orders, fills=fills,
                         executed=execute, mode=mode, order_notes=order_notes)

    def _refresh_execution_quotes(self, scan: ScanResult, account: Account) -> None:
        """Refresh quotes for held + target names immediately before order sizing."""
        if "snapshot" in self.providers:
            return
        tickers = {p.ticker for p in account.positions}
        tickers.update(t.ticker for t in scan.targets)
        if not tickers:
            return
        freshness = self.cfg.get("data.freshness", {}) or {}
        max_age = float(freshness.get("quote_max_age_seconds", 120))
        refresh = getattr(self.md, "refresh_quotes", None)
        if refresh:
            refresh(list(tickers), max_age_seconds=max_age)
            self.clear_price_cache()

    def _apply_hold_discipline(self, scan: ScanResult, account: Account, equity: float) -> ScanResult:
        """Use a lower exit bar than the buy bar so intraday scans do not churn.

        Hard risk exits still happen in the daemon every tick. This layer only
        decides whether a held name missing from the latest target book should
        be kept at its current weight or consciously sold on clear deterioration.
        """
        reb = self.cfg.get("portfolio.rebalance", {}) or {}
        if not reb.get("exit_hysteresis_enabled", True) or not account.positions:
            return scan

        target_map = {t.ticker: t for t in scan.targets}
        verdict_map = {v.ticker: v for v in scan.verdicts}
        td_map = scan.td_map or {}
        targets = list(scan.targets)
        hold_missing = bool(reb.get("hold_on_missing_data", True))

        for pos in account.positions:
            tk = pos.ticker
            if tk in target_map:
                continue
            verdict = verdict_map.get(tk)
            if verdict is None:
                if hold_missing:
                    target = self._held_target(pos, None, td_map.get(tk), equity,
                                               "hold: missing scan data")
                    if target:
                        targets.append(target)
                continue
            if self._should_exit_held(verdict) or self._should_rotate_for_runner(
                    tk, verdict, scan, account):
                log.info("held %s failed exit discipline — allowing rebalance sell", tk)
                continue
            target = self._held_target(pos, verdict, td_map.get(tk), equity,
                                       "hold: exit hysteresis")
            if target:
                targets.append(target)

        if len(targets) != len(scan.targets):
            scan.targets = targets
        return scan

    def _should_rotate_for_runner(self, ticker: str, verdict: Verdict,
                                  scan: ScanResult, account: Account) -> bool:
        """Sell a stale hold when a materially stronger fresh runner is in the book."""
        hunter = self.cfg.get("hunter", {}) or {}
        if not hunter.get("rotation_enabled", False):
            return False
        held = {p.ticker for p in account.positions}
        if ticker not in held:
            return False
        margin = float(hunter.get("rotation_score_margin", 6.0))
        newcomers = [t for t in scan.targets if t.ticker not in held]
        if not newcomers:
            return False
        best_new = max(newcomers, key=lambda t: t.score)
        if best_new.score >= verdict.composite + margin:
            log.info("rotation: %s (%.0f) yields slot to snipe %s (%.0f)",
                     ticker, verdict.composite, best_new.ticker, best_new.score)
            return True
        return False

    def _should_exit_held(self, verdict: Verdict) -> bool:
        reb = self.cfg.get("portfolio.rebalance", {}) or {}
        exit_score = float(reb.get("exit_conviction_score", 45.0))
        exit_pillars = int(float(reb.get("exit_min_pillars", 1)))
        if verdict.composite < exit_score or verdict.pillars_passing < exit_pillars:
            return True
        if reb.get("exit_on_ai_caution", True) and "ai_caution" in verdict.flags:
            return True
        if reb.get("exit_on_high_volatility", False) and "high_volatility" in verdict.flags:
            return True
        earnings_days = int(float(reb.get("exit_on_earnings_within_days", 0) or 0))
        if earnings_days > 0:
            for fl in verdict.flags:
                if fl.startswith("earnings_in_"):
                    try:
                        if int(fl.split("_")[-1].rstrip("d")) <= earnings_days:
                            return True
                    except ValueError:
                        pass
        return False

    def _held_target(self, pos: Position, verdict: Verdict | None, td: TickerData | None,
                     equity: float, reason: str) -> TargetPosition | None:
        px = self.price_fn(pos.ticker) or pos.current_price or pos.avg_price
        if not px or equity <= 0:
            return None
        dollars = pos.quantity * px
        if dollars <= 0:
            return None
        rc = self.cfg.get("portfolio.risk_controls", {}) or {}
        stop = take = None
        sector = "Unknown"
        if td is not None:
            sector = td.sector
            atr = td.technicals.get("atr")
            try:
                from .risk import atr_stop, take_profit
                stop = atr_stop(px, atr, rc.get("stop_loss_atr_mult", 2.5),
                                rc.get("hard_stop_pct", 0.18))
                take = take_profit(px, atr, rc.get("take_profit_atr_mult", 6.0))
            except Exception:
                pass
        score = round(verdict.composite, 1) if verdict else 0.0
        rationale = f"{reason}; score={score}" if verdict else reason
        return TargetPosition(
            ticker=pos.ticker,
            weight=round(dollars / equity, 4),
            score=score,
            shares=round(pos.quantity, 4),
            dollars=round(dollars, 2),
            stop_price=stop,
            take_profit=take,
            sector=sector,
            rationale=rationale,
        )

    # ---- backtest ----
    def backtest(self, limit: int | None = None, tickers: list[str] | None = None) -> "object":
        from .backtest.engine import Backtester
        bench_sym = self.cfg.get("backtest.benchmark", "SPY")
        tickers = self.universe(limit, watchlist=tickers)
        prices: dict[str, pd.DataFrame] = {}
        for t in tickers:
            df = self.md.get_prices(t)
            if df is not None and len(df) > 252:
                prices[t] = df
        bench = self.md.get_index_prices(bench_sym)
        if bench is None or len(bench) < 252:
            raise RuntimeError(f"no benchmark price history for {bench_sym}")
        log.info("backtest universe: %d names with sufficient history", len(prices))
        return Backtester(self.cfg).run(prices, bench,
                                        start=self.cfg.get("backtest.start"),
                                        end=self.cfg.get("backtest.end"))
