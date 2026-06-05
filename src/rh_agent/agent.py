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
from .execution import build_orders
from .logging_setup import get_logger
from .models import Account, Order, TargetPosition, TickerData, Verdict
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
    fills: list[dict] = field(default_factory=list)
    executed: bool = False
    mode: str = "paper"


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
        self._quote_cache: dict[str, float | None] = {}

    # ---- helpers ----
    def default_equity(self) -> float:
        return float(os.getenv("PAPER_EQUITY", self.cfg.get("backtest.initial_equity", 100_000)))

    def price_fn(self, ticker: str) -> float | None:
        if ticker in self._quote_cache:
            return self._quote_cache[ticker]
        q = self.md.get_quote(ticker)
        px = q.price if q else None
        self._quote_cache[ticker] = px
        return px

    def universe(self, limit: int | None = None, watchlist: list[str] | None = None) -> list[str]:
        # explicit --tickers (even if empty) overrides config universe.watchlist;
        # only fall back to config when no watchlist was passed at all.
        wl = watchlist if watchlist is not None else (self.cfg.get("universe.watchlist") or [])
        if wl:
            tickers = [t.strip().upper() for t in wl if t and t.strip()]
            return tickers[:limit] if limit else tickers
        if "snapshot" in self.providers:
            tickers = self.providers["snapshot"].list_universe()
        else:
            from .data.universe import build_universe
            tickers = build_universe(self.md, self.cfg)
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
             tickers: list[str] | None = None) -> ScanResult:
        if equity is None:                 # 0.0 is a valid (empty-account) sizing -> keep it
            equity = self.default_equity()
        names = self.universe(limit, watchlist=tickers)
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

        log.info("deep-scoring %d tickers", len(names))
        data = self._gather(names, deep=True)
        for td in data:
            if td.quote:
                self._quote_cache[td.ticker] = td.quote.price
        verdicts = self.scorer.score(data, regime)
        td_map = {td.ticker: td for td in data}
        ai_read = self._apply_ai_overlay(verdicts, td_map, regime)
        eligible = self.scorer.eligible(verdicts)
        targets = self.builder.build(eligible, td_map, regime, equity)
        return ScanResult(regime=regime, verdicts=verdicts, eligible=eligible, targets=targets,
                          equity=equity, universe_size=full_n, scored_size=len(data),
                          td_map=td_map, ai_market_read=ai_read)

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
                "fundamentals": {k: round(f[k], 3) for k in
                                 ("roe", "net_margin", "revenue_growth", "earnings_growth",
                                  "pe_ratio", "debt_to_equity") if isinstance(f.get(k), (int, float))},
                "news_sentiment": td.news_sentiment.get("score"),
                "days_to_earnings": td.earnings.get("days_to_next"),
                "headlines": self.md.headlines(v.ticker, 5),
            })
        ctx = f"Regime: {regime.describe()}. Recent market headlines: {self.md.market_news(8)}"
        res = self.ai.assess(ctx, cands)
        if not res.views:
            return res.market_read or ""
        w = self.ai.weight
        for v in verdicts:
            av = res.views.get(v.ticker)
            if not av:
                continue
            v.analyst_scores["ai_analyst"] = round(av["score"], 1)
            v.composite = round((1 - w) * v.composite + w * av["score"], 1)
            v.rationale += f" | AI {av['stance']}: {av['rationale']}"
            if av["stance"] == "bearish" and av["score"] < 40:
                v.flags.append("ai_caution")
        verdicts.sort(key=lambda x: x.composite, reverse=True)
        log.info("AI overlay: blended %d names (weight %.2f) | %s",
                 len(res.views), w, res.market_read[:120])
        return res.market_read or ""

    # ---- broker ----
    def make_broker(self):
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
            log.error("Live armed but no Robinhood auth (run `rh-agent auth` or set "
                      "ROBINHOOD_MCP_TOKEN). Falling back to PAPER.")
        return PaperBroker(self.price_fn, starting_cash=self.default_equity(),
                           slippage_bps=self.cfg.get("backtest.slippage_bps", 5.0))

    # ---- full run ----
    def run(self, execute: bool = False, tickers: list[str] | None = None) -> RunResult:
        broker = self.make_broker()
        account = broker.get_account()
        # NEVER size a LIVE account on the paper default. If a live balance reads
        # as 0 (e.g. a portfolio-fetch hiccup), skip the cycle ENTIRELY — no scan,
        # no orders — rather than risk over-sizing on the default equity.
        if broker.supports_live and not (account.equity and account.equity > 0):
            log.error("live account equity read as 0 — skipping cycle entirely (no orders)")
            empty = ScanResult(regime=RegimeResult("halted", {}, 0.0), verdicts=[], eligible=[],
                               targets=[], equity=0.0, universe_size=0, scored_size=0)
            return RunResult(scan=empty, account=account, orders=[], fills=[], executed=False,
                             mode="live")
        equity = account.equity if (account.equity and account.equity > 0) else self.default_equity()
        scan = self.scan(equity=equity, tickers=tickers)
        orders = build_orders(account, scan.targets, self.cfg, self.price_fn)

        live = self.cfg.live_trading_armed and broker.supports_live
        # dry_run gates only the LIVE brokerage. The paper broker always
        # simulates fills when we intend to execute (it has no real account).
        fills: list[dict] = []
        if execute:
            for o in orders:
                fills.append(broker.place_order(o, dry_run=False))
        mode = "live" if live else "paper"
        if execute and not live:
            log.info("executed in PAPER mode (simulated fills on live prices)")
        return RunResult(scan=scan, account=account, orders=orders, fills=fills,
                         executed=execute, mode=mode)

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
