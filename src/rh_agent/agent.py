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

    def universe(self, limit: int | None = None) -> list[str]:
        if "snapshot" in self.providers:
            tickers = self.providers["snapshot"].list_universe()
        else:
            from .data.universe import build_universe
            tickers = build_universe(self.md, self.cfg)
        return tickers[:limit] if limit else tickers

    def _gather(self, tickers: list[str], deep: bool = True) -> list[TickerData]:
        out: list[TickerData] = []
        for i, t in enumerate(tickers, 1):
            try:
                td = self.md.build(t, deep=deep)
            except Exception as e:
                log.debug("build %s failed: %s", t, e)
                continue
            if td.price:
                out.append(td)
            if i % 25 == 0:
                log.info("  gathered %d/%d", i, len(tickers))
        return out

    # ---- scan & score ----
    def scan(self, equity: float | None = None, limit: int | None = None) -> ScanResult:
        equity = equity or self.default_equity()
        tickers = self.universe(limit)
        log.info("scanning %d tickers", len(tickers))
        data = self._gather(tickers, deep=True)
        for td in data:
            if td.quote:
                self._quote_cache[td.ticker] = td.quote.price
        regime = detect_regime(self.md, self.cfg)
        verdicts = self.scorer.score(data, regime)
        eligible = self.scorer.eligible(verdicts)
        td_map = {td.ticker: td for td in data}
        targets = self.builder.build(eligible, td_map, regime, equity)
        return ScanResult(regime=regime, verdicts=verdicts, eligible=eligible, targets=targets,
                          equity=equity, universe_size=len(tickers), scored_size=len(data),
                          td_map=td_map)

    # ---- broker ----
    def make_broker(self):
        from .broker.paper import PaperBroker
        if self.cfg.live_trading_armed:
            try:
                from .broker.robinhood_mcp import RobinhoodMCPBroker
                acct = os.getenv("ROBINHOOD_ACCOUNT_NUMBER")
                b = RobinhoodMCPBroker(self.cfg.robinhood_url(), self.cfg.robinhood_token(),
                                       account_number=acct)
                log.warning("LIVE broker active (Robinhood Agentic MCP)")
                return b
            except Exception as e:
                log.error("Robinhood MCP broker unavailable (%s); refusing to fall back to live. "
                          "Using PAPER.", e)
        return PaperBroker(self.price_fn, starting_cash=self.default_equity(),
                           slippage_bps=self.cfg.get("backtest.slippage_bps", 5.0))

    # ---- full run ----
    def run(self, execute: bool = False) -> RunResult:
        broker = self.make_broker()
        account = broker.get_account()
        scan = self.scan(equity=account.equity or self.default_equity())
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
    def backtest(self, limit: int | None = None) -> "object":
        from .backtest.engine import Backtester
        bench_sym = self.cfg.get("backtest.benchmark", "SPY")
        tickers = self.universe(limit)
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
