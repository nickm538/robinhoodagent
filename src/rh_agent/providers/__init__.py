"""Provider package + factory that wires up only the sources that have keys."""
from __future__ import annotations

from ..config import Config
from ..logging_setup import get_logger
from .base import DataProvider, DiskCache

log = get_logger("providers")


def build_providers(cfg: Config, snapshot_path: str | None = None) -> dict[str, DataProvider]:
    """Construct the available providers. If ``snapshot_path`` is given, the
    SnapshotProvider is returned alone (used for offline/reproducible runs)."""
    cache = DiskCache()
    if snapshot_path:
        from .snapshot import SnapshotProvider
        sp = SnapshotProvider(snapshot_path)
        log.info("snapshot provider: %d tickers, captured %s", len(sp.tickers), sp.captured_at)
        return {"snapshot": sp}

    providers: dict[str, DataProvider] = {}

    fd_key = cfg.api_key("financialdatasets")
    if fd_key:
        from .financial_datasets import FinancialDatasetsProvider
        providers["financialdatasets"] = FinancialDatasetsProvider(fd_key, cache)

    mb_key = cfg.api_key("mboum")
    if mb_key:
        from .mboum import MboumProvider
        providers["mboum"] = MboumProvider(mb_key, cache)

    av_key = cfg.api_key("alphavantage")
    if av_key:
        from .alpha_vantage import AlphaVantageProvider
        providers["alphavantage"] = AlphaVantageProvider(av_key, cache)

    td_key = cfg.api_key("twelvedata")
    if td_key:
        from .twelvedata import TwelveDataProvider
        providers["twelvedata"] = TwelveDataProvider(td_key, cache)

    fc_key, exa_key = cfg.api_key("firecrawl"), cfg.api_key("exa")
    if fc_key or exa_key:
        from .web_research import WebResearchProvider
        web = WebResearchProvider(fc_key, exa_key, cache)
        if web.enabled:
            providers["web"] = web

    log.info("providers active: %s", list(providers))
    return providers


def snapshot_priorities() -> dict:
    """Route every data section to the snapshot provider."""
    sections = ["fundamentals", "prices", "quote", "technicals", "insider", "institutional",
                "news_sentiment", "analyst_ratings", "short_interest", "options_flow",
                "pro_scores", "macro", "universe"]
    return {s: ["snapshot"] for s in sections}
