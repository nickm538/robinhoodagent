"""rh-agent — a quantamental, multi-factor trading agent for the
Robinhood Agentic Trading MCP.

The package is organised as a classic quant pipeline:

    universe -> providers (live data) -> factors -> analysts (panel of 5)
             -> regime-weighted composite -> risk/portfolio -> broker

Nothing in this package fabricates market data. Every number flows from a
live provider (FinancialDatasets.AI, Mboum, Alpha Vantage, Twelve Data) or
from a timestamped cache of a real provider response.
"""

__version__ = "1.0.0"
