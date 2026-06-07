# rh-agent — autonomous quantamental trading agent for Robinhood

A production-grade, **always-on** trading agent that scans the equity universe,
scores every name through a **panel of five veteran-trader personas**, adapts its
weighting to the **current market regime**, sizes positions with real risk
controls, and executes through the **Robinhood Agentic Trading MCP**.

> **Goal:** strong, risk-adjusted outperformance of the S&P 500 / Dow over 1–3
> month horizons. **Honest reality:** no system can *guarantee* beating — let
> alone doubling — the index. Markets are adversarial and this trades **real
> money**. The agent is engineered to tilt the odds with disciplined, multi-factor,
> regime-aware decisions and hard risk limits — not to promise miracles. See
> [Safety & expectations](#safety--honest-expectations).

---

## The Panel of Five (how it thinks)

Every candidate is judged by five independent "analysts" — an ensemble that
turns five noisy opinions into one robust decision. A name must win the
agreement of **multiple pillars** to make the book (kills single-signal noise).

| Persona | Lens | Core factors |
|---|---|---|
| 🏃 **Momentum Trader** | trend & price action | 12-1 / 6-1 / 3-1 momentum, risk-adjusted momentum, 50/200 SMA trend, distance to 52w high |
| 📊 **Quant** | quality & value | ROE, ROIC, margins, revenue/earnings growth, FCF yield, balance-sheet strength |
| ⚡ **Catalyst Trader** | estimate momentum | EPS/revenue estimate revisions, earnings-surprise history, upcoming catalysts |
| 🕵️ **Smart-Money Tracker** | informed flow | insider net buying, institutional 13F changes, short-squeeze setups |
| 📰 **Sentiment Analyst** | crowd & pros | news sentiment, options put/call, **Zacks Rank, Morningstar, Danelfin, TipRanks** |

The **Chief PM** (regime engine) reads SPX trend, VIX, breadth (equal- vs
cap-weight) and the yield curve, classifies the regime
(`risk_on_trend` / `neutral` / `risk_off` / `high_volatility`), and **re-weights
the five personas** plus throttles gross exposure accordingly.

```
universe → live data providers → 27 factors → winsorised cross-sectional rank
        → 5 analyst personas → regime-weighted composite → conviction + pillar gates
        → vol-targeted sizing (per-name & sector caps, ATR stops) → broker (paper/live)
```

## Data sources (priority order)

**FinancialDatasets.AI** and **Mboum** are the primary sources (per mandate),
with Alpha Vantage / Twelve Data as fallbacks and unique-data fills, and
**Firecrawl + Exa** for pro-source web research (Zacks / Morningstar / Danelfin /
TipRanks). Every provider is a real HTTP client; nothing is mocked. Missing data
neutralises a factor rather than inventing a value.

| Source | Used for |
|---|---|
| **FinancialDatasets.AI** | prices, fundamentals, insider trades, 13F institutional, news, company facts |
| **Mboum** | quotes, history, analyst ratings & price targets, financials, short interest, options flow |
| **Alpha Vantage** | technicals, **news sentiment**, earnings estimates, options put/call, macro/regime, full listing universe |
| **Twelve Data** | quote / price fallback |
| **Firecrawl + Exa** | Zacks Rank, Morningstar, Danelfin AI score, TipRanks |
| **Robinhood Agentic MCP** | account, positions, buying power, **order execution** |

## Quickstart

```bash
git clone <this repo> && cd robinhood_agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then fill in your API keys
python -m rh_agent.cli doctor      # checks keys + connectivity
```

`rh-agent doctor` reports which keys are present and whether each data host is
reachable from your network.

## Connect to Robinhood (one-time)

Robinhood launched **Agentic Trading** (May 2026): agents connect over MCP and
can **only place trades in a separate "Agentic" account** you fund explicitly —
they cannot touch your main portfolio. That wall is by design.

**If you drive the agent from Claude Code / Claude Desktop:**
```bash
claude mcp add robinhood-trading --transport http https://agent.robinhood.com/mcp/trading
# then in Claude:  /mcp  → select robinhood-trading → authenticate (browser OAuth)
```
Claude can then place the agent's recommended orders by calling the Robinhood
MCP tools directly.

**If you run the standalone bot (recommended for 24/7):** authenticate the bot
itself, once, on the machine that will run it:

```bash
pip install "mcp>=1.2.0"      # the `live` extra
python -m rh_agent.cli auth    # opens a browser → approve → tokens cached to state/
```

`rh-agent auth` runs Robinhood's OAuth and saves the tokens to
`state/robinhood_oauth.json` (chmod 600). From then on the bot authenticates
non-interactively and the SDK **auto-refreshes** the access token — so it trades
hands-off indefinitely. (Alternative: paste a token as `ROBINHOOD_MCP_TOKEN=...`
in `.env`; the lightweight broker will use it, but it won't auto-refresh.)

Equities only in beta. Fund the Agentic account with the capital you want the
agent to manage.

## Run it

```bash
# Rank the universe and print the target book (no orders placed)
python -m rh_agent.cli scan --md

# Scan + reconcile vs your account + show orders
python -m rh_agent.cli run --execute

# Walk-forward backtest vs SPY
python -m rh_agent.cli backtest

# Account / positions
python -m rh_agent.cli status
```

Default config/.env is armed for live execution:
`EXECUTION_MODE=live` and `LIVE_TRADING_CONFIRM=I_UNDERSTAND_REAL_MONEY`.
Set `EXECUTION_MODE=paper` when you want simulated fills on live prices.

### Try it right now with the bundled live snapshot

This repo ships a reproducible run on **real data captured 2026-06-02** (11
liquid names + SPY/RSP, via FinancialDatasets.AI). See [`RESULTS.md`](RESULTS.md).
To regenerate a snapshot from your own captured price files, use
`scripts/assemble_snapshot.py`.

## Always-on, hands-off (the autonomous loop)

```bash
# Foreground
python -m rh_agent.cli loop --execute

# Resilient wrapper (auto-restarts), logs to logs/loop.log
./scripts/run_loop.sh --execute
```

The loop runs **non-stop**: every cycle (default 15 min, market-hours aware) it
manages risk on open positions (trailing/hard stops, take-profits), and on the
configured cadence (weekly by default) it re-scans, rebuilds the book, and
executes. A **daily-drawdown circuit breaker** suspends new buying after a -6%
day. It is crash-resistant — an error in one cycle is logged and the loop keeps
going.

For **live** trading first run `python -m rh_agent.cli auth` once (above), then
arm and deploy. **Deploy it to stay up 24/7** (this is what makes it truly
"always-on" — a laptop or a sandbox that sleeps will not do):

* **systemd** — `deploy/rh-agent.service` (auto-restart, boots with the host)
* **Docker** — `docker build -t rh-agent . && docker run -d --env-file .env rh-agent`
* **Any VPS / cloud VM** — `nohup ./scripts/run_loop.sh --execute &`

## Connectivity / egress allowlist

The agent needs outbound HTTPS to its data + trading hosts. On a locked-down
network (e.g. a Claude Code **web** environment with a restrictive policy) these
may be blocked (`Host not in allowlist`). Allow these hosts in your environment's
network egress policy (see https://code.claude.com/docs/en/claude-code-on-the-web):

```
api.financialdatasets.ai   api.mboum.com   www.alphavantage.co
api.twelvedata.com   api.firecrawl.dev   api.exa.ai   agent.robinhood.com
```

On your own server/VPS there is no such restriction.

## Safety & honest expectations

* **Real money, real risk.** You can lose money.
* **No performance guarantee.** The 2× S&P "north star" is an aspiration, not a
  promise. Backtests have selection/look-ahead caveats (see `RESULTS.md`).
* **Defense in depth:** paper default · explicit live opt-in · Robinhood's
  separate Agentic account · per-name (10%) and per-sector (35%) caps · ATR
  trailing stops · hard -18% stop · daily drawdown halt · liquidity floor
  (no penny stocks / illiquid names).
* **Not investment advice.** You are responsible for your account.

## Configuration

Everything tunable lives in [`config/config.yaml`](config/config.yaml): universe
& liquidity filters, the five personas and their factor weights, regime weight
tables and exposure throttles, normalisation/noise controls, portfolio caps and
risk controls, rebalance cadence, and backtest settings. Secrets live only in
`.env` (gitignored) — never in the repo.

## Repository layout

```
src/rh_agent/
  providers/     real HTTP clients: financial_datasets, mboum, alpha_vantage,
                 twelvedata, web_research (firecrawl+exa), snapshot, base
  factors/       indicators, library (27 factors), normalize
  analysts/      panel (the five personas + Chief PM blend)
  regime.py      market-regime detection + weighting
  scoring.py     raw factors → cross-sectional rank → composite
  portfolio.py   sizing, per-name & sector caps, ATR stops
  risk.py        vol, stops, drawdown guard
  broker/        base, paper (default), mcp_client, robinhood_mcp
  backtest/      walk-forward engine + metrics
  execution.py   reconcile target vs account → orders
  agent.py       orchestrator     daemon.py  always-on loop     cli.py  CLI
config/          config.yaml
scripts/         assemble_snapshot.py, run_loop.sh
deploy/          rh-agent.service          tests/  pytest suite
```

## Tests

```bash
PYTHONPATH=src python -m pytest -q
```
