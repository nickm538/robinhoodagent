# rh-agent — autonomous quantamental trading agent for Robinhood

A production-grade, **always-on** trading agent that scans the equity universe,
scores every name through a **panel of five veteran-trader personas**, adapts its
weighting to the **current market regime**, sizes positions with real risk
controls, and executes through the **Robinhood Agentic Trading MCP** in your
**live Agentic account** (real money). Paper mode remains available for testing.

> **Goal:** strong, risk-adjusted outperformance of the S&P 500 / Dow over 1–3
> month horizons. **Honest reality:** no system can *guarantee* beating — let
> alone doubling — the index. Markets are adversarial and this trades **real
> money**. Run `EXECUTION_MODE=paper` in `.env` if you want to test without
> placing orders. See [Safety & expectations](#safety--honest-expectations).

---

![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/nickm538/robinhoodagent?utm_source=oss&utm_medium=github&utm_campaign=nickm538%2Frobinhoodagent&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)

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

| Source | Niche it owns |
|---|---|
| **FinancialDatasets.AI** | depth: prices, fundamentals, insider trades, 13F institutional, news, company facts |
| **Massive** (ex-Polygon) | volume: uncapped calls — bulk snapshot radar prefetch, deep price history, top-gainers movers, FINRA short interest, news sentiment insights, universe |
| **Mboum** | listing universe, analyst ratings & price targets, short-interest/options backup |
| **Alpha Vantage** | structured **news sentiment**, macro/regime (yield curve), secondary movers feed |
| **Twelve Data** | batch-quote fallback, quote/price fallback, technicals enrichment |
| **Firecrawl + Exa** | Zacks Rank, Morningstar, Danelfin AI score, TipRanks |
| **Robinhood Agentic MCP** | account, positions, buying power, **order execution** |

Every chain falls through on outage or quota (each paid source has a fail-fast
rate-limit cooldown), so a single provider dying degrades data richness — never
correctness or safety.

## Quickstart

```bash
git clone <this repo> && cd robinhoodagent
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

# Scan + reconcile vs your account + show orders (live when armed in .env)
python -m rh_agent.cli run --execute

# Walk-forward backtest vs SPY
python -m rh_agent.cli backtest

# Account / positions
python -m rh_agent.cli status
```

`config/config.yaml` defaults to **live mode**, and `.env.example` includes the
required `LIVE_TRADING_CONFIRM=I_UNDERSTAND_REAL_MONEY` confirmation string for
the production VM. Together, after `rh-agent auth`, that places **real orders**
in your Robinhood *Agentic* account. Use `EXECUTION_MODE=paper` only for
explicit non-production testing.

Emergency flatten: `python -m rh_agent.cli cancel-open` (live armed only).

If `systemctl status rh-agent` shows `status=2`, the CLI refused to start on
purpose. Check the exact message with `journalctl -u rh-agent -n 80 --no-pager`;
the usual causes are missing `LIVE_TRADING_CONFIRM`, expired/missing OAuth
(`python -m rh_agent.cli auth`), or a duplicate daemon lock.

### Try it right now with an offline snapshot

No snapshot is committed to the repo. For offline demos, assemble one from
captured provider responses with `scripts/assemble_snapshot.py`, then run
`python -m rh_agent.cli scan --snapshot /path/to/snapshot.json`. See
[`RESULTS.md`](RESULTS.md) for a historical example.

## Always-on, hands-off (the autonomous loop)

```bash
# Foreground (live — ensure .env is configured and `rh-agent auth` is done)
python -m rh_agent.cli loop --execute

# Resilient wrapper (auto-restarts), logs to logs/loop.log
./scripts/run_loop.sh --execute
```

The loop runs **non-stop**: every cycle (default **1 min** poll, market-hours aware) it
manages risk on open positions (ATR trailing/hard stops, take-profits), and about every
20 minutes by default it runs a dynamic intraday radar across the provider-listed
equity universe, deep-scores the strongest candidates, rebuilds the book, and
executes. A **daily-drawdown circuit breaker** suspends new buying after a -6%
day (de-risking sells still run). It is crash-resistant — an error in one cycle is logged and the loop keeps
going.

For **live** trading first run `python -m rh_agent.cli auth` once (above), then
arm and deploy. **Deploy it to stay up 24/7** (this is what makes it truly
"always-on" — a laptop or a sandbox that sleeps will not do):

* **systemd** — `deploy/rh-agent.service` (auto-restart, boots with the host)
* **Docker** — `docker build -t rh-agent . && docker run -d --env-file .env rh-agent`
* **Any VPS / cloud VM** — `nohup ./scripts/run_loop.sh --execute &`

## "Is it working?" — reading the agent's activity

A quiet tape is often **discipline, not a stall**: once the book matches the
target weights, every 20-minute hunt that re-confirms the same names produces
**zero orders by design** (no-trade band + exit hysteresis). Stops/take-profits
then tend to fire on open/close volatility — which can *look* like "it only
trades at the start or end of the day". The agent now keeps a flight recorder
so you never have to guess:

```bash
# Human diagnosis of the last 24h: hunts started/completed/abandoned, scan
# durations vs the configured cadence, why rebalances produced no orders,
# protective exits, cooldowns, halts.
python -m rh_agent.cli why            # add --hours 72 for a longer window
```

Under the hood every scan, rebalance, stop/TP fire, halt and heartbeat is
appended to `state/activity.jsonl` (`daemon.activity_log`). What to look for:

* **`scan-bound` cadence** — scans taking longer than `intraday_hours` means
  hunts run back-to-back as fast as data allows; lower `universe.scan_cap` /
  `deep_top_k` or accept the pace.
* **`abandoned` scans** — the watchdog killed a hung scan (provider rate limits
  are the usual cause; run `rh-agent doctor`). Those cycles never rebalance.
* **Zero-order rebalances with `hold_within_band` notes** — healthy: it hunted,
  and the held book still ranked best.

The hunt cadence itself is protected by three guards added for exactly the
sparse-trading failure modes: the universe light-pass is now threaded (hunts
finish minutes faster), a pending scan that idles past
`daemon.pending_scan_expiry_seconds` (e.g. finished after the close) is
**discarded instead of fired blind at the next open**, and market hours stay
correct even on hosts without tzdata (built-in US-Eastern DST fallback).

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

* **Real money, real risk.** You can lose money. Use `EXECUTION_MODE=paper` to test.
* **No performance guarantee.** The 2× S&P "north star" is an aspiration, not a
  promise. Backtests have selection/look-ahead caveats (see `RESULTS.md`).
* **Defense in depth:** paper `.env.example` default · explicit live opt-in ·
  pre-execution quote refresh · order acceptance checks · `cancel-open` · Robinhood's
  separate Agentic account · per-name (10%) and per-sector (35%) caps · ATR
  trailing stops (monotonic — a rebalance can never lower a ratcheted stop) ·
  hard -18% stop · **breakeven ratchet** (a name up ≥1.5×ATR ratchets the stop
  toward breakeven near the entry price, reducing the chance of round-trip loss
  on ideal fills but may still incur losses due to live fills, spread, slippage,
  or partial fills) · **stop re-entry cooldown** (a stopped-out name can't
  be re-BOUGHT for 6h — exits are never blocked — and it doesn't waste a book
  slot meanwhile) · **intraday tape shock** (SPY -1.5%/-2.5% on the day downgrades
  the regime to neutral/risk_off immediately, instead of waiting for the daily
  bar) · stale-scan expiry (never trade yesterday's conviction at the open) ·
  daily drawdown halt · liquidity floor (no penny stocks / illiquid names).
* **Provider quota resilience:** when a data provider exhausts its plan (e.g.
  Mboum's monthly call cap), it trips a cooldown (re-probes hourly,
  `MBOUM_RATE_LIMIT_COOLDOWN_SECONDS` to tune) instead of burning a doomed HTTP
  round trip per name per scan; the priority chain serves from the remaining
  providers and the cooled provider resumes automatically when its quota resets.
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
