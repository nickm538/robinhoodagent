# AGENTS.md

## Cursor Cloud specific instructions

### Product overview

**rh-agent** is a single-process Python CLI/daemon for quantamental equity trading
(live Agentic account by default). There is no web UI, database, or docker-compose stack. The only local runtime is the `rh-agent` Python package.

### One-time VM prerequisites

Ubuntu images need the venv module before the first `python3 -m venv .venv`:

```bash
sudo apt-get install -y python3.12-venv
```

### Dependency install (see update script)

After the update script runs, activate the venv:

```bash
source .venv/bin/activate
```

Or invoke tools directly via `.venv/bin/python` / `.venv/bin/rh-agent`.

### Standard commands

| Task | Command |
|---|---|
| Lint | `ruff check src tests` (pre-existing style warnings in repo; CI does not run ruff) |
| Compile | `PYTHONPATH=src python -m compileall -q src` |
| Tests | `PYTHONPATH=src python -m pytest -q` |
| Env check | `python -m rh_agent.cli doctor` |
| Paper scan | `python -m rh_agent.cli scan --snapshot <file>` |
| Paper trade | `python -m rh_agent.cli run --execute --snapshot <file>` |
| Account | `python -m rh_agent.cli status` |

### Configuration

- Copy `.env.example` → `.env` and fill API keys for live data providers.
- Strategy tuning lives in `config/config.yaml`.
- `.env.example` defaults to `EXECUTION_MODE=paper` with an empty `LIVE_TRADING_CONFIRM`
  for safe simulated fills.
- For the production VM only, set `EXECUTION_MODE=live` with
  `LIVE_TRADING_CONFIRM=I_UNDERSTAND_REAL_MONEY` after `rh-agent auth`.

### Running without API keys (offline / cloud demo)

No bundled snapshot ships in the repo. For restricted egress or missing keys, generate a snapshot JSON and pass `--snapshot`:

1. Use `scripts/assemble_snapshot.py` with captured provider responses, **or**
2. Generate a minimal demo snapshot (e.g. `/tmp/gen_demo_snapshot.py` pattern) with ≥60 price bars per ticker plus SPY/RSP benchmarks.

Then run scan/run/backtest with `--snapshot /path/to/snapshot.json`.

### Live data / trading (optional)

- **Live scans** need at least one market-data API key (`FINANCIALDATASETS_API_KEY`, `MBOUM_API_KEY`, etc.) in `.env`.
- **Live trading** requires `pip install -e ".[live]"`, `python -m rh_agent.cli auth` (OAuth on port 8765), and `.env` with `EXECUTION_MODE=live` + `LIVE_TRADING_CONFIRM=I_UNDERSTAND_REAL_MONEY`.

### Gotchas

- `rh-agent doctor` probes external HTTPS hosts; most are reachable from this cloud VM, but API keys may still be absent.
- Paper broker state persists under `state/` (gitignored).
- Reports are written to `reports/` on scan/run.
- The always-on loop (`python -m rh_agent.cli loop --execute`) is long-running; use tmux for background sessions.
