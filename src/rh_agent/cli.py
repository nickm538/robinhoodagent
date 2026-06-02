"""Command-line interface.

    rh-agent doctor                 # environment & connectivity check
    rh-agent status                 # active providers + paper account
    rh-agent scan   [--snapshot f]  # rank universe, print target book
    rh-agent run    [--execute]     # scan + reconcile + (paper/live) orders
    rh-agent backtest [--snapshot f]# walk-forward vs benchmark
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import Config, load_config
from .logging_setup import get_logger, setup_logging

log = get_logger("cli")


def _agent(args):
    from .agent import TradingAgent
    cfg = load_config(args.config)
    return TradingAgent(cfg, snapshot_path=getattr(args, "snapshot", None)), cfg


def cmd_doctor(args) -> int:
    import requests
    cfg = load_config(args.config)
    print("== rh-agent doctor ==")
    print("\nAPI keys present:")
    for p in ["financialdatasets", "mboum", "alphavantage", "twelvedata", "firecrawl", "exa"]:
        print(f"  {p:18} {'✓' if cfg.api_key(p) else '— (missing)'}")
    print(f"  robinhood token    {'✓' if cfg.robinhood_token() else '— (authorise MCP first)'}")
    print(f"\nExecution mode: {cfg.execution_mode}  |  live armed: {cfg.live_trading_armed}")
    print("\nDirect egress (needs your environment's network policy to allow these hosts):")
    hosts = {"financialdatasets": "https://api.financialdatasets.ai",
             "mboum": "https://api.mboum.com", "alphavantage": "https://www.alphavantage.co",
             "twelvedata": "https://api.twelvedata.com", "firecrawl": "https://api.firecrawl.dev",
             "robinhood": "https://agent.robinhood.com/mcp/trading"}
    for name, url in hosts.items():
        try:
            r = requests.get(url, timeout=6)
            print(f"  {name:18} HTTP {r.status_code}")
        except Exception as e:
            print(f"  {name:18} unreachable ({type(e).__name__})")
    print("\nIf hosts show 'Host not in allowlist', add them to your Claude Code web "
          "environment's network egress policy (see README → Connectivity).")
    return 0


def cmd_status(args) -> int:
    agent, cfg = _agent(args)
    print("Active providers:", list(agent.providers))
    broker = agent.make_broker()
    acct = broker.get_account()
    print(f"Broker: {broker.name} | equity ${acct.equity:,.2f} | cash ${acct.cash:,.2f} "
          f"| positions {len(acct.positions)}")
    for p in acct.positions:
        print(f"  {p.ticker:6} {p.quantity:.4f} @ {p.avg_price:.2f}  "
              f"now {p.current_price:.2f}  P&L {p.unrealized_pnl:+.2f}")
    return 0


def cmd_scan(args) -> int:
    from . import report
    agent, cfg = _agent(args)
    scan = agent.scan(equity=args.equity, limit=args.limit)
    report.render_scan(scan)
    if args.md:
        path = report.write_markdown(scan)
        print(f"\nMarkdown brief written to {path}")
    if args.json:
        out = {"regime": scan.regime.name, "equity": scan.equity,
               "targets": [t.__dict__ for t in scan.targets]}
        print(json.dumps(out, indent=2, default=str) if args.json == "-" else "")
        if args.json != "-":
            with open(args.json, "w") as f:
                json.dump(out, f, indent=2, default=str)
    return 0


def cmd_run(args) -> int:
    from . import report
    agent, cfg = _agent(args)
    if args.execute and cfg.execution_mode == "live" and not cfg.live_trading_armed:
        print("Refusing: EXECUTION_MODE=live but LIVE_TRADING_CONFIRM is not set to "
              "'I_UNDERSTAND_REAL_MONEY'. Aborting.")
        return 2
    run = agent.run(execute=args.execute)
    report.render_run(run)
    path = report.write_markdown(run.scan, run=run)
    print(f"\nMarkdown brief written to {path}")
    return 0


def cmd_backtest(args) -> int:
    from . import report
    agent, cfg = _agent(args)
    res = agent.backtest(limit=args.limit)
    report.render_backtest(res)
    return 0


def cmd_auth(args) -> int:
    """One-time Robinhood OAuth for the standalone bot (opens a browser)."""
    cfg = load_config(args.config)
    from .broker.oauth import authenticate
    url = cfg.robinhood_url()
    print(f"Authenticating with Robinhood Agentic MCP at {url}")
    print("A browser window will open — approve access, then return here.\n")
    try:
        tools = authenticate(url, port=args.port)
    except Exception as e:
        print(f"Auth failed: {e}")
        return 1
    print("\n✅ Authenticated. Tokens saved to state/robinhood_oauth.json (chmod 600).")
    print(f"Discovered {len(tools)} tools: {', '.join(tools[:12])}"
          f"{'...' if len(tools) > 12 else ''}")
    print("\nNow arm live trading and run the loop:")
    print("  EXECUTION_MODE=live LIVE_TRADING_CONFIRM=I_UNDERSTAND_REAL_MONEY "
          "./scripts/run_loop.sh --execute")
    return 0


def cmd_loop(args) -> int:
    """Run the always-on autonomous agent."""
    from .daemon import AlwaysOnAgent
    cfg = load_config(args.config)
    if args.execute and cfg.execution_mode == "live" and not cfg.live_trading_armed:
        print("Refusing live loop: set LIVE_TRADING_CONFIRM=I_UNDERSTAND_REAL_MONEY to arm.")
        return 2
    AlwaysOnAgent(cfg, snapshot_path=args.snapshot).run_forever(
        execute=args.execute, once=args.once, max_cycles=args.max_cycles)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rh-agent", description="Quantamental Robinhood trading agent")
    p.add_argument("--config", default=None, help="path to config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor").set_defaults(func=cmd_doctor)

    s = sub.add_parser("auth", help="one-time Robinhood OAuth for the standalone bot")
    s.add_argument("--port", type=int, default=8765, help="localhost OAuth callback port")
    s.set_defaults(func=cmd_auth)
    s = sub.add_parser("status"); s.add_argument("--snapshot"); s.set_defaults(func=cmd_status)

    s = sub.add_parser("scan")
    s.add_argument("--snapshot"); s.add_argument("--limit", type=int)
    s.add_argument("--equity", type=float); s.add_argument("--md", action="store_true")
    s.add_argument("--json", nargs="?", const="-")
    s.set_defaults(func=cmd_scan)

    s = sub.add_parser("run")
    s.add_argument("--snapshot"); s.add_argument("--limit", type=int)
    s.add_argument("--execute", action="store_true", help="place orders (paper unless live armed)")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("backtest")
    s.add_argument("--snapshot"); s.add_argument("--limit", type=int)
    s.add_argument("--start"); s.add_argument("--end")
    s.set_defaults(func=cmd_backtest)

    s = sub.add_parser("loop", help="run the always-on autonomous agent")
    s.add_argument("--snapshot")
    s.add_argument("--execute", action="store_true")
    s.add_argument("--once", action="store_true", help="run a single cycle and exit")
    s.add_argument("--max-cycles", type=int, dest="max_cycles")
    s.set_defaults(func=cmd_loop)
    return p


def main(argv=None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        log.error("command failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
