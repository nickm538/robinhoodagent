"""Human-readable reporting: rich console tables + a saved markdown brief."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .config import REPO_ROOT
from .models import utcnow

REPORT_DIR = REPO_ROOT / "reports"

try:
    from rich.console import Console
    from rich.table import Table
    _con = Console()
except Exception:  # pragma: no cover
    _con = None


def _print(x):
    if _con:
        _con.print(x)
    else:
        print(x)


def render_scan(scan) -> None:
    _print(f"\n[bold]Regime:[/bold] {scan.regime.describe()}" if _con
           else f"\nRegime: {scan.regime.describe()}")
    _print(f"Universe scanned: {scan.universe_size} | priced & scored: {scan.scored_size} "
           f"| eligible: {len(scan.eligible)} | target book: {len(scan.targets)} "
           f"| equity ${scan.equity:,.0f}")
    if not scan.targets:
        _print("No positions clear the conviction + multi-pillar gates this run.")
        return
    if _con:
        t = Table(title="Target Portfolio", show_lines=False, header_style="bold cyan")
        for col in ["#", "Ticker", "Sector", "Score", "Weight", "$", "Stop", "Target", "Drivers"]:
            t.add_column(col, overflow="fold")
        for i, p in enumerate(scan.targets, 1):
            t.add_row(str(i), p.ticker, (p.sector or "")[:16], f"{p.score:.0f}",
                      f"{p.weight:.1%}", f"{p.dollars:,.0f}",
                      f"{p.stop_price:.2f}" if p.stop_price else "-",
                      f"{p.take_profit:.2f}" if p.take_profit else "-",
                      p.rationale[:46])
        _con.print(t)
    else:
        for i, p in enumerate(scan.targets, 1):
            print(f"{i:2}. {p.ticker:6} {p.weight:5.1%} score={p.score:.0f} {p.rationale[:50]}")


def render_run(run) -> None:
    a = run.account
    _print(f"\n[bold]Account[/bold] ({a.source}): equity ${a.equity:,.0f} | cash ${a.cash:,.0f} "
           f"| positions {len(a.positions)} | mode={run.mode}" if _con
           else f"\nAccount ({a.source}): equity ${a.equity:,.0f} cash ${a.cash:,.0f} mode={run.mode}")
    render_scan(run.scan)
    if run.orders:
        _print(f"\n[bold]Orders[/bold] ({len(run.orders)}):" if _con else f"\nOrders ({len(run.orders)}):")
        for o in run.orders:
            _print(f"  {o.side.upper():4} {o.ticker:6} "
                   f"{('$%.0f' % o.notional) if o.notional else ('%.2f sh' % o.quantity)}  — {o.reason}")
    if run.fills:
        _print(f"\n[bold]Fills[/bold]:" if _con else "\nFills:")
        for f in run.fills:
            _print(f"  {f}")


def render_backtest(res) -> None:
    s = res.stats
    _print("\n[bold]Backtest (momentum sleeve, point-in-time)[/bold]" if _con
           else "\nBacktest (momentum sleeve, point-in-time)")
    rows = [
        ("Total return", f"{s['total_return']:.1%}", f"{s.get('benchmark_total_return', 0):.1%}"),
        ("CAGR", f"{s['cagr']:.1%}", f"{s.get('benchmark_cagr', 0):.1%}"),
        ("Sharpe", f"{s['sharpe']:.2f}", "-"),
        ("Sortino", f"{s['sortino']:.2f}", "-"),
        ("Max drawdown", f"{s['max_drawdown']:.1%}", "-"),
        ("Annual alpha", f"{s.get('alpha_annual', 0):.1%}", "-"),
        ("Beta", f"{s.get('beta', 0):.2f}", "-"),
        ("Return vs bench (x)", f"{s.get('return_multiple_vs_benchmark') or 0:.2f}x", "-"),
    ]
    if _con:
        t = Table(header_style="bold cyan")
        t.add_column("Metric"); t.add_column("Strategy"); t.add_column("Benchmark")
        for r in rows:
            t.add_row(*r)
        _con.print(t)
    else:
        for r in rows:
            print(f"  {r[0]:22} {r[1]:>10}  bench {r[2]}")


def write_markdown(scan, path: str | Path | None = None, *, run=None) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = utcnow().strftime("%Y%m%d_%H%M%SZ")
    path = Path(path) if path else REPORT_DIR / f"scan_{ts}.md"
    L = []
    L.append(f"# rh-agent target book — {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}\n")
    L.append(f"**Regime:** {scan.regime.describe()}\n")
    L.append(f"- Universe scanned: **{scan.universe_size}**, scored: **{scan.scored_size}**, "
             f"eligible: **{len(scan.eligible)}**, positions: **{len(scan.targets)}**")
    L.append(f"- Sizing equity: **${scan.equity:,.0f}**\n")
    if run is not None:
        L.append(f"- Broker mode: **{run.mode}** | account equity ${run.account.equity:,.0f}\n")
    L.append("## Target portfolio\n")
    L.append("| # | Ticker | Sector | Score | Weight | $ | Stop | Target | Drivers |")
    L.append("|---|--------|--------|------:|-------:|--:|-----:|-------:|---------|")
    for i, p in enumerate(scan.targets, 1):
        L.append(f"| {i} | {p.ticker} | {p.sector} | {p.score:.0f} | {p.weight:.1%} | "
                 f"{p.dollars:,.0f} | {p.stop_price or '-'} | {p.take_profit or '-'} | {p.rationale} |")
    L.append("\n## Analyst panel — top 10 by composite\n")
    L.append("| Ticker | Composite | Momentum | Quant | Catalyst | SmartMoney | Sentiment | Pillars | Flags |")
    L.append("|--------|----------:|---------:|------:|---------:|-----------:|----------:|--------:|-------|")
    for v in scan.verdicts[:10]:
        a = v.analyst_scores
        L.append(f"| {v.ticker} | {v.composite:.1f} | {a.get('momentum_trader','-')} | "
                 f"{a.get('quant','-')} | {a.get('catalyst_trader','-')} | {a.get('smart_money','-')} | "
                 f"{a.get('sentiment_analyst','-')} | {v.pillars_passing} | {','.join(v.flags)} |")
    if run is not None and run.orders:
        L.append("\n## Orders\n")
        for o in run.orders:
            amt = f"${o.notional:,.0f}" if o.notional else f"{o.quantity:.2f} sh"
            L.append(f"- **{o.side.upper()}** {o.ticker} {amt} — {o.reason}")
    L.append("\n---\n*Not investment advice. Live trading risks real capital. "
             "No system can guarantee outperformance.*\n")
    path.write_text("\n".join(L))
    return path
