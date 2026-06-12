"""Reconcile the current account against target positions and emit the minimal
set of orders, honouring a no-trade band (to cut churn) and a turnover cap.
"""
from __future__ import annotations

from typing import Callable

from .broker.orders import order_succeeded
from .config import Config
from .logging_setup import get_logger
from .models import Account, Order, TargetPosition

log = get_logger("execution")


def build_orders(account: Account, targets: list[TargetPosition], cfg: Config,
                 price_fn: Callable[[str], float | None], *,
                 allow_buys: bool = True,
                 exclude_tickers: set[str] | None = None,
                 exclude_buys: set[str] | None = None,
                 explain: list | None = None) -> list[Order]:
    """exclude_tickers suppresses BOTH sides (unresolved protective orders must
    not be double-traded); exclude_buys suppresses buys only (re-entry cooldowns
    must never block an exit)."""
    equity = account.equity or 0.0
    if equity <= 0:
        return []
    reb = cfg.get("portfolio.rebalance", {})
    band = float(reb.get("no_trade_band", 0.015))
    entry_band = float(reb.get("entry_no_trade_band", 0.0))
    max_turn = float(reb.get("max_turnover_per_rebalance", 0.30))
    min_notional = float(reb.get("min_order_notional", 20.0))

    # `explain` (optional) collects one note per suppressed/shaped decision so a
    # zero-order rebalance can tell the operator WHY it stayed quiet.
    def note(ticker: str, action: str, detail: str = "") -> None:
        if explain is not None:
            explain.append({"ticker": ticker, "action": action, "detail": detail})

    exclude = exclude_tickers or set()
    no_buy = (exclude_buys or set()) | exclude
    cur = account.position_map()
    tmap = {t.ticker: t for t in targets}
    orders: list[Order] = []

    # --- exits & trims (sells) ---
    for tk, pos in cur.items():
        if tk in exclude:
            note(tk, "excluded", "unresolved protective order")
            continue
        px = price_fn(tk) or pos.current_price or pos.avg_price
        cur_w = (pos.quantity * px) / equity if px else 0.0
        tgt_w = tmap[tk].weight if tk in tmap else 0.0
        if tk not in tmap:
            orders.append(Order(tk, "sell", pos.quantity, reason="exit: dropped from target"))
        elif (cur_w - tgt_w) > band:
            sell_dollars = (cur_w - tgt_w) * equity
            if sell_dollars >= min_notional and px:
                orders.append(Order(tk, "sell", round(sell_dollars / px, 4),
                                    reason=f"trim {cur_w:.1%}->{tgt_w:.1%}"))
            else:
                note(tk, "skipped_min_notional",
                     f"trim ${sell_dollars:.0f} < ${min_notional:.0f} min")

    # --- entries & adds (buys) ---
    buys: list[Order] = []
    if not allow_buys and tmap:
        note("*", "buys_halted", "daily drawdown halt or sell-only cycle")
    for tk, t in tmap.items():
        if tk in no_buy:
            if tk not in cur:
                note(tk, "excluded", "protective order pending or re-entry cooldown")
            continue
        if not allow_buys:
            continue
        px = price_fn(tk)
        if not px:
            note(tk, "skipped_no_price", "no live quote at order time")
            continue
        cur_w = (cur[tk].quantity * px) / equity if tk in cur else 0.0
        trade_band = entry_band if tk not in cur else band
        if (t.weight - cur_w) <= trade_band:
            if tk in cur:
                note(tk, "hold_within_band",
                     f"drift {abs(t.weight - cur_w):.2%} <= band {trade_band:.2%}")
            continue
        buy_dollars = (t.weight - cur_w) * equity
        if buy_dollars < min_notional:
            note(tk, "skipped_min_notional",
                 f"buy ${buy_dollars:.0f} < ${min_notional:.0f} min")
            continue
        buys.append(Order(tk, "buy", round(buy_dollars / px, 4), notional=round(buy_dollars, 2),
                          reason=f"{'enter' if tk not in cur else 'add'} score={t.score}"))

    # --- turnover cap: scale buys down if needed (sells always allowed) ---
    buy_turnover = sum((o.notional or 0) for o in buys) / equity
    if buy_turnover > max_turn:
        scale = max_turn / buy_turnover
        for o in buys:
            o.notional = round((o.notional or 0) * scale, 2)
            px = price_fn(o.ticker) or 0
            o.quantity = round(o.notional / px, 4) if px else o.quantity
        log.info("turnover cap hit: scaled buys by %.2f", scale)
        note("*", "turnover_capped", f"buys scaled by {scale:.2f}")
        # scaling can push small orders under the minimum — drop those (-> cash)
        kept = [o for o in buys if (o.notional or 0) >= min_notional]
        for o in buys:
            if o not in kept:
                note(o.ticker, "dropped_below_min_after_scale", "")
        buys = kept

    # --- buying-power cap: never order more cash than the account actually has,
    # or the broker rejects it ("Not enough buying power"). Cushion for the
    # slippage between our quote and the market fill. ---
    bp = float(account.buying_power or 0.0)
    if bp > 0:
        budget = bp * 0.97
        buy_total = sum((o.notional or 0) for o in buys)
        if buy_total > budget:
            scale = (budget / buy_total) if buy_total else 0.0
            for o in buys:
                o.notional = round((o.notional or 0) * scale, 2)
                px = price_fn(o.ticker) or 0
                o.quantity = round(o.notional / px, 4) if px else o.quantity
            log.info("buying-power cap: $%.0f available -> scaled buys by %.2f", bp, scale)
            note("*", "buying_power_capped", f"${bp:.0f} available, buys scaled by {scale:.2f}")
            buys = [o for o in buys if (o.notional or 0) >= min_notional]

    orders.extend(buys)
    log.info("orders: %d (%d sells, %d buys)", len(orders),
             sum(1 for o in orders if o.side == "sell"), len(buys))
    return orders


def execute_orders(
    broker,
    orders: list[Order],
    cfg: Config,
    *,
    account: Account | None = None,
    get_account: Callable[[], Account] | None = None,
) -> tuple[list[dict], Account | None]:
    """Place orders sequentially with live-safety checks.

    For live brokers: re-check arming before each order, refresh the account
    snapshot before each buy (buying-power drift), verify broker acceptance,
    and return a post-trade account when possible.
    """
    if not orders:
        return [], account

    live = cfg.live_trading_armed and getattr(broker, "supports_live", False)
    fills: list[dict] = []
    post = account

    for o in orders:
        if live and not cfg.live_trading_armed:
            log.error("live trading disarmed mid-batch — aborting remaining orders")
            break

        if live and o.side == "buy" and get_account:
            try:
                fresh = get_account()
                post = fresh
                if not fresh.reliable:
                    log.error("unreliable account before buy %s — skipping buy", o.ticker)
                    fills.append({
                        "status": "skipped",
                        "ticker": o.ticker,
                        "side": o.side,
                        "reason": "unreliable_account",
                    })
                    continue
                if not _cap_live_buy_to_buying_power(o, fresh, cfg):
                    fills.append({
                        "status": "skipped",
                        "ticker": o.ticker,
                        "side": o.side,
                        "reason": "insufficient_buying_power",
                    })
                    continue
            except Exception as e:
                log.error("account refresh before live buy %s failed — skipping buy: %s",
                          o.ticker, e)
                fills.append({
                    "status": "skipped",
                    "ticker": o.ticker,
                    "side": o.side,
                    "reason": "account_refresh_failed",
                })
                continue

        res = broker.place_order(o, dry_run=False)
        fills.append(res)
        if not order_succeeded(res, executing=True):
            log.error("order not accepted: %s %s qty=%s — %s",
                      o.side, o.ticker, o.quantity, res)
            if live:
                log.error("aborting remaining live orders after rejected/error response")
                break

    if get_account:
        try:
            post = get_account()
        except Exception as e:
            log.warning("post-trade account refresh failed: %s", e)

    return fills, post


def _cap_live_buy_to_buying_power(order: Order, account: Account, cfg: Config) -> bool:
    """Conservatively cap a live market buy to the fresh buying-power snapshot."""
    budget = max(float(account.buying_power or 0.0) * 0.97, 0.0)
    min_notional = float(cfg.get("portfolio.rebalance.min_order_notional", 20.0))
    if budget < min_notional:
        log.error("buying power $%.2f below min order $%.2f before %s",
                  budget, min_notional, order.ticker)
        return False
    if order.notional is None:
        # Share-quantity buys cannot be safely rescaled without a fresh quote here.
        return True
    if order.notional <= budget:
        return True
    log.info("fresh buying-power cap before %s: $%.2f -> $%.2f",
             order.ticker, order.notional, budget)
    order.notional = round(budget, 2)
    return order.notional >= min_notional
