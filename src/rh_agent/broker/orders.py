"""Shared order helpers for broker implementations."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from ..models import Order


def order_succeeded(result: dict | None, *, executing: bool = True) -> bool:
    """True when a broker response indicates the order was accepted (not rejected)."""
    if not result:
        return False
    status = (result.get("status") or "").lower()
    if status in ("error", "rejected"):
        return False
    if executing:
        return status in ("submitted", "filled")
    return status in ("submitted", "filled", "preview")


def stable_ref_id(order: Order, account_number: str | None, *, day_key: str | None = None) -> str:
    """Deterministic idempotency key for live order placement (retries dedupe)."""
    day = day_key or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    qty = order.quantity if order.quantity is not None else ""
    notional = order.notional if order.notional is not None else ""
    blob = "|".join(
        str(x)
        for x in (
            account_number or "",
            order.ticker.upper(),
            order.side.lower(),
            order.order_type,
            qty,
            notional,
            day,
            (order.reason or "")[:48],
        )
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:32]
