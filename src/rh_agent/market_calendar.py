"""US equity session calendar (NYSE-style regular + early-close days).

If the host lacks tzdata (``zoneinfo`` unavailable), a built-in US-Eastern
DST rule keeps session times correct instead of silently drifting to UTC —
a daemon on a mis-provisioned VM would otherwise hunt 4–5 hours off and look
like it "only trades at the start of the day".
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo

    _ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    _ET = None

# Full-day closures (extend annually or replace with exchange_calendars later).
_US_HOLIDAYS: set[str] = {
    # 2024
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
    "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
    # 2025
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
    # 2028
    "2028-01-01", "2028-01-17", "2028-02-21", "2028-04-14", "2028-05-29",
    "2028-06-19", "2028-07-04", "2028-09-04", "2028-11-23", "2028-12-25",
}

# Early close at 13:00 ET (day after Thanksgiving, Christmas Eve when weekday, etc.)
_EARLY_CLOSE: set[str] = {
    "2024-07-03", "2024-11-29", "2024-12-24",
    "2025-07-03", "2025-11-28", "2025-12-24",
    "2026-07-02", "2026-11-27", "2026-12-24",
    "2027-07-02", "2027-11-26", "2027-12-31",
    "2028-07-03", "2028-11-24", "2028-12-22",
}


def _nth_sunday_utc(year: int, month: int, nth: int) -> int:
    """Day-of-month of the nth Sunday."""
    d = datetime(year, month, 1, tzinfo=timezone.utc)
    first_sunday = 1 + (6 - d.weekday()) % 7
    return first_sunday + 7 * (nth - 1)


def _eastern_offset_hours(now_utc: datetime) -> int:
    """US-Eastern UTC offset by statute: EDT (-4) from 2:00 EST on the second
    Sunday of March until 2:00 EDT on the first Sunday of November, else EST (-5)."""
    y = now_utc.year
    dst_start = datetime(y, 3, _nth_sunday_utc(y, 3, 2), 7, 0, tzinfo=timezone.utc)
    dst_end = datetime(y, 11, _nth_sunday_utc(y, 11, 1), 6, 0, tzinfo=timezone.utc)
    return -4 if dst_start <= now_utc < dst_end else -5


def to_eastern(now_utc: datetime | None = None) -> datetime:
    """Convert to US-Eastern, via zoneinfo when present, else the built-in rule."""
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if _ET is not None:
        return now.astimezone(_ET)
    return now.astimezone(timezone(timedelta(hours=_eastern_offset_hours(now))))


def tz_source() -> str:
    return "zoneinfo" if _ET is not None else "built-in US-Eastern DST fallback"


def is_market_open(now_utc: datetime | None = None) -> bool:
    """True during regular US equity session (09:30–16:00 ET, or 13:00 on early-close days)."""
    et = to_eastern(now_utc)
    if et.weekday() >= 5:
        return False
    day = et.strftime("%Y-%m-%d")
    if day in _US_HOLIDAYS:
        return False
    minutes = et.hour * 60 + et.minute
    open_m = 9 * 60 + 30
    close_m = 13 * 60 if day in _EARLY_CLOSE else 16 * 60
    return open_m <= minutes <= close_m


def minutes_to_close(now_utc: datetime | None = None) -> float | None:
    """Minutes until today's closing bell; None when the market is closed."""
    if not is_market_open(now_utc):
        return None
    et = to_eastern(now_utc)
    close_m = 13 * 60 if et.strftime("%Y-%m-%d") in _EARLY_CLOSE else 16 * 60
    return close_m - (et.hour * 60 + et.minute + et.second / 60.0)


def session_state(now_utc: datetime | None = None) -> dict:
    """Session snapshot for diagnostics: phase + ET clock + tz source."""
    et = to_eastern(now_utc)
    day = et.strftime("%Y-%m-%d")
    minutes = et.hour * 60 + et.minute
    close_m = 13 * 60 if day in _EARLY_CLOSE else 16 * 60
    if et.weekday() >= 5:
        phase = "weekend"
    elif day in _US_HOLIDAYS:
        phase = "holiday"
    elif minutes < 9 * 60 + 30:
        phase = "pre-market"
    elif minutes <= close_m:
        phase = "regular session" + (" (early close)" if day in _EARLY_CLOSE else "")
    else:
        phase = "after-hours"
    return {"phase": phase, "et": et.strftime("%Y-%m-%d %H:%M"), "tz_source": tz_source(),
            "open": is_market_open(now_utc)}
