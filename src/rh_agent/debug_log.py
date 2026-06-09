from __future__ import annotations

import json
from pathlib import Path
from time import time

DEBUG_LOG_PATH = Path("/opt/cursor/logs/debug.log")


def write_debug_log(*, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(time() * 1000),
            }, default=str) + "\n")
    except Exception:
        pass
