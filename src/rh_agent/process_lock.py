"""Single-instance lock so two daemon loops cannot double-trade."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from .config import REPO_ROOT
from .logging_setup import get_logger

log = get_logger("lock")
DEFAULT_LOCK = REPO_ROOT / "state" / "daemon.lock"


class ProcessLockError(RuntimeError):
    pass


def _harden_state_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        # Best effort on filesystems/platforms that ignore chmod.
        pass


@contextmanager
def daemon_lock(path: Path = DEFAULT_LOCK):
    """Acquire an exclusive non-blocking lock; raise if another instance holds it."""
    _harden_state_parent(path)
    fh = open(path, "a+")
    try:
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as e:
                raise ProcessLockError(
                    f"Another rh-agent daemon holds {path} — refusing to start."
                ) from e
        else:
            import fcntl

            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as e:
                raise ProcessLockError(
                    f"Another rh-agent daemon holds {path} — refusing to start."
                ) from e
        fh.seek(0)
        fh.truncate()
        fh.write(str(os.getpid()))
        fh.flush()
        try:
            path.chmod(0o600)
        except OSError:
            pass
        log.info("acquired daemon lock %s", path)
        yield
    finally:
        try:
            if os.name != "nt":
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        fh.close()
