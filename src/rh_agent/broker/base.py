"""Broker abstraction. Two real implementations:
  * PaperBroker        — simulates fills on live prices (DEFAULT, no risk)
  * RobinhoodMCPBroker — places real orders via the Robinhood Agentic MCP
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Account, Order


class Broker(ABC):
    name: str = "base"
    supports_live: bool = False

    @abstractmethod
    def get_account(self) -> Account: ...

    @abstractmethod
    def place_order(self, order: Order, dry_run: bool = True) -> dict: ...

    def get_orders(self) -> list:
        return []

    def cancel_all(self) -> None:
        return None
