"""Broker-level errors for live trading safety."""


class LiveBrokerUnavailable(RuntimeError):
    """Live trading is armed but no authenticated Robinhood broker is available."""
