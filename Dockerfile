FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY config ./config
COPY scripts ./scripts

ENV PYTHONPATH=/app/src \
    EXECUTION_MODE=live \
    LIVE_TRADING_CONFIRM=I_UNDERSTAND_REAL_MONEY

# Default: run the live always-on loop. Override CMD for scan/backtest.
# Provide API keys + (optional) Robinhood OAuth token via -e / --env-file.
ENTRYPOINT ["python", "-m", "rh_agent.cli"]
CMD ["loop", "--execute"]
