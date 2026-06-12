"""The 6th analyst — an LLM-in-the-loop catalyst & macro reviewer (Claude).

Once per rebalance, the agent hands the quant shortlist + a market-news context
to Claude in a SINGLE batched call. Claude reasons about catalysts, earnings
setups, and geopolitical/macro risk — the narrative judgment the pure-quant
engine can't do — and returns a 0..100 conviction + stance + rationale per name,
plus an overall market read. The Chief PM blends that into the composite.

Design choices (per the claude-api skill):
  * Official `anthropic` SDK, structured-JSON output (json_schema), adaptive
    thinking, effort configurable.
  * Static system prompt is prompt-cached; only the volatile per-cycle data
    goes in the user turn (after the cached prefix).
  * Model selectable via config / RH_AI_MODEL (default: claude-opus-4-8).
  * GRACEFUL NO-OP: if ANTHROPIC_API_KEY or the SDK is missing, or the call
    errors, it returns nothing and the agent runs pure-quant exactly as before.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from ..config import Config
from ..logging_setup import get_logger

log = get_logger("ai_analyst")

SYSTEM_PROMPT = """You are the sixth member of a quantitative trading desk's \
analyst panel — the macro & catalyst specialist. The other five are systematic \
factor models (momentum, quality, catalyst/estimates, smart-money, sentiment). \
Your job is the judgment they cannot do: read the current market context and \
each candidate's recent news, and reason about real-world catalysts and risks \
over a 1–3 month horizon without inventing missing facts.

For every candidate you are given, weigh:
  - Idiosyncratic catalysts: earnings setups, product/regulatory events, \
guidance, M&A, management/insider signals, sector rotation.
  - Macro & geopolitical overlay: rates/Fed path, inflation, growth, USD, oil, \
elections/conflict/tariffs — and how each name is exposed.
  - Whether the quant scores you're shown are confirmed or contradicted by the \
narrative (a high-momentum name into a known negative catalyst is a trap).
  - Options flow, short interest, sector/industry sensitivity, and whether \
apparent correlations have a plausible causal channel or are just noise.
  - Historical price-pattern evidence: trend, drawdown, volatility, beta/correlation \
to broad market, and whether the setup implies favorable forward asymmetry.

Output discipline:
  - Score 0–100 = your forward conviction for the next 1–3 months (50 = neutral).
  - stance ∈ {bullish, neutral, bearish}.
  - rationale = ONE tight sentence naming the dominant driver or risk.
  - Be calibrated and skeptical, not promotional. Penalize hype, stale/low-quality \
data, fragile correlations, and crowded trades into binary events. You are not \
given positions or P&L — judge the setup on its merits. This is decision support \
for a real-money system; do not invent facts you weren't given — state uncertainty \
when data is absent or ambiguous."""

# Array form (not a dict keyed by ticker) so it's a valid strict json_schema.
_SCHEMA = {
    "type": "object",
    "properties": {
        "market_read": {"type": "string",
                        "description": "2-3 sentences on the current macro/market regime and risks."},
        "assessments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "score": {"type": "integer"},
                    "stance": {"type": "string", "enum": ["bullish", "neutral", "bearish"]},
                    "rationale": {"type": "string"},
                },
                "required": ["ticker", "score", "stance", "rationale"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["market_read", "assessments"],
    "additionalProperties": False,
}


@dataclass
class AIResult:
    market_read: str = ""
    views: dict = field(default_factory=dict)   # ticker -> {score, stance, rationale}


class AIAnalyst:
    def __init__(self, cfg: Config):
        a = cfg.get("ai_analyst", {}) or {}
        self.cfg = cfg
        self.model = os.getenv("RH_AI_MODEL", a.get("model", "claude-opus-4-8"))
        self.effort = a.get("effort", "low")          # low|medium|high (cost vs depth)
        self.weight = max(0.0, min(1.0, float(a.get("weight", 0.25))))  # blend weight, clamped [0,1]
        self.max_candidates = int(a.get("max_candidates", 15))
        self.max_tokens = int(a.get("max_tokens", 6000))
        # A too-tight timeout silently drops the AI voice every cycle (graceful
        # no-op) — make it tunable so deep-effort calls actually land.
        self.timeout = float(a.get("timeout_seconds", 60.0))
        self.api_key = Config.api_key("anthropic")
        cfg_enabled = a.get("enabled", True)
        self.enabled = bool(cfg_enabled and self.api_key)
        self._client = None
        if cfg_enabled and not self.api_key:
            log.info("AI analyst disabled: no ANTHROPIC_API_KEY")

    def _client_or_none(self):
        if self._client is not None:
            return self._client
        # Guard BOTH the import and the constructor — a missing/incompatible SDK
        # must degrade to a graceful no-op, never abort a rebalance. Bounded
        # timeout + single retry so a hung call can't wedge the window either
        # (SDK default is a 600s timeout with 2 retries).
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout,
                                               max_retries=1)
        except Exception as e:
            log.warning("AI analyst: client unavailable (%s) — running pure-quant", e)
            self.enabled = False
            return None
        return self._client

    def assess(self, market_context: str, candidates: list[dict]) -> AIResult:
        """One batched call. Returns AIResult (empty on any failure)."""
        if not self.enabled or not candidates:
            return AIResult()
        client = self._client_or_none()
        if client is None:
            return AIResult()

        user_payload = {
            "market_context": market_context[:4000],
            "candidates": candidates[: self.max_candidates],
            "instructions": "Assess EVERY candidate. Return market_read + one "
                            "assessment per ticker (score 0-100, stance, one-sentence rationale).",
        }
        try:
            resp = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                thinking={"type": "adaptive"},
                system=[{"type": "text", "text": SYSTEM_PROMPT,
                         "cache_control": {"type": "ephemeral"}}],
                output_config={"format": {"type": "json_schema", "schema": _SCHEMA},
                               "effort": self.effort},
                messages=[{"role": "user", "content": json.dumps(user_payload, default=str)}],
            )
        except Exception as e:
            log.warning("AI analyst call failed (%s) — continuing pure-quant", e)
            return AIResult()

        text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
        try:
            data = json.loads(text)
        except Exception:
            log.warning("AI analyst: unparseable response — skipping overlay")
            return AIResult()

        views = {}
        for a in data.get("assessments", []):
            tk = (a.get("ticker") or "").upper()
            if not tk:
                continue
            score = a.get("score")
            try:
                score = max(0.0, min(100.0, float(score)))
            except (TypeError, ValueError):
                continue
            stance = a.get("stance", "neutral")
            if stance not in ("bullish", "neutral", "bearish"):
                stance = "neutral"
            views[tk] = {"score": score, "stance": stance,
                         "rationale": (a.get("rationale") or "")[:160]}
        cr = getattr(resp, "usage", None)
        if cr is not None:
            log.info("AI analyst: %d views | model=%s | cache_read=%s in=%s out=%s",
                     len(views), self.model, getattr(cr, "cache_read_input_tokens", "?"),
                     getattr(cr, "input_tokens", "?"), getattr(cr, "output_tokens", "?"))
        return AIResult(market_read=data.get("market_read", ""), views=views)
