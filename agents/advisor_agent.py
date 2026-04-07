import json
from typing import Iterator, List, Optional

from utils import config
from agents.base_agent import BaseAgent
from models.data_models import AdvisorDataLayer, ChatTurn

try:
    from google import genai
except Exception:  # pragma: no cover
    genai = None


class AdvisorAgent(BaseAgent):
    """
    Gemini reasoning layer: consumes structured pipeline output (AdvisorDataLayer)
    and produces analytical prose. Streaming is delegated to generate_content_stream.
    """

    def __init__(self) -> None:
        super().__init__()
        self.client = None
        provider = getattr(config, "ADVISOR_PROVIDER", "gemini")
        api_key = getattr(config, "ADVISOR_GEMINI_API_KEY", None)
        if provider == "gemini" and genai and api_key:
            self.client = genai.Client(api_key=api_key)

    def is_configured(self) -> bool:
        return bool(self.client)

    def _format_data_layer_json(self, data_layer: AdvisorDataLayer) -> str:
        d = data_layer.model_dump(mode="json")
        if isinstance(d.get("price_prediction"), dict):
            d["price_prediction"].pop("explainability", None)
        return json.dumps(d, indent=2, ensure_ascii=False)

    def _format_explainability_json(self, data_layer: AdvisorDataLayer) -> str:
        expl = data_layer.price_prediction.explainability
        if not expl:
            return "(none — run full pipeline with price history for attribution)"
        return json.dumps(expl.model_dump(mode="json"), indent=2, ensure_ascii=False)

    def _format_conversation_memory(self, history: List[ChatTurn]) -> str:
        if not history:
            return "(none — first turn in this session)"
        max_chars = int(getattr(config, "ADVISOR_ASSISTANT_MEMORY_MAX_CHARS", 4000))
        lines: List[str] = []
        for turn in history:
            label = "User" if turn.role == "user" else "Assistant"
            text = turn.content.strip()
            if turn.role == "assistant" and len(text) > max_chars:
                text = text[:max_chars] + "…"
            lines.append(f"- {label}: {text}")
        return "\n".join(lines)

    def _build_reasoning_prompt(
        self,
        user_message: str,
        data_layer: AdvisorDataLayer,
        history: List[ChatTurn],
    ) -> str:
        data_block = self._format_data_layer_json(data_layer)
        memory_block = self._format_conversation_memory(history)

        return f"""You are a financial reasoning analyst, not a generic chatbot.

You will receive:
1) DATA LAYER — structured pipeline signals (JSON). Authoritative for numbers, headline text, and social/news volume.
2) EXPLAINABILITY — approximate attribution for the price prediction (drivers, sentiment role, recent events). Use it to explain *why* the model output may lean a certain way; it is not a guarantee of future prices.
3) CONVERSATION MEMORY — short prior turns for continuity only; not a fact source.

Rules:
- Ground every factual claim in the DATA LAYER. If a fact is not present there, say you do not have that data.
- Do not invent prices, sentiment scores, headlines, or events.
- When model confidence is low or confidence intervals are wide, state uncertainty explicitly.
- Prefer precise, analytical language over platitudes. Reference specific fields (e.g. sentiment score, dominant emotion, headline themes) when reasoning.

--- DATA LAYER (JSON; facts only) ---
{data_block}

--- EXPLAINABILITY (prediction attribution JSON; approximate) ---
{self._format_explainability_json(data_layer)}

--- CONVERSATION MEMORY (context only; not a fact source) ---
{memory_block}

--- USER QUESTION ---
{user_message.strip()}

--- OUTPUT FORMAT ---
Respond in Markdown with **exactly** these four sections, in this order, using these headings:
### What is happening?
### What might happen next?
### Why?
### Actionable insight

In each section, tie conclusions to evidence from the DATA LAYER. In **Actionable insight**, give risk-aware, non-prescriptive guidance (what to monitor, scenarios to watch, information gaps)—not explicit buy/sell instructions."""

    def stream_advice_tokens(
        self,
        user_message: str,
        data_layer: AdvisorDataLayer,
        history: Optional[List[ChatTurn]] = None,
    ) -> Iterator[str]:
        """
        Synchronous token iterator for the WebSocket layer (LLM reasoning only).
        """
        if not self.client:
            yield (
                "Robo-advisor is not configured. "
                "Set ADVISOR_PROVIDER=gemini and ADVISOR_GEMINI_API_KEY "
                "(or GEMINI_API_KEY / GOOGLE_API_KEY) in .env."
            )
            return

        prompt = self._build_reasoning_prompt(user_message, data_layer, history or [])
        stream = self.client.models.generate_content_stream(
            model=getattr(config, "ADVISOR_GEMINI_MODEL", config.GEMINI_MODEL),
            contents=prompt,
        )
        for chunk in stream:
            token_text = getattr(chunk, "text", "") or ""
            if token_text:
                yield token_text

    async def generate_advice(
        self,
        user_message: str,
        data_layer: AdvisorDataLayer,
        history: Optional[List[ChatTurn]] = None,
    ) -> str:
        """Non-streaming helper; same reasoning path as stream_advice_tokens."""

        def _collect() -> str:
            return "".join(self.stream_advice_tokens(user_message, data_layer, history or []))

        return await self.run_blocking(_collect)
