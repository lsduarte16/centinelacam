"""LLM-based contextual analyzer using Ollama (TinyLlama/Phi-3-mini)."""

import json
import logging
from datetime import datetime

import httpx

from src.config import settings
from src.gate_logic.events import GateEvent, Severity

logger = logging.getLogger(__name__)


class LLMAnalyzer:
    """Uses a local SLM via Ollama for contextual event analysis."""

    def __init__(self):
        self.model = settings.llm.model
        self.base_url = settings.llm.base_url
        self.timeout = settings.llm.timeout
        self.system_prompt = settings.llm.system_prompt
        self._client = httpx.Client(timeout=self.timeout)
        self._event_buffer: list[dict] = []
        self._buffer_size = 10

    async def analyze_event(self, event: GateEvent) -> dict | None:
        """Send event to LLM for contextual analysis."""
        self._event_buffer.append(event.to_dict())

        if len(self._event_buffer) < 3:
            return None

        prompt = self._build_prompt()

        try:
            response = await self._query_ollama(prompt)
            self._event_buffer.clear()
            return response
        except Exception as e:
            logger.error("LLM analysis failed: %s", e)
            return None

    def analyze_event_sync(self, event: GateEvent) -> dict | None:
        """Synchronous version for non-async contexts."""
        self._event_buffer.append(event.to_dict())

        if len(self._event_buffer) < 3:
            return None

        prompt = self._build_prompt()

        try:
            response = self._query_ollama_sync(prompt)
            self._event_buffer.clear()
            return response
        except Exception as e:
            logger.error("LLM analysis failed: %s", e)
            return None

    def _build_prompt(self) -> str:
        """Build analysis prompt from buffered events."""
        events_text = json.dumps(self._event_buffer[-self._buffer_size :], indent=2)
        return (
            f"Analiza los siguientes eventos de seguridad del andén y determina "
            f"si hay patrones anómalos o acciones requeridas:\n\n"
            f"Eventos recientes:\n{events_text}\n\n"
            f"Hora actual: {datetime.now().isoformat()}\n"
            f"Responde en JSON con: action (allow/alert/block), "
            f"severity (low/medium/high/critical), description (breve)."
        )

    def _query_ollama_sync(self, prompt: str) -> dict | None:
        """Query Ollama API synchronously."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": self.system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 256,
            },
        }

        resp = self._client.post(f"{self.base_url}/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()

        return self._parse_response(data.get("response", ""))

    async def _query_ollama(self, prompt: str) -> dict | None:
        """Query Ollama API asynchronously."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": self.system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 256,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()

        return self._parse_response(data.get("response", ""))

    def _parse_response(self, text: str) -> dict | None:
        """Extract JSON from LLM response."""
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            logger.warning("Could not parse LLM response: %s", text[:200])
        return None

    def assess_severity(self, events: list[GateEvent]) -> Severity:
        """Quick heuristic severity assessment without LLM."""
        if any(e.event_type.value == "unauthorized_access" for e in events):
            return Severity.CRITICAL
        if sum(1 for e in events if "entry" in e.event_type.value) > 10:
            return Severity.HIGH
        if any(e.event_type.value == "zone_crowded" for e in events):
            return Severity.MEDIUM
        return Severity.LOW

    def health_check(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            resp = self._client.get(f"{self.base_url}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    def shutdown(self):
        self._client.close()
