"""
BaseAgent — Abstract base for every RF-1.5 agent.
Each agent owns a system prompt, a role name, and a reference to the shared LLM client.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from rf.llm.client import LLMClient

logger = logging.getLogger("rf")


def _extract_json(text: str) -> dict[str, Any]:
    """
    Extract JSON from LLM output that may be wrapped in ```json ... ``` blocks
    or contain leading/trailing text.
    """
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fence
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding the first { ... } block
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    # Last resort: return raw text in a wrapper
    logger.warning("Could not parse JSON from LLM output, wrapping as raw_text")
    return {"raw_text": text, "_parse_error": True}


class BaseAgent(ABC):
    """Base class for all Research Factory agents."""

    role: str = "BaseAgent"
    system_prompt: str = "You are a helpful research assistant."

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.history: list[dict[str, str]] = []
        self.spec = llm.config.agent_settings.get(self.role, {})
        
    # ------------------------------------------------------------------
    # Core LLM interaction
    # ------------------------------------------------------------------

    async def _ask(self, user_msg: str, *, json_mode: bool = False) -> str:
        """Send a message with the agent's system prompt, return raw text."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.history[-6:],  # Keep last 3 turns to manage context window
            {"role": "user", "content": user_msg},
        ]
        
        req_config = {
            "model": self.spec.get("model"),
            "temperature": self.spec.get("temperature"),
            "max_tokens": self.spec.get("max_tokens"),
        }
        
        if json_mode:
            result = await self.llm.chat_json(messages, **req_config)
        else:
            result = await self.llm.chat(messages, **req_config)
        # Append to conversational memory
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": result})
        logger.info("[%s] responded (%d chars)", self.role, len(result))
        return result

    async def _ask_structured(self, user_msg: str) -> dict[str, Any]:
        """Ask and parse JSON response with robust extraction."""
        raw = await self._ask(user_msg, json_mode=True)
        return _extract_json(raw)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent's primary mission. Returns enriched context."""
        ...

    def reset(self) -> None:
        self.history.clear()
