"""
Unified LLM Client — wraps GitHub Models API (OpenAI-compatible).
Supports any model available on the endpoint: GPT-4.1, o4-mini, Claude, etc.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import yaml
from openai import AsyncOpenAI


@dataclass
class LLMConfig:
    api_base: str = "https://models.github.ai/inference"
    api_key: str = ""
    default_model: str = "openai/gpt-4o"
    temperature: float = 0.4
    max_tokens: int = 8192

    @classmethod
    def from_yaml(cls, path: str = "config/settings.yaml") -> "LLMConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        llm_cfg = raw.get("llm", {})
        api_key = llm_cfg.get("api_key", "")
        # Resolve ${ENV_VAR} patterns
        if isinstance(api_key, str) and api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            api_key = os.environ.get(env_var, "")
        if not api_key:
            api_key = os.environ.get("GITHUB_TOKEN", "")
        return cls(
            api_base=llm_cfg.get("api_base", cls.api_base),
            api_key=api_key,
            default_model=llm_cfg.get("default_model", cls.default_model),
            temperature=llm_cfg.get("temperature", cls.temperature),
            max_tokens=llm_cfg.get("max_tokens", cls.max_tokens),
        )


class LLMClient:
    """Async LLM client backed by GitHub Models API."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_yaml()
        if not self.config.api_key:
            raise ValueError(
                "No API key found. Set GITHUB_TOKEN environment variable or "
                "configure llm.api_key in config/settings.yaml"
            )
        self._client = AsyncOpenAI(
            base_url=self.config.api_base,
            api_key=self.config.api_key,
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Send a chat completion request and return the assistant content."""
        kwargs: dict[str, Any] = dict(
            model=model or self.config.default_model,
            messages=messages,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
        )
        if response_format is not None:
            kwargs["response_format"] = response_format
        resp = await self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Chat completion that forces JSON output."""
        return await self.chat(
            messages,
            response_format={"type": "json_object"},
            **kwargs,
        )
