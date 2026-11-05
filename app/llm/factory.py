"""Select the LLM implementation from settings."""

from __future__ import annotations

from app.config import Settings
from app.core.errors import LLMProviderError
from app.llm.base import LLMClient
from app.llm.deepseek_client import DeepSeekClient
from app.llm.scripted import DemoScenarioLLM


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "deepseek":
        return DeepSeekClient(settings)
    if settings.llm_provider == "scripted":
        return DemoScenarioLLM(settings)
    raise LLMProviderError(f"unknown LLM provider: {settings.llm_provider}")
