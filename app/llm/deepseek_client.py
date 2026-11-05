"""DeepSeek chat client (OpenAI-compatible API).

Thinking-mode findings (verified against the DeepSeek docs):
- https://api-docs.deepseek.com/guides/thinking_mode : thinking is ON by default for the V4
  models and is toggled per request with ``"thinking": {"type": "enabled" | "disabled"}``;
  with the OpenAI SDK it is passed via ``extra_body``. Without tools ``reasoning_content`` need
  not be sent back (ignored); WITH tool calls it must be passed back fully in all later
  requests, otherwise the API answers 400.
- https://api-docs.deepseek.com/guides/tool_calls : non-thinking sample uses
  ``extra_body={"thinking": {"type": "disabled"}}``.

Implementation: ``settings.deepseek_thinking=False`` (default) sends an explicit
``thinking: disabled`` so no ``reasoning_content`` is produced and history replay stays simple.
When True we send ``thinking: enabled``; the parser keeps ``reasoning_content`` in the
assistant message so the runner replays it verbatim.
"""

from __future__ import annotations

import time
from typing import Any

import openai
from openai import AsyncOpenAI

from app.config import Settings
from app.core.errors import LLMProviderError
from app.core.retry import retry_async
from app.llm.base import LLMResponse, LLMUsage
from app.llm.parser import normalize_assistant_message
from app.observability.logging import get_logger

logger = get_logger("harness.llm.deepseek")

# Transport-level failures worth retrying: timeout, connection, 429 and 5xx.
_RETRYABLE = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
)


class DeepSeekClient:
    provider = "deepseek"

    def __init__(self, settings: Settings) -> None:
        if settings.deepseek_api_key is None or not settings.deepseek_api_key.get_secret_value():
            raise LLMProviderError("DEEPSEEK_API_KEY is not set; required for LLM_PROVIDER=deepseek")
        self._settings = settings
        self.model = settings.llm_model
        self._client = AsyncOpenAI(
            base_url=settings.deepseek_base_url,
            api_key=settings.deepseek_api_key.get_secret_value(),
            timeout=settings.llm_timeout_s,
            max_retries=0,
        )

    async def _create(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "extra_body": {"thinking": {"type": "enabled" if self._settings.deepseek_thinking else "disabled"}},
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return await self._client.chat.completions.create(**kwargs)

    async def decide(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        start = time.perf_counter()

        def log_retry(n: int, exc: BaseException) -> None:
            logger.warning("llm.retry", provider="deepseek", retry=n, error_type=type(exc).__name__)

        try:
            response = await retry_async(
                lambda: self._create(messages, tools),
                retries=self._settings.llm_max_retries,
                base_delay=0.5,
                retry_on=_RETRYABLE,
                on_retry=log_retry,
            )
        except openai.APIStatusError as exc:
            raise LLMProviderError(
                f"DeepSeek API error {exc.status_code}: {str(exc.message)[:300]}", status=exc.status_code
            ) from exc
        except openai.OpenAIError as exc:
            raise LLMProviderError(f"DeepSeek request failed: {type(exc).__name__}", status=None) from exc

        if not response.choices:
            raise LLMProviderError("DeepSeek returned no choices", status=None)
        raw_message = response.choices[0].message.model_dump(exclude_none=True)
        decision, assistant_message = normalize_assistant_message(raw_message, self._settings.max_tool_calls_per_step)
        usage = LLMUsage()
        if response.usage is not None:
            usage = LLMUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
        return LLMResponse(
            decision=decision,
            assistant_message=assistant_message,
            usage=usage,
            model=getattr(response, "model", None) or self.model,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )
