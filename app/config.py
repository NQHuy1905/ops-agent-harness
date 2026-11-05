"""Runtime configuration loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM ---
    llm_provider: Literal["scripted", "deepseek"] = "scripted"
    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-v4-flash"
    deepseek_thinking: bool = False
    llm_timeout_s: float = Field(30.0, gt=0)
    llm_max_retries: int = Field(2, ge=0)
    agent_name: str = "ops-assistant"

    # --- Storage ---
    database_url: str = "sqlite+aiosqlite:///./harness.db"
    mock_data_dir: Path = Path("./data")

    # --- Execution limits & safety ---
    max_steps: int = Field(10, ge=1)
    max_wall_time_s: float = Field(60.0, gt=0)
    max_tool_calls_per_step: int = Field(5, ge=1)
    max_malformed_llm_responses: int = Field(2, ge=0)
    loop_detect_threshold: int = Field(3, ge=2)

    # --- Tool execution ---
    tool_timeout_s: float = Field(5.0, gt=0)
    tool_max_retries: int = Field(2, ge=0)
    tool_retry_base_delay_s: float = Field(0.2, ge=0)

    # --- Observability ---
    log_level: str = "INFO"
    log_file: Path | None = None
    otel_exporter_otlp_endpoint: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
