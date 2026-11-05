"""Settings parsing from env files."""

from app.config import Settings


def test_blank_values_in_env_file_are_treated_as_unset(tmp_path, monkeypatch):
    for var in ("LOG_FILE", "OTEL_EXPORTER_OTLP_ENDPOINT", "DEEPSEEK_API_KEY", "MAX_STEPS"):
        monkeypatch.delenv(var, raising=False)
    env = tmp_path / ".env"
    env.write_text("LOG_FILE=\nOTEL_EXPORTER_OTLP_ENDPOINT=\nDEEPSEEK_API_KEY=\nMAX_STEPS=\n")

    settings = Settings(_env_file=env)

    assert settings.log_file is None
    assert settings.otel_exporter_otlp_endpoint is None
    assert settings.deepseek_api_key is None
    assert settings.max_steps == 10
