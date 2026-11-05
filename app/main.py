"""FastAPI application entrypoint: ``uvicorn app.main:app``."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import Settings, get_settings
from app.container import Container, build_container
from app.core.errors import HarnessError
from app.observability.logging import configure_logging, get_logger

log = get_logger("harness.api")


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    """``container`` may be pre-built (tests inject a scripted LLM / fake clock)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        cfg = settings or get_settings()
        if container is None:
            configure_logging(cfg.log_level, cfg.log_file)
        app.state.container = container or await build_container(cfg)
        log.info("api.started", llm_provider=app.state.container.llm.provider)
        try:
            yield
        finally:
            if container is None:
                await app.state.container.close()

    app = FastAPI(
        title="Ops Agent Harness",
        version="0.1.0",
        description="Agent harness for an operations assistant (LLM-tool loop, approvals, limits, traces).",
        lifespan=lifespan,
    )

    @app.exception_handler(HarnessError)
    async def harness_error_handler(_request: Request, exc: HarnessError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content={"error": exc.to_dict()})

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        # Same envelope as harness errors; keep field-level details for clients.
        parts = [f"{'.'.join(str(p) for p in e['loc'] if p != 'body')}: {e['msg']}" for e in exc.errors()[:5]]
        return JSONResponse(
            status_code=422,
            content={
                "error": {"code": "validation_error", "message": "; ".join(parts)},
                "detail": jsonable_encoder(exc.errors()),
            },
        )

    app.include_router(router)
    return app


app = create_app()
