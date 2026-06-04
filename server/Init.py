"""FastAPI application: CORS, lifespan-driven engine startup, router wiring.

Replaces the sample's blocking ``on_event`` + ``loadOpenAI()``-gated startup
with a FastAPI ``lifespan`` that builds the engine exactly once and fails fast
if it cannot.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import settings
from engine.EngineClient import initEngine, shutdownEngine
from engine.ServingClient import initServing
from helper.LoggingHelper import configureLogging, getLogger
from server.ChatEndpoint import router as chat_router
from server.HealthEndpoint import router as health_router
from server.SchemaEndpoint import router as schema_router

logger = getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configureLogging()
    logger.info("Starting platform-ai-service: loading model '%s'...", settings.MODEL)

    result = await initEngine()
    if result.Status != 1:
        # Fail fast: a half-started inference service is worse than a crash.
        raise RuntimeError(f"Engine startup failed: {result.Message}")

    result = await initServing()
    if result.Status != 1:
        raise RuntimeError(f"Serving startup failed: {result.Message}")

    logger.info("platform-ai-service ready on %s:%s", settings.HOST, settings.PORT)
    try:
        yield
    finally:
        await shutdownEngine()
        logger.info("platform-ai-service stopped.")


app = FastAPI(
    title="platform-ai-service",
    description="OpenAI-compatible local LLM inference, powered by vLLM.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOW_ORIGINS,
    # Never pair wildcard origins with credentials (browser-rejected + unsafe).
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(schema_router)

# Expose vLLM's Prometheus metrics at /metrics if prometheus_client is present.
try:
    from prometheus_client import make_asgi_app

    app.mount("/metrics", make_asgi_app())
    logger.info("Prometheus metrics mounted at /metrics")
except Exception:  # pragma: no cover - metrics are optional
    logger.warning("prometheus_client unavailable; /metrics not mounted")
