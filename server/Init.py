"""FastAPI gateway application.

Brings up the vLLM backend (or connects to an external one) via the lifespan,
then proxies the OpenAI-compatible surface to it while owning auth, CORS, health,
and the custom /api/v1/infer-schema feature.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import settings
from engine.Backend import startBackend, stopBackend
from helper.LoggingHelper import configureLogging, getLogger
from server.HealthEndpoint import router as health_router
from server.Proxy import router as proxy_router
from server.SchemaEndpoint import router as schema_router

logger = getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configureLogging()
    logger.info("Starting gateway; bringing up vLLM backend (model '%s')...", settings.MODEL)

    result = await startBackend()
    if result.Status != 1:
        # Fail fast: a gateway with no reachable model is useless.
        raise RuntimeError(f"Backend startup failed: {result.Message}")

    logger.info(
        "Gateway ready on %s:%s -> backend %s",
        settings.HOST,
        settings.PORT,
        settings.LLM_BACKEND_URL,
    )
    try:
        yield
    finally:
        await stopBackend()
        logger.info("Gateway stopped.")


app = FastAPI(
    title="platform-ai-service",
    description="OpenAI-compatible LLM inference gateway over a vLLM backend.",
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
app.include_router(schema_router)
# Proxy router last: its /v1/{path:path} catch-all must not shadow specific routes.
app.include_router(proxy_router)
