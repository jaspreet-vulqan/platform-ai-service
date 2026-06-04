"""Health, model listing, and info endpoints.

/health* are intentionally unauthenticated so liveness/readiness probes work.
/v1/models is OpenAI-compatible (auth-protected). /info returns our Result
envelope (the sample idiom, kept for auxiliary routes).
"""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

import settings
from engine.EngineClient import isReady
from engine.ServingClient import getServingModels
from helper.ResponseHelper import ok
from model.HealthModel import Health, ServiceInfo
from model.ResultModel import Result
from server.ValidateRequest import requireApiKey

try:
    from vllm import __version__ as _VLLM_VERSION
except Exception:  # pragma: no cover
    _VLLM_VERSION = None

router = APIRouter()


@router.get("/health", response_model=Health)
async def health() -> Health:
    """Liveness: process is up and serving HTTP."""
    return Health(status="ok", ready=isReady())


@router.get("/health/ready")
async def ready():
    """Readiness: engine built and able to serve. 503 until ready."""
    if isReady():
        return Health(status="ok", ready=True)
    return JSONResponse(
        status_code=503,
        content=Health(status="starting", ready=False).model_dump(),
    )


@router.get("/v1/models", dependencies=[Depends(requireApiKey)])
async def list_models():
    """OpenAI-compatible model listing, served by vLLM's handler."""
    return await getServingModels().show_available_models()


@router.get("/info", response_model=Result[ServiceInfo], dependencies=[Depends(requireApiKey)])
async def info() -> Result[ServiceInfo]:
    return ok(
        ServiceInfo(
            model=settings.MODEL,
            served_model_name=settings.SERVED_MODEL_NAME,
            dtype=settings.DTYPE,
            max_model_len=settings.MAX_MODEL_LEN,
            tensor_parallel_size=settings.TENSOR_PARALLEL_SIZE,
            quantization=settings.QUANTIZATION,
            auth_enabled=bool(settings.API_KEYS),
            vllm_version=_VLLM_VERSION,
        ),
        message="Service info.",
    )
