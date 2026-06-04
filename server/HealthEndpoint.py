"""Health and info endpoints.

/health* are unauthenticated so liveness/readiness probes work. /info returns the
Result envelope. (/v1/models and /metrics are served by the proxy router.)
"""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

import settings
from engine.Backend import ping
from helper.ResponseHelper import ok
from model.HealthModel import Health, ServiceInfo
from model.ResultModel import Result
from server.ValidateRequest import requireApiKey

router = APIRouter()


@router.get("/health", response_model=Health)
async def health() -> Health:
    """Liveness: the gateway process is up and serving HTTP."""
    return Health(status="ok", ready=True)


@router.get("/health/ready")
async def ready():
    """Readiness: the vLLM backend is reachable and has loaded the model."""
    if await ping():
        return Health(status="ok", ready=True)
    return JSONResponse(
        status_code=503,
        content=Health(status="starting", ready=False).model_dump(),
    )


@router.get(
    "/info", response_model=Result[ServiceInfo], dependencies=[Depends(requireApiKey)]
)
async def info() -> Result[ServiceInfo]:
    return ok(
        ServiceInfo(
            model=settings.MODEL,
            served_model_name=settings.SERVED_MODEL_NAME,
            dtype=settings.DTYPE,
            max_model_len=settings.MAX_MODEL_LEN,
            tensor_parallel_size=settings.TENSOR_PARALLEL_SIZE,
            quantization=settings.QUANTIZATION,
            load_format=settings.LOAD_FORMAT,
            auth_enabled=bool(settings.API_KEYS),
            backend_url=settings.LLM_BACKEND_URL,
            backend_ready=await ping(),
        ),
        message="Service info.",
    )
