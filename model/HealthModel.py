from typing import Optional

from pydantic import BaseModel


class Health(BaseModel):
    """Liveness/readiness payload."""

    status: str  # "ok" | "starting" | "error"
    ready: bool


class ServiceInfo(BaseModel):
    """Human-friendly summary of what this instance is serving (for /info)."""

    model: str
    served_model_name: str
    dtype: str
    max_model_len: Optional[int] = None
    tensor_parallel_size: int
    quantization: Optional[str] = None
    auth_enabled: bool
    vllm_version: Optional[str] = None
