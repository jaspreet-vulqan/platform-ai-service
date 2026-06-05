"""Runtime configuration, loaded from the environment (.env supported).
"""

import os

from dotenv import load_dotenv

from constants import DEFAULT_MODEL

load_dotenv()

# python-dotenv turns a bare `HF_HOME=` line into HF_HOME="" in the environment,
# and huggingface_hub then resolves its cache to a *relative* "./hub" — dumping
# tens of GB of model weights into the current working directory (the repo!).
# Drop empty values so HF falls back to its real default (~/.cache/huggingface).
for _empty_var in ("HF_HOME", "HF_TOKEN"):
    if os.environ.get(_empty_var) == "":
        del os.environ[_empty_var]


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _get_csv(name: str) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


# -------------------- Model / engine -------------------- #
# HuggingFace id or local path of the model to serve.
MODEL = os.getenv("MODEL", DEFAULT_MODEL)
# Public name advertised on /v1/models and accepted in the request "model" field.
SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", MODEL)
# "auto" | "half" | "bfloat16" | "float16" | "float32"
DTYPE = os.getenv("DTYPE", "auto")
# Fraction of GPU memory vLLM may use for weights + KV cache (0-1).
GPU_MEMORY_UTILIZATION = _get_float("GPU_MEMORY_UTILIZATION", 0.90)
# Max context length; None lets vLLM use the model's default.
MAX_MODEL_LEN = _get_int("MAX_MODEL_LEN", 0) or None
# GPUs to shard the model across (single node).
TENSOR_PARALLEL_SIZE = _get_int("TENSOR_PARALLEL_SIZE", 1)
# Upper bound on concurrently batched sequences.
MAX_NUM_SEQS = _get_int("MAX_NUM_SEQS", 256)
# e.g. "awq" | "gptq" | "fp8" | "bitsandbytes"; empty means none.
QUANTIZATION = os.getenv("QUANTIZATION") or None
# vLLM weight loader. For pre-quantized bitsandbytes (e.g. unsloth bnb-4bit)
# checkpoints this MUST be "bitsandbytes". Empty lets vLLM auto-pick.
LOAD_FORMAT = os.getenv("LOAD_FORMAT") or None
# Disable CUDA graph capture. Recommended (true) for bitsandbytes models and a
# safe choice for the first load of a large model; costs some decode throughput.
ENFORCE_EAGER = _get_bool("ENFORCE_EAGER", False)
# Needed for some custom-code models (e.g. certain Qwen/Phi variants).
TRUST_REMOTE_CODE = _get_bool("TRUST_REMOTE_CODE", False)
# Optional explicit chat template path/string; empty uses the model's own.
CHAT_TEMPLATE = os.getenv("CHAT_TEMPLATE") or None

# Tool / function calling (OpenAI-compatible).
ENABLE_AUTO_TOOL_CHOICE = _get_bool("ENABLE_AUTO_TOOL_CHOICE", False)
TOOL_CALL_PARSER = os.getenv("TOOL_CALL_PARSER") or None

# -------------------- Sampling defaults -------------------- #
# Applied only when the caller omits the corresponding field.
DEFAULT_MAX_TOKENS = _get_int("DEFAULT_MAX_TOKENS", 1024)
DEFAULT_TEMPERATURE = _get_float("DEFAULT_TEMPERATURE", 0.7)

# -------------------- HTTP server -------------------- #
HOST = os.getenv("HOST", "0.0.0.0")
PORT = _get_int("PORT", 8000)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# -------------------- LLM backend (vllm serve) -------------------- #
# The gateway proxies to vLLM's own OpenAI server, run as a separate process.
# Host/port the vLLM backend listens on (kept on localhost).
BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = _get_int("BACKEND_PORT", 8001)
# Full base URL of the backend; defaults to the host/port above.
LLM_BACKEND_URL = os.getenv("LLM_BACKEND_URL") or f"http://{BACKEND_HOST}:{BACKEND_PORT}"
# If true, this process launches and supervises `vllm serve`. Set false to run
# vLLM as its own service (e.g. a separate systemd unit) and only proxy to it.
MANAGE_BACKEND = _get_bool("MANAGE_BACKEND", True)
# Optional API key the backend itself requires (gateway -> backend auth).
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY") or None
# Max seconds to wait for the backend to finish loading the model.
BACKEND_STARTUP_TIMEOUT = _get_int("BACKEND_STARTUP_TIMEOUT", 1800)
# Per-request timeout to the backend for non-streaming calls (seconds).
REQUEST_TIMEOUT = _get_float("REQUEST_TIMEOUT", 600.0)
# vLLM launcher executable (on PATH inside the venv).
VLLM_BIN = os.getenv("VLLM_BIN", "vllm")
# Extra raw flags appended verbatim to `vllm serve` (space-separated). Escape
# hatch for any vLLM flag we don't model, e.g.
#   "--limit-mm-per-prompt image=0,video=0,audio=0"  (skip multimodal memory)
VLLM_EXTRA_ARGS = os.getenv("VLLM_EXTRA_ARGS", "")

# -------------------- Auth / CORS -------------------- #
# Comma-separated API keys accepted via "Authorization: Bearer <key>" or
# "X-API-Key: <key>". Empty list disables auth (dev only).
API_KEYS = set(_get_csv("API_KEYS"))
# Comma-separated allowed CORS origins. Default "*" with credentials OFF.
CORS_ALLOW_ORIGINS = _get_csv("CORS_ALLOW_ORIGINS") or ["*"]
CORS_ALLOW_CREDENTIALS = _get_bool("CORS_ALLOW_CREDENTIALS", False)

# -------------------- HuggingFace -------------------- #
# Token for gated models; vLLM/transformers read HF_TOKEN from the env directly,
# we just surface it here for /info diagnostics.
HF_TOKEN = os.getenv("HF_TOKEN")
HF_HOME = os.getenv("HF_HOME")
