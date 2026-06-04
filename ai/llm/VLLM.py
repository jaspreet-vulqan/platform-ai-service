"""Builds the `vllm serve` command line for the backend process.

We run vLLM's own OpenAI-compatible server as a separate process and proxy to it,
so this module just turns our settings into a CLI argv.

NOTE: this deliberately does NOT import vllm. The gateway is fully decoupled from
vLLM's Python internals — that decoupling is the whole point of this design, and
is why vLLM can be upgraded (or its modules reorganised) without touching us.
"""

import shlex
from typing import List

import settings
from helper.LoggingHelper import getLogger

logger = getLogger(__name__)


def buildServeCommand() -> List[str]:
    """Assemble the `vllm serve ...` argv from settings."""
    cmd: List[str] = [
        settings.VLLM_BIN,
        "serve",
        settings.MODEL,
        "--host",
        settings.BACKEND_HOST,
        "--port",
        str(settings.BACKEND_PORT),
        "--served-model-name",
        settings.SERVED_MODEL_NAME,
        "--dtype",
        settings.DTYPE,
        "--gpu-memory-utilization",
        str(settings.GPU_MEMORY_UTILIZATION),
        "--tensor-parallel-size",
        str(settings.TENSOR_PARALLEL_SIZE),
        "--max-num-seqs",
        str(settings.MAX_NUM_SEQS),
    ]

    if settings.MAX_MODEL_LEN:
        cmd += ["--max-model-len", str(settings.MAX_MODEL_LEN)]
    if settings.QUANTIZATION:
        cmd += ["--quantization", settings.QUANTIZATION]
    if settings.LOAD_FORMAT:
        cmd += ["--load-format", settings.LOAD_FORMAT]
    if settings.ENFORCE_EAGER:
        cmd += ["--enforce-eager"]
    if settings.TRUST_REMOTE_CODE:
        cmd += ["--trust-remote-code"]
    if settings.CHAT_TEMPLATE:
        cmd += ["--chat-template", settings.CHAT_TEMPLATE]
    if settings.ENABLE_AUTO_TOOL_CHOICE:
        cmd += ["--enable-auto-tool-choice"]
    if settings.TOOL_CALL_PARSER:
        cmd += ["--tool-call-parser", settings.TOOL_CALL_PARSER]
    if settings.BACKEND_API_KEY:
        cmd += ["--api-key", settings.BACKEND_API_KEY]
    if settings.VLLM_EXTRA_ARGS:
        cmd += shlex.split(settings.VLLM_EXTRA_ARGS)

    logger.info("vLLM serve command: %s", " ".join(cmd))
    return cmd
