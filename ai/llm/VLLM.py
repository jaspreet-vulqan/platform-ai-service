"""vLLM engine configuration assembly.

Translates our `settings` into vLLM's `AsyncEngineArgs`. Kept separate from the
engine lifecycle (engine/EngineClient.py) so the "what to load" concern is
isolated from the "load it / hold it" concern, mirroring the sample's
ai/llm -> engine split.
"""

from vllm.engine.arg_utils import AsyncEngineArgs

import settings
from helper.LoggingHelper import getLogger

logger = getLogger(__name__)


def buildEngineArgs() -> AsyncEngineArgs:
    """Build AsyncEngineArgs from environment-driven settings."""
    kwargs = dict(
        model=settings.MODEL,
        served_model_name=settings.SERVED_MODEL_NAME,
        dtype=settings.DTYPE,
        gpu_memory_utilization=settings.GPU_MEMORY_UTILIZATION,
        tensor_parallel_size=settings.TENSOR_PARALLEL_SIZE,
        max_num_seqs=settings.MAX_NUM_SEQS,
        trust_remote_code=settings.TRUST_REMOTE_CODE,
        enforce_eager=settings.ENFORCE_EAGER,
        # Keep vLLM's stats logger enabled so the /metrics endpoint has data.
        disable_log_stats=False,
    )

    if settings.MAX_MODEL_LEN:
        kwargs["max_model_len"] = settings.MAX_MODEL_LEN
    if settings.QUANTIZATION:
        kwargs["quantization"] = settings.QUANTIZATION
    # Pre-quantized bitsandbytes checkpoints need an explicit weight loader;
    # without this the unsloth bnb-4bit weights fail to load.
    if settings.LOAD_FORMAT:
        kwargs["load_format"] = settings.LOAD_FORMAT

    logger.info(
        "Engine args: model=%s tp=%s dtype=%s gpu_util=%s max_len=%s "
        "quant=%s load_format=%s enforce_eager=%s",
        settings.MODEL,
        settings.TENSOR_PARALLEL_SIZE,
        settings.DTYPE,
        settings.GPU_MEMORY_UTILIZATION,
        settings.MAX_MODEL_LEN,
        settings.QUANTIZATION,
        settings.LOAD_FORMAT,
        settings.ENFORCE_EAGER,
    )

    return AsyncEngineArgs(**kwargs)
