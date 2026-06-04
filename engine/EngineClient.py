"""Owns the vLLM AsyncLLMEngine lifecycle.

The engine is a process-wide singleton built once at startup (see
server/Init.py lifespan) and shared by every request — vLLM's continuous
batching handles concurrency internally, so there is no per-request loading and
no need for asyncio.to_thread (the engine is natively async).
"""

from typing import Optional

from vllm.engine.async_llm_engine import AsyncLLMEngine

from ai.llm.VLLM import buildEngineArgs
from helper.LoggingHelper import getLogger
from model.ResultModel import Result

logger = getLogger(__name__)

# Process-wide handles. Populated by initEngine(), read via the getters below.
_engine: Optional[AsyncLLMEngine] = None
_model_config = None  # vllm.config.ModelConfig, resolved after load.


async def initEngine() -> Result:
    """Build the AsyncLLMEngine. Call once at application startup."""
    global _engine, _model_config
    try:
        if _engine is not None:
            return Result(Status=1, Message="Engine already initialised.")

        logger.info("Building vLLM AsyncLLMEngine (this loads weights)...")
        engine_args = buildEngineArgs()
        _engine = AsyncLLMEngine.from_engine_args(engine_args)

        # Cache the resolved model config (needed by the serving handlers).
        _model_config = await _engine.get_model_config()

        logger.info("vLLM engine ready.")
        return Result(Status=1, Message="Engine initialised successfully!")
    except Exception as ex:
        logger.exception("Failed to initialise vLLM engine")
        return Result(Status=0, Message=f"Error initialising engine: {ex}")


def getEngine() -> AsyncLLMEngine:
    if _engine is None:
        raise RuntimeError("Engine not initialised. Call initEngine() at startup.")
    return _engine


def getModelConfig():
    if _model_config is None:
        raise RuntimeError("Model config unavailable. Call initEngine() at startup.")
    return _model_config


def isReady() -> bool:
    return _engine is not None and _model_config is not None


async def shutdownEngine() -> None:
    """Release the engine on application shutdown."""
    global _engine, _model_config
    if _engine is not None:
        logger.info("Shutting down vLLM engine.")
        # AsyncLLMEngine has no public async close; drop references so the
        # background loop and GPU memory are released on interpreter teardown.
        try:
            shutdown = getattr(_engine, "shutdown_background_loop", None)
            if callable(shutdown):
                shutdown()
        except Exception:
            logger.exception("Error while stopping engine background loop")
    _engine = None
    _model_config = None
