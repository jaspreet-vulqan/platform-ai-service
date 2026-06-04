"""OpenAI-compatible serving handlers (the ONE version-sensitive module).

=============================== VERSION-SENSITIVE ===============================
This is the only place that imports from ``vllm.entrypoints.openai.*``. Those
serving-handler constructors change between vLLM releases. The wiring below
mirrors ``vllm/entrypoints/openai/api_server.py::init_app_state``.

To reduce breakage on a vLLM upgrade, every handler is built through
``_construct`` which inspects the constructor signature and passes ONLY the
keyword arguments that version accepts (extra ones are dropped, so a removed/
renamed optional kwarg won't crash startup). If startup still fails here after a
version bump, open ``init_app_state`` in the installed vLLM and reconcile the
kwargs in ``_candidate_*`` below.

Pinned target: vLLM >= 0.19 (first release with the ``gemma4`` architecture).
================================================================================
"""

import inspect
from typing import Optional

from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_models import (
    BaseModelPath,
    OpenAIServingModels,
)

import settings
from constants import RESPONSE_ROLE
from engine.EngineClient import getEngine, getModelConfig
from helper.LoggingHelper import getLogger
from model.ResultModel import Result

logger = getLogger(__name__)

_serving_chat: Optional[OpenAIServingChat] = None
_serving_completion: Optional[OpenAIServingCompletion] = None
_serving_models: Optional[OpenAIServingModels] = None


def _construct(cls, **candidate_kwargs):
    """Instantiate ``cls`` passing only the kwargs its __init__ accepts.

    Makes the wiring tolerant to vLLM versions that add/remove optional kwargs.
    If __init__ accepts **kwargs, everything is passed through.
    """
    sig = inspect.signature(cls.__init__)
    has_var_kw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if has_var_kw:
        accepted = candidate_kwargs
    else:
        allowed = set(sig.parameters) - {"self"}
        accepted = {k: v for k, v in candidate_kwargs.items() if k in allowed}
        dropped = set(candidate_kwargs) - set(accepted)
        if dropped:
            logger.warning(
                "%s: dropping kwargs not accepted by this vLLM version: %s",
                cls.__name__,
                sorted(dropped),
            )
    return cls(**accepted)


async def initServing() -> Result:
    """Construct the serving handlers. Call after initEngine() at startup."""
    global _serving_chat, _serving_completion, _serving_models
    try:
        engine = getEngine()
        model_config = getModelConfig()

        base_model_paths = [
            BaseModelPath(name=settings.SERVED_MODEL_NAME, model_path=settings.MODEL)
        ]

        _serving_models = _construct(
            OpenAIServingModels,
            engine_client=engine,
            model_config=model_config,
            base_model_paths=base_model_paths,
            lora_modules=None,
            prompt_adapters=None,
        )
        # Present in recent vLLM; harmless to skip on versions without it.
        init_loras = getattr(_serving_models, "init_static_loras", None)
        if callable(init_loras):
            await init_loras()

        _serving_chat = _construct(
            OpenAIServingChat,
            engine_client=engine,
            model_config=model_config,
            models=_serving_models,
            response_role=RESPONSE_ROLE,
            request_logger=None,
            chat_template=settings.CHAT_TEMPLATE,
            chat_template_content_format="auto",
            return_tokens_as_token_ids=False,
            enable_auto_tools=settings.ENABLE_AUTO_TOOL_CHOICE,
            tool_parser=settings.TOOL_CALL_PARSER,
        )

        _serving_completion = _construct(
            OpenAIServingCompletion,
            engine_client=engine,
            model_config=model_config,
            models=_serving_models,
            request_logger=None,
        )

        logger.info("OpenAI serving handlers initialised.")
        return Result(Status=1, Message="Serving handlers initialised successfully!")
    except Exception as ex:
        logger.exception("Failed to initialise serving handlers")
        return Result(Status=0, Message=f"Error initialising serving handlers: {ex}")


def getServingChat() -> OpenAIServingChat:
    if _serving_chat is None:
        raise RuntimeError("Chat handler not initialised.")
    return _serving_chat


def getServingCompletion() -> OpenAIServingCompletion:
    if _serving_completion is None:
        raise RuntimeError("Completion handler not initialised.")
    return _serving_completion


def getServingModels() -> OpenAIServingModels:
    if _serving_models is None:
        raise RuntimeError("Models handler not initialised.")
    return _serving_models
