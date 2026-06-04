"""In-process chat helper for internal features (e.g. schema inference).

Calls the already-loaded model through the same vLLM serving handler that backs
/v1/chat/completions - no HTTP hop, no self-auth. This is the single place the
schema-inference feature talks to the LLM; when that feature moves to its own
service, only this function changes (swap for an OpenAI HTTP client pointed at
the vLLM service's base_url).

vLLM-internal imports stay inside engine/ by design.
"""

from typing import List, Optional

from fastapi import Request
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ErrorResponse

import settings
from engine.ServingClient import getServingChat
from helper.LoggingHelper import getLogger
from model.ResultModel import Result

logger = getLogger(__name__)


async def chatComplete(
    messages: List[dict],
    raw_request: Request,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> Result:
    """Run a non-streaming chat completion. Result.Data is the assistant text."""
    try:
        request = ChatCompletionRequest(
            model=settings.SERVED_MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens or settings.DEFAULT_MAX_TOKENS,
            stream=False,
        )

        response = await getServingChat().create_chat_completion(request, raw_request)

        if isinstance(response, ErrorResponse):
            message = getattr(response, "message", str(response))
            return Result(Status=0, Message=f"LLM error: {message}")

        content = response.choices[0].message.content or ""
        return Result(Data=content, Status=1, Message="Completion generated.")
    except Exception as ex:
        logger.exception("chatComplete failed")
        return Result(Status=0, Message=f"Error calling LLM: {ex}")
