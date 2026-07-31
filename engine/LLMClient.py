"""HTTP chat helper for internal features (e.g. schema inference).

Calls the vLLM backend's OpenAI-compatible /v1/chat/completions over HTTP via the
shared client. This is the single place the schema-inference feature talks to the
LLM; when that feature becomes its own service, only LLM_BACKEND_URL changes.
"""

from typing import List, Optional

import settings
from engine.Backend import backendHeaders, getClient
from helper.LoggingHelper import getLogger
from model.ResultModel import Result

logger = getLogger(__name__)


async def chatComplete(
    messages: List[dict],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    response_format: Optional[dict] = None,
) -> Result:
    """Run a non-streaming chat completion. Result.Data is the assistant text.

    Pass ``response_format`` (e.g. an OpenAI-style ``{"type": "json_schema",
    ...}``) to enable vLLM structured outputs and constrain the model to valid,
    schema-matching JSON. Omitted for free-form completions.
    """
    try:
        payload = {
            "model": settings.SERVED_MODEL_NAME,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or settings.DEFAULT_MAX_TOKENS,
            "stream": False,
        }
        if response_format:
            payload["response_format"] = response_format
        resp = await getClient().post(
            "/v1/chat/completions",
            json=payload,
            headers=backendHeaders(),
            timeout=settings.REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            return Result(
                Status=0,
                Message=f"LLM backend error {resp.status_code}: {resp.text[:500]}",
            )

        data = resp.json()
        content = data["choices"][0]["message"].get("content") or ""
        return Result(Data=content, Status=1, Message="Completion generated.")
    except Exception as ex:
        logger.exception("chatComplete failed")
        return Result(Status=0, Message=f"Error calling LLM backend: {ex}")
