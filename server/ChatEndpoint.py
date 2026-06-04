"""OpenAI-compatible inference endpoints.

Thin HTTP handlers that delegate all heavy lifting (chat templating, sampling,
streaming, tool calling, usage accounting) to the vLLM serving handlers built in
engine/ServingClient.py. This mirrors vLLM's own api_server handlers so the
responses are byte-for-byte OpenAI-compatible.

These return native OpenAI shapes (NOT the Result envelope) so consumers can use
the stock OpenAI SDK by pointing base_url at this service.
"""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

# OpenAI request/response schemas come straight from vLLM so they always match
# the serving handlers. (The only vLLM coupling outside engine/ — kept to the
# request models the HTTP layer must parse.)
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ErrorResponse,
)

from engine.ServingClient import getServingChat, getServingCompletion
from server.ValidateRequest import requireApiKey

router = APIRouter()


@router.post("/v1/chat/completions", dependencies=[Depends(requireApiKey)])
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    handler = getServingChat()
    generator = await handler.create_chat_completion(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)

    if request.stream:
        return StreamingResponse(content=generator, media_type="text/event-stream")

    return JSONResponse(content=generator.model_dump())


@router.post("/v1/completions", dependencies=[Depends(requireApiKey)])
async def create_completion(request: CompletionRequest, raw_request: Request):
    handler = getServingCompletion()
    generator = await handler.create_completion(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)

    if request.stream:
        return StreamingResponse(content=generator, media_type="text/event-stream")

    return JSONResponse(content=generator.model_dump())
