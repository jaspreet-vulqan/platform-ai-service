"""Reverse proxy to the vLLM OpenAI-compatible backend.

Forwards /v1/* (chat/completions, completions, embeddings, models, ...) and
/metrics to the backend, streaming responses through unchanged so SSE works.
Our gateway concerns — API-key auth, CORS — are applied here; the backend stays
on localhost.
"""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from engine.Backend import backendHeaders, getClient
from server.ValidateRequest import requireApiKey

router = APIRouter()

# Hop-by-hop headers must not be forwarded (RFC 7230), plus host/content-length
# which httpx/uvicorn recompute.
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}
# Strip the caller's gateway credentials before talking to the backend; we add
# the backend's own auth (if any) via backendHeaders().
_STRIP_REQUEST = _HOP_BY_HOP | {"authorization", "x-api-key"}


async def _proxy(request: Request, upstream_path: str) -> StreamingResponse:
    client = getClient()

    fwd_headers = {
        k: v for k, v in request.headers.items() if k.lower() not in _STRIP_REQUEST
    }
    fwd_headers.update(backendHeaders())

    upstream_req = client.build_request(
        request.method,
        upstream_path,
        params=request.query_params,
        content=await request.body(),
        headers=fwd_headers,
    )
    upstream = await client.send(upstream_req, stream=True)

    resp_headers = {
        k: v for k, v in upstream.headers.items() if k.lower() not in _HOP_BY_HOP
    }
    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        headers=resp_headers,
        media_type=upstream.headers.get("content-type"),
        background=BackgroundTask(upstream.aclose),
    )


@router.api_route(
    "/v1/{path:path}", methods=["GET", "POST"], dependencies=[Depends(requireApiKey)]
)
async def proxy_v1(request: Request, path: str):
    return await _proxy(request, f"/v1/{path}")


@router.get("/metrics")
async def proxy_metrics(request: Request):
    # Unauthenticated, like the backend's own /metrics (for Prometheus scraping).
    return await _proxy(request, "/metrics")
