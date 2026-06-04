"""vLLM backend supervisor + shared HTTP client.

Replaces the old in-process engine. Responsibilities:
  - optionally launch and supervise `vllm serve` as a child process,
  - hold one shared httpx.AsyncClient pointed at the backend,
  - expose readiness so the gateway can gate /health/ready.

No vllm import here either — we talk to the backend purely over HTTP.
"""

import asyncio
import atexit
import time
from typing import Optional

import httpx

import settings
from ai.llm.VLLM import buildServeCommand
from helper.LoggingHelper import getLogger
from model.ResultModel import Result

logger = getLogger(__name__)

_process: Optional[asyncio.subprocess.Process] = None
_client: Optional[httpx.AsyncClient] = None
_ready = False


def getClient() -> httpx.AsyncClient:
    if _client is None:
        raise RuntimeError("Backend client not initialised.")
    return _client


def backendHeaders() -> dict:
    """Auth headers for gateway -> backend calls (if the backend requires a key)."""
    if settings.BACKEND_API_KEY:
        return {"Authorization": f"Bearer {settings.BACKEND_API_KEY}"}
    return {}


def isReady() -> bool:
    return _ready


async def ping() -> bool:
    """Live readiness check against the backend's /health."""
    if _client is None:
        return False
    try:
        resp = await _client.get("/health", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


def _terminateAtExit() -> None:
    # Backstop so a crashed gateway doesn't orphan the vLLM child.
    if _process is not None and _process.returncode is None:
        try:
            _process.terminate()
        except Exception:
            pass


async def _launchProcess() -> None:
    global _process
    cmd = buildServeCommand()
    logger.info("Launching vLLM backend process...")
    # Inherit stdout/stderr so vLLM's logs flow to the console / journald.
    _process = await asyncio.create_subprocess_exec(*cmd)
    atexit.register(_terminateAtExit)


async def _waitReady(timeout: float) -> bool:
    global _ready
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        # If we manage the process and it died, stop waiting.
        if _process is not None and _process.returncode is not None:
            logger.error("vLLM backend exited early (code %s).", _process.returncode)
            return False
        if await ping():
            _ready = True
            return True
        await asyncio.sleep(3.0)
    return False


async def startBackend() -> Result:
    """Start (if managed) and wait for the backend. Call from the app lifespan."""
    global _client
    _client = httpx.AsyncClient(
        base_url=settings.LLM_BACKEND_URL,
        # No read timeout: generation/streaming can run long. Connect/write bounded.
        timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=None),
    )

    if settings.MANAGE_BACKEND:
        await _launchProcess()
        logger.info("Waiting up to %ss for the model to load...", settings.BACKEND_STARTUP_TIMEOUT)
    else:
        logger.info("MANAGE_BACKEND=false; expecting an external backend at %s", settings.LLM_BACKEND_URL)

    ok = await _waitReady(settings.BACKEND_STARTUP_TIMEOUT)
    if ok:
        return Result(Status=1, Message="Backend ready.")
    if settings.MANAGE_BACKEND:
        return Result(Status=0, Message="vLLM backend did not become ready in time.")
    logger.warning("Backend not reachable yet; /health/ready will report not-ready until it is.")
    return Result(Status=1, Message="Gateway up; backend not ready yet.")


async def stopBackend() -> None:
    """Stop the backend (if managed) and close the HTTP client."""
    global _ready
    _ready = False

    if _process is not None and _process.returncode is None:
        logger.info("Stopping vLLM backend...")
        try:
            _process.terminate()
            await asyncio.wait_for(_process.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("Backend did not stop gracefully; killing.")
            _process.kill()
        except Exception:
            logger.exception("Error stopping backend process")

    if _client is not None:
        await _client.aclose()
