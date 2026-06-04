"""Server runner (mirrors the sample's RESTClient.runServer)."""

import uvicorn

import settings
from server.Init import app


def runServer() -> None:
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        workers=1,
        log_level=settings.LOG_LEVEL.lower(),
    )
