"""Entry point (mirrors the sample's main.py).

Unlike the sample, the model is NOT loaded here — it's loaded by the FastAPI
lifespan (server/Init.py) so startup, readiness, and shutdown are managed in one
place. This file just configures logging and hands off to the server runner.
"""

from helper.LoggingHelper import configureLogging, getLogger
from RESTClient import runServer

logger = getLogger(__name__)


if __name__ == "__main__":
    configureLogging()
    logger.info("Launching platform-ai-service...")
    runServer()
