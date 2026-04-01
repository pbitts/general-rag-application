import os
import logging
from config import settings

logger = logging.getLogger(__name__)


def configure_langsmith():
    if not settings.LANGSMITH_TRACING:
        logger.info("LangSmith tracing disabled")
        return

    if not settings.LANGSMITH_API_KEY:
        logger.warning("LANGSMITH_TRACING enabled but no API key provided")
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY

    if settings.LANGSMITH_PROJECT:
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT

    logger.info("LangSmith tracing enabled")
