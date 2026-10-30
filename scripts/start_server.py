import os
import sys
import uvicorn

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("sage.server")


def main():
    """Start SAGE FastAPI Uvicorn Server."""
    host = settings.host
    port = settings.port
    log_level = settings.log_level.lower()

    logger.info("Starting SAGE Inference Server at http://%s:%d ...", host, port)
    logger.info("Swagger Documentation: http://%s:%d/docs", host, port)

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        log_level=log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()
