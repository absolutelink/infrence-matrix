"""Logging configuration for Inference Matrix."""

import logging
import os

# Configure root logger
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create logger instance
logger = logging.getLogger("inference_matrix")

# websockets' keepalive ping timeout on a disconnecting UI client surfaces as an
# "exception in shielded future" ERROR traceback from asyncio; the disconnect is
# handled by our own reconnect logic, so silence the noise.
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
