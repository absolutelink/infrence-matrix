"""Logging configuration for Inference Matrix."""

import logging

from app.core.config import settings

# Configure root logger
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create logger instance
logger = logging.getLogger("inference_matrix")
