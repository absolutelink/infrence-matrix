#!/usr/bin/env python3
"""Agent startup script to initialize services."""

import asyncio
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

from app.core.logging import logger
from app.services.frontend_client import frontend_client


async def main():
    """Initialize and run agent services."""
    logger.info("Starting Inference Matrix Agent...")

    # Try to register with frontend
    logger.info("Registering with frontend...")
    try:
        success = await frontend_client.register()
        if success:
            logger.info("Successfully registered with frontend")
        else:
            logger.warning("Failed to register with frontend")
    except Exception as e:
        logger.error(f"Registration error: {e}")

    # Start background tasks
    try:
        await frontend_client.start_background_tasks()
        logger.info("Started frontend client background tasks")
    except Exception as e:
        logger.error(f"Failed to start background tasks: {e}")

    # Keep the agent running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down agent...")
        await frontend_client.close()
        logger.info("Agent shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
