"""Select the process manager for the configured agent platform."""

from app.core.config import settings
from app.services.halogen_server import halogen_server_manager
from app.services.llama_server import llama_server_manager

server_manager = (
    halogen_server_manager
    if settings.AGENT_PLATFORM == "halogen"
    else llama_server_manager
)
