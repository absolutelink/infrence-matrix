"""Select the process manager for the configured agent platform."""

from app.core.config import settings
from app.services.gufo_server import gufo_server_manager
from app.services.halogen_flash_server import halogen_flash_server_manager
from app.services.halogen_server import halogen_server_manager
from app.services.llama_server import llama_server_manager

server_manager = (
    halogen_flash_server_manager
    if settings.AGENT_PLATFORM == "halogen-flash"
    else halogen_server_manager
    if settings.AGENT_PLATFORM == "halogen"
    else gufo_server_manager
    if settings.AGENT_PLATFORM == "gufo"
    else llama_server_manager
)
