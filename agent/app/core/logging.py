import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("inference_matrix_agent")

# Silence asyncio's "exception in shielded future" ERROR tracebacks emitted when
# a websockets keepalive ping times out during a UI disconnect; the agent's own
# reconnect loop handles that case.
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
