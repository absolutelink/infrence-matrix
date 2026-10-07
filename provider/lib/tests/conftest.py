"""Test-suite defaults for provider_lib tests.

Phase 16 made ``MACHINE_SECRET`` and ``AGENT_ID`` required on
``ProviderSettings``. A few tests construct a bare ``ProviderSettings()``
(relying on the environment); these defaults let the whole suite run with no
externally-exported env. Explicit kwargs in individual tests still take
precedence over these.
"""

import os

os.environ.setdefault("MACHINE_UID", "lib-test")
os.environ.setdefault("MACHINE_SECRET", "lib-test-secret")
os.environ.setdefault("AGENT_ID", "lib-test-agent")
os.environ.setdefault("ADMIN_BASE_URL", "http://localhost:9999")
