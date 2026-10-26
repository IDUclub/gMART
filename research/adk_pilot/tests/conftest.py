"""Environment for importing ``src.agents`` without a live stack.

Same defaults as ``tests/conftest.py``; real env vars win.
"""

from __future__ import annotations

import os
import tempfile

os.environ.setdefault("OLLAMA_API_URL", "http://localhost:11434")
os.environ.setdefault("IDU_MCP_SERVER", "http://localhost:8000/mcp")
os.environ.setdefault("OBJECTS_EFFECTS_MCP_SERVER", "http://localhost:8080/mcp")
os.environ.setdefault("DVD_MCP_SERVER", "http://localhost:8000/mcp")
os.environ.setdefault("URBAN_MCP_SERVER", "https://urban-mcp.testing.idulab.ru")
os.environ.setdefault("CHAT_STORAGE", "http://localhost:8010")
os.environ.setdefault("URBAN_API_URL", "http://localhost/api")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
os.environ.setdefault("SERVICE_AUTH_SERVER_URL", "http://localhost:9999")
os.environ.setdefault("SERVICE_AUTH_REALM", "test")
os.environ.setdefault("SERVICE_AUTH_CLIENT_ID", "test-client")
os.environ.setdefault("SERVICE_AUTH_CLIENT_SECRET", "test-secret")
os.environ.setdefault("GMART_CACHE_DIR", tempfile.mkdtemp(prefix="gmart-tests-"))
