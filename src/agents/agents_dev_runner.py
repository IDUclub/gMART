"""
Module for local development app debugging
"""

import uvicorn

from src.agents.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=80)
