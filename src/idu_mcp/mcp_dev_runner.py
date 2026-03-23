import uvicorn

from src.idu_mcp.main import mcp_app, mcp_deps

if __name__ == "__main__":
    uvicorn.run(
        mcp_app, host="0.0.0.0", port=8000, workers=mcp_deps["server_deps"].workers
    )
