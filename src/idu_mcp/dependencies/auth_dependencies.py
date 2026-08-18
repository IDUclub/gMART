from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_headers


def extract_user_id() -> str:
    headers = get_http_headers(include_all=True)
    user_id = headers.get("x-user-id", "").strip()
    if not user_id:
        raise ToolError("X-User-Id header is required")
    return user_id


def extract_workspace_owner() -> str:
    """Use the stable JWT subject as the workspace ownership namespace.

    Authentication is still enforced by the mounted FastMCP verifier. This helper only
    extracts the already authenticated identity so refreshed tokens keep access to the
    same chat artifacts.
    """

    return extract_user_id()
