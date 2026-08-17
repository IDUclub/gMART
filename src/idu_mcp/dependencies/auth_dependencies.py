import base64
import json

from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_headers


def extract_token():

    headers = get_http_headers(include_all=True)
    auth_header = headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise ValueError("Unauthorized: Bearer token is missing")
    return auth_header.removeprefix("Bearer ").strip()


def extract_workspace_owner() -> str:
    """Use the stable JWT subject as the workspace ownership namespace.

    Authentication is still enforced by the mounted FastMCP verifier. This helper only
    extracts the already authenticated identity so refreshed tokens keep access to the
    same chat artifacts.
    """

    token = extract_token()
    try:
        encoded = token.split(".", 2)[1]
        payload = json.loads(
            base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        )
        subject = payload.get("sub")
    except (IndexError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ToolError("JWT не содержит доступный идентификатор пользователя") from exc
    if not isinstance(subject, str) or not subject:
        raise ToolError("JWT не содержит subject для изоляции workspace")
    return subject
