from fastmcp.server.dependencies import get_http_headers


def extract_token():

    headers = get_http_headers(include_all=True)
    auth_header = headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise ValueError("Unauthorized: Bearer token is missing")
    return auth_header.removeprefix("Bearer ").strip()
