"""Urban API roots for direct connections and load-balancer mounts."""

from urllib.parse import urlsplit, urlunsplit


def normalize_urban_api_url(base_url: str) -> str:
    """Preserve explicit API roots; use /api only for an origin without a path."""
    url = urlsplit(base_url.strip())
    if (
        url.scheme not in {"http", "https"}
        or not url.netloc
        or url.query
        or url.fragment
    ):
        raise ValueError(
            "Urban API URL must be an HTTP(S) base URL without query or fragment"
        )
    path = url.path.rstrip("/") or "/api"
    return urlunsplit((url.scheme, url.netloc, path, "", ""))
