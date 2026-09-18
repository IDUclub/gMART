"""Canonical Urban API base URL used behind the /api load-balancer route."""

from urllib.parse import urlsplit, urlunsplit


def normalize_urban_api_url(base_url: str) -> str:
    """Accept an HTTP origin or API root and keep exactly one trailing /api."""
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
    path = url.path.rstrip("/")
    while path.endswith("/api"):
        path = path[:-4].rstrip("/")
    return urlunsplit((url.scheme, url.netloc, f"{path}/api", "", ""))
