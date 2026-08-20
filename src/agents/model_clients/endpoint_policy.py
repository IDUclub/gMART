from urllib.parse import urlparse

LOCAL_OLLAMA_HOSTS = frozenset(
    {"localhost", "127.0.0.1", "::1", "host.docker.internal", "ollama"}
)
FORBIDDEN_LLM_HOSTS = frozenset({"a.dgx"})


def require_local_ollama_url(url: str, variable: str) -> None:
    """Reject native Ollama endpoints outside the local host/compose network."""

    host = _hostname(url, variable)
    if host not in LOCAL_OLLAMA_HOSTS:
        allowed = ", ".join(sorted(LOCAL_OLLAMA_HOSTS))
        raise ValueError(
            f"{variable} must point to local Ollama; host {host!r} is not allowed "
            f"(allowed: {allowed})"
        )


def reject_forbidden_llm_host(url: str | None, variable: str) -> None:
    """Keep generative LLM traffic off hosts reserved for non-LLM services."""

    if not url:
        return
    host = _hostname(url, variable)
    if host in FORBIDDEN_LLM_HOSTS:
        raise ValueError(
            f"{variable} must not target {host!r} for language-model requests"
        )


def _hostname(url: str, variable: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{variable} must be an absolute http(s) URL")
    return parsed.hostname.rstrip(".").lower()
