from dataclasses import dataclass, field


@dataclass
class AuthConfig:
    """Keycloak JWT authentication settings."""

    verify: bool
    server_url: str
    client_id: str
    verify_aud: bool = False
    valid_audiences: list[str] = field(default_factory=list)
    user_cache_ttl: int = 300
    user_cache_size: int = 10_000
    jwks_cache_ttl: int = 600
    timeout: int = 5

    def __post_init__(self):
        if self.server_url and not self.server_url.startswith("http"):
            self.server_url = "http://" + self.server_url

    @property
    def jwks_url(self) -> str:
        return f"{self.server_url}/protocol/openid-connect/certs"

    @property
    def authorization_url(self) -> str:
        return f"{self.server_url}/protocol/openid-connect/auth"

    @property
    def token_url(self) -> str:
        return f"{self.server_url}/protocol/openid-connect/token"
