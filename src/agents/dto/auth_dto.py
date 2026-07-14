from pydantic import BaseModel, Field


class LoginRequestDTO(BaseModel):
    """
    Login request DTO for the auth helper token proxy.
    Attributes:
        username (str): Keycloak username (IDU realm).
        password (str): Keycloak password.
    """

    username: str = Field(
        examples=["user@example.com"],
        description="Keycloak username of the IDU realm",
    )
    password: str = Field(
        examples=["password"],
        description="Keycloak password",
    )
