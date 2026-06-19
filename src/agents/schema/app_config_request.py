from pydantic import BaseModel


class AppConfigRequest(BaseModel):
    """
    Request body for retrieving the agents service runtime configuration.
    """

    password: str
