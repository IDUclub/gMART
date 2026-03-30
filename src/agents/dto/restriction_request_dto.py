from pydantic import Field

from .llm_request_dto import SimpleRequestDTO


class RestrictionRequestDTO(SimpleRequestDTO):
    """
    Restriction Request DTO to LLM.
    Attributes:
        model (str): Model name.
        request (str): Request text.
        scenario_id (int): Scenario ID from Urban API.
    """

    scenario_id: int = Field(examples=[772], description="Scenario ID from Urban API")
