from .base_exceptions import AgentsNotFound


class ModelNotFound(AgentsNotFound):
    """
    Exception for ollama model not found.
    Attributes:
        status_code (int): HTTP status code.
        base_message (str): Base message for error.
        message (str): Exception message.
        error_input (str): Exception message.
        model (str): Model name requested by user.
        available_models (list): List of available models.
    """

    base_message: str = "model not found in Ollama"

    def __init__(self, model: str, available_models: list[str]):
        """
        Initialization function for ModelNotFound class.
        Args:
            model (str): Model name requested by user.
            available_models (list): List of available models.
        """

        self.model = model
        self.available_models = available_models
        self.message = f"{self.model} {self.base_message}"
        super().__init__(
            self.message,
            {"requested_model": self.model, "available_models": self.available_models},
        )
