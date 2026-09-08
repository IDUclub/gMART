"""
This module is aimed to describe sse errors models for handling internal errors.
"""

from pydantic import BaseModel


class SseBaseError(BaseModel):
    """
    Class describes base error entity in sse streaming endpoints.
    Attributes:
        traceback (str): Legacy field kept for response compatibility. The generic
            public stream wrapper leaves it empty; full tracebacks belong in logs.
        message (str): Natural language description of error.
    """

    traceback: str
    message: str
