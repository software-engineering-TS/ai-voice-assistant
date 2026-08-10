from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ApiResponse(BaseModel):
    """
    Standard successful API response.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    success: bool = Field(
        ...,
        description="Whether the operation was successful.",
    )

    message: str = Field(
        ...,
        description="Human-readable result message.",
    )

    data: Any | None = Field(
        default=None,
        description="Optional response payload.",
    )


class ErrorResponse(BaseModel):
    """
    Standard API error response.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    success: bool = Field(
        default=False,
        description="Always false for an error response.",
    )

    message: str = Field(
        ...,
        description="Human-readable error message.",
    )