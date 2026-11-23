from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    """
    Request model for chat messages.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    session_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$",
        description="Unique conversation session identifier (letters, numbers, hyphens and underscores).",
        examples=["tarun"],
    )

    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User message sent to the AI assistant.",
        examples=["Explain my experience."],
    )

    documents: list[str] | None = Field(
        default=None,
        description=(
            "Optional list of document names to use "
            "for document-grounded responses."
        ),
        examples=[
            ["resume.pdf"]
        ],
    )


class ChatResponse(BaseModel):
    """
    Response model for chat messages.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    response: str = Field(
        ...,
        description="Generated assistant response.",
    )

    sources: list[str] = Field(
        default_factory=list,
        description="Documents used to generate the response.",
    )

    processing_time: float | None = Field(
        default=None,
        description="Response generation time in seconds.",
    )