from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    "Request model for chat messages."

    session_id: str = Field(
        ...,
        min_length=1,
        description="Unique session identifier",
    )
    
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The message to be sent to the AI voice assistant.",
    )

class ChatResponse(BaseModel):
    "Response model for chat messages."

    response: str 