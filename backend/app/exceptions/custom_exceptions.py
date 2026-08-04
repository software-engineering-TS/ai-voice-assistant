class AssistantException(Exception):
    """
    Base exception for the AI Assistant.
    """

    pass


class LLMConnectionError(AssistantException):
    """
    Raised when the application cannot connect to the LLM.
    """

    pass


class LLMResponseError(AssistantException):
    """
    Raised when the LLM returns an invalid response.
    """

    pass