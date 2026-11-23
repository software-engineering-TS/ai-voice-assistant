"""Small, dependency-light validation helpers for public API inputs."""

import re

SESSION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$"
_SESSION_ID_RE = re.compile(SESSION_ID_PATTERN)


def validate_session_id(value: str) -> str:
    """Validate and return a session ID safe for use as an identifier.

    Session IDs are not used as filesystem paths; this validation also
    rejects path separators and control characters at API boundaries.
    """
    if not isinstance(value, str):
        raise ValueError("Session ID must be a string.")

    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("Session ID cannot contain control characters.")

    normalized = value.strip()
    if not normalized or not _SESSION_ID_RE.fullmatch(normalized):
        raise ValueError(
            "Session ID must be 1-100 characters and contain only "
            "letters, numbers, hyphens or underscores."
        )
    return normalized