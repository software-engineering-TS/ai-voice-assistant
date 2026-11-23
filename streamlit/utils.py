"""Compatibility imports for older Streamlit page modules."""

from api import delete_conversation, send_chat, send_voice

send_message = send_chat

__all__ = ["delete_conversation", "send_chat", "send_message", "send_voice"]