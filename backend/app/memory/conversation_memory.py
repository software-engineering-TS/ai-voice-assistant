from collections import defaultdict

from app.models.conversation import Message


class ConversationMemory:

    def __init__(self):
        self.sessions = defaultdict(list)

    def get_messages(self, session_id: str):

        return [
            message.model_dump()
            for message in self.sessions[session_id]
        ]

    def add_user_message(
        self,
        session_id: str,
        message: str,
    ):

        self.sessions[session_id].append(
            Message(
                role="user",
                content=message,
            )
        )

    def add_assistant_message(
        self,
        session_id: str,
        message: str,
    ):

        self.sessions[session_id].append(
            Message(
                role="assistant",
                content=message,
            )
        )

    def clear_session(
        self,
        session_id: str,
    ):
        self.sessions[session_id].clear()


conversation_memory = ConversationMemory()