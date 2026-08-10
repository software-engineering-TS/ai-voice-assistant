from fastapi import APIRouter

from app.api.routes.chat import router as chat_router
from app.api.routes.conversation import (
    router as conversation_router,
)
from app.api.routes.document import (
    router as document_router,
)
from app.api.routes.health import (
    router as health_router,
)
from app.api.routes.rag import (
    router as rag_router,
)
from app.api.routes.speech import (
    router as speech_router,
)


api_router = APIRouter()


# ============================================================
# Health
# ============================================================

api_router.include_router(
    health_router
)


# ============================================================
# Chat
# ============================================================

api_router.include_router(
    chat_router
)


# ============================================================
# Speech
# ============================================================

api_router.include_router(
    speech_router
)


# ============================================================
# RAG
# ============================================================

api_router.include_router(
    rag_router
)


# ============================================================
# Documents
# ============================================================

api_router.include_router(
    document_router
)


# ============================================================
# Conversations
# ============================================================

api_router.include_router(
    conversation_router
)