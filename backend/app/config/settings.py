from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central application configuration.

    Values are loaded from environment variables and .env.
    Environment variables override .env values.
    """

    # ============================================================
    # Application
    # ============================================================

    APP_NAME: str = "AI Voice Assistant"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    HOST: str = "127.0.0.1"

    PORT: int = Field(
        default=8000,
        gt=0,
        le=65535,
    )

    # ============================================================
    # Logging
    # ============================================================

    LOG_LEVEL: str = "INFO"

    # ============================================================
    # Database
    # ============================================================

    DATABASE_URL: str = (
        "sqlite:///./data/assistant.db"
    )

    # ============================================================
    # Ollama / LLM
    # ============================================================

    OLLAMA_BASE_URL: str = (
        "http://localhost:11434"
    )

    OLLAMA_MODEL: str = "llama3.2:3b"
    
    OLLAMA_TEMPERATURE: float = 0.2

    OLLAMA_NUM_PREDICT: int = 400

    MAX_RETRIES: int = Field(
        default=3,
        ge=0,
    )

    REQUEST_TIMEOUT: int = Field(
        default=180,
        gt=0,
    )

    # ============================================================
    # Conversation Memory
    # ============================================================

    MAX_HISTORY: int = Field(
        default=10,
        gt=0,
    )

    # ============================================================
    # Speech
    # ============================================================

    TTS_VOICE: str = (
        "en-IN-NeerjaNeural"
    )

    WHISPER_MODEL: str = "base"

    WHISPER_DEVICE: str = "cpu"

    WHISPER_COMPUTE_TYPE: str = "int8"

    # ============================================================
    # File Storage
    # ============================================================

    UPLOAD_FOLDER: str = "uploads"

    OUTPUT_FOLDER: str = "outputs"

    MAX_UPLOAD_SIZE_MB: int = Field(
        default=20,
        gt=0,
    )

    # ============================================================
    # ChromaDB
    # ============================================================

    CHROMA_DB_PATH: str = "vector_db"

    # ============================================================
    # Embedding
    # ============================================================

    EMBEDDING_MODEL: str = (
        "BAAI/bge-small-en-v1.5"
    )

    # ============================================================
    # Reranker
    # ============================================================

    RERANKER_MODEL: str = (
        "BAAI/bge-reranker-base"
    )

    RERANK_SCORE_THRESHOLD: float = 0.0

    # ============================================================
    # Text Splitting
    # ============================================================

    CHUNK_SIZE: int = Field(
        default=800,
        gt=0,
    )

    CHUNK_OVERLAP: int = Field(
        default=150,
        ge=0,
    )

    # ============================================================
    # RAG Retrieval
    # ============================================================

    RAG_SEARCH_TOP_K: int = Field(
        default=15,
        gt=0,
    )

    RAG_TOP_K: int = Field(
        default=5,
        gt=0,
    )

    RAG_MIN_RELEVANCE: int = Field(
        default=1,
        ge=0,
    )

    # ============================================================
    # API
    # ============================================================

    API_PREFIX: str = ""

    # ============================================================
    # CORS
    # ============================================================

    ALLOWED_ORIGINS: str = (
        "http://localhost:5173"
    )

    # ============================================================
    # Pydantic Settings
    # ============================================================

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


# ================================================================
# Cached Settings
# ================================================================

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached Settings instance.
    """

    return Settings()


settings = get_settings()