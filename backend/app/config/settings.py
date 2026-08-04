from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    "Application Configuration loaded from .env file."

    APP_NAME: str
    APP_VERSION: str
    ENVIRONMENT: str

    HOST: str
    PORT: int

    LOG_LEVEL: str

    OLLAMA_BASE_URL: str
    OLLAMA_MODEL: str

    MAX_HISTORY: int

    MAX_RETRIES: int
    REQUEST_TIMEOUT: int

    DATABASE_URL: str

    TTS_VOICE: str
    WHISPER_MODEL: str
    WHISPER_DEVICE: str
    WHISPER_COMPUTE_TYPE: str
    UPLOAD_FOLDER: str
    OUTPUT_FOLDER: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

@lru_cache()
def get_settings() -> Settings:
    "Returns a cached settings instance to avoid reloading the .env file multiple times."
    return Settings()

settings = get_settings()