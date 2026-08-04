from fastapi import FastAPI

from app.api.router import api_router
from app.config.logging_config import app_logger
from app.config.settings import settings
from app.database.init_db import init_db
from app.exceptions.handlers import register_exception_handlers
from app.middleware.request_logger import log_requests
from app.middleware.request_id import request_id_middleware
# from app.speech.speech_to_text import speech_to_text_service

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production Ready AI Voice Assistant Backend",
)

app.include_router(api_router)
register_exception_handlers(app)
app.middleware("http")(log_requests)
app.middleware("http")(request_id_middleware)


@app.on_event("startup")
async def startup_event():
    init_db()

    app_logger.info("Database initialized successfully.")

    app_logger.info("Application started successfully.")

@app.get("/")
async def startup_event():
    return {"message": "Application started successfully."}

@app.get("/", tags=["Home"])
async def home():
    return {
        "message": "Welcome to AI Voice Assistant",
        "docs": "/docs",
        "health": "/health",
    }

# @app.post("/voice")
# async def voice_chat(file: UploadFile):
#     stt = get_speech_to_text_service()
#     text = stt.transcribe(audio_path)