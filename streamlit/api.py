"""HTTP client for the FastAPI backend."""

from __future__ import annotations

import os
from io import BytesIO
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
API_TIMEOUT_SECONDS = float(os.getenv("API_TIMEOUT_SECONDS", "300"))
API_ACCESS_TOKEN = os.getenv("API_ACCESS_TOKEN", "").strip()


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if API_ACCESS_TOKEN:
        headers["Authorization"] = f"Bearer {API_ACCESS_TOKEN}"
    return headers


def _unwrap_api_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Unwrap the standard FastAPI ApiResponse envelope."""
    if payload.get("success") is False:
        raise RuntimeError(payload.get("message", "The backend request failed."))
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("The backend returned an unexpected response format.")
    return data


def send_chat(session_id: str, message: str) -> dict[str, Any]:
    response = requests.post(
        f"{API_URL}/chat",
        json={"session_id": session_id, "message": message},
        headers=_headers(),
        timeout=API_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return _unwrap_api_response(response.json())


def send_voice(session_id: str, audio_bytes: bytes) -> dict[str, Any]:
    if not audio_bytes:
        raise ValueError("No audio data was recorded.")

    response = requests.post(
        f"{API_URL}/speech/chat",
        params={"session_id": session_id},
        files={"file": ("voice.wav", BytesIO(audio_bytes), "audio/wav")},
        headers=_headers(),
        timeout=API_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    result = _unwrap_api_response(response.json())

    audio_path = str(result.get("audio_file", "")).lstrip("/")
    if not audio_path or ".." in audio_path.split("/"):
        raise RuntimeError("The backend returned an invalid audio path.")

    audio_response = requests.get(
        f"{API_URL}/{audio_path}",
        headers=_headers(),
        timeout=API_TIMEOUT_SECONDS,
    )
    audio_response.raise_for_status()
    result["audio_bytes"] = audio_response.content
    return result


def delete_conversation(session_id: str) -> dict[str, Any]:
    response = requests.delete(
        f"{API_URL}/conversations/{session_id}",
        headers=_headers(),
        timeout=API_TIMEOUT_SECONDS,
    )
    if response.status_code == 404:
        return {"deleted": False}
    response.raise_for_status()
    return _unwrap_api_response(response.json())