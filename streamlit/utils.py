import requests

API_URL = "http://127.0.0.1:8000"


def send_message(session_id: str, message: str):

    response = requests.post(
        f"{API_URL}/chat",
        json={
            "session_id": session_id,
            "message": message,
        },
    )

    response.raise_for_status()

    return response.json()


def send_voice(session_id: str, audio_file):

    files = {
        "file": (
            audio_file.name,
            audio_file,
            "audio/wav",
        )
    }

    response = requests.post(
        f"{API_URL}/speech/chat",
        params={
            "session_id": session_id,
        },
        files=files,
    )

    response.raise_for_status()

    return response.json()