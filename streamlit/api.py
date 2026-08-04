import requests

API_URL = "http://127.0.0.1:8000"


def send_chat(session_id, message):

    response = requests.post(
        f"{API_URL}/chat",
        json={
            "session_id": session_id,
            "message": message,
        },
    )

    response.raise_for_status()

    return response.json()


def send_voice(session_id, audio_file):

    files = {
        "file": (
            "voice.wav",
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

    result = response.json()

    result["audio_url"] = (
        f"{API_URL}/{result['audio_file']}"
    )

    return result