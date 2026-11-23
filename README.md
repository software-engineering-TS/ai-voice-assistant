# AI Voice Assistant

A local-first conversational assistant with a FastAPI backend and Streamlit interface. It combines Ollama-based chat, persistent conversation history, document-grounded retrieval (RAG), speech-to-text and text-to-speech.

> **Deployment boundary:** This repository is designed for local development by default. The API has no per-user identity system. Keep it bound to loopback for personal use. Before exposing it to another machine, configure `API_ACCESS_TOKEN` and put the service behind HTTPS. The optional bearer token is a shared access gate, not multi-user authentication or authorization.

## Features

- Text conversations through FastAPI and Streamlit.
- Local LLM inference through Ollama.
- Conversation persistence through SQLAlchemy and SQLite by default.
- RAG document ingestion with supported PDF, DOCX, PPTX, TXT, CSV and Excel loaders.
- Persistent ChromaDB vector storage, sentence-transformer embeddings and cross-encoder reranking.
- Voice input using Faster-Whisper and generated speech using Edge-TTS.
- Conversation history, session management and health checks.
- Optional bearer-token protection for API and generated audio routes.

## Architecture

```text
Streamlit UI
  ├── Text chat ──────────────┐
  └── Voice recording ────────┤
                              ▼
                         FastAPI API
                              │
              ┌───────────────┼─────────────────┐
              ▼               ▼                 ▼
        Assistant / LLM    Speech pipeline    Document / RAG
              │          Whisper → Ollama      │
              │             → Edge-TTS         ▼
              ▼                         Loaders → Splitter
        SQLite / SQLAlchemy                     │
                                               ▼
                                    Embeddings → ChromaDB
                                               │
                                               ▼
                                         Retrieval / Reranker
```

More detail: [`docs/architecture.md`](docs/architecture.md).

## Requirements

- Python 3.12 is the recommended development version.
- Ollama installed locally with a model available (default: `llama3.2:3b`).
- Internet access is needed for first-time model downloads and Edge-TTS speech generation. After model downloads, Ollama and Faster-Whisper inference can run locally; the overall application is **not fully offline** because Edge-TTS is an online service.

## Quick start — backend

From the repository root:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install all backend runtime and test dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r backend/requirements-dev.txt
```

Configure the backend:

```powershell
# Windows PowerShell
Copy-Item backend/.env.example backend/.env
```

```bash
# macOS / Linux
cp backend/.env.example backend/.env
```

Start Ollama and download the configured model:

```bash
ollama pull llama3.2:3b
ollama serve
```

Start FastAPI from the backend directory:

```bash
cd backend
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API documentation is available at `http://127.0.0.1:8000/docs`.

## Quick start — Streamlit frontend

In a second terminal, from the repository root:

```bash
python -m pip install -r streamlit/requirements.txt
```

Configure and run:

```powershell
# Windows PowerShell
Copy-Item streamlit/.env.example streamlit/.env
cd streamlit
python -m streamlit run app.py
```

```bash
# macOS / Linux
cp streamlit/.env.example streamlit/.env
cd streamlit
python -m streamlit run app.py
```

The UI opens at `http://localhost:8501`. Set `API_URL` in `streamlit/.env` if the backend uses a different local address.

## Run tests

From the repository root:

```bash
python -m pytest -q
```

The tests use fakes and mocks for model inference; running the suite should not download Whisper, embedding or reranker models. CI runs the backend suite on Python 3.12 using GitHub Actions.

## Configuration

Backend settings are read from `backend/.env` and can be overridden by environment variables. Common settings include:

| Variable | Default | Purpose |
|---|---|---|
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Backend bind address |
| `DATABASE_URL` | `sqlite:///./data/assistant.db` | SQLAlchemy database |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama API |
| `OLLAMA_MODEL` | `llama3.2:3b` | Chat model |
| `WHISPER_MODEL` | `base` | Faster-Whisper model size |
| `MAX_AUDIO_UPLOAD_SIZE_MB` | `20` | Audio upload limit |
| `MAX_UPLOAD_SIZE_MB` | `20` | Document upload limit |
| `UPLOAD_FOLDER` / `OUTPUT_FOLDER` | `uploads` / `outputs` | Runtime file locations |
| `CHROMA_DB_PATH` | `vector_db` | Persistent vector-store directory |
| `ALLOWED_ORIGINS` | localhost Streamlit origins | Comma-separated CORS origins |
| `API_ACCESS_TOKEN` | empty | Optional shared bearer token |

The Streamlit client reads `API_URL`, `API_ACCESS_TOKEN` and `API_TIMEOUT_SECONDS` from `streamlit/.env`.

## Security notes

- Default backend binding is loopback (`127.0.0.1`); do not change it to `0.0.0.0` on an untrusted network without the deployment controls described in [`docs/security.md`](docs/security.md).
- CORS is a browser policy, not authentication.
- `API_ACCESS_TOKEN` protects API routes and generated audio when configured. It is a single shared token; it does not provide per-user identity, roles, rate limiting or audit attribution.
- Use HTTPS at a reverse proxy for any non-local access. Do not send bearer tokens over plain HTTP on untrusted networks.
- Uploaded documents/audio and generated audio are runtime data. Apply appropriate retention, backup and access policies before using personal or sensitive material.

## Project layout

```text
backend/
  app/                 FastAPI application, services, RAG and speech modules
  tests/unit/          Backend unit and integration-style tests
  requirements.txt     Backend runtime dependencies
  requirements-dev.txt Backend runtime + test dependencies
streamlit/
  app.py               Streamlit entry point
  api.py               Backend HTTP client
  requirements.txt     Frontend dependencies
docs/                  Architecture, testing, limitations and security guides
.github/workflows/     Continuous integration
```

## Current limitations

See [`docs/limitations.md`](docs/limitations.md). In particular, remote multi-user deployment requires a real identity/authorization layer; document upload is available in the backend API but its Streamlit control is not yet wired; Edge-TTS requires network access; and model/vector data requires explicit lifecycle and retention management.

## License

No license has been added yet. Reuse and redistribution rights are not explicitly granted until a license is selected.