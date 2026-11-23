# Architecture

## Overview

AI Voice Assistant is a local-first application with a Streamlit presentation layer and a FastAPI service layer. The backend coordinates conversation persistence, Ollama inference, speech processing and document-grounded retrieval.

## Components

- **Streamlit**: local browser UI for text and voice interaction.
- **FastAPI**: HTTP routes for chat, streaming chat, speech, documents, RAG, conversations and health.
- **Assistant service**: orchestrates conversation history, RAG retrieval, prompt construction, Ollama requests and persistence.
- **Ollama client**: asynchronous HTTP client for a locally hosted language model.
- **SQLAlchemy / SQLite**: stores conversation records by session identifier.
- **Document ingestion**: validates, saves and extracts supported document content, builds chunks and metadata, generates embeddings and indexes them.
- **Sentence Transformers**: embedding and cross-encoder reranking models, loaded lazily on first use.
- **ChromaDB**: persistent vector store for document chunks and metadata.
- **Faster-Whisper**: local speech-to-text processing, loaded lazily when the first audio request is handled.
- **Edge-TTS**: text-to-speech generation. It requires network access and means the complete application is not fully offline.

## Request flows

### Text conversation

1. Streamlit posts a session ID and message to `/chat`.
2. FastAPI validates the request model and session identifier.
3. The assistant service loads conversation context and retrieves document context when relevant.
4. The prompt builder creates the model input.
5. Ollama returns the assistant response.
6. The conversation repository persists user and assistant messages.
7. FastAPI returns the standard `ApiResponse` envelope.

### Voice conversation

1. Streamlit sends a WAV upload and session ID to `/speech/chat`.
2. The backend validates the session ID, extension and upload size.
3. Audio is written to a random temporary filename.
4. Faster-Whisper transcribes the audio.
5. The assistant service generates and persists a response.
6. Edge-TTS generates an MP3 under a random output filename.
7. The temporary input audio is removed, including on error paths.
8. Streamlit fetches the generated audio through the backend using the same optional bearer token and plays it.

### Document-grounded retrieval

1. `/rag/extract` accepts a supported document.
2. A format-specific loader extracts text.
3. The text splitter creates chunks and metadata builder attaches source metadata.
4. Sentence Transformers generate vectors.
5. ChromaDB persists chunks, vectors and metadata.
6. At query time, retrieval searches matching chunks and the cross-encoder reranks candidates before prompt construction.

## Startup and model loading

Database and vector-store initialization are performed during FastAPI lifespan startup. Embedding, reranker and Whisper models are lazy: importing the API does not immediately download or initialize them. They load on first actual use (or when the health service explicitly checks them).

## Data locations

The default relative directories are `data/`, `uploads/`, `outputs/` and `vector_db/`, resolved from the backend working directory. These paths are runtime data and must not be committed to source control.

## Security boundary

The default host is loopback. An optional shared bearer token can protect API routes and generated audio, but there is no multi-user identity or per-user data authorization layer. See [`security.md`](security.md) before any network exposure.