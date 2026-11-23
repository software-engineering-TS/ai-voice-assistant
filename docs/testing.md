# Testing guide

## Install backend test dependencies

From the repository root:

```bash
python -m pip install -r backend/requirements-dev.txt
```

## Run the test suite

```bash
python -m pytest -q
```

The root `pytest.ini` adds `backend` to the Python import path, discovers tests under `backend/tests` and enables pytest-asyncio automatic mode.

## What is covered

The existing suite covers assistant orchestration, chat/conversation APIs, SQLAlchemy repository behavior, document loaders, document ingestion, embeddings, vector storage, retrieval, reranking, middleware, speech processing and prompt construction. Additional regression tests cover session ID validation, audio upload size limits, safe temporary filenames and unique generated audio names.

## External integrations

The automated suite should not make live calls to Ollama, Edge-TTS or Hugging Face model downloads. Those integrations require separate local validation:

1. Start Ollama and pull the configured model.
2. Start the backend and inspect `/health`.
3. Submit a text request through Streamlit or `/chat`.
4. Record a short voice sample and verify transcription, response generation and audio playback.
5. Upload a non-sensitive sample document through `/rag/extract` and verify that the document appears in `/documents` and can be used in a grounded query.

## Continuous integration

`.github/workflows/tests.yml` runs the backend tests on Python 3.12 for pushes to `main` and `hardening/**` branches and for pull requests targeting `main`.