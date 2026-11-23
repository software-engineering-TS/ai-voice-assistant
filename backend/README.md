# Backend development

The FastAPI application lives in `backend/app`.

From the repository root, create and activate a virtual environment, then install:

```bash
python -m pip install -r backend/requirements-dev.txt
```

Copy the environment template and configure Ollama:

```powershell
Copy-Item backend/.env.example backend/.env
```

Run the API from this directory:

```bash
cd backend
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Run the full backend suite from the repository root with `python -m pytest -q`.

For security and remote deployment boundaries, read [`../docs/security.md`](../docs/security.md).