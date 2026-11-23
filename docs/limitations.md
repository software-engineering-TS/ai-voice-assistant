# Current limitations

This document records boundaries that should be understood before treating the application as production-ready.

1. **No multi-user identity system.** The optional bearer token is shared across clients. It does not associate conversation/document ownership with individual users.
2. **Local-first deployment.** The default bind address is loopback. A public deployment needs HTTPS, network restrictions, rate limiting, proper identity/authorization, operational monitoring and deployment review.
3. **Edge-TTS is online.** Ollama and Faster-Whisper can run locally, but Edge-TTS requires network access. The full workflow is not completely offline.
4. **PDF ingestion is available in Streamlit.** The sidebar uploads PDFs through `/rag/extract`, with a 20 MB limit. Other supported document formats are currently available through the backend API rather than the PDF-only UI.
5. **Several tool modules were placeholders.** Calculator, file reader, search, weather and wake-word modules were empty and have been removed from the source rather than presented as implemented features.
6. **Runtime retention needs an operational policy.** Uploaded files are cleaned during normal processing, but output-audio retention, vector-store retention, database backups and log retention require an explicit operator policy.
7. **Model downloads and resource requirements.** Whisper, embedding and reranker models may download on first use and can require significant disk space, RAM and CPU/GPU resources.
8. **No end-to-end hosted-model test in CI.** Unit tests mock model and external service calls. Live Ollama, Edge-TTS and microphone testing must be performed separately on a configured machine.
9. **No license yet.** The repository does not currently grant external reuse rights.