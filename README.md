# RAG Project

OpenAI-compatible FastAPI service that performs retrieval‑augmented generation (RAG) over ingested text using Qdrant and local embeddings.

## Features
- `/admin/ingest_text` endpoint to chunk + index text
- `/v1/chat/completions` OpenAI-style chat endpoint
- Safe behavior when no relevant context is found

## Stack
- FastAPI, Qdrant, local embeddings model
- Designed to run against a local LLM gateway (e.g., Ollama)

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start Qdrant (example)
docker run -p 6333:6333 qdrant/qdrant

# Run API
uvicorn app.main:app --reload
```

## Environment
Configure via `.env`:
- `LLAMA_BASE` (default `http://localhost:11434`)
- `EMBEDDINGS_BASE`
- `EMBEDDINGS_MODEL`
- `QDRANT_URL` (default `http://localhost:6333`)
- `QDRANT_COLLECTION` (default `it_poc`)

## Notes
`data/pasted_text.txt` is a sample document used for indexing tests.
