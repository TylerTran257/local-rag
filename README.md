# Local RAG

Local-first FastAPI service for uploading documents, indexing them into lexical and vector stores, and querying them through retrieval and optional answer generation.

## Why This Exists

This repository is a small, inspectable RAG system for experimenting with the full document-to-answer loop on local infrastructure. It is useful if you want to evaluate how upload, chunking, embedding, retrieval, and generation fit together without starting from a larger platform.

## What It Does

- Accepts `.txt` and `.pdf` uploads.
- Extracts text, chunks documents, and stores chunk metadata in SQLite.
- Indexes embeddings in on-disk Qdrant.
- Supports dense search and hybrid search.
- Exposes `/ask` for answer generation with citations.
- Exposes `/ws/chat` and `/chat` for a small streaming chat UI.
- Supports background indexing jobs with polling via `/upload_async` and `/jobs/{job_id}`.
- Includes a golden-eval runner for retrieval and optional answer checks.

## Quick Start

Assumption: the commands below use the direct Python workflow in this repo because there is no higher-level task runner such as a `Makefile` or project script wrapper.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the project:

```bash
uvicorn asgi:app --reload
```

Run tests:

```bash
pytest
```

Notes:

- The app starts at `http://127.0.0.1:8000`.
- Swagger UI is at `http://127.0.0.1:8000/docs`.
- `/ask` and `/ws/chat` require a local OpenAI-compatible generation endpoint. By default the app expects `http://127.0.0.1:8080/v1` from `.env.example`.

## Example Usage

Upload a document, then run retrieval against it:

```bash
curl -F "file=@tests/fixtures/python_rag_intro.txt" \
  http://127.0.0.1:8000/upload_v2

curl -X POST http://127.0.0.1:8000/semantic-search \
  -H "Content-Type: application/json" \
  -d '{"query":"what does chunk overlap help with in rag","limit":3}'
```

If you also have a local OpenAI-compatible model server running, ask a question over the indexed corpus:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"what does chunk overlap help with in rag","limit":3}'
```

## Architecture

High-level flow:

1. `POST /upload_v1`, `POST /upload_v2`, or `POST /upload_async` stores the file and creates a document record.
2. `TextExtractor` reads `.txt` or `.pdf` content.
3. `DocumentService` chunks the text with `langchain-text-splitters`.
4. `EmbeddingService` creates embeddings with `sentence-transformers`.
5. `VectorStoreService` writes vectors to on-disk Qdrant.
6. `LexicalSearchService` indexes chunk text into a SQLite FTS table.
7. Search routes build a retrieval request and send it through the Retrieval Core (`RetrieveUseCase` plus a gateway adapter).
8. `/ask` and `/ws/chat` pass retrieved chunks to `GenerationService`, which calls an external OpenAI-compatible endpoint.

Storage and runtime data:

- `app.db`: SQLite metadata, chunks, jobs, and FTS tables.
- `uploads/`: persisted source files.
- `qdrant_data/`: local vector index.

## Project Structure

- `app/`: FastAPI routes, document pipeline services, retrieval core, eval runner, and chat template.
- `tests/`: API, retrieval, WebSocket, and eval tests.
- `docs/`: ADRs and project workflow docs.
- `.scratch/`: local issue-tracker and roadmap-style implementation notes.
- `asgi.py`: ASGI entry point.
- `.env.example`: runtime configuration defaults.

## Development

Common commands:

```bash
uvicorn asgi:app --reload
pytest
python -m app.evals.golden_eval
python -m app.evals.golden_eval --with-answer-eval
docker compose up --build
```

Useful local URLs:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/chat`
