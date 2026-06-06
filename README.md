# Document Upload RAG Prototype

Small FastAPI project for learning a local-first RAG pipeline with:

- document upload for `.txt` and `.pdf`
- text extraction
- chunking with `langchain-text-splitters`
- embeddings with `sentence-transformers`
- local vector storage with Qdrant-on-disk
- dense, lexical, and hybrid retrieval
- answer generation through a local OpenAI-compatible `llama-server`
- a small mobile-first chat UI backed by WebSocket streaming
- background indexing jobs with polling

## Requirements

- Python 3.11+
- virtual environment recommended
- a local OpenAI-compatible generation endpoint for `/ask` and `WS /ws/chat`

## Install

```bash
pip install -r requirements.txt
```

## Run Locally

Start the app with Uvicorn:

```bash
uvicorn asgi:app --reload --host 0.0.0.0 --port 8000
```

App URLs:

- App: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`
- Chat UI: `http://127.0.0.1:8000/chat`

## Local Model

`POST /ask` and `WS /ws/chat` expect a local OpenAI-compatible model endpoint.

Current host setup:

```bash
~/llama.cpp/build/bin/llama-server -m Qwen_Qwen3-14B-Q4_K_M.gguf --host 0.0.0.0 --port 8080
```

Current generation target:

- Base URL: `http://127.0.0.1:8080/v1`
- Endpoint: `/chat/completions`
- Combined target: `http://127.0.0.1:8080/v1/chat/completions`

## Runtime Data

The app stores local runtime data in:

- `app.db`
- `uploads/`
- `qdrant_data/`

If you want a fresh local reset, those are the main paths to clear.

## Current Flow

1. Upload a document
2. Extract text
3. Chunk the text
4. Index chunks into lexical and vector stores
5. Search across the indexed corpus
6. Ask a question over HTTP or WebSocket
7. Optionally index in the background with job polling

There are three upload flows:

- `POST /upload_v1`
  - upload only
- `POST /upload_v2`
  - upload, extract, chunk, and embed in one request
- `POST /upload_async`
  - upload, create a background indexing job, and return immediately

## API Endpoints

- `GET /health`
- `GET /chat`
  - serve the chat UI
- `POST /upload_v1`
  - upload a document only
- `POST /upload_v2`
  - upload and run the full indexing pipeline
- `POST /upload_async`
  - upload and enqueue the indexing pipeline as a background job
- `GET /{document_id}`
  - fetch persisted document metadata and chunks
- `POST /{document_id}/extract`
- `POST /{document_id}/chunk`
- `POST /{document_id}/search`
  - document-scoped keyword search
- `POST /{document_id}/embed`
  - embed and index one document into Qdrant
- `POST /semantic-search`
  - corpus-wide dense retrieval
- `POST /hybrid-search`
  - corpus-wide hybrid retrieval using dense + lexical fusion
- `POST /ask`
  - retrieve relevant chunks and generate one non-streaming answer from the indexed corpus
- `GET /jobs/{job_id}`
  - fetch the current status of a background indexing job
- `WS /ws/chat`
  - retrieve relevant chunks and stream answer tokens to the chat UI

## Retrieval Modes

The repo currently has three retrieval styles:

- document-scoped keyword search via `POST /{document_id}/search`
- dense corpus retrieval via `POST /semantic-search`
- hybrid corpus retrieval via `POST /hybrid-search`

Hybrid retrieval combines:

- dense vector retrieval from Qdrant
- lexical retrieval from SQLite FTS5
- reciprocal rank fusion (RRF)

## Background Job Flow

`POST /upload_async` stores the uploaded document, creates a job with status `queued`, and schedules the indexing pipeline in a FastAPI background task.

Initial response example:

```json
{
  "job_id": "job-123",
  "job_type": "document_index",
  "document_id": "doc-123",
  "status": "queued",
  "error_message": null,
  "created_at": "2026-05-23T12:00:00",
  "started_at": null,
  "finished_at": null
}
```

Poll the job with:

```text
GET /jobs/{job_id}
```

Current job statuses:

- `queued`
- `running`
- `completed`
- `failed`

## Ask And Chat Responses

`POST /ask` returns:

- `answer`
- `match_count`
- `sources`
- `citations`

Current citations are model-attributed inline citations backed by the retrieved source list.

The browser chat page at `GET /chat` opens a WebSocket connection to `WS /ws/chat`.

Client message example:

```json
{"query":"what is retrieval augmented generation","limit":3}
```

Server event examples:

```json
{"type":"status","message":"retrieving context"}
{"type":"status","message":"generating answer"}
{"type":"token","value":"Retrieval "}
{"type":"done","answer":"Retrieval augmented generation...[1]","sources":[...],"citations":[...]}
{"type":"error","message":"Generation service is unavailable"}
```

`POST /ask` is still useful for non-streaming request/response testing in Swagger.

## Example Manual Workflow

### One-Step Upload And Indexing

```bash
curl -X POST "http://127.0.0.1:8000/upload_v2" \
  -F "file=@tests/fixtures/python_rag_intro.txt;type=text/plain"
```

### Step-By-Step Upload Flow

```bash
curl -X POST "http://127.0.0.1:8000/upload_v1" \
  -F "file=@tests/fixtures/python_rag_intro.txt;type=text/plain"
```

Then use the returned `document_id`:

```bash
curl -X POST "http://127.0.0.1:8000/<document_id>/extract"
curl -X POST "http://127.0.0.1:8000/<document_id>/chunk"
curl -X POST "http://127.0.0.1:8000/<document_id>/embed"
```

### Dense Search

```bash
curl -X POST "http://127.0.0.1:8000/semantic-search" \
  -H "Content-Type: application/json" \
  -d '{"query":"what is retrieval augmented generation","limit":3}'
```

### Hybrid Search

```bash
curl -X POST "http://127.0.0.1:8000/hybrid-search" \
  -H "Content-Type: application/json" \
  -d '{"query":"what is retrieval augmented generation","limit":3}'
```

### Ask A Question Over HTTP

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query":"what is retrieval augmented generation","limit":3}'
```

### Upload And Index In The Background

```bash
curl -X POST "http://127.0.0.1:8000/upload_async" \
  -F "file=@tests/fixtures/python_rag_intro.txt;type=text/plain"
```

Then poll the returned `job_id`:

```bash
curl "http://127.0.0.1:8000/jobs/<job_id>"
```

### Open The Chat UI

Visit:

```text
http://127.0.0.1:8000/chat
```

The UI connects to `WS /ws/chat` automatically and streams the answer into the assistant bubble.

## Tests

Run the test suite with:

```bash
pytest
```

Current coverage includes:

- upload pipeline behavior
- async upload job creation and polling behavior
- `/ask` request/response behavior
- websocket happy path
- websocket no-context path
- websocket generation-error path

## Golden Eval Set

Run the Golden eval set with:

```bash
python -m app.evals.golden_eval
```

Useful options:

- `--eval-set /path/to/eval.json`
- `--example-id rag-chunk-overlap`
- `--keep-artifacts`
- `--with-answer-eval`

Default behavior:

- rebuilds an isolated eval workspace for each run
- ingests the checked-in Eval corpus through the real pipeline
- evaluates the blocking Retrieval gate against dense retrieval
- reports diagnostic `MRR`
- runs Informational answer eval only when `--with-answer-eval` is requested

## Docker

Build the image:

```bash
docker build -t document-upload-app .
```

Verify the image:

```bash
docker images document-upload-app
```

Run the app container directly:

```bash
docker run --rm \
  --name document-upload-app \
  -p 8000:8000 \
  --add-host=host.docker.internal:host-gateway \
  -e GENERATION_BASE_URL=http://host.docker.internal:8080/v1 \
  -v "$(pwd)/app.db:/app/app.db" \
  -v "$(pwd)/uploads:/app/uploads" \
  -v "$(pwd)/qdrant_data:/app/qdrant_data" \
  document-upload-app
```

Run with Docker Compose:

```bash
docker compose up --build
```

Current Docker assumptions:

- only the FastAPI application is containerized
- `llama-server` stays on the host machine
- host `llama-server` must be reachable from Docker
- for Linux host access, `host.docker.internal` is mapped through `host-gateway`

## TODOs:
7. Docker Compose (1 session)
8. RAG eval tests (1 session)
9. breakdown main.py more (ask.py, health.py...) (1/2 session)
