# Local RAG

Local-first FastAPI service for document ingestion, metadata-scoped retrieval, and optional answer generation on local infrastructure.

## Current Stage

This repo is no longer a broad RAG demo app. The current app is a narrowed HTTP service centered on six public endpoints:

- `GET /health`
- `POST /documents/upload`
- `POST /documents/ingest`
- `POST /retrieve`
- `POST /answer`
- `POST /answer/stream`

What is implemented now:

- Metadata-aware ingestion and retrieval are the default runtime path.
- Manual uploads are a thin adapter over the same ingest pipeline used by service-to-service ingestion.
- Retrieval is scoped by `service_name`, `tenant_id`, `collections`, and optional metadata filters.
- Full-answer and streaming-answer endpoints both reuse the same retrieval flow.
- Local persistence uses SQLite for metadata/lexical indexing and on-disk Qdrant for vectors.

Important limitation:

- The default runtime uses `PassthroughScopePolicy`, which validates request structure but is not a full authorization policy. A stricter `NamespacePolicy` exists in code, but it is not the default wiring.

## Architecture

Request flow:

1. Ingest text from uploaded files or direct JSON requests.
2. Validate required metadata.
3. Chunk text.
4. Generate embeddings with `sentence-transformers`.
5. Store vectors in local Qdrant.
6. Store lexical search records in SQLite FTS.
7. Retrieve through the metadata-aware retrieval gateway.
8. Optionally generate an answer through an OpenAI-compatible chat-completions endpoint.

Local runtime data:

- `app.db`: SQLite metadata and lexical index data
- `qdrant_data/`: local vector store

## Quick Start

Install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the app:

```bash
uvicorn asgi:app --reload
```

Run tests:

```bash
venv/bin/pytest tests/ -x -q
```

Local URLs:

- App: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

Answer generation notes:

- `/answer` and `/answer/stream` require an OpenAI-compatible endpoint.
- Default base URL: `http://127.0.0.1:8080/v1`
- Default endpoint path: `/chat/completions`

## Environment

Main runtime settings come from `.env`.

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./app.db` | SQLite metadata database |
| `UPLOAD_DIR` | `uploads` | Upload storage directory |
| `MAX_FILE_SIZE` | `2097152` | File-size setting in config |
| `QDRANT_PATH` | `./qdrant_data` | Local Qdrant storage path |
| `QDRANT_COLLECTION_NAME` | `document_chunks` | Qdrant collection name |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `GENERATION_BASE_URL` | `http://127.0.0.1:8080/v1` | OpenAI-compatible base URL |
| `GENERATION_ENDPOINT` | `/chat/completions` | Chat completions endpoint |
| `GENERATION_TIMEOUT` | `600` | Generation timeout in seconds |
| `GENERATION_TEMPERATURE` | `0.2` | Generation temperature |
| `GENERATION_MAX_OUTPUT_TOKENS` | `300` | Max answer tokens |
| `GENERATION_MAX_CONTEXT_CHARS` | `6000` | Total context char budget |
| `GENERATION_MAX_CHARS_PER_CHUNK` | `1800` | Per-source char cap |
| `DENSE_RETRIEVAL_LIMIT` | `15` | Dense backend candidate limit |
| `LEXICAL_RETRIEVAL_LIMIT` | `15` | Lexical backend candidate limit |
| `FUSION_RRF_K` | `60` | Hybrid fusion constant |

## API Reference

### `GET /health`

Purpose: simple liveness check.

Response:

```json
{
  "status": "ok"
}
```

### `POST /documents/upload`

Purpose: manual upload for local use. This route extracts text from a file, then ingests it with default metadata.

Content type: `multipart/form-data`

Parameters:

| Field | Type | Required | Notes |
|---|---|---|---|
| `file` | file | yes | Supported file extensions: `.txt`, `.pdf` |

Default metadata applied internally:

| Field | Value |
|---|---|
| `service_name` | `manual` |
| `tenant_id` | `local` |
| `collection` | `general` |
| `source_type` | `uploaded_file` |
| `source_label` | original filename |
| `domain_metadata` | `{}` |

Success response:

```json
{
  "success": true,
  "chunk_count": 3,
  "source_label": "test.txt"
}
```

Error behavior:

- `422` if the upload has no filename
- `422` if the file type is unsupported
- `422` if the extracted document has no ingestible text
- `422` if metadata validation fails internally

Example:

```bash
curl -X POST http://127.0.0.1:8000/documents/upload \
  -F "file=@tests/fixtures/python_rag_intro.txt"
```

### `POST /documents/ingest`

Purpose: service-to-service ingestion of a whole text document with explicit metadata.

Content type: `application/json`

Request body:

| Field | Type | Required | Notes |
|---|---|---|---|
| `text` | string | yes | Must be non-empty |
| `service_name` | string | yes | Retrieval scope owner |
| `tenant_id` | string | yes | Retrieval tenant scope |
| `collection` | string | yes | Logical collection name |
| `source_type` | string | yes | Caller-defined source type such as `pdf`, `text`, `api`, `kb_article` |
| `source_label` | string | yes | Human-readable source identifier |
| `domain_metadata` | object | no | Extra metadata stored with the indexed content |

Example request:

```json
{
  "text": "The API service provides REST endpoints for data access.",
  "service_name": "api-service",
  "tenant_id": "tenant-456",
  "collection": "documentation",
  "source_type": "pdf",
  "source_label": "api-docs.pdf",
  "domain_metadata": {
    "author": "docs-team",
    "topic": "rest"
  }
}
```

Success response:

```json
{
  "chunk_count": 3
}
```

Error behavior:

- `422` for missing required fields
- `422` for empty or whitespace-only text
- `422` for invalid metadata

Example:

```bash
curl -X POST http://127.0.0.1:8000/documents/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The API service provides REST endpoints for data access.",
    "service_name": "api-service",
    "tenant_id": "tenant-456",
    "collection": "documentation",
    "source_type": "pdf",
    "source_label": "api-docs.pdf"
  }'
```

### `POST /retrieve`

Purpose: retrieve scoped chunks without generation.

Content type: `application/json`

Request body:

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `query` | string | yes | - | Must be non-empty |
| `service_name` | string | yes | - | Required scope field |
| `tenant_id` | string | yes | - | Required scope field |
| `collections` | string[] | yes | - | Must contain at least one non-empty value |
| `filters` | object | no | `{}` | Public schema is `dict[str, str]` |
| `limit` | integer | no | `5` | Must be `1` to `50` |
| `mode` | string | no | `hybrid` | One of `dense`, `lexical`, `hybrid` |

Filter rules:

- `filters` cannot contain `service_name`, `tenant_id`, `collection`, or `collections`.
- Those keys are reserved for scope enforcement.
- Additional filter keys are matched against stored metadata.

Example request:

```json
{
  "query": "how do I use the API",
  "service_name": "api-service",
  "tenant_id": "tenant-456",
  "collections": ["documentation"],
  "filters": {
    "topic": "rest"
  },
  "limit": 5,
  "mode": "hybrid"
}
```

Success response:

```json
{
  "chunks": [
    {
      "text": "Service-specific content about APIs.",
      "score": 0.92,
      "source_label": "api-docs.pdf",
      "collection": "documentation",
      "service_name": "api-service",
      "tenant_id": "tenant-456",
      "chunk_id": "chunk-service-1",
      "domain_metadata": {
        "topic": "rest",
        "is_external": false
      }
    }
  ],
  "trace_id": "trace-456"
}
```

Response fields:

| Field | Type | Notes |
|---|---|---|
| `chunks[].text` | string | Retrieved chunk text |
| `chunks[].score` | number | Backend/fusion relevance score |
| `chunks[].source_label` | string | Original source label |
| `chunks[].collection` | string | Chunk collection |
| `chunks[].service_name` | string | Chunk service namespace |
| `chunks[].tenant_id` | string | Chunk tenant namespace |
| `chunks[].chunk_id` | string | Stable chunk identifier from retrieval result |
| `chunks[].domain_metadata` | object | Arbitrary non-core metadata stored on the chunk |
| `trace_id` | string | Retrieval trace ID |

Error behavior:

- `422` for invalid request shape or reserved filter keys
- `422` for empty query or invalid limit
- `500` for retrieval backend/runtime failures

Example:

```bash
curl -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "how do I use the API",
    "service_name": "api-service",
    "tenant_id": "tenant-456",
    "collections": ["documentation"],
    "filters": {"topic": "rest"},
    "limit": 5,
    "mode": "hybrid"
  }'
```

### `POST /answer`

Purpose: retrieve scoped chunks, then generate one complete answer.

Content type: `application/json`

Request body: identical to `POST /retrieve`.

Success response:

```json
{
  "answer": "The API service provides REST endpoints. [1]",
  "sources": [
    {
      "text": "Service-specific content about APIs.",
      "score": 0.92,
      "source_label": "api-docs.pdf",
      "collection": "documentation",
      "service_name": "api-service",
      "tenant_id": "tenant-456",
      "chunk_id": "chunk-service-1",
      "domain_metadata": {
        "topic": "rest",
        "is_external": false
      }
    }
  ],
  "trace_id": "trace-456"
}
```

Behavior notes:

- If retrieval returns no chunks, the endpoint returns `200` with a fallback answer and an empty `sources` list.
- The generation service is instructed to answer only from provided context and use inline citations like `[1]`.

Error behavior:

- `422` for invalid retrieval request input
- `500` if answer generation fails
- `500` if retrieval fails internally

Example:

```bash
curl -X POST http://127.0.0.1:8000/answer \
  -H "Content-Type: application/json" \
  -d '{
    "query": "how do I use the API",
    "service_name": "api-service",
    "tenant_id": "tenant-456",
    "collections": ["documentation"]
  }'
```

### `POST /answer/stream`

Purpose: retrieve scoped chunks, then stream answer text over Server-Sent Events.

Content type: `application/json`

Request body: identical to `POST /retrieve`.

Response content type:

```text
text/event-stream
```

SSE event payload shape:

| Field | Type | Notes |
|---|---|---|
| `event` | string | `content`, `done`, or `error` |
| `data` | string | Token text, empty string for `done`, or error message |
| `done` | boolean | `false` for content, `true` for terminal events |

Example event stream:

```text
data: {"event":"content","data":"The API ","done":false}

data: {"event":"content","data":"service provides REST endpoints.","done":false}

data: {"event":"done","data":"","done":true}
```

Behavior notes:

- If retrieval returns no chunks, the stream emits one fallback `content` event and then a `done` event.
- Generation failures are emitted as an `error` event with `done: true`.

Example:

```bash
curl -N -X POST http://127.0.0.1:8000/answer/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "how do I use the API",
    "service_name": "api-service",
    "tenant_id": "tenant-456",
    "collections": ["documentation"]
  }'
```

## Supported File Types

- `.txt`
- `.pdf`

PDF notes:

- PDF extraction uses `pypdf`.
- If a PDF contains no extractable text, the upload route returns `422`.

## Development Notes

Current implementation centers on the metadata-aware runtime factory in `app/composition.py` and the narrowed route set registered in `app/main.py`.

Useful files:

- `app/main.py`: app creation and router wiring
- `app/composition.py`: runtime assembly
- `app/api/routes/`: public HTTP endpoints
- `app/api/schemas/schemas.py`: request/response schemas
- `app/retrieval/`: retrieval core and scope policy
- `app/ingest/`: ingest contracts and use case
- `tests/`: API and end-to-end behavior tests

## Project Structure

- `app/`: application code
- `docs/`: specs and design notes
- `tests/`: API and end-to-end tests
- `.scratch/`: local issue tracker notes
- `asgi.py`: ASGI entry point
- `.env.example`: example runtime settings
