# Context

Local-RAG is a **shared Retrieval Service**: one runtime that many consuming
services and agents call to ingest documents and retrieve scoped chunks (and
optionally generate grounded answers). It is exposed over REST and MCP.

## Domain vocabulary

- **Service (`service_name`)** — a consumer of the retrieval service. The unit
  of ownership: a service owns its corpus, its scope grant, and its profile.
- **Tenant (`tenant_id`)** — an isolation boundary within a service.
- **Collection** — a named bucket of documents within a service's corpus.
- **Scope** — the `(service_name, tenant_id, collections, filters)` tuple that
  bounds every ingest and retrieve operation.
- **Principal / Grant** — an authenticated API key and the scope it is allowed
  to act within (allowed services/tenants/collections, plus an admin flag).
- **Scope policy** — the retrieval-core component that validates a scope.
  `PassthroughScopePolicy` checks structure only; `NamespacePolicy` enforces
  allowed services/collections (fail-closed).
- **Profile (`ServiceProfile`)** — a service's ingestion + retrieval
  configuration: embedding model, chunk size/overlap, retrieval limits, default
  mode. Defaults mirror platform settings when no profile is registered.
- **Collection-per-model** — each embedding model maps to its own Qdrant
  collection (dimensions cannot share a collection). The default model maps to
  the default collection.
- **Gateway** — `MetadataAwareRetrievalGateway`, which executes scope-filtered
  dense/lexical/hybrid retrieval against the backends.
- **Trace id** — per-request correlation id, returned in the `X-Trace-Id`
  header and in every error envelope.
- **Grounded answer** — an answer generated only from retrieved in-scope
  chunks. When retrieval returns no chunks, the service returns a fixed
  *no-grounded-answer* response (the same fallback text across REST and MCP,
  sync and streaming) and does not call generation.

## Invariants

- Every ingest/retrieve/answer/profile call is authenticated and scope-checked:
  the request scope must be a subset of the key's grant.
- Ingest and retrieve resolve the **same** profile for a `service_name`, so a
  document is always read back with the model + collection it was written with.
- A profile's embedding model is immutable once the profile exists.
- The MCP tool surface and REST API share the same use cases, profile store,
  and auth — they cannot diverge in behavior.

## Boundaries

- Vectors: Qdrant (on-disk local by default, remote via `QDRANT_URL`).
- Metadata + lexical (FTS5): SQLite via `DATABASE_URL`.
- Answer generation: an external OpenAI-compatible chat-completions endpoint.

See `docs/adr/` for the decisions behind the shared-service design.
