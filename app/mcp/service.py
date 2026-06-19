"""Transport-agnostic MCP tool logic.

These methods are the single implementation behind the MCP tools. They reuse
the same use cases, profile store, and auth as the REST API, so the MCP surface
and the HTTP surface can never drift. Keeping them free of any MCP/transport
types makes them directly unit-testable.
"""
from __future__ import annotations

from typing import Any

from app.auth import ApiKeyRegistry, Principal, enforce_scope
from app.auth.errors import AuthenticationError
from app.composition import MetadataAwareRuntime, build_metadata_aware_runtime
from app.ingest.contracts import IngestDocument
from app.profiles import ServiceProfile
from app.retrieval import RetrievalScope, RetrieveRequest
from app.retrieval.types import RetrievalMode
from app.services.generation_service import GenerationService

_CORE_CHUNK_METADATA_KEYS = frozenset(
    {
        "service_name",
        "tenant_id",
        "collection",
        "source_type",
        "source_label",
        "document_id",
        "original_filename",
        "chunk_index",
    }
)


def _chunk_to_dict(chunk: Any) -> dict[str, Any]:
    metadata = chunk.metadata
    return {
        "text": chunk.content,
        "score": chunk.score,
        "chunk_id": chunk.chunk_id,
        "source_label": metadata.get("source_label", "unknown"),
        "collection": metadata.get("collection", "unknown"),
        "service_name": metadata.get("service_name", "unknown"),
        "tenant_id": metadata.get("tenant_id", "unknown"),
        "domain_metadata": {
            k: v for k, v in metadata.items() if k not in _CORE_CHUNK_METADATA_KEYS
        },
    }


def _merge_profile(existing: ServiceProfile, **overrides: Any) -> ServiceProfile:
    mode = overrides.get("default_mode")
    return ServiceProfile(
        service_name=existing.service_name,
        embedding_model=overrides.get("embedding_model") or existing.embedding_model,
        chunk_size=overrides.get("chunk_size") or existing.chunk_size,
        chunk_overlap=overrides.get("chunk_overlap")
        if overrides.get("chunk_overlap") is not None
        else existing.chunk_overlap,
        dense_limit=overrides.get("dense_limit") or existing.dense_limit,
        lexical_limit=overrides.get("lexical_limit") or existing.lexical_limit,
        fusion_rrf_k=overrides.get("fusion_rrf_k") or existing.fusion_rrf_k,
        default_mode=RetrievalMode(mode) if mode else existing.default_mode,
        generation_overrides=overrides.get("generation_overrides")
        or existing.generation_overrides,
    )


class McpService:
    def __init__(
        self,
        runtime: MetadataAwareRuntime,
        generation_service: GenerationService,
        registry: ApiKeyRegistry,
    ) -> None:
        self._runtime = runtime
        self._generation = generation_service
        self._registry = registry

    @classmethod
    def from_config(cls, *, api_keys_file=None) -> "McpService":
        runtime = build_metadata_aware_runtime()
        if runtime.profile_store is not None:
            runtime.profile_store.seed_from_file(None)
        return cls(
            runtime=runtime,
            generation_service=GenerationService(),
            registry=ApiKeyRegistry.from_config(api_keys_file=api_keys_file),
        )

    # -- auth -------------------------------------------------------------
    def resolve_principal(self, api_key: str | None) -> Principal:
        principal = self._registry.resolve(api_key)
        if principal is None:
            raise AuthenticationError("Missing or invalid API key")
        return principal

    def _require_service(self, principal: Principal, service_name: str) -> None:
        if not principal.allows_service(service_name):
            from app.auth.errors import AuthorizationError

            raise AuthorizationError(
                f"Key '{principal.key_id}' may not manage service '{service_name}'",
                details={"service_name": service_name},
            )

    # -- tools ------------------------------------------------------------
    def retrieve(
        self,
        principal: Principal,
        *,
        query: str,
        service_name: str,
        tenant_id: str,
        collections: list[str],
        filters: dict[str, str] | None = None,
        limit: int = 5,
        mode: str = "hybrid",
    ) -> dict[str, Any]:
        enforce_scope(
            principal,
            service_name=service_name,
            tenant_id=tenant_id,
            collections=collections,
        )
        scope = RetrievalScope(
            service_name=service_name,
            tenant_id=tenant_id,
            collections=collections,
            filters=filters or {},
        )
        result = self._runtime.retrieve_use_case.execute(
            RetrieveRequest(
                query=query,
                retrieval_mode=RetrievalMode(mode),
                limit=limit,
                scope=scope,
            )
        )
        return {
            "chunks": [_chunk_to_dict(c) for c in result.chunks],
            "trace_id": result.trace_id or "unknown",
        }

    def answer(
        self,
        principal: Principal,
        *,
        query: str,
        service_name: str,
        tenant_id: str,
        collections: list[str],
        filters: dict[str, str] | None = None,
        limit: int = 5,
        mode: str = "hybrid",
    ) -> dict[str, Any]:
        retrieved = self.retrieve(
            principal,
            query=query,
            service_name=service_name,
            tenant_id=tenant_id,
            collections=collections,
            filters=filters,
            limit=limit,
            mode=mode,
        )
        if not retrieved["chunks"]:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "trace_id": retrieved["trace_id"],
            }
        sources = [
            {
                "document_id": c.get("chunk_id", "unknown"),
                "original_filename": c["source_label"],
                "chunk_index": 0,
                "score": c["score"],
                "text": c["text"],
            }
            for c in retrieved["chunks"]
        ]
        answer_text = self._generation.answer_question(query, sources)
        return {
            "answer": answer_text,
            "sources": retrieved["chunks"],
            "trace_id": retrieved["trace_id"],
        }

    def ingest_document(
        self,
        principal: Principal,
        *,
        text: str,
        service_name: str,
        tenant_id: str,
        collection: str,
        source_type: str,
        source_label: str,
        domain_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        enforce_scope(
            principal,
            service_name=service_name,
            tenant_id=tenant_id,
            collections=[collection],
        )
        result = self._runtime.ingest_use_case.ingest_document(
            IngestDocument(
                text=text,
                service_name=service_name,
                tenant_id=tenant_id,
                collection=collection,
                source_type=source_type,
                source_label=source_label,
                domain_metadata=domain_metadata or {},
            )
        )
        return {"chunk_count": result.chunk_count}

    def configure_profile(
        self, principal: Principal, *, service_name: str, **overrides: Any
    ) -> dict[str, Any]:
        self._require_service(principal, service_name)
        store = self._runtime.profile_store
        if store is None:
            raise RuntimeError("Profile store is not configured")
        merged = _merge_profile(store.get(service_name), **overrides)
        return store.upsert(merged).to_dict()

    def get_profile(self, principal: Principal, *, service_name: str) -> dict[str, Any]:
        self._require_service(principal, service_name)
        store = self._runtime.profile_store
        if store is None:
            raise RuntimeError("Profile store is not configured")
        return store.get(service_name).to_dict()

    def health(self) -> dict[str, Any]:
        return {"status": "ok"}
