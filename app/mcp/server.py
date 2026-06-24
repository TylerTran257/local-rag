"""FastMCP server exposing the retrieval service as agent-callable tools.

Every tool reuses :class:`McpService`, which shares the REST API's use cases,
profile store, and auth. The server process authenticates with the API key in
the ``LOCAL_RAG_API_KEY`` environment variable: each consuming agent runs its
MCP connection with its own key, so every tool call is scoped to that grant.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from app.mcp.service import McpService

logger = logging.getLogger(__name__)

API_KEY_ENV = "LOCAL_RAG_API_KEY"

_service: McpService | None = None


def get_service() -> McpService:
    global _service
    if _service is None:
        from app.settings import settings

        _service = McpService.from_config(api_keys_file=settings.api_keys_file)
    return _service


def set_service(service: McpService) -> None:
    """Inject a service (used by tests)."""
    global _service
    _service = service


def _principal():
    return get_service().resolve_principal(os.environ.get(API_KEY_ENV))


mcp = FastMCP("local-rag")


@mcp.tool()
def retrieve(
    query: str,
    service_name: str,
    tenant_id: str,
    collections: list[str],
    filters: dict[str, str] | None = None,
    limit: int = 5,
    mode: str = "hybrid",
) -> dict[str, Any]:
    """Retrieve scoped chunks for a query (dense/lexical/hybrid)."""
    return get_service().retrieve(
        _principal(),
        query=query,
        service_name=service_name,
        tenant_id=tenant_id,
        collections=collections,
        filters=filters,
        limit=limit,
        mode=mode,
    )


@mcp.tool()
def answer(
    query: str,
    service_name: str,
    tenant_id: str,
    collections: list[str],
    filters: dict[str, str] | None = None,
    limit: int = 5,
    mode: str = "hybrid",
) -> dict[str, Any]:
    """Retrieve scoped chunks and generate a grounded answer."""
    return get_service().answer(
        _principal(),
        query=query,
        service_name=service_name,
        tenant_id=tenant_id,
        collections=collections,
        filters=filters,
        limit=limit,
        mode=mode,
    )


@mcp.tool()
def ingest_document(
    text: str,
    service_name: str,
    tenant_id: str,
    collection: str,
    source_type: str,
    source_label: str,
    domain_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ingest a document into the service's scoped corpus."""
    return get_service().ingest_document(
        _principal(),
        text=text,
        service_name=service_name,
        tenant_id=tenant_id,
        collection=collection,
        source_type=source_type,
        source_label=source_label,
        domain_metadata=domain_metadata,
    )


@mcp.tool()
def delete_collection(
    service_name: str,
    tenant_id: str,
    collections: list[str],
    filters: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Delete all chunks matching the given scope (service/tenant/collections)."""
    return get_service().delete_collection(
        _principal(),
        service_name=service_name,
        tenant_id=tenant_id,
        collections=collections,
        filters=filters,
    )


@mcp.tool()
def configure_profile(
    service_name: str,
    embedding_model: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    dense_limit: int | None = None,
    lexical_limit: int | None = None,
    fusion_rrf_k: int | None = None,
    default_mode: str | None = None,
) -> dict[str, Any]:
    """Create or update the service's config profile (chunking, embedding, defaults)."""
    overrides = {
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "dense_limit": dense_limit,
        "lexical_limit": lexical_limit,
        "fusion_rrf_k": fusion_rrf_k,
        "default_mode": default_mode,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    return get_service().configure_profile(
        _principal(), service_name=service_name, **overrides
    )


@mcp.tool()
def get_profile(service_name: str) -> dict[str, Any]:
    """Return the service's config profile (defaults if none registered)."""
    return get_service().get_profile(_principal(), service_name=service_name)


@mcp.tool()
def health() -> dict[str, Any]:
    """Liveness check."""
    return get_service().health()
