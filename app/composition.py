"""Centralized metadata-aware runtime composition factory.

Constructs a fully-wired metadata-aware retrieval runtime consisting of
RetrieveUseCase and IngestUseCase with all their dependencies.

This module centralizes the wiring and makes it reusable across
create_app(), evals, and tests.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.answer.use_case import AnswerUseCase
from app.delete.use_case import DeleteUseCase
from app.ingest.use_case import IngestUseCase
from app.profiles import ProfileResolver
from app.profiles.store import ProfileStore
from app.retrieval import (
    NamespacePolicy,
    PassthroughScopePolicy,
    StructuredLoggingTraceSink,
    SystemClock,
    UuidTraceIdGenerator,
)
from app.retrieval.contracts import (
    Clock,
    RetrievalTraceSink,
    ScopePolicy,
    TraceIdGenerator,
)
from app.retrieval.metadata_gateway import MetadataAwareRetrievalGateway
from app.retrieval.use_case import RetrieveUseCase
from app.services.embedding_service import EmbeddingService
from app.services.generation_service import GenerationService
from app.services.lexical_search_service import LexicalSearchService
from app.services.vector_store_service import VectorStoreService
from app.settings import settings


def _default_scope_policy() -> ScopePolicy:
    if settings.scope_policy_mode == "namespace":
        return NamespacePolicy()
    return PassthroughScopePolicy()


@dataclass(frozen=True)
class MetadataAwareRuntime:
    """Container for a fully-wired metadata-aware retrieval runtime.

    Holds the main use-case objects that routes and higher-level code need,
    together with the gateway for introspection and testing.
    """

    retrieve_use_case: RetrieveUseCase
    ingest_use_case: IngestUseCase
    gateway: MetadataAwareRetrievalGateway
    # Always populated by ``build_metadata_aware_runtime``; defaulted so unit
    # tests can construct a runtime with mocked use cases.
    profile_store: ProfileStore | None = None
    delete_use_case: DeleteUseCase | None = None
    answer_use_case: AnswerUseCase | None = None


def build_metadata_aware_runtime(
    *,
    # Infrastructure services -- callers may supply their own instances
    # (e.g. for eval workspaces with isolated storage).  When omitted the
    # factory creates production defaults.
    embedding_service: EmbeddingService | None = None,
    vector_store_service: VectorStoreService | None = None,
    lexical_search_service: LexicalSearchService | None = None,
    profile_store: ProfileStore | None = None,
    generation_service: GenerationService | None = None,
    # Retrieval core overrides -- primarily useful for tests.
    scope_policy: ScopePolicy | None = None,
    clock: Clock | None = None,
    trace_id_generator: TraceIdGenerator | None = None,
    trace_sink: RetrievalTraceSink | None = None,
) -> MetadataAwareRuntime:
    """Build a complete metadata-aware retrieval runtime.

    All parameters are optional.  When omitted the factory creates
    production-ready defaults (real embedding model, on-disk Qdrant, etc.).
    Pass explicit instances to swap in fakes or isolated storage for tests
    and evals.

    Returns:
        A ``MetadataAwareRuntime`` dataclass with ``retrieve_use_case``,
        ``ingest_use_case``, and ``gateway``.
    """
    # --- Infrastructure services ---
    resolved_embedding = embedding_service or EmbeddingService()
    resolved_vector_store = vector_store_service or VectorStoreService()
    resolved_lexical = lexical_search_service or LexicalSearchService()
    resolved_profile_store = profile_store or ProfileStore()

    # Single resolver shared by ingest, retrieve, and delete so all paths turn a
    # service_name into the same (profile, collection) pair.
    profile_resolver = ProfileResolver(resolved_profile_store)

    # --- Retrieval gateway ---
    gateway = MetadataAwareRetrievalGateway(
        vector_store_service=resolved_vector_store,
        lexical_search_service=resolved_lexical,
        embedding_service=resolved_embedding,
        profile_resolver=profile_resolver,
    )

    # --- Retrieval use case ---
    # Scope policy: explicit override wins; otherwise pick by configured mode.
    # "namespace" adds fail-closed structural enforcement in the retrieval core
    # (defense in depth on top of the API-key scope check at the route layer).
    resolved_scope_policy = scope_policy or _default_scope_policy()
    resolved_clock = clock or SystemClock()
    resolved_trace_id_gen = trace_id_generator or UuidTraceIdGenerator()
    resolved_trace_sink = trace_sink or StructuredLoggingTraceSink()

    retrieve_use_case = RetrieveUseCase(
        gateway=gateway,
        scope_policy=resolved_scope_policy,
        clock=resolved_clock,
        trace_id_generator=resolved_trace_id_gen,
        trace_sink=resolved_trace_sink,
    )

    # --- Ingest use case ---
    ingest_use_case = IngestUseCase(
        embedding_service=resolved_embedding,
        vector_store_service=resolved_vector_store,
        lexical_search_service=resolved_lexical,
        profile_resolver=profile_resolver,
    )

    # --- Delete use case ---
    delete_use_case = DeleteUseCase(
        vector_store_service=resolved_vector_store,
        lexical_search_service=resolved_lexical,
        profile_resolver=profile_resolver,
    )

    # --- Answer use case ---
    # Generation is wired here (default OpenAI-compatible client) so the answer
    # orchestration is shared by REST and MCP, like retrieve and ingest.
    resolved_generation = generation_service or GenerationService()
    answer_use_case = AnswerUseCase(
        retrieve_use_case=retrieve_use_case,
        generation_service=resolved_generation,
    )

    return MetadataAwareRuntime(
        retrieve_use_case=retrieve_use_case,
        ingest_use_case=ingest_use_case,
        gateway=gateway,
        profile_store=resolved_profile_store,
        delete_use_case=delete_use_case,
        answer_use_case=answer_use_case,
    )
