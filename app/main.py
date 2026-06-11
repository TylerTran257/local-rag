import logging

from fastapi import FastAPI

from app.api.routes import (
    ask_router,
    chat_router,
    documents_router,
    health_router,
    ingest_router,
    jobs_router,
    pages_router,
    search_router,
    social_style_router,
    uploads_router,
)
from app.composition import MetadataAwareRuntime, build_metadata_aware_runtime
from app.core.logging import configure_logging
from app.core.middleware import register_request_timing_middleware
from app.db.database import Base, engine
from app.services.document_service import DocumentService
from app.services.embedding_service import EmbeddingService
from app.services.generation_service import GenerationService
from app.services.lexical_search_service import LexicalSearchService
from app.services.text_extractor import TextExtractor
from app.services.vector_store_service import VectorStoreService
from app.retrieval.use_case import RetrieveUseCase
from app.retrieval.legacy_adapter import LegacyDocumentRetrievalAdapter
from app.retrieval import (
    PassthroughScopePolicy,
    StructuredLoggingTraceSink,
    SystemClock,
    UuidTraceIdGenerator,
)

logger = logging.getLogger(__name__)


def create_app(
    document_service=None,
    generation_service=None,
    retrieve_use_case=None,
    *,
    metadata_aware: bool = False,
    metadata_aware_runtime: MetadataAwareRuntime | None = None,
) -> FastAPI:
    """Create and wire the FastAPI application.

    Args:
        document_service: Override for DocumentService (legacy mode).
        generation_service: Override for GenerationService.
        retrieve_use_case: Override for RetrieveUseCase (legacy mode).
        metadata_aware: When True, use the metadata-aware retrieval path
            instead of the legacy LegacyDocumentRetrievalAdapter path.
        metadata_aware_runtime: Pre-built MetadataAwareRuntime to use when
            ``metadata_aware=True``.  When omitted the factory builds
            production defaults via ``build_metadata_aware_runtime()``.

    Returns:
        Configured FastAPI application.
    """
    configure_logging()
    logger.info("event=app_started metadata_aware=%s", metadata_aware)
    app = FastAPI()
    register_request_timing_middleware(app)

    Base.metadata.create_all(bind=engine)

    if metadata_aware:
        # --- Metadata-aware path ---
        runtime = metadata_aware_runtime or build_metadata_aware_runtime()

        # Generation service is still needed for ask / chat routes
        resolved_generation_service = generation_service
        if generation_service is None:
            resolved_generation_service = GenerationService()

        app.state.retrieve_use_case = runtime.retrieve_use_case
        app.state.ingest_use_case = runtime.ingest_use_case
        app.state.social_style_service = runtime.social_style_service
        app.state.generation_service = resolved_generation_service

        # Legacy routes that depend on document_service still need an
        # instance for uploads / jobs / documents pages.  Build one from
        # the same infra services when not explicitly provided.
        resolved_document_service = document_service
        if document_service is None:
            embedding_service = EmbeddingService()
            vector_store_service = VectorStoreService()
            text_extractor = TextExtractor()
            lexical_search_service = LexicalSearchService()
            resolved_document_service = DocumentService(
                embedding_service,
                vector_store_service,
                text_extractor,
                lexical_search_service,
            )
        app.state.document_service = resolved_document_service
    else:
        # --- Legacy path (default) ---
        resolved_document_service = document_service
        if document_service is None:
            embedding_service = EmbeddingService()
            vector_store_service = VectorStoreService()
            text_extractor = TextExtractor()
            lexical_search_service = LexicalSearchService()
            resolved_document_service = DocumentService(
                embedding_service,
                vector_store_service,
                text_extractor,
                lexical_search_service,
            )

        resolved_generation_service = generation_service
        if generation_service is None:
            resolved_generation_service = GenerationService()

        # Wire Retrieval Core
        resolved_retrieve_use_case = retrieve_use_case
        if retrieve_use_case is None:
            gateway = LegacyDocumentRetrievalAdapter(document_service=resolved_document_service)
            scope_policy = PassthroughScopePolicy()
            clock = SystemClock()
            trace_id_generator = UuidTraceIdGenerator()
            trace_sink = StructuredLoggingTraceSink()

            resolved_retrieve_use_case = RetrieveUseCase(
                gateway=gateway,
                scope_policy=scope_policy,
                clock=clock,
                trace_id_generator=trace_id_generator,
                trace_sink=trace_sink,
            )

        app.state.document_service = resolved_document_service
        app.state.generation_service = resolved_generation_service
        app.state.retrieve_use_case = resolved_retrieve_use_case

    app.include_router(health_router)
    app.include_router(pages_router)
    app.include_router(uploads_router)
    app.include_router(search_router)
    app.include_router(ask_router)
    app.include_router(chat_router)
    app.include_router(jobs_router)
    app.include_router(documents_router)

    # Metadata-aware routes are only available when metadata_aware=True
    if metadata_aware:
        app.include_router(ingest_router)
        app.include_router(social_style_router)

    return app
