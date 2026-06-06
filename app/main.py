import logging

from fastapi import FastAPI

from app.api.routes import (
    ask_router,
    chat_router,
    documents_router,
    health_router,
    jobs_router,
    pages_router,
    search_router,
    uploads_router,
)
from app.core.logging import configure_logging
from app.core.middleware import register_request_timing_middleware
from app.db.database import Base, engine
from app.services.document_service import DocumentService
from app.services.embedding_service import EmbeddingService
from app.services.generation_service import GenerationService
from app.services.lexical_search_service import LexicalSearchService
from app.services.text_extractor import TextExtractor
from app.services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)


def create_app(document_service=None, generation_service=None) -> FastAPI:
    configure_logging()
    logger.info("event=app_started")
    app = FastAPI()
    register_request_timing_middleware(app)

    Base.metadata.create_all(bind=engine)

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
    app.state.document_service = resolved_document_service
    app.state.generation_service = resolved_generation_service

    app.include_router(health_router)
    app.include_router(pages_router)
    app.include_router(uploads_router)
    app.include_router(search_router)
    app.include_router(ask_router)
    app.include_router(chat_router)
    app.include_router(jobs_router)
    app.include_router(documents_router)
    return app
