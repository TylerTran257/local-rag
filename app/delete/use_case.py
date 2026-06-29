from __future__ import annotations

import logging

from app.delete.contracts import DeleteRequest, DeleteResult
from app.profiles import ProfileResolver
from app.retrieval.types import RetrievalScope
from app.services.lexical_search_service import LexicalSearchService
from app.services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)


class DeleteUseCase:
    def __init__(
        self,
        vector_store_service: VectorStoreService,
        lexical_search_service: LexicalSearchService,
        profile_resolver: ProfileResolver | None = None,
    ):
        self.vector_store_service = vector_store_service
        self.lexical_search_service = lexical_search_service
        self.profile_resolver = profile_resolver or ProfileResolver(None)

    def execute(self, request: DeleteRequest) -> DeleteResult:
        qdrant_collection = self.profile_resolver.resolve(
            request.service_name
        ).collection

        scope = RetrievalScope(
            service_name=request.service_name,
            tenant_id=request.tenant_id,
            collections=request.collections,
            filters=request.filters,
        )

        vector_count = self.vector_store_service.delete_by_scope(
            scope, collection_name=qdrant_collection
        )

        lexical_filters = {
            "service_name": request.service_name,
            "tenant_id": request.tenant_id,
            "collections": request.collections,
            **{k: v for k, v in request.filters.items()},
        }
        self.lexical_search_service.delete_by_scope(lexical_filters)

        logger.info(
            "event=delete_completed service_name=%s tenant_id=%s collections=%s deleted_count=%s",
            request.service_name,
            request.tenant_id,
            request.collections,
            vector_count,
        )

        return DeleteResult(deleted_count=vector_count)
