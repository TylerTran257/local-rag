"""Metadata-aware retrieval gateway with backend-level filtering."""
import logging
from typing import Any

from app.retrieval.errors import (
    NoIndexedCorpusError,
    RetrievalError,
    RetrievalExecutionError,
)
from app.retrieval.types import (
    EffectiveRetrieveRequest,
    RetrievalGatewayResult,
    RetrievalMode,
    RetrievedChunk,
)
from app.services.embedding_service import EmbeddingService
from app.services.lexical_search_service import LexicalSearchService
from app.services.vector_store_service import VectorStoreService
from app.settings import settings

logger = logging.getLogger(__name__)


class MetadataAwareRetrievalGateway:
    """
    Retrieval gateway that applies validated scope filters at the backend level.

    Supports dense, lexical, and hybrid retrieval modes. Hybrid mode runs
    scoped dense and lexical retrieval internally and fuses results using
    reciprocal rank fusion (RRF).

    This gateway talks directly to VectorStoreService and LexicalSearchService
    with metadata-aware filter translation.
    """

    def __init__(
        self,
        vector_store_service: VectorStoreService,
        lexical_search_service: LexicalSearchService,
        embedding_service: EmbeddingService,
    ):
        self.vector_store_service = vector_store_service
        self.lexical_search_service = lexical_search_service
        self.embedding_service = embedding_service

    def retrieve(self, request: EffectiveRetrieveRequest) -> RetrievalGatewayResult:
        """
        Execute retrieval with scope-filtered backends.

        Args:
            request: Effective retrieve request with validated scope

        Returns:
            RetrievalGatewayResult with normalized chunks, warnings, and diagnostics

        Raises:
            NoIndexedCorpusError: When the required backends have no indexed content
            RetrievalExecutionError: When backend execution fails unexpectedly
        """
        try:
            self._check_corpus(request.retrieval_mode)

            if request.retrieval_mode == RetrievalMode.DENSE:
                return self._retrieve_dense(request)
            elif request.retrieval_mode == RetrievalMode.LEXICAL:
                return self._retrieve_lexical(request)
            elif request.retrieval_mode == RetrievalMode.HYBRID:
                return self._retrieve_hybrid(request)
            else:
                raise RetrievalExecutionError(
                    trace_id="unknown",  # Will be set by RetrieveUseCase
                    internal_message=f"Unsupported retrieval mode: {request.retrieval_mode}",
                    details={"mode": str(request.retrieval_mode)},
                )
        except RetrievalError:
            raise
        except Exception as e:
            raise RetrievalExecutionError(
                trace_id="unknown",
                internal_message=f"Backend execution failed: {str(e)}",
                details={"exception_type": type(e).__name__, "exception_message": str(e)},
            )

    def _check_corpus(self, mode: RetrievalMode) -> None:
        """Raise NoIndexedCorpusError when the backends required by the mode are empty.

        Hybrid retrieval can serve results from either backend, so it only
        fails when both are empty.
        """
        if mode == RetrievalMode.DENSE:
            if not self.vector_store_service.has_indexed_chunks():
                raise NoIndexedCorpusError(
                    trace_id="unknown",
                    internal_message="No indexed corpus available in vector backend",
                    details={"backend": "vector"},
                )
        elif mode == RetrievalMode.LEXICAL:
            if not self.lexical_search_service.has_indexed_chunks():
                raise NoIndexedCorpusError(
                    trace_id="unknown",
                    internal_message="No indexed corpus available in lexical backend",
                    details={"backend": "lexical"},
                )
        elif mode == RetrievalMode.HYBRID:
            if (
                not self.vector_store_service.has_indexed_chunks()
                and not self.lexical_search_service.has_indexed_chunks()
            ):
                raise NoIndexedCorpusError(
                    trace_id="unknown",
                    internal_message="No indexed corpus available in vector or lexical backend",
                    details={"backend": "vector,lexical"},
                )

    def _retrieve_dense(
        self, request: EffectiveRetrieveRequest
    ) -> RetrievalGatewayResult:
        """Execute dense retrieval with scope filters."""
        # Embed query
        query_embedding = self.embedding_service.embed_text(request.normalized_query)

        # Build qdrant filter from scope
        query_filter = self.vector_store_service.build_query_filter(
            request.validated_scope
        )

        # Query vector store
        results = self.vector_store_service.search(
            query_embedding=query_embedding,
            limit=request.limit,
            query_filter=query_filter,
        )

        # Build filters dict for diagnostics
        filters = self._build_filters_dict(request.validated_scope)

        # Normalize results
        chunks = [
            self._normalize_dense_result(result, rank)
            for rank, result in enumerate(results)
        ]

        # Build diagnostics
        diagnostics = self._build_diagnostics(
            retrieval_mode="dense",
            backend_names=["vector"],
            requested_collections=request.validated_scope.collections,
            filter_keys=list(filters.keys()),
            dense_candidate_count=len(results),
            filters_applied_by_dense=True,
        )

        return RetrievalGatewayResult(
            chunks=chunks,
            warnings=[],
            diagnostics=diagnostics,
        )

    def _retrieve_lexical(
        self, request: EffectiveRetrieveRequest
    ) -> RetrievalGatewayResult:
        """Execute lexical retrieval with scope filters."""
        # Translate scope to filters
        filters = self._build_filters(request.validated_scope)

        # Query lexical search
        results = self.lexical_search_service.search(
            query=request.normalized_query,
            limit=request.limit,
            filters=filters,
        )

        # Normalize results
        chunks = [
            self._normalize_lexical_result(result, rank)
            for rank, result in enumerate(results)
        ]

        # Build diagnostics
        diagnostics = self._build_diagnostics(
            retrieval_mode="lexical",
            backend_names=["lexical"],
            requested_collections=request.validated_scope.collections,
            filter_keys=list(filters.keys()),
            lexical_candidate_count=len(results),
            filters_applied_by_lexical=True,
        )

        return RetrievalGatewayResult(
            chunks=chunks,
            warnings=[],
            diagnostics=diagnostics,
        )

    def _retrieve_hybrid(
        self, request: EffectiveRetrieveRequest
    ) -> RetrievalGatewayResult:
        """Execute hybrid retrieval with RRF fusion of scoped sub-retrievals."""
        # Build filters for both backends
        query_filter = self.vector_store_service.build_query_filter(
            request.validated_scope
        )
        lexical_filters = self._build_filters(request.validated_scope)

        # Dense retrieval
        query_embedding = self.embedding_service.embed_text(request.normalized_query)
        dense_results = self.vector_store_service.search(
            query_embedding=query_embedding,
            limit=request.limit,
            query_filter=query_filter,
        )

        # Lexical retrieval
        lexical_results = self.lexical_search_service.search(
            query=request.normalized_query,
            limit=request.limit,
            filters=lexical_filters,
        )

        # Fuse results using RRF
        fused_results = self._fuse_rankings_rrf(
            dense_results, lexical_results, request.limit
        )

        # Normalize fused results
        chunks = [
            self._normalize_hybrid_result(result, rank)
            for rank, result in enumerate(fused_results)
        ]

        # Build diagnostics
        filters_dict = self._build_filters_dict(request.validated_scope)
        diagnostics = self._build_diagnostics(
            retrieval_mode="hybrid",
            backend_names=["vector", "lexical"],
            requested_collections=request.validated_scope.collections,
            filter_keys=list(filters_dict.keys()),
            dense_candidate_count=len(dense_results),
            lexical_candidate_count=len(lexical_results),
            filters_applied_by_dense=True,
            filters_applied_by_lexical=True,
        )

        return RetrievalGatewayResult(
            chunks=chunks,
            warnings=[],
            diagnostics=diagnostics,
        )

    def _build_filters(self, scope: Any) -> dict[str, Any]:
        """
        Translate validated scope to lexical search filters.

        Returns dict with:
        - service_name: scope enforcement
        - tenant_id: scope enforcement
        - collections: list of collections (special key mapped to "collection" field)
        - Additional scope.filters fields pass through
        """
        reserved_keys = {"service_name", "tenant_id", "collections", "collection"}

        filters = {
            "service_name": scope.service_name,
            "tenant_id": scope.tenant_id,
            "collections": scope.collections,
        }

        # Merge additional filters, never letting them shadow the
        # scope-enforcement keys above
        for key, value in (scope.filters or {}).items():
            if key not in reserved_keys:
                filters[key] = value

        return filters

    def _build_filters_dict(self, scope: Any) -> dict[str, Any]:
        """Build a dict representation of filters for diagnostics."""
        return self._build_filters(scope)

    def _build_diagnostics(self, **kwargs: Any) -> dict[str, Any]:
        """Build diagnostics dict from provided fields."""
        return {k: v for k, v in kwargs.items() if v is not None}

    def _normalize_dense_result(self, result: dict, rank: int) -> RetrievedChunk:
        """Normalize vector store result to RetrievedChunk."""
        return RetrievedChunk(
            chunk_id=f"{result['document_id']}:{result['chunk_index']}",
            document_id=result["document_id"],
            content=result["text"],
            score=result["score"],
            rank=rank,
            retrieval_mode=RetrievalMode.DENSE,
            metadata=self._normalize_metadata(result),
        )

    def _normalize_lexical_result(self, result: dict, rank: int) -> RetrievedChunk:
        """Normalize lexical search result to RetrievedChunk."""
        return RetrievedChunk(
            chunk_id=f"{result['document_id']}:{result['chunk_index']}",
            document_id=result["document_id"],
            content=result["text"],
            score=result["score"],
            rank=rank,
            retrieval_mode=RetrievalMode.LEXICAL,
            metadata=self._normalize_metadata(result),
        )

    def _normalize_hybrid_result(self, result: dict, rank: int) -> RetrievedChunk:
        """Normalize hybrid fused result to RetrievedChunk."""
        # Hybrid results retain their original mode from fusion
        return RetrievedChunk(
            chunk_id=f"{result['document_id']}:{result['chunk_index']}",
            document_id=result["document_id"],
            content=result["text"],
            score=result["score"],
            rank=rank,
            retrieval_mode=result.get("retrieval_mode", RetrievalMode.HYBRID),
            metadata=self._normalize_metadata(result),
        )

    def _normalize_metadata(self, result: dict[str, Any]) -> dict[str, Any]:
        """Preserve metadata fields returned by backends while excluding chunk payload fields."""
        excluded_keys = {
            "document_id",
            "chunk_index",
            "text",
            "score",
            "retrieval_mode",
        }
        return {
            key: value
            for key, value in result.items()
            if key not in excluded_keys
        }

    def _fuse_rankings_rrf(
        self, dense_results: list[dict], lexical_results: list[dict], limit: int
    ) -> list[dict]:
        """
        Fuse dense and lexical rankings using reciprocal rank fusion.

        Uses same RRF algorithm as DocumentService with k=60 from settings.

        Args:
            dense_results: Results from vector store
            lexical_results: Results from lexical search
            limit: Maximum number of results to return

        Returns:
            Fused and ranked results
        """
        if limit <= 0:
            return []

        fused_by_key: dict[tuple[str, int], dict] = {}
        rrf_k = settings.fusion_rrf_k

        def add_results(results: list[dict], mode: RetrievalMode) -> None:
            for rank, result in enumerate(results, start=1):
                key = (result["document_id"], result["chunk_index"])
                rrf_score = 1 / (rrf_k + rank)

                existing = fused_by_key.get(key)
                if existing is None:
                    fused_by_key[key] = {
                        **result,
                        "score": rrf_score,
                        "retrieval_mode": mode,
                    }
                else:
                    existing["score"] += rrf_score

        add_results(dense_results, RetrievalMode.DENSE)
        add_results(lexical_results, RetrievalMode.LEXICAL)

        fused_results = list(fused_by_key.values())
        fused_results.sort(
            key=lambda item: (-item["score"], item["document_id"], item["chunk_index"])
        )

        return fused_results[:limit]
