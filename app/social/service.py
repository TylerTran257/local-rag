"""Social style retrieval service."""
import logging

from app.retrieval.types import RetrievalMode, RetrievalScope, RetrieveRequest
from app.retrieval.use_case import RetrieveUseCase, RetrieveResult
from app.social.types import (
    StyleCategory,
    StyleRetrievalRequest,
    StyleContext,
)
from app.social.mapper import StyleResponseMapper

logger = logging.getLogger(__name__)


class SocialStyleRetrievalService:
    """
    Service layer for social style retrieval.

    Translates style-specific retrieval requests into generic Retrieval Core
    calls and maps results into style-specific response contract.

    Framework-independent - no FastAPI dependencies.
    """

    def __init__(
        self,
        retrieve_use_case: RetrieveUseCase,
        service_name: str = "social-style",
    ):
        """
        Initialize social style retrieval service.

        Args:
            retrieve_use_case: Core retrieval use case
            service_name: Service name for scope enforcement (default: "social-style")
        """
        self.retrieve_use_case = retrieve_use_case
        self.service_name = service_name
        self.mapper = StyleResponseMapper()

    def retrieve(self, request: StyleRetrievalRequest) -> StyleContext:
        """
        Retrieve style context for the given request.

        Executes one retrieval per requested category, collects results,
        and maps them into a StyleContext response.

        Args:
            request: Style retrieval request

        Returns:
            StyleContext with results grouped by category
        """
        chunks_by_category = {}
        all_warnings = []
        trace_ids = []

        # Execute one retrieval per category
        for category in request.style_categories:
            retrieve_result = self._retrieve_category(
                request=request,
                category=category,
            )

            chunks_by_category[category] = retrieve_result.chunks
            all_warnings.extend(retrieve_result.warnings)
            if retrieve_result.trace_id is not None:
                trace_ids.append(retrieve_result.trace_id)

        # Map results to style context
        context = self.mapper.map_to_context(
            chunks_by_category=chunks_by_category,
            warnings=all_warnings,
            trace_ids=trace_ids,
            requested_categories=request.style_categories,
        )

        logger.info(
            "event=style_retrieval_completed tenant_id=%s categories=%s total_entries=%s missing_categories=%s",
            request.tenant_id,
            [cat.value for cat in request.style_categories],
            sum(len(chunks) for chunks in chunks_by_category.values()),
            [cat.value for cat in context.missing_categories],
        )

        return context

    def _retrieve_category(
        self,
        request: StyleRetrievalRequest,
        category: StyleCategory,
    ) -> RetrieveResult:
        """
        Execute retrieval for a single style category.

        Args:
            request: Original style request
            category: Category to retrieve

        Returns:
            RetrieveResult from core retrieval
        """
        # Build filters
        filters = {
            "style_category": category.value,
        }

        # Add platform filter if provided
        if request.platform is not None:
            filters["platform"] = request.platform

        # Build scope
        scope = RetrievalScope(
            service_name=self.service_name,
            tenant_id=request.tenant_id,
            collections=[request.collection],
            filters=filters,
        )

        # Build retrieve request
        retrieve_request = RetrieveRequest(
            query=request.query,
            retrieval_mode=RetrievalMode.HYBRID,
            limit=request.per_category_limit,
            scope=scope,
        )

        # Execute retrieval
        return self.retrieve_use_case.execute(retrieve_request)
