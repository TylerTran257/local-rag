"""Style response mapper."""
from app.retrieval.types import RetrievedChunk, RetrievalWarning
from app.social.types import (
    StyleCategory,
    StyleEntry,
    StyleContext,
    SourceReference,
)


class StyleResponseMapper:
    """
    Maps retrieved chunks to style-specific response contract.

    Groups chunks by category, converts to StyleEntry objects,
    collects source references, and reports missing categories.
    """

    def map_to_context(
        self,
        chunks_by_category: dict[StyleCategory, list[RetrievedChunk]],
        warnings: list[RetrievalWarning],
        trace_ids: list[str],
        requested_categories: list[StyleCategory],
    ) -> StyleContext:
        """
        Map retrieved chunks to StyleContext response.

        Args:
            chunks_by_category: Chunks grouped by style category
            warnings: Warnings from all per-category retrievals
            trace_ids: Trace IDs from all retrievals
            requested_categories: Categories that were requested

        Returns:
            StyleContext with results grouped by category
        """
        # Convert chunks to style entries by category
        voice_rules = self._convert_to_entries(
            chunks_by_category.get(StyleCategory.VOICE_RULES, [])
        )
        hook_patterns = self._convert_to_entries(
            chunks_by_category.get(StyleCategory.HOOK_PATTERNS, [])
        )
        cta_patterns = self._convert_to_entries(
            chunks_by_category.get(StyleCategory.CTA_PATTERNS, [])
        )
        past_post_patterns = self._convert_to_entries(
            chunks_by_category.get(StyleCategory.PAST_POST_PATTERNS, [])
        )
        avoid_rules = self._convert_to_entries(
            chunks_by_category.get(StyleCategory.AVOID_RULES, [])
        )
        offer_positioning = self._convert_to_entries(
            chunks_by_category.get(StyleCategory.OFFER_POSITIONING, [])
        )

        # Collect source references (deduplicated)
        all_chunks = []
        for chunks in chunks_by_category.values():
            all_chunks.extend(chunks)
        source_references = self._collect_source_references(all_chunks)

        # Identify missing categories
        missing_categories = self._identify_missing_categories(
            requested_categories=requested_categories,
            chunks_by_category=chunks_by_category,
        )

        return StyleContext(
            voice_rules=voice_rules,
            hook_patterns=hook_patterns,
            cta_patterns=cta_patterns,
            past_post_patterns=past_post_patterns,
            avoid_rules=avoid_rules,
            offer_positioning=offer_positioning,
            source_references=source_references,
            warnings=warnings,
            trace_ids=trace_ids,
            missing_categories=missing_categories,
        )

    def _convert_to_entries(
        self, chunks: list[RetrievedChunk]
    ) -> list[StyleEntry]:
        """Convert retrieved chunks to style entries."""
        return [
            StyleEntry(
                content=chunk.content,
                source_label=chunk.metadata.get("source_label", "unknown"),
                score=chunk.score,
                metadata=chunk.metadata,
            )
            for chunk in chunks
        ]

    def _collect_source_references(
        self, chunks: list[RetrievedChunk]
    ) -> list[SourceReference]:
        """Collect unique source references from chunks."""
        seen_documents = set()
        references = []

        for chunk in chunks:
            doc_id = chunk.document_id
            if doc_id not in seen_documents:
                seen_documents.add(doc_id)
                references.append(
                    SourceReference(
                        source_label=chunk.metadata.get("source_label", "unknown"),
                        document_id=doc_id,
                    )
                )

        return references

    def _identify_missing_categories(
        self,
        requested_categories: list[StyleCategory],
        chunks_by_category: dict[StyleCategory, list[RetrievedChunk]],
    ) -> list[StyleCategory]:
        """Identify categories that were requested but returned zero results."""
        missing = []
        for category in requested_categories:
            if len(chunks_by_category.get(category, [])) == 0:
                missing.append(category)
        return missing
