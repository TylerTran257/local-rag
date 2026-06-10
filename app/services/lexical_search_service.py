import logging
from time import perf_counter
from typing import Any

from sqlalchemy import text

from app.db.database import SessionLocal
from app.db.models import DocumentChunk

logger = logging.getLogger(__name__)

FTS_TABLE_NAME = "document_chunks_fts"

# New FTS5 schema with metadata columns
NEW_SCHEMA_SQL = f"""
CREATE VIRTUAL TABLE {FTS_TABLE_NAME} USING fts5(
    document_chunk_id UNINDEXED,
    document_id UNINDEXED,
    chunk_index UNINDEXED,
    original_filename UNINDEXED,
    service_name UNINDEXED,
    tenant_id UNINDEXED,
    collection UNINDEXED,
    source_type UNINDEXED,
    source_label UNINDEXED,
    text
)
"""


class LexicalSearchService:
    def __init__(self, session_factory=None) -> None:
        self.session_factory = session_factory or SessionLocal
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Ensure FTS table exists with new schema, migrating if needed."""
        with self.session_factory() as session:
            # Check if table exists
            result = session.execute(
                text("SELECT sql FROM sqlite_master WHERE name = :table_name"),
                {"table_name": FTS_TABLE_NAME},
            )
            existing_schema = result.scalar()

            if existing_schema is None:
                # Table doesn't exist, create with new schema
                session.execute(text(NEW_SCHEMA_SQL))
                session.commit()
            elif "service_name" not in existing_schema:
                # Old schema exists, need to migrate
                logger.info(
                    "event=fts_schema_migration action=recreating_table reason=old_schema_detected"
                )
                session.execute(text(f"DROP TABLE {FTS_TABLE_NAME}"))
                session.execute(text(NEW_SCHEMA_SQL))
                session.commit()

    def _normalize_query(self, query: str) -> str:
        # TODO: need to further normalize to strip unsafe character
        tokens = query.lower().strip().split()
        return " ".join(tokens)

    def _build_filter_clauses(
        self, filters: dict[str, Any]
    ) -> tuple[list[str], dict[str, Any]]:
        """Build SQL WHERE clauses from filter dict.

        Returns:
            Tuple of (list of WHERE clause strings, dict of parameter values)
        """
        if not filters:
            return [], {}

        where_clauses = []
        params = {}
        param_counter = 0

        for key, value in filters.items():
            if key == "collections":
                # Special case: collections is a list field
                if isinstance(value, list) and value:
                    placeholders = []
                    for item in value:
                        param_name = f"collection_{param_counter}"
                        params[param_name] = item
                        placeholders.append(f":{param_name}")
                        param_counter += 1
                    where_clauses.append(f"collection IN ({', '.join(placeholders)})")
            elif isinstance(value, list):
                # List filter: field IN (...)
                if value:
                    placeholders = []
                    for item in value:
                        param_name = f"{key}_{param_counter}"
                        params[param_name] = item
                        placeholders.append(f":{param_name}")
                        param_counter += 1
                    where_clauses.append(f"{key} IN ({', '.join(placeholders)})")
            else:
                # Primitive filter: field = value
                param_name = f"{key}_{param_counter}"
                params[param_name] = value
                where_clauses.append(f"{key} = :{param_name}")
                param_counter += 1

        return where_clauses, params

    def delete_document_chunks(self, document_id: str) -> None:
        with self.session_factory() as session:
            session.execute(
                text(f"""
                DELETE FROM {FTS_TABLE_NAME}
                WHERE document_id = :document_id;
            """),
                {"document_id": document_id},
            )
            session.commit()

    def index_document_chunks(
        self,
        document_id: str,
        original_filename: str,
        chunks: list[DocumentChunk],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Index document chunks with optional metadata.

        Args:
            document_id: Unique document identifier
            original_filename: Original filename
            chunks: List of document chunks to index
            metadata: Optional metadata dict with keys: service_name, tenant_id,
                     collection, source_type, source_label
        """
        started_at = perf_counter()
        self.delete_document_chunks(document_id)

        if len(chunks) == 0:
            logger.info(
                "event=lexical_index_completed document_id=%s chunk_count=0 duration_ms=%s",
                document_id,
                round((perf_counter() - started_at) * 1000, 2),
            )
            return

        # Extract metadata or use None for backward compatibility
        service_name = metadata.get("service_name") if metadata else None
        tenant_id = metadata.get("tenant_id") if metadata else None
        collection = metadata.get("collection") if metadata else None
        source_type = metadata.get("source_type") if metadata else None
        source_label = metadata.get("source_label") if metadata else None

        rows = [
            {
                "document_chunk_id": chunk.id,
                "document_id": document_id,
                "chunk_index": chunk.chunk_index,
                "original_filename": original_filename,
                "service_name": service_name,
                "tenant_id": tenant_id,
                "collection": collection,
                "source_type": source_type,
                "source_label": source_label,
                "text": chunk.text,
            }
            for chunk in chunks
        ]

        with self.session_factory() as session:
            session.execute(
                text(f"""
                    INSERT INTO {FTS_TABLE_NAME} (
                        document_chunk_id,
                        document_id,
                        chunk_index,
                        original_filename,
                        service_name,
                        tenant_id,
                        collection,
                        source_type,
                        source_label,
                        text
                    )
                    VALUES (
                        :document_chunk_id,
                        :document_id,
                        :chunk_index,
                        :original_filename,
                        :service_name,
                        :tenant_id,
                        :collection,
                        :source_type,
                        :source_label,
                        :text
                    )
                """),
                rows,
            )
            session.commit()

        logger.info(
            "event=lexical_index_completed document_id=%s chunk_count=%s duration_ms=%s",
            document_id,
            len(chunks),
            round((perf_counter() - started_at) * 1000, 2),
        )

    def search(
        self, query: str, limit: int, filters: dict[str, Any] | None = None
    ) -> list[dict]:
        """Search indexed chunks with optional metadata filtering.

        Args:
            query: Search query
            limit: Maximum number of results
            filters: Optional metadata filters. Supports:
                    - Primitive equality: {"field": "value"}
                    - List membership: {"field": ["value1", "value2"]}
                    - Special "collections" key maps to "collection" field

        Returns:
            List of result dicts with text, score, and metadata fields
        """
        started_at = perf_counter()
        normalized_query = self._normalize_query(query)
        if not normalized_query:
            logger.info(
                "event=retrieval_completed mode=lexical query_length=%s requested_limit=%s result_count=0 duration_ms=%s",
                len(query),
                limit,
                round((perf_counter() - started_at) * 1000, 2),
            )
            return []

        # Build filter clauses
        where_clauses, filter_params = self._build_filter_clauses(filters or {})

        # Build WHERE clause combining FTS MATCH and metadata filters
        if where_clauses:
            where_clause = f"WHERE {FTS_TABLE_NAME} MATCH :query AND " + " AND ".join(
                where_clauses
            )
        else:
            where_clause = f"WHERE {FTS_TABLE_NAME} MATCH :query"

        # Combine query params and filter params
        params = {"query": normalized_query, "limit": limit}
        params.update(filter_params)

        with self.session_factory() as session:
            result = session.execute(
                text(f"""
                SELECT
                    document_id,
                    original_filename,
                    chunk_index,
                    service_name,
                    tenant_id,
                    collection,
                    source_type,
                    source_label,
                    text,
                    bm25({FTS_TABLE_NAME}) as score
                FROM {FTS_TABLE_NAME}
                {where_clause}
                ORDER BY bm25({FTS_TABLE_NAME})
                LIMIT :limit
            """),
                params,
            )

            rows = result.mappings().all()

        results = [
            {
                "document_id": row["document_id"],
                "original_filename": row["original_filename"],
                "chunk_index": row["chunk_index"],
                "service_name": row["service_name"],
                "tenant_id": row["tenant_id"],
                "collection": row["collection"],
                "source_type": row["source_type"],
                "source_label": row["source_label"],
                "score": row["score"],
                "text": row["text"],
            }
            for row in rows
        ]

        logger.info(
            "event=retrieval_completed mode=lexical query_length=%s requested_limit=%s result_count=%s duration_ms=%s",
            len(query),
            limit,
            len(results),
            round((perf_counter() - started_at) * 1000, 2),
        )
        return results
