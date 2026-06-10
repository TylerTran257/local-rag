import json
import logging
import re
from time import perf_counter
from typing import Any

from sqlalchemy import text

from app.db.database import SessionLocal
from app.db.models import DocumentChunk

logger = logging.getLogger(__name__)

FTS_TABLE_NAME = "document_chunks_fts"

# Core metadata stored as dedicated columns. All other metadata keys are
# domain metadata, stored as JSON in the domain_metadata column so the
# lexical backend stays domain-agnostic.
CORE_METADATA_FIELDS = (
    "service_name",
    "tenant_id",
    "collection",
    "source_type",
    "source_label",
)

# Columns that may be referenced directly in filter WHERE clauses. Any other
# filter key is resolved against the domain_metadata JSON column.
FILTERABLE_COLUMNS = frozenset(
    {
        "document_chunk_id",
        "document_id",
        "chunk_index",
        "original_filename",
        *CORE_METADATA_FIELDS,
    }
)

_FILTER_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]+$")

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
    domain_metadata UNINDEXED,
    text
)
"""


class LexicalSearchService:
    def __init__(self, session_factory=None) -> None:
        self.session_factory = session_factory or SessionLocal
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Ensure FTS table exists with the current schema, migrating if needed."""
        with self.session_factory() as session:
            result = session.execute(
                text("SELECT sql FROM sqlite_master WHERE name = :table_name"),
                {"table_name": FTS_TABLE_NAME},
            )
            existing_schema = result.scalar()

            if existing_schema is None:
                session.execute(text(NEW_SCHEMA_SQL))
                session.commit()
            elif "domain_metadata" not in existing_schema:
                # Old schema (pre-metadata or hardcoded domain columns) - recreate.
                # Existing FTS data is re-indexable from source documents.
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

    def _split_metadata(
        self, metadata: dict[str, Any] | None
    ) -> tuple[dict[str, Any], str | None]:
        """Split metadata into core column values and a domain-metadata JSON string."""
        if not metadata:
            return {field: None for field in CORE_METADATA_FIELDS}, None

        core = {field: metadata.get(field) for field in CORE_METADATA_FIELDS}
        domain = {
            key: value
            for key, value in metadata.items()
            if key not in CORE_METADATA_FIELDS
        }
        return core, json.dumps(domain) if domain else None

    def _build_filter_clauses(
        self, filters: dict[str, Any]
    ) -> tuple[list[str], dict[str, Any]]:
        """Build SQL WHERE clauses from a filter dict.

        Core metadata fields filter against their dedicated columns. Any other
        key filters against the domain_metadata JSON column via json_extract,
        with the JSON path passed as a bound parameter. Filter values are
        always bound parameters; filter keys never reach the SQL string unless
        they are in the FILTERABLE_COLUMNS allowlist.

        Raises:
            ValueError: For invalid filter keys or empty list filter values.
        """
        if not filters:
            return [], {}

        where_clauses: list[str] = []
        params: dict[str, Any] = {}
        param_counter = 0

        def bind(value: Any) -> str:
            nonlocal param_counter
            name = f"filter_{param_counter}"
            param_counter += 1
            params[name] = value
            return name

        for key, value in filters.items():
            if not _FILTER_KEY_PATTERN.match(key):
                raise ValueError(f"Invalid filter key: {key!r}")

            if isinstance(value, list) and not value:
                # Fail closed: an empty list filter must not silently widen scope
                raise ValueError(f"List filter for {key!r} must be non-empty")

            if key == "collections":
                if not isinstance(value, list):
                    raise ValueError("collections filter must be a list")
                names = [bind(item) for item in value]
                placeholders = ", ".join(f":{name}" for name in names)
                where_clauses.append(f"collection IN ({placeholders})")
            elif key in FILTERABLE_COLUMNS:
                if isinstance(value, list):
                    names = [bind(item) for item in value]
                    placeholders = ", ".join(f":{name}" for name in names)
                    where_clauses.append(f"{key} IN ({placeholders})")
                else:
                    where_clauses.append(f"{key} = :{bind(value)}")
            else:
                path_name = bind(f'$."{key}"')
                if isinstance(value, list):
                    names = [bind(item) for item in value]
                    placeholders = ", ".join(f":{name}" for name in names)
                    where_clauses.append(
                        f"json_extract(domain_metadata, :{path_name}) IN ({placeholders})"
                    )
                else:
                    where_clauses.append(
                        f"json_extract(domain_metadata, :{path_name}) = :{bind(value)}"
                    )

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

    def has_indexed_chunks(self) -> bool:
        """Return True if any chunks are indexed."""
        with self.session_factory() as session:
            result = session.execute(
                text(f"SELECT EXISTS(SELECT 1 FROM {FTS_TABLE_NAME})")
            )
            return bool(result.scalar())

    def index_document_chunks(
        self,
        document_id: str,
        original_filename: str,
        chunks: list[DocumentChunk],
        metadata: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> None:
        """Index document chunks with optional metadata.

        Args:
            document_id: Unique document identifier
            original_filename: Original filename
            chunks: List of document chunks to index
            metadata: Optional metadata. A single dict applies to every chunk;
                     a list provides per-chunk metadata and must match the
                     chunk count. Core fields (service_name, tenant_id,
                     collection, source_type, source_label) are stored as
                     columns; remaining keys are stored as domain metadata JSON.
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

        if isinstance(metadata, list):
            if len(metadata) != len(chunks):
                raise ValueError("metadata length must match chunks length")
            metadata_by_chunk: list[dict[str, Any] | None] = list(metadata)
        else:
            metadata_by_chunk = [metadata] * len(chunks)

        rows = []
        for chunk, chunk_metadata in zip(chunks, metadata_by_chunk):
            core, domain_json = self._split_metadata(chunk_metadata)
            rows.append(
                {
                    "document_chunk_id": chunk.id,
                    "document_id": document_id,
                    "chunk_index": chunk.chunk_index,
                    "original_filename": original_filename,
                    **core,
                    "domain_metadata": domain_json,
                    "text": chunk.text,
                }
            )

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
                        domain_metadata,
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
                        :domain_metadata,
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
                    - Keys beyond the core columns filter domain metadata

        Returns:
            List of result dicts with text, score, core metadata fields, and
            any stored domain metadata keys merged in.
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

        where_clauses, filter_params = self._build_filter_clauses(filters or {})

        if where_clauses:
            where_clause = f"WHERE {FTS_TABLE_NAME} MATCH :query AND " + " AND ".join(
                where_clauses
            )
        else:
            where_clause = f"WHERE {FTS_TABLE_NAME} MATCH :query"

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
                    domain_metadata,
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

        results = []
        for row in rows:
            entry = {
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
            if row["domain_metadata"]:
                domain = json.loads(row["domain_metadata"])
                for key, value in domain.items():
                    entry.setdefault(key, value)
            results.append(entry)

        logger.info(
            "event=retrieval_completed mode=lexical query_length=%s requested_limit=%s result_count=%s duration_ms=%s",
            len(query),
            limit,
            len(results),
            round((perf_counter() - started_at) * 1000, 2),
        )
        return results
