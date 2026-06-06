import logging
from time import perf_counter

from sqlalchemy import text

from app.db.database import SessionLocal
from app.db.models import DocumentChunk

logger = logging.getLogger(__name__)

FTS_TABLE_NAME = "document_chunks_fts"


class LexicalSearchService:
    def __init__(self, session_factory=None) -> None:
        self.session_factory = session_factory or SessionLocal
        self._ensure_table()

    def _ensure_table(self) -> None:
        with self.session_factory() as session:
            session.execute(text(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE_NAME}
                USING fts5(
                    document_chunk_id UNINDEXED,
                    document_id UNINDEXED,
                    chunk_index UNINDEXED,
                    original_filename UNINDEXED,
                    text
                )
            """))
            session.commit()

    def _normalize_query(self, query: str) -> str:
        # TODO: need to further normalize to strip unsafe character
        tokens = query.lower().strip().split()
        return " ".join(tokens)

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
        self, document_id: str, original_filename: str, chunks: list[DocumentChunk]
    ) -> None:
        started_at = perf_counter()
        self.delete_document_chunks(document_id)

        if len(chunks) == 0:
            logger.info(
                "event=lexical_index_completed document_id=%s chunk_count=0 duration_ms=%s",
                document_id,
                round((perf_counter() - started_at) * 1000, 2),
            )
            return

        rows = [
            {
                "document_chunk_id": chunk.id,
                "document_id": document_id,
                "chunk_index": chunk.chunk_index,
                "original_filename": original_filename,
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
                        text
                    )
                    VALUES (
                        :document_chunk_id,
                        :document_id,
                        :chunk_index,
                        :original_filename,
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

    def search(self, query: str, limit: int) -> list[dict]:
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

        with self.session_factory() as session:
            result = session.execute(
                text(f"""
                SELECT
                    document_id,
                    original_filename,
                    chunk_index,
                    text,
                    bm25({FTS_TABLE_NAME}) as score
                FROM {FTS_TABLE_NAME}
                WHERE {FTS_TABLE_NAME} MATCH :query
                ORDER BY bm25({FTS_TABLE_NAME})
                LIMIT :limit
            """),
                {"query": normalized_query, "limit": limit},
            )

            rows = result.mappings().all()

        results = [
            {
                "document_id": row["document_id"],
                "original_filename": row["original_filename"],
                "chunk_index": row["chunk_index"],
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
