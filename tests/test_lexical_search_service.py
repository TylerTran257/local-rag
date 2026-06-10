import pytest
from sqlalchemy import text

from app.db.database import SessionLocal
from app.db.models import DocumentChunk
from app.services.lexical_search_service import LexicalSearchService


@pytest.fixture
def lexical_service():
    """Fixture providing a fresh LexicalSearchService instance with clean FTS table."""
    # Drop existing FTS table to ensure clean state
    with SessionLocal() as session:
        session.execute(text("DROP TABLE IF EXISTS document_chunks_fts"))
        session.commit()

    # Create fresh service (will recreate table)
    service = LexicalSearchService(session_factory=SessionLocal)

    yield service

    # Cleanup after test
    with SessionLocal() as session:
        session.execute(text("DROP TABLE IF EXISTS document_chunks_fts"))
        session.commit()


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    return [
        DocumentChunk(
            id="chunk-1",
            chunk_index=0,
            text="Python is a programming language",
        ),
        DocumentChunk(
            id="chunk-2",
            chunk_index=1,
            text="JavaScript is also a programming language",
        ),
    ]


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return {
        "service_name": "test-service",
        "tenant_id": "tenant-123",
        "collection": "documents",
        "source_type": "pdf",
        "source_label": "test-document.pdf",
    }


class TestSchemaAndMigration:
    """Tests for FTS5 schema and migration handling."""

    def test_table_creation_includes_metadata_columns(self, lexical_service):
        """Verify FTS table includes all required metadata columns."""
        with SessionLocal() as session:
            # Query table schema
            result = session.execute(
                text("SELECT sql FROM sqlite_master WHERE name = 'document_chunks_fts'")
            )
            schema = result.scalar()

            assert schema is not None
            assert "service_name" in schema
            assert "tenant_id" in schema
            assert "collection" in schema
            assert "source_type" in schema
            assert "source_label" in schema
            assert "UNINDEXED" in schema

    def test_schema_migration_from_old_table(self):
        """Verify graceful migration from old FTS schema to new schema."""
        # Create old-style FTS table
        with SessionLocal() as session:
            # Drop existing table if present
            session.execute(text("DROP TABLE IF EXISTS document_chunks_fts"))

            # Create old schema
            session.execute(text("""
                CREATE VIRTUAL TABLE document_chunks_fts USING fts5(
                    document_chunk_id UNINDEXED,
                    document_id UNINDEXED,
                    chunk_index UNINDEXED,
                    original_filename UNINDEXED,
                    text
                )
            """))

            # Insert old-style data
            session.execute(text("""
                INSERT INTO document_chunks_fts (
                    document_chunk_id, document_id, chunk_index, original_filename, text
                )
                VALUES ('old-chunk-1', 'old-doc-1', 0, 'old.txt', 'old content')
            """))
            session.commit()

        # Initialize service (should trigger migration)
        service = LexicalSearchService(session_factory=SessionLocal)

        # Verify new schema exists
        with SessionLocal() as session:
            result = session.execute(
                text("SELECT sql FROM sqlite_master WHERE name = 'document_chunks_fts'")
            )
            schema = result.scalar()
            assert "service_name" in schema
            assert "tenant_id" in schema


class TestMetadataStorage:
    """Tests for metadata storage functionality."""

    def test_index_chunks_with_metadata(self, lexical_service, sample_chunks, sample_metadata):
        """Verify chunks can be indexed with metadata."""
        lexical_service.index_document_chunks(
            document_id="doc-1",
            original_filename="test.txt",
            chunks=sample_chunks,
            metadata=sample_metadata,
        )

        # Verify data was stored
        with SessionLocal() as session:
            result = session.execute(
                text("""
                    SELECT service_name, tenant_id, collection, source_type, source_label, text
                    FROM document_chunks_fts
                    WHERE document_id = :doc_id
                    ORDER BY chunk_index
                """),
                {"doc_id": "doc-1"},
            )
            rows = result.fetchall()

            assert len(rows) == 2
            assert rows[0][0] == "test-service"  # service_name
            assert rows[0][1] == "tenant-123"     # tenant_id
            assert rows[0][2] == "documents"      # collection
            assert rows[0][3] == "pdf"            # source_type
            assert rows[0][4] == "test-document.pdf"  # source_label
            assert rows[0][5] == "Python is a programming language"  # text

    def test_metadata_round_trip(self, lexical_service, sample_chunks, sample_metadata):
        """Verify metadata survives index -> search round-trip."""
        lexical_service.index_document_chunks(
            document_id="doc-1",
            original_filename="test.txt",
            chunks=sample_chunks,
            metadata=sample_metadata,
        )

        results = lexical_service.search(query="programming", limit=10)

        assert len(results) == 2
        for result in results:
            assert result["service_name"] == "test-service"
            assert result["tenant_id"] == "tenant-123"
            assert result["collection"] == "documents"
            assert result["source_type"] == "pdf"
            assert result["source_label"] == "test-document.pdf"

    def test_index_chunks_without_metadata_uses_null(self, lexical_service, sample_chunks):
        """Verify backward compat: chunks without metadata use NULL."""
        lexical_service.index_document_chunks(
            document_id="doc-1",
            original_filename="test.txt",
            chunks=sample_chunks,
        )

        with SessionLocal() as session:
            result = session.execute(
                text("""
                    SELECT service_name, tenant_id, collection, source_type, source_label
                    FROM document_chunks_fts
                    WHERE document_id = :doc_id
                """),
                {"doc_id": "doc-1"},
            )
            row = result.fetchone()

            # All metadata fields should be NULL
            assert row[0] is None or row[0] == ""
            assert row[1] is None or row[1] == ""
            assert row[2] is None or row[2] == ""
            assert row[3] is None or row[3] == ""
            assert row[4] is None or row[4] == ""


class TestFilteredSearch:
    """Tests for metadata filtering during search."""

    def test_filter_by_service_name(self, lexical_service, sample_chunks):
        """Verify filtering by service_name."""
        # Index chunks for two different services
        lexical_service.index_document_chunks(
            document_id="doc-1",
            original_filename="test1.txt",
            chunks=[sample_chunks[0]],
            metadata={
                "service_name": "service-a",
                "tenant_id": "tenant-1",
                "collection": "docs",
                "source_type": "pdf",
                "source_label": "doc1.pdf",
            },
        )
        lexical_service.index_document_chunks(
            document_id="doc-2",
            original_filename="test2.txt",
            chunks=[sample_chunks[1]],
            metadata={
                "service_name": "service-b",
                "tenant_id": "tenant-1",
                "collection": "docs",
                "source_type": "pdf",
                "source_label": "doc2.pdf",
            },
        )

        # Search with service_name filter
        results = lexical_service.search(
            query="programming",
            limit=10,
            filters={"service_name": "service-a"},
        )

        assert len(results) == 1
        assert results[0]["service_name"] == "service-a"

    def test_filter_by_tenant_id(self, lexical_service, sample_chunks):
        """Verify filtering by tenant_id."""
        lexical_service.index_document_chunks(
            document_id="doc-1",
            original_filename="test1.txt",
            chunks=[sample_chunks[0]],
            metadata={
                "service_name": "service-1",
                "tenant_id": "tenant-a",
                "collection": "docs",
                "source_type": "pdf",
                "source_label": "doc1.pdf",
            },
        )
        lexical_service.index_document_chunks(
            document_id="doc-2",
            original_filename="test2.txt",
            chunks=[sample_chunks[1]],
            metadata={
                "service_name": "service-1",
                "tenant_id": "tenant-b",
                "collection": "docs",
                "source_type": "pdf",
                "source_label": "doc2.pdf",
            },
        )

        results = lexical_service.search(
            query="programming",
            limit=10,
            filters={"tenant_id": "tenant-b"},
        )

        assert len(results) == 1
        assert results[0]["tenant_id"] == "tenant-b"

    def test_filter_by_collections_list(self, lexical_service, sample_chunks):
        """Verify filtering by collections using IN clause."""
        lexical_service.index_document_chunks(
            document_id="doc-1",
            original_filename="test1.txt",
            chunks=[sample_chunks[0]],
            metadata={
                "service_name": "service-1",
                "tenant_id": "tenant-1",
                "collection": "medical",
                "source_type": "pdf",
                "source_label": "doc1.pdf",
            },
        )
        lexical_service.index_document_chunks(
            document_id="doc-2",
            original_filename="test2.txt",
            chunks=[sample_chunks[1]],
            metadata={
                "service_name": "service-1",
                "tenant_id": "tenant-1",
                "collection": "social",
                "source_type": "pdf",
                "source_label": "doc2.pdf",
            },
        )

        results = lexical_service.search(
            query="programming",
            limit=10,
            filters={"collections": ["medical", "social"]},
        )

        assert len(results) == 2

    def test_filter_primitive_field(self, lexical_service, sample_chunks):
        """Verify primitive filter (equality)."""
        lexical_service.index_document_chunks(
            document_id="doc-1",
            original_filename="test1.txt",
            chunks=[sample_chunks[0]],
            metadata={
                "service_name": "service-1",
                "tenant_id": "tenant-1",
                "collection": "docs",
                "source_type": "pdf",
                "source_label": "doc1.pdf",
            },
        )
        lexical_service.index_document_chunks(
            document_id="doc-2",
            original_filename="test2.txt",
            chunks=[sample_chunks[1]],
            metadata={
                "service_name": "service-1",
                "tenant_id": "tenant-1",
                "collection": "docs",
                "source_type": "html",
                "source_label": "doc2.html",
            },
        )

        results = lexical_service.search(
            query="programming",
            limit=10,
            filters={"source_type": "pdf"},
        )

        assert len(results) == 1
        assert results[0]["source_type"] == "pdf"

    def test_filter_list_field(self, lexical_service, sample_chunks):
        """Verify list filter (IN clause)."""
        lexical_service.index_document_chunks(
            document_id="doc-1",
            original_filename="test1.txt",
            chunks=[sample_chunks[0]],
            metadata={
                "service_name": "service-1",
                "tenant_id": "tenant-1",
                "collection": "docs",
                "source_type": "pdf",
                "source_label": "doc1.pdf",
            },
        )
        lexical_service.index_document_chunks(
            document_id="doc-2",
            original_filename="test2.txt",
            chunks=[sample_chunks[1]],
            metadata={
                "service_name": "service-1",
                "tenant_id": "tenant-1",
                "collection": "docs",
                "source_type": "html",
                "source_label": "doc2.html",
            },
        )

        results = lexical_service.search(
            query="programming",
            limit=10,
            filters={"source_type": ["pdf", "html"]},
        )

        assert len(results) == 2

    def test_combined_filters(self, lexical_service, sample_chunks):
        """Verify multiple filters work together."""
        lexical_service.index_document_chunks(
            document_id="doc-1",
            original_filename="test1.txt",
            chunks=[sample_chunks[0]],
            metadata={
                "service_name": "service-a",
                "tenant_id": "tenant-1",
                "collection": "docs",
                "source_type": "pdf",
                "source_label": "doc1.pdf",
            },
        )
        lexical_service.index_document_chunks(
            document_id="doc-2",
            original_filename="test2.txt",
            chunks=[sample_chunks[1]],
            metadata={
                "service_name": "service-b",
                "tenant_id": "tenant-1",
                "collection": "docs",
                "source_type": "pdf",
                "source_label": "doc2.pdf",
            },
        )

        results = lexical_service.search(
            query="programming",
            limit=10,
            filters={
                "service_name": "service-a",
                "tenant_id": "tenant-1",
                "source_type": "pdf",
            },
        )

        assert len(results) == 1
        assert results[0]["service_name"] == "service-a"

    def test_no_filters_returns_all(self, lexical_service, sample_chunks, sample_metadata):
        """Verify search without filters returns all matching chunks."""
        lexical_service.index_document_chunks(
            document_id="doc-1",
            original_filename="test.txt",
            chunks=sample_chunks,
            metadata=sample_metadata,
        )

        results = lexical_service.search(query="programming", limit=10)

        assert len(results) == 2

    def test_filters_exclude_non_matching(self, lexical_service, sample_chunks):
        """Verify filters exclude non-matching chunks."""
        lexical_service.index_document_chunks(
            document_id="doc-1",
            original_filename="test1.txt",
            chunks=[sample_chunks[0]],
            metadata={
                "service_name": "service-a",
                "tenant_id": "tenant-1",
                "collection": "docs",
                "source_type": "pdf",
                "source_label": "doc1.pdf",
            },
        )
        lexical_service.index_document_chunks(
            document_id="doc-2",
            original_filename="test2.txt",
            chunks=[sample_chunks[1]],
            metadata={
                "service_name": "service-b",
                "tenant_id": "tenant-1",
                "collection": "docs",
                "source_type": "pdf",
                "source_label": "doc2.pdf",
            },
        )

        results = lexical_service.search(
            query="programming",
            limit=10,
            filters={"service_name": "service-c"},
        )

        assert len(results) == 0


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing code."""

    def test_existing_index_call_path_works(self, lexical_service, sample_chunks):
        """Verify DocumentService.chunk_document call path still works."""
        # This simulates the existing call from DocumentService
        lexical_service.index_document_chunks(
            document_id="doc-1",
            original_filename="legacy.txt",
            chunks=sample_chunks,
        )

        results = lexical_service.search(query="programming", limit=10)

        assert len(results) == 2
        assert results[0]["text"] is not None

    def test_search_without_filters_unchanged(self, lexical_service, sample_chunks):
        """Verify search behavior unchanged when no filters provided."""
        lexical_service.index_document_chunks(
            document_id="doc-1",
            original_filename="test.txt",
            chunks=sample_chunks,
        )

        results = lexical_service.search(query="programming", limit=10)

        assert len(results) == 2
        assert "score" in results[0]
        assert "text" in results[0]


class TestSQLInjectionProtection:
    """Tests to verify parameterized queries prevent SQL injection."""

    def test_filter_values_are_parameterized(self, lexical_service, sample_chunks):
        """Verify filter values use parameterized queries."""
        # Try to inject SQL via filter value
        lexical_service.index_document_chunks(
            document_id="doc-1",
            original_filename="test.txt",
            chunks=[sample_chunks[0]],
            metadata={
                "service_name": "service-1",
                "tenant_id": "tenant-1",
                "collection": "docs",
                "source_type": "pdf",
                "source_label": "doc1.pdf",
            },
        )

        # Attempt SQL injection
        results = lexical_service.search(
            query="programming",
            limit=10,
            filters={"service_name": "service-1' OR '1'='1"},
        )

        # Should return 0 results (injection attempt fails)
        assert len(results) == 0

    def test_collection_list_values_are_parameterized(self, lexical_service, sample_chunks):
        """Verify collection list values are parameterized."""
        lexical_service.index_document_chunks(
            document_id="doc-1",
            original_filename="test.txt",
            chunks=[sample_chunks[0]],
            metadata={
                "service_name": "service-1",
                "tenant_id": "tenant-1",
                "collection": "docs",
                "source_type": "pdf",
                "source_label": "doc1.pdf",
            },
        )

        # Attempt SQL injection via list
        results = lexical_service.search(
            query="programming",
            limit=10,
            filters={"collections": ["docs", "' OR '1'='1"]},
        )

        # Should only return legitimate matches
        assert len(results) == 1
        assert results[0]["collection"] == "docs"
