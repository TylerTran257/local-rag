from pathlib import Path

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from app.retrieval import RetrievalScope
from app.services.vector_store_service import VectorStoreService


def make_embedding(index: int) -> list[float]:
    return [1.0 if position == index else 0.0 for position in range(384)]


def make_metadata(**overrides):
    metadata = {
        "service_name": "svc",
        "tenant_id": "tenant-123",
        "collection": "documents",
        "source_type": "document",
        "source_label": "guide.pdf",
        "topic": "rag",
    }
    metadata.update(overrides)
    return metadata


def make_service(tmp_path: Path) -> VectorStoreService:
    return VectorStoreService(
        qdrant_path=str(tmp_path / "qdrant"),
        collection_name="test_chunks",
    )


def test_build_query_filter_translates_scope_and_filters(tmp_path):
    service = make_service(tmp_path)
    scope = RetrievalScope(
        service_name="svc",
        tenant_id="tenant-123",
        collections=["documents", "notes"],
        filters={"topic": "rag", "version": [1, 2]},
    )

    query_filter = service.build_query_filter(scope)

    must_conditions = query_filter.must
    assert isinstance(must_conditions, list)
    assert len(must_conditions) == 5

    service_name_condition = must_conditions[0]
    assert isinstance(service_name_condition, FieldCondition)
    assert service_name_condition.key == "service_name"
    assert isinstance(service_name_condition.match, MatchValue)
    assert service_name_condition.match.value == "svc"

    collection_condition = must_conditions[2]
    assert isinstance(collection_condition, FieldCondition)
    assert collection_condition.key == "collection"
    assert isinstance(collection_condition.match, MatchAny)
    assert collection_condition.match.any == ["documents", "notes"]

    primitive_filter_condition = must_conditions[3]
    assert isinstance(primitive_filter_condition, FieldCondition)
    assert primitive_filter_condition.key == "topic"
    assert isinstance(primitive_filter_condition.match, MatchValue)
    assert primitive_filter_condition.match.value == "rag"

    list_filter_condition = must_conditions[4]
    assert isinstance(list_filter_condition, FieldCondition)
    assert list_filter_condition.key == "version"
    assert isinstance(list_filter_condition.match, MatchAny)
    assert list_filter_condition.match.any == [1, 2]


def test_build_query_filter_supports_bool_and_float_constraints(tmp_path):
    service = make_service(tmp_path)
    scope = RetrievalScope(
        service_name="svc",
        tenant_id="tenant-123",
        collections=["documents"],
        filters={"published": [True, False], "score": 0.75},
    )

    query_filter = service.build_query_filter(scope)

    must_conditions = query_filter.must
    assert isinstance(must_conditions, list)
    assert len(must_conditions) == 5

    bool_list_condition = must_conditions[3]
    assert isinstance(bool_list_condition, Filter)
    assert isinstance(bool_list_condition.should, list)
    assert len(bool_list_condition.should) == 2

    float_condition = must_conditions[4]
    assert isinstance(float_condition, FieldCondition)
    assert float_condition.key == "score"
    assert float_condition.range is not None
    assert float_condition.range.gte == 0.75
    assert float_condition.range.lte == 0.75


def test_upsert_and_search_round_trip_with_metadata(tmp_path):
    service = make_service(tmp_path)
    service.upsert_document_chunks(
        document_id="doc-1",
        original_filename="guide.pdf",
        chunks=["chunk one"],
        embeddings=[make_embedding(0)],
        metadata=[make_metadata(topic="rag", audience="search")],
    )

    results = service.search(make_embedding(0), limit=1)

    assert len(results) == 1
    result = results[0]
    assert result["document_id"] == "doc-1"
    assert result["original_filename"] == "guide.pdf"
    assert result["service_name"] == "svc"
    assert result["tenant_id"] == "tenant-123"
    assert result["collection"] == "documents"
    assert result["source_type"] == "document"
    assert result["source_label"] == "guide.pdf"
    assert result["topic"] == "rag"
    assert result["audience"] == "search"


def test_domain_metadata_cannot_override_stored_chunk_fields(tmp_path):
    service = make_service(tmp_path)
    service.upsert_document_chunks(
        document_id="doc-1",
        original_filename="guide.pdf",
        chunks=["chunk one"],
        embeddings=[make_embedding(0)],
        metadata=[
            make_metadata(
                document_id="bad-doc",
                original_filename="bad.pdf",
                chunk_index=999,
                text="bad text",
            )
        ],
    )

    results = service.search(make_embedding(0), limit=1)

    assert len(results) == 1
    result = results[0]
    assert result["document_id"] == "doc-1"
    assert result["original_filename"] == "guide.pdf"
    assert result["chunk_index"] == 0
    assert result["text"] == "chunk one"


def test_filtered_search_returns_only_matching_chunks(tmp_path):
    service = make_service(tmp_path)
    service.upsert_document_chunks(
        document_id="doc-1",
        original_filename="guide.pdf",
        chunks=["chunk one", "chunk two", "chunk three"],
        embeddings=[make_embedding(0), make_embedding(1), make_embedding(2)],
        metadata=[
            make_metadata(collection="documents", topic="rag"),
            make_metadata(collection="notes", topic="rag"),
            make_metadata(collection="documents", topic="ops"),
        ],
    )

    query_filter = service.build_query_filter(
        RetrievalScope(
            service_name="svc",
            tenant_id="tenant-123",
            collections=["documents"],
            filters={"topic": "rag"},
        )
    )

    results = service.search(make_embedding(0), limit=10, query_filter=query_filter)

    assert len(results) == 1
    assert results[0]["collection"] == "documents"
    assert results[0]["topic"] == "rag"
    assert results[0]["text"] == "chunk one"


def test_upsert_document_chunks_without_metadata_preserves_backward_compatibility(tmp_path):
    service = make_service(tmp_path)
    service.upsert_document_chunks(
        document_id="doc-1",
        original_filename="guide.pdf",
        chunks=["chunk one"],
        embeddings=[make_embedding(0)],
    )

    results = service.search(make_embedding(0), limit=1)

    assert len(results) == 1
    assert results[0] == {
        "document_id": "doc-1",
        "original_filename": "guide.pdf",
        "chunk_index": 0,
        "score": results[0]["score"],
        "text": "chunk one",
    }
