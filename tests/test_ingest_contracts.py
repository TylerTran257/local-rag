import pytest

from app.ingest import (
    IngestBatch,
    IngestChunk,
    IngestDocument,
    MetadataValidationError,
    MetadataValidator,
    REQUIRED_METADATA_FIELDS,
    ValidatedMetadata,
)


def make_valid_metadata(**overrides):
    metadata = {
        "service_name": "svc",
        "tenant_id": "tenant-123",
        "collection": "documents",
        "source_type": "document",
        "source_label": "guide.pdf",
        "topic": "rag",
        "priority": 2,
    }
    metadata.update(overrides)
    return metadata


def test_ingest_document_type_exists():
    document = IngestDocument(
        text="raw text",
        service_name="svc",
        tenant_id="tenant-123",
        collection="documents",
        source_type="document",
        source_label="guide.pdf",
        domain_metadata={"topic": "rag"},
    )

    assert document.text == "raw text"
    assert document.service_name == "svc"
    assert document.domain_metadata == {"topic": "rag"}


def test_ingest_chunk_type_exists():
    chunk = IngestChunk(
        chunk_id="doc-1:0",
        text="chunk text",
        service_name="svc",
        tenant_id="tenant-123",
        collection="documents",
        source_type="document",
        source_label="guide.pdf",
        domain_metadata={"topic": "rag"},
    )

    assert chunk.chunk_id == "doc-1:0"
    assert chunk.text == "chunk text"
    assert chunk.domain_metadata == {"topic": "rag"}


def test_ingest_batch_accepts_documents_and_chunks():
    batch: IngestBatch = [
        IngestDocument(
            text="raw text",
            service_name="svc",
            tenant_id="tenant-123",
            collection="documents",
            source_type="document",
            source_label="guide.pdf",
        ),
        IngestChunk(
            chunk_id="doc-1:0",
            text="chunk text",
            service_name="svc",
            tenant_id="tenant-123",
            collection="documents",
            source_type="document",
            source_label="guide.pdf",
        ),
    ]

    assert len(batch) == 2


@pytest.mark.parametrize("field_name", REQUIRED_METADATA_FIELDS)
def test_metadata_validator_rejects_missing_required_field(field_name):
    metadata = make_valid_metadata()
    metadata.pop(field_name)

    with pytest.raises(MetadataValidationError) as exc_info:
        MetadataValidator.validate(metadata)

    assert exc_info.value.invalid_fields == [field_name]
    assert field_name in str(exc_info.value)


@pytest.mark.parametrize("field_name", REQUIRED_METADATA_FIELDS)
def test_metadata_validator_rejects_empty_required_field(field_name):
    metadata = make_valid_metadata(**{field_name: "   "})

    with pytest.raises(MetadataValidationError) as exc_info:
        MetadataValidator.validate(metadata)

    assert exc_info.value.invalid_fields == [field_name]
    assert field_name in str(exc_info.value)


def test_metadata_validator_accepts_valid_metadata():
    validated = MetadataValidator.validate(make_valid_metadata())

    assert validated == ValidatedMetadata(
        service_name="svc",
        tenant_id="tenant-123",
        collection="documents",
        source_type="document",
        source_label="guide.pdf",
        domain_metadata={"topic": "rag", "priority": 2},
    )


def test_metadata_validator_preserves_optional_domain_metadata():
    validated = MetadataValidator.validate(
        make_valid_metadata(topic="rag", audience={"team": "search"}, enabled=True)
    )

    assert validated.domain_metadata == {
        "topic": "rag",
        "priority": 2,
        "audience": {"team": "search"},
        "enabled": True,
    }
