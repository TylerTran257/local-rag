def test_upload_v2_runs_full_pipeline_and_returns_final_document(
    client, fake_document_service, fake_generation_service
):
    response = client.post(
        "/upload_v2",
        files={"file": ("python_rag_intro.txt", b"hello world", "text/plain")},
    )

    assert response.status_code == 200

    assert response.json() == {
        "document_id": "doc-123",
        "original_filename": "python_rag_intro.txt",
        "stored_filename": "doc-123.txt",
        "status": "embedded",
        "size_bytes": 123,
        "extracted_text": "some extracted text",
        "chunks": ["chunk one", "chunk two"],
    }

    assert fake_document_service.calls == [
        ("create_document", "python_rag_intro.txt"),
        ("extract_text", "doc-123"),
        ("chunk_document", "doc-123"),
        ("embed_document", "doc-123"),
        ("get_document", "doc-123"),
    ]
