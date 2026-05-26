def test_upload_async_returns_202_and_queued_job(
    client, fake_document_service, fake_generation_service
):
    response = client.post(
        "/upload_async",
        files={"file": ("python_rag_intro.txt", b"hello world", "text/plain")},
    )

    assert response.status_code == 202

    assert response.json() == {
        "job_id": "job-123",
        "job_type": "document_index",
        "document_id": "doc-123",
        "status": "queued",
        "error_message": None,
        "created_at": "2026-05-23T12:00:00",
        "started_at": None,
        "finished_at": None,
    }

    assert fake_document_service.calls == [
        ("create_document", "python_rag_intro.txt"),
        ("create_job", "doc-123"),
        ("run_indexing_pipeline", "doc-123", "job-123"),
        ("mark_job_running", "job-123"),
        ("extract_text", "doc-123"),
        ("chunk_document", "doc-123"),
        ("embed_document", "doc-123"),
        ("mark_job_completed", "job-123"),
    ]
