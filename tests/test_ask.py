from app.services.generation_service import GenerationServiceError

CITATIONS = [
    {
        "id": 1,
        "document_id": "doc-1",
        "original_filename": "python_rag_intro.txt",
        "chunk_index": 0,
        "score": 0.91,
        "text": "Retrieval augmented generation combines retrieval with generation.",
    }
]

CONTEXTS = [
    {
        "document_id": "doc-1",
        "original_filename": "python_rag_intro.txt",
        "chunk_index": 0,
        "score": 0.91,
        "text": "Retrieval augmented generation combines retrieval with generation.",
    }
]


def test_ask_returns_answer_and_sources(
    client, fake_document_service, fake_generation_service
):
    fake_document_service.contexts = CONTEXTS
    fake_generation_service.answer = (
        "Retrieval augmented generation combines retrieval with generation."
    )

    response = client.post(
        "/ask",
        json={"query": "what is retrieval augmented generation", "limit": 3},
    )

    assert response.status_code == 200

    assert response.json() == {
        "query": "what is retrieval augmented generation",
        "answer": "Retrieval augmented generation combines retrieval with generation.",
        "match_count": 1,
        "sources": CONTEXTS,
        "citations": CITATIONS,
    }


def test_ask_returns_502_when_generation_fails(
    client, fake_document_service, fake_generation_service
):

    fake_document_service.contexts = CONTEXTS
    fake_generation_service.error = GenerationServiceError("Some errors")

    response = client.post(
        "/ask",
        json={"query": "what is retrieval augmented generation", "limit": 3},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "Some errors"}


def test_ask_returns_empty_answer_when_no_context_found(
    client, fake_document_service, fake_generation_service
):
    fake_document_service.contexts = []
    fake_generation_service.error = AssertionError("Generation should not be called")

    response = client.post(
        "/ask",
        json={"query": "what is retrieval augmented generation", "limit": 3},
    )

    assert response.status_code == 200

    assert response.json() == {
        "query": "what is retrieval augmented generation",
        "answer": "",
        "match_count": 0,
        "sources": [],
        "citations": [],
    }
    assert fake_generation_service.calls == []
