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


def test_chat_websocket_streams_answer_and_sources(
    client, fake_document_service, fake_generation_service
):
    fake_document_service.contexts = CONTEXTS
    fake_generation_service.streamed_tokens = [
        "Retrieval ",
        "augmented generation combines retrieval with generation.",
    ]

    with client.websocket_connect("/ws/chat") as websocket:
        websocket.send_json(
            {"query": "what is retrieval augmented generation", "limit": 3}
        )

        retrieving_event = websocket.receive_json()
        generating_event = websocket.receive_json()
        first_token_event = websocket.receive_json()
        second_token_event = websocket.receive_json()
        done_event = websocket.receive_json()

    assert retrieving_event == {
        "type": "status",
        "message": "retrieving context",
    }

    assert generating_event == {
        "type": "status",
        "message": "generating answer",
    }

    assert first_token_event == {
        "type": "token",
        "value": "Retrieval ",
    }

    assert second_token_event == {
        "type": "token",
        "value": "augmented generation combines retrieval with generation.",
    }

    assert done_event == {
        "type": "done",
        "answer": "Retrieval augmented generation combines retrieval with generation.",
        "sources": CONTEXTS,
        "citations": CITATIONS,
    }

    assert fake_generation_service.calls == [
        ("stream_answer_question", "what is retrieval augmented generation", CONTEXTS)
    ]


def test_chat_websocket_returns_done_when_no_context_found(
    client, fake_document_service, fake_generation_service
):
    fake_document_service.contexts = []

    with client.websocket_connect("/ws/chat") as websocket:
        websocket.send_json(
            {"query": "what is retrieval augmented generation", "limit": 3}
        )

        retrieving_event = websocket.receive_json()
        done_event = websocket.receive_json()

    assert retrieving_event == {
        "type": "status",
        "message": "retrieving context",
    }

    assert done_event == {
        "type": "done",
        "answer": "",
        "sources": [],
        "citations": [],
    }

    assert fake_document_service.calls == [
        ("retrieve_context", "what is retrieval augmented generation", 3)
    ]

    assert fake_generation_service.calls == []


def test_chat_websocket_returns_error_when_streaming_generation_fails(
    client, fake_document_service, fake_generation_service
):
    fake_document_service.contexts = CONTEXTS
    fake_generation_service.stream_error = GenerationServiceError("Some errors")

    with client.websocket_connect("/ws/chat") as websocket:
        websocket.send_json(
            {"query": "what is retrieval augmented generation", "limit": 3}
        )

        retrieving_event = websocket.receive_json()
        generating_event = websocket.receive_json()
        error_event = websocket.receive_json()

    assert retrieving_event == {
        "type": "status",
        "message": "retrieving context",
    }

    assert generating_event == {
        "type": "status",
        "message": "generating answer",
    }

    assert error_event == {
        "type": "error",
        "message": "Some errors",
    }

    assert fake_generation_service.calls == [
        ("stream_answer_question", "what is retrieval augmented generation", CONTEXTS)
    ]
