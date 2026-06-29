from pathlib import Path
from unittest.mock import Mock

from app.evals.workspace import (
    EVAL_COLLECTION,
    EVAL_SERVICE,
    EVAL_TENANT,
    EvalRig,
)
from app.retrieval.types import RetrievalMode, RetrievedChunk
from app.retrieval.use_case import RetrieveResult


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="doc-a.txt:0",
        document_id="d1",
        content="hello world",
        score=0.91,
        rank=0,
        retrieval_mode=RetrievalMode.HYBRID,
        metadata={
            "service_name": EVAL_SERVICE,
            "tenant_id": EVAL_TENANT,
            "collection": EVAL_COLLECTION,
            "source_type": "document",
            "source_label": "doc-a.txt",
            "document_id": "d1",
            "chunk_index": 0,
        },
    )


def _rig(*, ingest=None, retrieve=None, extractor=None) -> EvalRig:
    return EvalRig(
        ingest_use_case=ingest or Mock(),
        retrieve_use_case=retrieve or Mock(),
        text_extractor=extractor or Mock(),
    )


class TestEvalRigRetrieve:
    def test_maps_chunks_to_eval_contexts(self):
        retrieve = Mock()
        retrieve.execute.return_value = RetrieveResult(
            chunks=[_chunk()], warnings=[], trace_id="t"
        )
        rig = _rig(retrieve=retrieve)

        contexts = rig.retrieve("what is x?", limit=3)

        assert contexts == [
            {
                "document_id": "d1",
                "original_filename": "doc-a.txt",
                "chunk_index": 0,
                "score": 0.91,
                "text": "hello world",
            }
        ]

    def test_retrieves_under_eval_scope(self):
        retrieve = Mock()
        retrieve.execute.return_value = RetrieveResult(
            chunks=[], warnings=[], trace_id="t"
        )
        rig = _rig(retrieve=retrieve)

        rig.retrieve("q", limit=3)

        request = retrieve.execute.call_args[0][0]
        assert request.scope.service_name == EVAL_SERVICE
        assert request.scope.tenant_id == EVAL_TENANT
        assert request.scope.collections == [EVAL_COLLECTION]
        assert request.limit == 3


class TestEvalRigIngest:
    def test_extracts_text_and_ingests_under_eval_scope(self):
        extractor = Mock()
        extractor.extract.return_value = "body text"
        ingest = Mock()
        rig = _rig(ingest=ingest, extractor=extractor)

        rig.ingest(Path("/corpus/doc-a.txt"), original_filename="doc-a.txt")

        extractor.extract.assert_called_once_with(Path("/corpus/doc-a.txt"))
        document = ingest.ingest_document.call_args[0][0]
        assert document.text == "body text"
        assert document.source_label == "doc-a.txt"
        assert document.service_name == EVAL_SERVICE
        assert document.tenant_id == EVAL_TENANT
        assert document.collection == EVAL_COLLECTION
