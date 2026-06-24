from unittest.mock import Mock

import pytest

from app.delete.contracts import DeleteRequest
from app.delete.use_case import DeleteUseCase


@pytest.fixture
def vector_store():
    mock = Mock()
    mock.delete_by_scope.return_value = 5
    return mock


@pytest.fixture
def lexical_search():
    mock = Mock()
    mock.delete_by_scope.return_value = 5
    return mock


@pytest.fixture
def use_case(vector_store, lexical_search):
    return DeleteUseCase(
        vector_store_service=vector_store,
        lexical_search_service=lexical_search,
        profile_store=None,
    )


class TestDeleteUseCase:
    def test_execute_deletes_from_both_backends(self, use_case, vector_store, lexical_search):
        request = DeleteRequest(
            service_name="svc",
            tenant_id="t1",
            collections=["docs"],
        )
        result = use_case.execute(request)

        assert result.deleted_count == 5
        vector_store.delete_by_scope.assert_called_once()
        lexical_search.delete_by_scope.assert_called_once()

    def test_execute_passes_scope_to_vector_store(self, use_case, vector_store):
        request = DeleteRequest(
            service_name="svc",
            tenant_id="t1",
            collections=["docs", "faq"],
        )
        use_case.execute(request)

        scope = vector_store.delete_by_scope.call_args[0][0]
        assert scope.service_name == "svc"
        assert scope.tenant_id == "t1"
        assert scope.collections == ["docs", "faq"]

    def test_execute_passes_filters_to_lexical(self, use_case, lexical_search):
        request = DeleteRequest(
            service_name="svc",
            tenant_id="t1",
            collections=["docs"],
            filters={"document_id": "abc"},
        )
        use_case.execute(request)

        filters = lexical_search.delete_by_scope.call_args[0][0]
        assert filters["service_name"] == "svc"
        assert filters["tenant_id"] == "t1"
        assert filters["collections"] == ["docs"]
        assert filters["document_id"] == "abc"

    def test_execute_with_zero_count(self, vector_store, lexical_search):
        vector_store.delete_by_scope.return_value = 0
        lexical_search.delete_by_scope.return_value = 0
        use_case = DeleteUseCase(
            vector_store_service=vector_store,
            lexical_search_service=lexical_search,
        )
        result = use_case.execute(
            DeleteRequest(service_name="svc", tenant_id="t1", collections=["empty"])
        )
        assert result.deleted_count == 0
