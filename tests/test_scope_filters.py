from app.retrieval.scope_filters import lexical_filters_for
from app.retrieval.types import RetrievalScope


def _scope(filters=None) -> RetrievalScope:
    return RetrievalScope(
        service_name="svc",
        tenant_id="ten",
        collections=["docs", "faq"],
        filters=filters or {},
    )


class TestLexicalFiltersFor:
    def test_includes_scope_enforcement_keys(self):
        assert lexical_filters_for(_scope()) == {
            "service_name": "svc",
            "tenant_id": "ten",
            "collections": ["docs", "faq"],
        }

    def test_merges_non_reserved_filters(self):
        filters = lexical_filters_for(_scope({"department": "eng", "status": ["a", "b"]}))

        assert filters["department"] == "eng"
        assert filters["status"] == ["a", "b"]
        assert filters["service_name"] == "svc"

    def test_reserved_filter_keys_cannot_shadow_scope(self):
        filters = lexical_filters_for(
            _scope(
                {
                    "service_name": "evil",
                    "tenant_id": "evil",
                    "collections": ["evil"],
                    "collection": "evil",
                }
            )
        )

        assert filters == {
            "service_name": "svc",
            "tenant_id": "ten",
            "collections": ["docs", "faq"],
        }
