from unittest.mock import Mock

from app.profiles import collection_for, default_profile
from app.profiles.models import ServiceProfile
from app.profiles.resolver import ProfileResolver


class TestProfileResolver:
    def test_resolve_uses_store_profile_and_derives_collection(self):
        profile = ServiceProfile(service_name="svc", embedding_model="custom-model")
        store = Mock()
        store.get.return_value = profile
        resolver = ProfileResolver(store)

        resolved = resolver.resolve("svc")

        assert resolved.profile is profile
        assert resolved.collection == collection_for("custom-model")
        store.get.assert_called_once_with("svc")

    def test_resolve_without_store_uses_default_profile(self):
        resolver = ProfileResolver(None)

        resolved = resolver.resolve("svc")

        assert resolved.profile == default_profile("svc")
        assert resolved.collection == collection_for(
            default_profile("svc").embedding_model
        )
