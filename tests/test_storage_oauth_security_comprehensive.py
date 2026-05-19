"""
Comprehensive tests for storage.py, oauth.py, and security.py.
"""

import pytest

from flamehaven_filesearch.storage import (
    MemoryMetadataStore,
    create_metadata_store,
)
from flamehaven_filesearch.oauth import (
    OAuthTokenInfo,
    _normalize_list,
    is_jwt_format,
    oauth_permissions,
    validate_oauth_token,
)
from flamehaven_filesearch.security import (
    RequestContext,
    REQUEST_CONTEXT_KEY,
    _store_request_context,
    _oauth_to_api_key_info,
)
from flamehaven_filesearch.config import Config


# ===========================================================================
# storage.py — MemoryMetadataStore
# ===========================================================================


class TestMemoryMetadataStore:
    def test_ensure_store_creates(self):
        store = MemoryMetadataStore()
        store.ensure_store("s1")
        assert "s1" in store._stores

    def test_add_doc(self):
        store = MemoryMetadataStore()
        store.add_doc("s1", {"title": "doc"})
        docs = store.get_docs("s1")
        assert len(docs) == 1
        assert docs[0]["title"] == "doc"

    def test_get_docs_empty_store(self):
        store = MemoryMetadataStore()
        docs = store.get_docs("nonexistent")
        assert docs == []

    def test_get_docs_returns_copy(self):
        store = MemoryMetadataStore()
        store.add_doc("s1", {"title": "doc"})
        docs1 = store.get_docs("s1")
        docs2 = store.get_docs("s1")
        assert docs1 == docs2

    def test_list_store_names(self):
        store = MemoryMetadataStore()
        store.ensure_store("a")
        store.ensure_store("b")
        names = store.list_store_names()
        assert "a" in names
        assert "b" in names

    def test_delete_store(self):
        store = MemoryMetadataStore()
        store.ensure_store("todel")
        store.add_doc("todel", {"title": "x"})
        store.delete_store("todel")
        assert store.get_docs("todel") == []
        assert "todel" not in store.list_store_names()

    def test_delete_nonexistent_noop(self):
        store = MemoryMetadataStore()
        store.delete_store("doesnt_exist")  # should not raise

    def test_backing_dict_shared(self):
        backing = {}
        store = MemoryMetadataStore(backing=backing)
        store.ensure_store("x")
        assert "x" in backing

    def test_add_doc_without_ensure(self):
        store = MemoryMetadataStore()
        store.add_doc("newstore", {"title": "auto"})
        assert store.get_docs("newstore")[0]["title"] == "auto"

    def test_multiple_docs(self):
        store = MemoryMetadataStore()
        for i in range(5):
            store.add_doc("multi", {"idx": i})
        assert len(store.get_docs("multi")) == 5


class TestCreateMetadataStore:
    def test_returns_memory_store_by_default(self):
        cfg = Config.__new__(Config)
        cfg.postgres_enabled = False
        cfg.postgres_dsn = None
        cfg.postgres_schema = "public"
        result = create_metadata_store(cfg)
        assert isinstance(result, MemoryMetadataStore)

    def test_postgres_enabled_without_dsn_raises(self):
        cfg = Config.__new__(Config)
        cfg.postgres_enabled = True
        cfg.postgres_dsn = None
        cfg.postgres_schema = "public"
        with pytest.raises(RuntimeError, match="POSTGRES_DSN"):
            create_metadata_store(cfg)


# ===========================================================================
# oauth.py
# ===========================================================================


class TestIsJwtFormat:
    def test_valid_jwt_format(self):
        assert is_jwt_format("header.payload.signature") is True

    def test_not_jwt_one_dot(self):
        assert is_jwt_format("header.payload") is False

    def test_not_jwt_no_dots(self):
        assert is_jwt_format("plaintoken") is False

    def test_three_dots(self):
        # 3 dots means count(".") == 3, not 2
        assert is_jwt_format("a.b.c.d") is False


class TestNormalizeList:
    def test_none_returns_empty(self):
        assert _normalize_list(None) == []

    def test_string_split_by_space(self):
        result = _normalize_list("read write")
        assert "read" in result
        assert "write" in result

    def test_string_split_by_comma(self):
        result = _normalize_list("read,write")
        assert "read" in result
        assert "write" in result

    def test_list_passthrough(self):
        result = _normalize_list(["a", "b"])
        assert result == ["a", "b"]

    def test_list_converts_to_str(self):
        result = _normalize_list([1, 2])
        assert result == ["1", "2"]

    def test_other_type_returns_empty(self):
        result = _normalize_list(42)
        assert result == []


class TestOAuthPermissions:
    def _make_info(self, scopes=None, roles=None):
        return OAuthTokenInfo(
            subject="user1",
            roles=roles or [],
            scopes=scopes or [],
            issuer=None,
            audience=None,
            claims={},
        )

    def test_scope_mapped(self):
        info = self._make_info(scopes=["filesearch:search"])
        cfg = Config.__new__(Config)
        cfg.oauth_required_roles = []
        perms = oauth_permissions(info, cfg)
        assert "search" in perms

    def test_upload_scope(self):
        info = self._make_info(scopes=["filesearch:upload"])
        cfg = Config.__new__(Config)
        cfg.oauth_required_roles = []
        perms = oauth_permissions(info, cfg)
        assert "upload" in perms

    def test_admin_role(self):
        info = self._make_info(roles=["admin"])
        cfg = Config.__new__(Config)
        cfg.oauth_required_roles = []
        perms = oauth_permissions(info, cfg)
        assert "admin" in perms

    def test_filesearch_admin_role(self):
        info = self._make_info(roles=["filesearch-admin"])
        cfg = Config.__new__(Config)
        cfg.oauth_required_roles = []
        perms = oauth_permissions(info, cfg)
        assert "admin" in perms

    def test_required_role_grants_admin(self):
        info = self._make_info(roles=["special-role"])
        cfg = Config.__new__(Config)
        cfg.oauth_required_roles = ["special-role"]
        perms = oauth_permissions(info, cfg)
        assert "admin" in perms

    def test_no_permissions_empty(self):
        info = self._make_info()
        cfg = Config.__new__(Config)
        cfg.oauth_required_roles = []
        perms = oauth_permissions(info, cfg)
        assert perms == []

    def test_deduplication(self):
        info = self._make_info(scopes=["filesearch:admin", "admin"], roles=["admin"])
        cfg = Config.__new__(Config)
        cfg.oauth_required_roles = []
        perms = oauth_permissions(info, cfg)
        assert perms.count("admin") == 1

    def test_bare_scope_names(self):
        info = self._make_info(scopes=["search", "upload"])
        cfg = Config.__new__(Config)
        cfg.oauth_required_roles = []
        perms = oauth_permissions(info, cfg)
        assert "search" in perms
        assert "upload" in perms


class TestValidateOAuthToken:
    def test_oauth_disabled_returns_none(self):
        cfg = Config.__new__(Config)
        cfg.oauth_enabled = False
        cfg.oauth_jwt_secret = None
        cfg.oauth_jwks_url = None
        result = validate_oauth_token("header.payload.sig", config=cfg)
        assert result is None

    def test_empty_token_returns_none(self):
        cfg = Config.__new__(Config)
        cfg.oauth_enabled = True
        cfg.oauth_jwt_secret = "secret"
        cfg.oauth_audience = None
        cfg.oauth_issuer = None
        result = validate_oauth_token("", config=cfg)
        assert result is None

    def test_non_jwt_format_returns_none(self):
        cfg = Config.__new__(Config)
        cfg.oauth_enabled = True
        cfg.oauth_jwt_secret = "secret"
        cfg.oauth_audience = None
        cfg.oauth_issuer = None
        result = validate_oauth_token("plaintoken", config=cfg)
        assert result is None

    def test_no_secret_no_jwks_returns_none(self):
        cfg = Config.__new__(Config)
        cfg.oauth_enabled = True
        cfg.oauth_jwt_secret = None
        cfg.oauth_jwks_url = None
        cfg.oauth_audience = None
        cfg.oauth_issuer = None
        result = validate_oauth_token("a.b.c", config=cfg)
        assert result is None

    def test_invalid_token_with_secret_returns_none(self):
        cfg = Config.__new__(Config)
        cfg.oauth_enabled = True
        cfg.oauth_jwt_secret = "mysecret"
        cfg.oauth_audience = None
        cfg.oauth_issuer = None
        result = validate_oauth_token("bad.token.value", config=cfg)
        assert result is None

    def test_valid_hs256_token(self):
        import jwt as pyjwt

        secret = "testsecret"
        payload = {"sub": "user123", "scope": "search upload"}
        token = pyjwt.encode(payload, secret, algorithm="HS256")

        cfg = Config.__new__(Config)
        cfg.oauth_enabled = True
        cfg.oauth_jwt_secret = secret
        cfg.oauth_audience = None
        cfg.oauth_issuer = None
        result = validate_oauth_token(token, config=cfg)
        assert result is not None
        assert result.subject == "user123"
        assert "search" in result.scopes

    def test_valid_hs256_token_with_roles(self):
        import jwt as pyjwt

        secret = "testsecret2"
        payload = {"sub": "admin_user", "roles": ["admin"], "groups": ["devs"]}
        token = pyjwt.encode(payload, secret, algorithm="HS256")

        cfg = Config.__new__(Config)
        cfg.oauth_enabled = True
        cfg.oauth_jwt_secret = secret
        cfg.oauth_audience = None
        cfg.oauth_issuer = None
        result = validate_oauth_token(token, config=cfg)
        assert result is not None
        assert "admin" in result.roles

    def test_subject_fallbacks(self):
        import jwt as pyjwt

        secret = "fallback_secret"
        payload = {"preferred_username": "fallback_user"}
        token = pyjwt.encode(payload, secret, algorithm="HS256")

        cfg = Config.__new__(Config)
        cfg.oauth_enabled = True
        cfg.oauth_jwt_secret = secret
        cfg.oauth_audience = None
        cfg.oauth_issuer = None
        result = validate_oauth_token(token, config=cfg)
        assert result is not None
        assert result.subject == "fallback_user"


# ===========================================================================
# security.py
# ===========================================================================


class TestRequestContext:
    def test_has_permission_true(self):
        ctx = RequestContext(
            api_key_id="k1",
            user_id="u1",
            key_name="key",
            permissions=["search", "upload"],
            rate_limit=100,
        )
        assert ctx.has_permission("search") is True

    def test_has_permission_false(self):
        ctx = RequestContext(
            api_key_id="k1",
            user_id="u1",
            key_name="key",
            permissions=["search"],
            rate_limit=100,
        )
        assert ctx.has_permission("admin") is False

    def test_auth_type_default(self):
        ctx = RequestContext("k", "u", "n", [], 60)
        assert ctx.auth_type == "api_key"

    def test_auth_type_custom(self):
        ctx = RequestContext("k", "u", "n", [], 60, auth_type="oauth")
        assert ctx.auth_type == "oauth"


class TestOAuthToApiKeyInfo:
    def test_builds_api_key_info(self):
        info = OAuthTokenInfo(
            subject="user_oauth",
            roles=["admin"],
            scopes=["filesearch:search"],
            issuer=None,
            audience=None,
            claims={},
        )
        cfg = Config.__new__(Config)
        cfg.oauth_required_roles = []

        result = _oauth_to_api_key_info(info, cfg)
        assert result.id == "oauth:user_oauth"
        assert result.user_id == "user_oauth"
        assert result.is_active is True

    def test_permissions_derived_from_scopes(self):
        info = OAuthTokenInfo(
            subject="u",
            roles=[],
            scopes=["filesearch:search", "filesearch:upload"],
            issuer=None,
            audience=None,
            claims={},
        )
        cfg = Config.__new__(Config)
        cfg.oauth_required_roles = []

        result = _oauth_to_api_key_info(info, cfg)
        assert "search" in result.permissions
        assert "upload" in result.permissions
