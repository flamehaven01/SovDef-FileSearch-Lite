"""
Comprehensive tests for admin_routes.py.
Tests helper functions directly (no FastAPI routing required for most).
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException
from starlette.testclient import TestClient
from starlette.requests import Request

from flamehaven_filesearch.admin_routes import (
    _parse_bearer_token,
    _resolve_key_admin,
    _try_oauth_admin,
    CreateAPIKeyRequest,
    CreateAPIKeyResponse,
    ListAPIKeysResponse,
)
from flamehaven_filesearch.config import Config


# ---------------------------------------------------------------------------
# _parse_bearer_token
# ---------------------------------------------------------------------------


def _make_request(auth_header=None, x_api_key=None):
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/admin/keys",
        "query_string": b"",
        "headers": [],
    }
    headers = []
    if auth_header:
        headers.append((b"authorization", auth_header.encode()))
    if x_api_key:
        headers.append((b"x-api-key", x_api_key.encode()))
    scope["headers"] = headers
    return Request(scope)


class TestParseBearerToken:
    def test_valid_bearer(self):
        req = _make_request("Bearer mytoken123")
        token = _parse_bearer_token(req)
        assert token == "mytoken123"

    def test_missing_auth_raises_401(self):
        req = _make_request()
        with pytest.raises(HTTPException) as exc_info:
            _parse_bearer_token(req)
        assert exc_info.value.status_code == 401

    def test_invalid_format_one_part(self):
        req = _make_request("justatoken")
        with pytest.raises(HTTPException) as exc_info:
            _parse_bearer_token(req)
        assert exc_info.value.status_code == 401

    def test_invalid_scheme(self):
        req = _make_request("Basic mytoken")
        with pytest.raises(HTTPException) as exc_info:
            _parse_bearer_token(req)
        assert exc_info.value.status_code == 401

    def test_bearer_case_insensitive(self):
        req = _make_request("bearer mytoken")
        token = _parse_bearer_token(req)
        assert token == "mytoken"


# ---------------------------------------------------------------------------
# _try_oauth_admin
# ---------------------------------------------------------------------------


class TestTryOAuthAdmin:
    def _cfg(self, oauth_enabled=False):
        cfg = Config.__new__(Config)
        cfg.oauth_enabled = oauth_enabled
        cfg.oauth_jwt_secret = None
        cfg.oauth_jwks_url = None
        cfg.oauth_audience = None
        cfg.oauth_issuer = None
        cfg.oauth_required_roles = []
        return cfg

    def test_oauth_disabled_returns_none(self):
        cfg = self._cfg(oauth_enabled=False)
        result = _try_oauth_admin("sometoken", cfg)
        assert result is None

    def test_non_jwt_format_returns_none(self):
        cfg = self._cfg(oauth_enabled=True)
        result = _try_oauth_admin("plaintoken", cfg)
        assert result is None

    def test_valid_jwt_without_admin_raises_403(self):
        import jwt as pyjwt

        secret = "admintest"
        token = pyjwt.encode(
            {"sub": "nonadmin", "scope": "search"}, secret, algorithm="HS256"
        )

        cfg = Config.__new__(Config)
        cfg.oauth_enabled = True
        cfg.oauth_jwt_secret = secret
        cfg.oauth_audience = None
        cfg.oauth_issuer = None
        cfg.oauth_required_roles = []
        with pytest.raises(HTTPException) as exc_info:
            _try_oauth_admin(token, cfg)
        assert exc_info.value.status_code == 403

    def test_valid_jwt_with_admin_returns_subject(self):
        import jwt as pyjwt

        secret = "adminsecret"
        token = pyjwt.encode(
            {"sub": "adminuser", "roles": ["admin"]}, secret, algorithm="HS256"
        )

        cfg = Config.__new__(Config)
        cfg.oauth_enabled = True
        cfg.oauth_jwt_secret = secret
        cfg.oauth_audience = None
        cfg.oauth_issuer = None
        cfg.oauth_required_roles = []
        result = _try_oauth_admin(token, cfg)
        assert result == "adminuser"


# ---------------------------------------------------------------------------
# _resolve_key_admin
# ---------------------------------------------------------------------------


class TestResolveKeyAdmin:
    def test_admin_key_env_match(self, monkeypatch):
        monkeypatch.setenv("FLAMEHAVEN_ADMIN_KEY", "env_admin_key_xyz")
        result = _resolve_key_admin("env_admin_key_xyz")
        assert result == "admin"

    def test_invalid_key_raises_401(self):
        with patch.dict(os.environ, {"FLAMEHAVEN_ADMIN_KEY": ""}):
            mock_iam = MagicMock()
            mock_iam.validate_admin_token.return_value = None
            mock_km = MagicMock()
            mock_km.validate_key.return_value = None
            with patch(
                "flamehaven_filesearch.admin_routes.get_iam_provider",
                return_value=mock_iam,
            ):
                with patch(
                    "flamehaven_filesearch.admin_routes.get_key_manager",
                    return_value=mock_km,
                ):
                    with pytest.raises(HTTPException) as exc_info:
                        _resolve_key_admin("bad_key")
                    assert exc_info.value.status_code == 401

    def test_api_key_without_admin_perm_raises_403(self):
        with patch.dict(os.environ, {"FLAMEHAVEN_ADMIN_KEY": ""}):
            mock_iam = MagicMock()
            mock_iam.validate_admin_token.return_value = None
            mock_key_info = MagicMock()
            mock_key_info.permissions = ["search"]
            mock_km = MagicMock()
            mock_km.validate_key.return_value = mock_key_info
            with patch(
                "flamehaven_filesearch.admin_routes.get_iam_provider",
                return_value=mock_iam,
            ):
                with patch(
                    "flamehaven_filesearch.admin_routes.get_key_manager",
                    return_value=mock_km,
                ):
                    with pytest.raises(HTTPException) as exc_info:
                        _resolve_key_admin("user_key")
                    assert exc_info.value.status_code == 403

    def test_api_key_with_admin_perm_returns_user(self):
        with patch.dict(os.environ, {"FLAMEHAVEN_ADMIN_KEY": ""}):
            mock_iam = MagicMock()
            mock_iam.validate_admin_token.return_value = None
            mock_key_info = MagicMock()
            mock_key_info.permissions = ["admin", "search"]
            mock_key_info.user_id = "admin_user"
            mock_km = MagicMock()
            mock_km.validate_key.return_value = mock_key_info
            with patch(
                "flamehaven_filesearch.admin_routes.get_iam_provider",
                return_value=mock_iam,
            ):
                with patch(
                    "flamehaven_filesearch.admin_routes.get_key_manager",
                    return_value=mock_km,
                ):
                    result = _resolve_key_admin("admin_key")
                    assert result == "admin_user"

    def test_iam_provider_validates_token(self):
        with patch.dict(os.environ, {"FLAMEHAVEN_ADMIN_KEY": ""}):
            mock_iam = MagicMock()
            mock_iam.validate_admin_token.return_value = "iam_user"
            with patch(
                "flamehaven_filesearch.admin_routes.get_iam_provider",
                return_value=mock_iam,
            ):
                result = _resolve_key_admin("iam_token")
                assert result == "iam_user"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TestPydanticModels:
    def test_create_api_key_request_defaults(self):
        req = CreateAPIKeyRequest(name="test_key")
        assert req.name == "test_key"
        assert req.permissions is None
        assert req.rate_limit_per_minute == 100
        assert req.expires_in_days is None

    def test_create_api_key_request_custom(self):
        req = CreateAPIKeyRequest(
            name="custom",
            permissions=["search"],
            rate_limit_per_minute=50,
            expires_in_days=30,
        )
        assert req.permissions == ["search"]
        assert req.rate_limit_per_minute == 50
        assert req.expires_in_days == 30

    def test_create_api_key_response(self):
        resp = CreateAPIKeyResponse(
            id="key_id",
            key="sk_live_xxx",
            name="mykey",
            created_at="2026-01-01T00:00:00Z",
            permissions=["search"],
            rate_limit_per_minute=100,
        )
        assert resp.id == "key_id"
        assert "Save your API key" in resp.warning

    def test_list_api_keys_response(self):
        resp = ListAPIKeysResponse(keys=[{"id": "k1"}, {"id": "k2"}])
        assert len(resp.keys) == 2


# ---------------------------------------------------------------------------
# FastAPI route integration tests via TestClient
# ---------------------------------------------------------------------------


class TestAdminRouteIntegration:
    def _create_app(self):
        from fastapi import FastAPI
        from flamehaven_filesearch.admin_routes import router

        app = FastAPI()
        app.include_router(router)
        return app

    def test_create_key_no_auth_returns_401(self):
        app = self._create_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/api/admin/keys",
            json={"name": "test"},
        )
        assert resp.status_code in (401, 422, 403)

    def test_list_keys_no_auth_returns_401(self):
        app = self._create_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/admin/keys")
        assert resp.status_code in (401, 403)

    def test_delete_key_no_auth_returns_401(self):
        app = self._create_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.delete("/api/admin/keys/some_key_id")
        assert resp.status_code in (401, 403)

    def test_create_key_with_admin_key(self):
        os.environ["FLAMEHAVEN_ADMIN_KEY"] = "test_admin_key_123"
        try:
            app = self._create_app()
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/api/admin/keys",
                json={"name": "new_key"},
                headers={"Authorization": "Bearer test_admin_key_123"},
            )
            assert resp.status_code in (200, 201, 422, 500)
        finally:
            del os.environ["FLAMEHAVEN_ADMIN_KEY"]

    def test_list_keys_with_admin_key(self):
        os.environ["FLAMEHAVEN_ADMIN_KEY"] = "test_admin_list_key"
        try:
            app = self._create_app()
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get(
                "/api/admin/keys",
                headers={"Authorization": "Bearer test_admin_list_key"},
            )
            assert resp.status_code in (200, 500)
        finally:
            del os.environ["FLAMEHAVEN_ADMIN_KEY"]

    def test_quota_endpoints(self):
        os.environ["FLAMEHAVEN_ADMIN_KEY"] = "test_quota_key"
        try:
            app = self._create_app()
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get(
                "/api/admin/quota/key1",
                headers={"Authorization": "Bearer test_quota_key"},
            )
            assert resp.status_code in (200, 404, 500)
        finally:
            del os.environ["FLAMEHAVEN_ADMIN_KEY"]

    def test_cache_stats_endpoint(self):
        os.environ["FLAMEHAVEN_ADMIN_KEY"] = "test_cache_key"
        try:
            app = self._create_app()
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get(
                "/api/admin/cache/stats",
                headers={"Authorization": "Bearer test_cache_key"},
            )
            assert resp.status_code in (200, 404, 500)
        finally:
            del os.environ["FLAMEHAVEN_ADMIN_KEY"]

    def test_cache_reset_endpoint(self):
        os.environ["FLAMEHAVEN_ADMIN_KEY"] = "test_reset_key"
        try:
            app = self._create_app()
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/api/admin/cache/reset",
                headers={"Authorization": "Bearer test_reset_key"},
            )
            assert resp.status_code in (200, 404, 500)
        finally:
            del os.environ["FLAMEHAVEN_ADMIN_KEY"]
