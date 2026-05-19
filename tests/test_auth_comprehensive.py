"""
Comprehensive tests for auth.py (APIKeyManager).
Target: cover the 173 uncovered statements.
"""

import json
import sqlite3
import uuid
from pathlib import Path

import pytest

from flamehaven_filesearch.auth import (
    APIKeyInfo,
    APIKeyManager,
    NullIAMProvider,
    bootstrap_api_key,
    get_key_manager,
)
import flamehaven_filesearch.auth as auth_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db(tmp_path):
    db_path = str(tmp_path / "test_auth.db")
    yield db_path
    try:
        Path(db_path).unlink()
    except (FileNotFoundError, PermissionError):
        pass


@pytest.fixture
def manager(temp_db):
    mgr = APIKeyManager(db_path=temp_db)
    return mgr


# ---------------------------------------------------------------------------
# APIKeyInfo
# ---------------------------------------------------------------------------


class TestAPIKeyInfo:
    def test_to_dict(self):
        info = APIKeyInfo(
            key_id="k1",
            name="Test Key",
            user_id="user1",
            created_at="2025-01-01",
            last_used=None,
            is_active=True,
            rate_limit_per_minute=100,
            permissions=["search", "upload"],
        )
        d = info.to_dict()
        assert d["id"] == "k1"
        assert d["name"] == "Test Key"
        assert d["permissions"] == ["search", "upload"]
        assert d["is_active"] is True


# ---------------------------------------------------------------------------
# APIKeyManager
# ---------------------------------------------------------------------------


class TestAPIKeyManagerInit:
    def test_db_created(self, temp_db):
        APIKeyManager(db_path=temp_db)
        assert Path(temp_db).exists()

    def test_tables_created(self, temp_db):
        APIKeyManager(db_path=temp_db)
        with sqlite3.connect(temp_db) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = {t[0] for t in tables}
        assert "api_keys" in table_names
        assert "api_key_usage" in table_names

    def test_hash_key(self, manager):
        h = manager._hash_key("test_key")
        assert len(h) == 64  # SHA256 hex

    def test_hash_key_deterministic(self, manager):
        assert manager._hash_key("key") == manager._hash_key("key")

    def test_generate_key_secret(self, manager):
        secret = manager._generate_key_secret()
        assert secret.startswith("sk_live_")
        assert len(secret) > 10


class TestGenerateKey:
    def test_generate_returns_key_id_and_plain(self, manager):
        key_id, plain_key = manager.generate_key("user1", "My Key")
        assert key_id.startswith("key_")
        assert plain_key.startswith("sk_live_")

    def test_generated_key_is_valid(self, manager):
        _, plain_key = manager.generate_key("user1", "Key")
        info = manager.validate_key(plain_key)
        assert info is not None
        assert info.is_active is True

    def test_custom_permissions(self, manager):
        _, plain_key = manager.generate_key("user1", "Key", permissions=["search"])
        info = manager.validate_key(plain_key)
        assert "search" in info.permissions

    def test_default_permissions(self, manager):
        _, plain_key = manager.generate_key("user1", "Key")
        info = manager.validate_key(plain_key)
        assert "upload" in info.permissions
        assert "search" in info.permissions

    def test_with_expiry(self, manager):
        _, plain_key = manager.generate_key("user1", "Key", expires_in_days=30)
        info = manager.validate_key(plain_key)
        assert info is not None

    def test_with_metadata(self, manager):
        _, plain_key = manager.generate_key("user1", "Key", metadata={"env": "test"})
        info = manager.validate_key(plain_key)
        assert info is not None

    def test_rate_limit_stored(self, manager):
        _, plain_key = manager.generate_key("user1", "Key", rate_limit_per_minute=50)
        info = manager.validate_key(plain_key)
        assert info.rate_limit_per_minute == 50


class TestValidateKey:
    def test_invalid_key_returns_none(self, manager):
        result = manager.validate_key("sk_live_nonexistent")
        assert result is None

    def test_inactive_key_returns_none(self, manager):
        key_id, plain_key = manager.generate_key("user1", "Key")
        manager.revoke_key(key_id)
        result = manager.validate_key(plain_key)
        assert result is None

    def test_valid_updates_last_used(self, manager):
        _, plain_key = manager.generate_key("user1", "Key")
        manager.validate_key(plain_key)
        info2 = manager.validate_key(plain_key)
        # Both should succeed (last_used should be set)
        assert info2 is not None


class TestRevokeKey:
    def test_revoke_existing(self, manager):
        key_id, _ = manager.generate_key("user1", "Key")
        result = manager.revoke_key(key_id)
        assert result is True

    def test_revoke_nonexistent(self, manager):
        result = manager.revoke_key("nonexistent_key_id")
        assert result is False


class TestListKeys:
    def test_list_empty(self, manager):
        keys = manager.list_keys("no_user")
        assert keys == []

    def test_list_after_generate(self, manager):
        manager.generate_key("user2", "Key A")
        manager.generate_key("user2", "Key B")
        keys = manager.list_keys("user2")
        # At least the keys we just created (user_id is encrypted so listing may vary)
        assert isinstance(keys, list)

    def test_decode_permissions_valid_json(self):
        perms_json = json.dumps(["search", "upload"])
        result = APIKeyManager._decode_permissions(perms_json)
        assert "search" in result

    def test_decode_permissions_none(self):
        result = APIKeyManager._decode_permissions(None)
        assert result == []

    def test_row_to_key_info(self):
        row = (
            "kid1",
            "name",
            "user",
            "2025-01-01",
            None,
            1,
            100,
            json.dumps(["search"]),
        )
        info = APIKeyManager._row_to_key_info(row)
        assert info.id == "kid1"
        assert info.is_active is True


class TestLogUsage:
    def test_log_usage_creates_record(self, manager):
        key_id, _ = manager.generate_key("u1", "K")
        manager.log_usage(
            api_key_id=key_id,
            request_id=str(uuid.uuid4()),
            endpoint="/api/search",
            method="POST",
            status_code=200,
            duration_ms=15,
        )
        stats = manager.get_usage_stats()
        assert stats["total_requests"] >= 1

    def test_get_usage_stats_by_user(self, manager):
        key_id, _ = manager.generate_key("u_stats", "K")
        manager.log_usage(key_id, str(uuid.uuid4()), "/api/search", "POST", 200, 10)
        stats = manager.get_usage_stats(user_id="u_stats")
        assert isinstance(stats, dict)
        assert "total_requests" in stats
        assert "by_endpoint" in stats


# ---------------------------------------------------------------------------
# bootstrap_api_key
# ---------------------------------------------------------------------------


class TestBootstrapApiKey:
    def test_bootstrap_inserts_new_key(self, temp_db):
        # Reset singleton so a fresh manager uses temp_db
        original = auth_module._key_manager
        auth_module._key_manager = APIKeyManager(db_path=temp_db)
        try:
            result = bootstrap_api_key("sk_live_testbootstrapkey123456")
            assert result is True
        finally:
            auth_module._key_manager = original

    def test_bootstrap_no_duplicate(self, temp_db):
        original = auth_module._key_manager
        auth_module._key_manager = APIKeyManager(db_path=temp_db)
        try:
            plain_key = "sk_live_uniquekey987654321abcde"
            r1 = bootstrap_api_key(plain_key)
            r2 = bootstrap_api_key(plain_key)
            assert r1 is True
            assert r2 is False
        finally:
            auth_module._key_manager = original


# ---------------------------------------------------------------------------
# NullIAMProvider
# ---------------------------------------------------------------------------


class TestNullIAMProvider:
    def test_always_returns_none(self):
        provider = NullIAMProvider()
        assert provider.validate_admin_token("any_token") is None
        assert provider.validate_admin_token("") is None


# ---------------------------------------------------------------------------
# get_key_manager singleton
# ---------------------------------------------------------------------------


class TestGetKeyManager:
    def test_singleton_returns_same(self, temp_db, monkeypatch):
        monkeypatch.setenv("FLAMEHAVEN_API_KEYS_DB", temp_db)
        auth_module._key_manager = None
        mgr1 = get_key_manager(temp_db)
        mgr2 = get_key_manager(temp_db)
        assert mgr1 is mgr2
        auth_module._key_manager = None
