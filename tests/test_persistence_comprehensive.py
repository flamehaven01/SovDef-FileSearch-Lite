"""
Comprehensive tests for persistence.py.
Target: cover FlamehavenPersistence and get_persistence.
"""

import json
import os

import pytest

from flamehaven_filesearch.persistence import (
    FlamehavenPersistence,
    get_persistence,
    _json_default,
)


@pytest.fixture
def persist(tmp_path):
    return FlamehavenPersistence(str(tmp_path / "persist"))


class TestFlamehavenPersistenceInit:
    def test_creates_root(self, tmp_path):
        p = FlamehavenPersistence(str(tmp_path / "mydata"))
        assert p.root.exists()

    def test_creates_stores_dir(self, tmp_path):
        p = FlamehavenPersistence(str(tmp_path / "mydata"))
        assert (p.root / "stores").exists()

    def test_expands_user(self, tmp_path):
        p = FlamehavenPersistence(str(tmp_path / "data"))
        assert p.root.is_absolute()


class TestSaveStore:
    def test_save_creates_file(self, persist):
        persist.save_store("test_store", [{"title": "doc1"}], {})
        target = persist._stores_dir / "test_store_docs.json"
        assert target.exists()

    def test_save_content_correct(self, persist):
        docs = [{"title": "a", "content": "hello"}]
        atoms = {"id1": {"chunk": "piece"}}
        persist.save_store("s1", docs, atoms)
        with open(persist._stores_dir / "s1_docs.json", encoding="utf-8") as f:
            data = json.load(f)
        assert data["store_name"] == "s1"
        assert data["doc_count"] == 1
        assert data["atom_count"] == 1
        assert data["docs"] == docs
        assert data["atoms"] == atoms

    def test_save_empty_store(self, persist):
        persist.save_store("empty", [], {})
        target = persist._stores_dir / "empty_docs.json"
        assert target.exists()
        with open(target, encoding="utf-8") as f:
            data = json.load(f)
        assert data["doc_count"] == 0

    def test_save_overwrites_previous(self, persist):
        persist.save_store("s2", [{"title": "v1"}], {})
        persist.save_store("s2", [{"title": "v2"}, {"title": "v3"}], {})
        with open(persist._stores_dir / "s2_docs.json", encoding="utf-8") as f:
            data = json.load(f)
        assert data["doc_count"] == 2

    def test_save_schema_version(self, persist):
        persist.save_store("sv", [], {})
        with open(persist._stores_dir / "sv_docs.json", encoding="utf-8") as f:
            data = json.load(f)
        assert data["schema_version"] == 1


class TestLoadStore:
    def test_load_nonexistent_returns_empty(self, persist):
        docs, atoms = persist.load_store("nonexistent")
        assert docs == []
        assert atoms == {}

    def test_load_after_save(self, persist):
        docs_in = [{"title": "note", "content": "text"}]
        atoms_in = {"k1": {"chunk": "val"}}
        persist.save_store("mystore", docs_in, atoms_in)
        docs_out, atoms_out = persist.load_store("mystore")
        assert docs_out == docs_in
        assert atoms_out == atoms_in

    def test_load_wrong_schema_version_returns_empty(self, persist):
        target = persist._stores_dir / "badver_docs.json"
        with open(target, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "schema_version": 99,
                    "store_name": "badver",
                    "docs": [{"title": "x"}],
                    "atoms": {},
                },
                f,
            )
        docs, atoms = persist.load_store("badver")
        assert docs == []
        assert atoms == {}

    def test_load_invalid_json_returns_empty(self, persist):
        target = persist._stores_dir / "corrupt_docs.json"
        target.write_text("not valid json", encoding="utf-8")
        docs, atoms = persist.load_store("corrupt")
        assert docs == []
        assert atoms == {}

    def test_load_missing_docs_key(self, persist):
        target = persist._stores_dir / "nodocs_docs.json"
        with open(target, "w", encoding="utf-8") as f:
            json.dump({"schema_version": 1, "store_name": "nodocs"}, f)
        docs, atoms = persist.load_store("nodocs")
        assert docs == []
        assert atoms == {}


class TestListPersistedStores:
    def test_empty_initially(self, persist):
        assert persist.list_persisted_stores() == []

    def test_lists_saved_stores(self, persist):
        persist.save_store("store_a", [], {})
        persist.save_store("store_b", [], {})
        names = persist.list_persisted_stores()
        assert "store_a" in names
        assert "store_b" in names

    def test_sorted_order(self, persist):
        persist.save_store("zz", [], {})
        persist.save_store("aa", [], {})
        names = persist.list_persisted_stores()
        assert names == sorted(names)


class TestDeleteStore:
    def test_delete_existing(self, persist):
        persist.save_store("todel", [], {})
        result = persist.delete_store("todel")
        assert result is True
        assert not (persist._stores_dir / "todel_docs.json").exists()

    def test_delete_nonexistent_returns_false(self, persist):
        result = persist.delete_store("doesnt_exist")
        assert result is False

    def test_delete_removes_from_list(self, persist):
        persist.save_store("temp", [], {})
        persist.delete_store("temp")
        assert "temp" not in persist.list_persisted_stores()


class TestGetPersistence:
    def test_none_when_no_path(self):
        old = os.environ.pop("PERSIST_PATH", None)
        try:
            result = get_persistence()
            assert result is None
        finally:
            if old is not None:
                os.environ["PERSIST_PATH"] = old

    def test_none_when_empty_string(self):
        result = get_persistence("")
        assert result is None

    def test_returns_instance_with_path(self, tmp_path):
        result = get_persistence(str(tmp_path / "data"))
        assert result is not None
        assert isinstance(result, FlamehavenPersistence)

    def test_uses_env_var(self, tmp_path):
        path = str(tmp_path / "envdata")
        os.environ["PERSIST_PATH"] = path
        try:
            result = get_persistence()
            assert result is not None
        finally:
            del os.environ["PERSIST_PATH"]


class TestJsonDefault:
    def test_str_fallback(self):
        class Custom:
            def __str__(self):
                return "custom_str"

        result = _json_default(Custom())
        assert result == "custom_str"

    def test_numpy_types_if_available(self):
        try:
            import numpy as np

            assert _json_default(np.int32(5)) == 5
            assert _json_default(np.float32(1.5)) == pytest.approx(1.5, abs=0.01)
            arr = np.array([1, 2, 3])
            assert _json_default(arr) == [1, 2, 3]
        except ImportError:
            pass
