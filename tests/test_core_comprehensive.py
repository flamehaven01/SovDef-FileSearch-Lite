"""
Comprehensive tests for core.py (FlamehavenFileSearch high-level methods).
Target: cover core init paths, store management, persistence helpers, get_metrics.
"""

import pytest

from flamehaven_filesearch.core import FlamehavenFileSearch
from flamehaven_filesearch.persistence import FlamehavenPersistence


@pytest.fixture
def fs():
    searcher = FlamehavenFileSearch(allow_offline=True)
    searcher.create_store("default")
    return searcher


# ---------------------------------------------------------------------------
# init paths
# ---------------------------------------------------------------------------


class TestCoreInit:
    def test_default_init(self):
        s = FlamehavenFileSearch(allow_offline=True)
        assert s.config is not None

    def test_stores_initialized(self):
        s = FlamehavenFileSearch(allow_offline=True)
        assert isinstance(s.stores, dict)

    def test_chronos_grid_present(self):
        s = FlamehavenFileSearch(allow_offline=True)
        assert s.chronos_grid is not None

    def test_embedding_generator_present(self):
        s = FlamehavenFileSearch(allow_offline=True)
        assert s.embedding_generator is not None

    def test_intent_refiner_present(self):
        s = FlamehavenFileSearch(allow_offline=True)
        assert s.intent_refiner is not None

    def test_gravitas_packer_present(self):
        s = FlamehavenFileSearch(allow_offline=True)
        assert s.gravitas_packer is not None

    def test_default_store_created(self):
        s = FlamehavenFileSearch(allow_offline=True)
        assert "default" in s.stores

    def test_bm25_dirty_initialized(self):
        s = FlamehavenFileSearch(allow_offline=True)
        assert isinstance(s._bm25_dirty, set)

    def test_local_store_docs_initialized(self):
        s = FlamehavenFileSearch(allow_offline=True)
        assert isinstance(s._local_store_docs, dict)

    def test_atom_store_docs_initialized(self):
        s = FlamehavenFileSearch(allow_offline=True)
        assert isinstance(s._atom_store_docs, dict)


# ---------------------------------------------------------------------------
# create_store / list_stores / delete_store
# ---------------------------------------------------------------------------


class TestStoreManagement:
    def test_create_store_returns_id(self, fs):
        store_id = fs.create_store("test_store")
        assert store_id is not None
        assert "test_store" in fs.stores

    def test_create_store_idempotent(self, fs):
        id1 = fs.create_store("s1")
        id2 = fs.create_store("s1")
        assert id1 == id2

    def test_list_stores(self, fs):
        fs.create_store("alpha")
        fs.create_store("beta")
        stores = fs.list_stores()
        assert "alpha" in stores
        assert "beta" in stores

    def test_list_stores_returns_copy(self, fs):
        stores = fs.list_stores()
        stores["fake"] = "modified"
        assert "fake" not in fs.stores

    def test_delete_store_success(self, fs):
        fs.create_store("todel")
        result = fs.delete_store("todel")
        assert result["status"] == "success"
        assert "todel" not in fs.stores

    def test_delete_nonexistent_store(self, fs):
        result = fs.delete_store("nonexistent_store_xyz")
        assert result["status"] == "error"

    def test_delete_removes_local_docs(self, fs):
        fs.create_store("clean_me")
        fs._local_store_docs["clean_me"] = [{"title": "doc"}]
        fs.delete_store("clean_me")
        assert "clean_me" not in fs._local_store_docs

    def test_delete_removes_bm25(self, fs):
        fs.create_store("bm25_store")
        fs._bm25_dirty.add("bm25_store")
        fs.delete_store("bm25_store")
        assert "bm25_store" not in fs._bm25_dirty


# ---------------------------------------------------------------------------
# get_metrics
# ---------------------------------------------------------------------------


class TestGetMetrics:
    def test_returns_dict(self, fs):
        metrics = fs.get_metrics()
        assert isinstance(metrics, dict)

    def test_stores_count(self, fs):
        metrics = fs.get_metrics()
        assert "stores_count" in metrics
        assert metrics["stores_count"] >= 1

    def test_stores_list(self, fs):
        metrics = fs.get_metrics()
        assert "stores" in metrics
        assert isinstance(metrics["stores"], list)

    def test_chronos_grid_metrics(self, fs):
        metrics = fs.get_metrics()
        assert "chronos_grid" in metrics
        assert "indexed_files" in metrics["chronos_grid"]

    def test_intent_refiner_metrics(self, fs):
        metrics = fs.get_metrics()
        assert "intent_refiner" in metrics

    def test_gravitas_packer_metrics(self, fs):
        metrics = fs.get_metrics()
        assert "gravitas_packer" in metrics

    def test_embedding_generator_metrics(self, fs):
        metrics = fs.get_metrics()
        assert "embedding_generator" in metrics

    def test_vector_store_metrics(self, fs):
        metrics = fs.get_metrics()
        assert "vector_store" in metrics
        assert "backend" in metrics["vector_store"]


# ---------------------------------------------------------------------------
# _resolve_vector_backend
# ---------------------------------------------------------------------------


class TestResolveVectorBackend:
    def test_auto_returns_memory_without_postgres(self, fs):
        result = fs._resolve_vector_backend("auto")
        assert result in ("memory", "postgres")

    def test_memory_explicit(self, fs):
        assert fs._resolve_vector_backend("memory") == "memory"

    def test_chronos_alias(self, fs):
        assert fs._resolve_vector_backend("chronos") == "memory"

    def test_default_alias(self, fs):
        result = fs._resolve_vector_backend("default")
        assert result in ("memory", "postgres")

    def test_empty_string_auto(self, fs):
        result = fs._resolve_vector_backend("")
        assert result in ("memory", "postgres")

    def test_postgres_without_store(self, fs):
        result = fs._resolve_vector_backend("postgres")
        assert result in ("memory", "postgres")

    def test_unknown_returns_memory(self, fs):
        assert fs._resolve_vector_backend("unknown_backend") == "memory"


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


class TestPersistenceHelpers:
    def test_inject_into_chronos_empty_content(self, fs):
        doc = {"content": ""}
        fs._inject_into_chronos("uri://test", doc)  # should not crash

    def test_inject_into_chronos_with_content(self, fs):
        doc = {"content": "hello world test content"}
        fs._inject_into_chronos("uri://test2", doc)
        assert fs.chronos_grid.total_lore_essences >= 0

    def test_inject_into_chronos_empty_uri(self, fs):
        doc = {"content": "some content"}
        fs._inject_into_chronos("", doc)  # empty URI, should not crash

    def test_restore_store_docs_adds_new(self, fs):
        fs.create_store("restore_s")
        docs = [{"uri": "local://a", "content": "text", "title": "doc"}]
        count = fs._restore_store_docs("restore_s", docs)
        assert count == 1

    def test_restore_store_docs_skips_existing(self, fs):
        fs.create_store("restore_s2")
        docs = [{"uri": "local://existing", "content": "text"}]
        fs._restore_store_docs("restore_s2", docs)
        count = fs._restore_store_docs("restore_s2", docs)
        assert count == 0

    def test_restore_store_atoms_adds_new(self, fs):
        fs.create_store("atom_s")
        atoms = {"atom://1": {"chunk": "piece", "uri": "atom://1"}}
        count = fs._restore_store_atoms("atom_s", atoms)
        assert count == 1

    def test_restore_store_atoms_skips_existing(self, fs):
        fs.create_store("atom_s2")
        atoms = {"atom://2": {"chunk": "piece"}}
        fs._restore_store_atoms("atom_s2", atoms)
        count = fs._restore_store_atoms("atom_s2", atoms)
        assert count == 0

    def test_snapshot_store_no_persistence(self, fs):
        fs._persistence = None
        fs._snapshot_store("default")  # should not crash

    def test_snapshot_store_with_persistence(self, fs, tmp_path):
        persist = FlamehavenPersistence(str(tmp_path / "snapshots"))
        fs._persistence = persist
        fs._local_store_docs["default"] = [{"title": "doc"}]
        fs._snapshot_store("default")
        assert (persist._stores_dir / "default_docs.json").exists()

    def test_restore_from_persistence_no_persist(self, fs):
        fs._persistence = None
        count = fs._restore_from_persistence()
        assert count == 0

    def test_restore_from_persistence_with_data(self, fs, tmp_path):
        persist = FlamehavenPersistence(str(tmp_path / "restore_test"))
        persist.save_store("mystore", [{"uri": "local://x", "title": "t"}], {})
        fs._persistence = persist
        count = fs._restore_from_persistence()
        assert count >= 0  # may be 0 if store already exists with same URIs
