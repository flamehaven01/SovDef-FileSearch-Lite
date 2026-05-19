"""
flamehaven_filesearch/persistence.py
──────────────────────────────────────
Lightweight JSON snapshot persistence for in-memory mode.

Solves the #1 operational pain point: ChronosGrid and _local_store_docs
are fully in-memory — every server restart wipes the index and requires
a full re-ingest.

Design principles:
  • Opt-in via PERSIST_PATH env var (or Config.persist_path)
  • Zero new dependencies (stdlib json + optional numpy for vectors)
  • Save after each upload (append-safe, atomic rename on POSIX)
  • Load on startup: restore docs → regenerate embeddings → refill ChronosGrid
  • No lock files — single-process use case (FAS is single-worker by default)

Usage:
    Set in .env:
        PERSIST_PATH=./.flamehaven_data

    The FAS server will automatically:
        - On startup: load all persisted stores
        - On each upload: snapshot the updated store
        - On delete_store: remove persisted files

File layout:
    {PERSIST_PATH}/
        stores/
            {store_name}_docs.json     # list of main docs + chunk atoms
        meta.json                       # store registry + schema version
"""

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1


class FlamehavenPersistence:
    """
    Snapshot-based persistence for FAS in-memory store.

    Thread safety: not thread-safe (FAS uses single worker by default).
    For multi-worker setups, use the PostgreSQL backend instead.
    """

    def __init__(self, persist_path: str):
        self.root = Path(persist_path).expanduser().resolve()
        self._stores_dir = self.root / "stores"
        self._stores_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[Persist] Root: %s", self.root)

    # ── Save ──────────────────────────────────────────────────────────────────

    def save_store(
        self,
        store_name: str,
        docs: List[Dict[str, Any]],
        atoms: Dict[str, Dict[str, Any]],
    ) -> None:
        """
        Snapshot docs + chunk atoms for one store to JSON.

        Uses atomic write (temp file + rename) where supported.
        """
        target = self._stores_dir / f"{store_name}_docs.json"
        tmp = target.with_suffix(".json.tmp")
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "store_name": store_name,
            "saved_at": time.time(),
            "doc_count": len(docs),
            "atom_count": len(atoms),
            "docs": docs,
            "atoms": atoms,
        }
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, default=_json_default)
            # Atomic rename (POSIX) / best-effort on Windows
            try:
                os.replace(tmp, target)
            except OSError:
                shutil.move(str(tmp), str(target))
            logger.debug(
                "[Persist] Saved store '%s' — %d docs, %d atoms",
                store_name,
                len(docs),
                len(atoms),
            )
        except Exception as exc:
            logger.warning("[Persist] Failed to save store '%s': %s", store_name, exc)
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────

    def load_store(
        self, store_name: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        Load persisted docs + atoms for one store.

        Returns ([], {}) if no snapshot exists or on parse error.
        """
        target = self._stores_dir / f"{store_name}_docs.json"
        if not target.exists():
            return [], {}
        try:
            with open(target, "r", encoding="utf-8") as f:
                payload = json.load(f)
            version = payload.get("schema_version", 0)
            if version != _SCHEMA_VERSION:
                logger.warning(
                    "[Persist] Store '%s' schema v%d != expected v%d — skipping",
                    store_name,
                    version,
                    _SCHEMA_VERSION,
                )
                return [], {}
            docs = payload.get("docs") or []
            atoms = payload.get("atoms") or {}
            logger.info(
                "[Persist] Loaded store '%s' — %d docs, %d atoms",
                store_name,
                len(docs),
                len(atoms),
            )
            return docs, atoms
        except Exception as exc:
            logger.warning("[Persist] Failed to load store '%s': %s", store_name, exc)
            return [], {}

    # ── Discovery ─────────────────────────────────────────────────────────────

    def list_persisted_stores(self) -> List[str]:
        """Return names of all stores with persisted snapshots."""
        suffix = "_docs"
        return [
            p.stem[: -len(suffix)] if p.stem.endswith(suffix) else p.stem
            for p in sorted(self._stores_dir.glob("*_docs.json"))
        ]

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_store(self, store_name: str) -> bool:
        """Remove persisted snapshot for a store. Returns True if file existed."""
        target = self._stores_dir / f"{store_name}_docs.json"
        if target.exists():
            try:
                target.unlink()
                logger.info("[Persist] Deleted snapshot for store '%s'", store_name)
                return True
            except Exception as exc:
                logger.warning(
                    "[Persist] Failed to delete store '%s': %s", store_name, exc
                )
        return False


# ── Module-level helpers ───────────────────────────────────────────────────────


def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy types and non-serializable objects."""
    try:
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    # Fallback: stringify unknown types
    return str(obj)


def get_persistence(
    persist_path: Optional[str] = None,
) -> Optional[FlamehavenPersistence]:
    """
    Factory — returns a FlamehavenPersistence if PERSIST_PATH is configured,
    otherwise None (persistence disabled).

    Args:
        persist_path: Override path. If None, reads PERSIST_PATH env var.

    Returns:
        FlamehavenPersistence instance or None.
    """
    resolved = persist_path or os.getenv("PERSIST_PATH", "").strip()
    if not resolved:
        return None
    try:
        p = FlamehavenPersistence(resolved)
        logger.info("[Persist] Enabled — path: %s", p.root)
        return p
    except Exception as exc:
        logger.warning("[Persist] Could not initialize at '%s': %s", resolved, exc)
        return None
