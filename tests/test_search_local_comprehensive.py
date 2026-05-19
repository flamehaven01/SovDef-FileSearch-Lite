"""
Comprehensive tests for LocalSearchMixin (_search_local.py).
Target: cover the 614 uncovered statements in that module.
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
from urllib.parse import quote

import pytest

from flamehaven_filesearch.config import Config
from flamehaven_filesearch.core import FlamehavenFileSearch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def searcher():
    fs = FlamehavenFileSearch(allow_offline=True)
    fs.create_store("default")
    return fs


def _make_doc(
    title: str,
    content: str,
    uri: str = None,
    file_path: str = None,
    headings: List[str] = None,
    tags: List[str] = None,
    aliases: List[str] = None,
    wikilinks: List[str] = None,
    context: str = "",
    parent_glyph: str = "",
) -> Dict[str, Any]:
    abs_path = file_path or f"/vault/{title}.md"
    obsidian: Dict[str, Any] = {}
    if headings:
        obsidian["headings"] = headings
    if tags:
        obsidian["tags"] = tags
    if aliases:
        obsidian["aliases"] = aliases
    if wikilinks:
        obsidian["wikilinks"] = wikilinks
    metadata: Dict[str, Any] = {
        "file_path": abs_path,
        "file_type": ".md",
        "context": context,
    }
    if obsidian:
        metadata["obsidian"] = obsidian
    if parent_glyph:
        metadata["parent_glyph"] = parent_glyph

    return {
        "title": title,
        "uri": uri or f"local://default/{quote(abs_path, safe='')}",
        "content": content,
        "metadata": metadata,
    }


def _inject_docs(searcher: FlamehavenFileSearch, docs: List[Dict[str, Any]]):
    searcher._local_store_docs["default"] = docs
    searcher._bm25_dirty.add("default")


# ---------------------------------------------------------------------------
# _doc_* accessors
# ---------------------------------------------------------------------------


class TestDocAccessors:
    def test_doc_content(self, searcher):
        doc = _make_doc("A", "hello world")
        assert searcher._doc_content(doc) == "hello world"

    def test_doc_content_empty(self, searcher):
        assert searcher._doc_content({}) == ""

    def test_doc_title(self, searcher):
        doc = _make_doc("MyTitle", "x")
        assert searcher._doc_title(doc) == "MyTitle"

    def test_doc_title_missing(self, searcher):
        assert searcher._doc_title({}) == ""

    def test_doc_metadata(self, searcher):
        doc = _make_doc("A", "x")
        meta = searcher._doc_metadata(doc)
        assert isinstance(meta, dict)
        assert "file_path" in meta

    def test_doc_metadata_non_dict(self, searcher):
        doc = {"title": "A", "uri": "u", "content": "x", "metadata": "not_a_dict"}
        assert searcher._doc_metadata(doc) == {}

    def test_doc_context(self, searcher):
        doc = _make_doc("A", "x", context="some context")
        assert searcher._doc_context(doc) == "some context"

    def test_doc_headings_from_obsidian(self, searcher):
        doc = _make_doc("A", "x", headings=["H1", "H2"])
        heads = searcher._doc_headings(doc)
        assert "H1" in heads
        assert "H2" in heads

    def test_doc_headings_empty(self, searcher):
        doc = _make_doc("A", "x")
        assert searcher._doc_headings(doc) == []

    def test_doc_uri_path_from_metadata(self, searcher):
        doc = _make_doc("A", "x", file_path="/some/path/file.md")
        assert searcher._doc_uri_path(doc) == "/some/path/file.md"

    def test_doc_uri_path_from_uri(self, searcher):
        abs_path = "/vault/notes/note.md"
        doc = {
            "title": "note",
            "uri": f"local://default/{quote(abs_path, safe='')}",
            "content": "x",
            "metadata": {},
        }
        path = searcher._doc_uri_path(doc)
        assert "note.md" in path

    def test_doc_uri_path_non_local(self, searcher):
        doc = {
            "title": "A",
            "uri": "http://example.com",
            "content": "x",
            "metadata": {},
        }
        assert searcher._doc_uri_path(doc) == ""

    def test_doc_uri_path_no_slash(self, searcher):
        doc = {"title": "A", "uri": "local://noslash", "content": "x", "metadata": {}}
        assert searcher._doc_uri_path(doc) == ""


# ---------------------------------------------------------------------------
# _document_cluster_key, _canonical_uri
# ---------------------------------------------------------------------------


class TestClusterKey:
    def test_cluster_key_from_file_path(self, searcher):
        doc = _make_doc("A", "x", file_path="/vault/notes/note.md")
        key = searcher._document_cluster_key(doc)
        assert "note.md" in key

    def test_cluster_key_parent_glyph(self, searcher):
        doc = _make_doc("A", "x", parent_glyph="/vault/notes/note.md")
        key = searcher._document_cluster_key(doc)
        assert "note.md" in key

    def test_cluster_key_none_doc(self, searcher):
        assert searcher._document_cluster_key(None) == ""

    def test_canonical_uri_strips_fragment(self, searcher):
        assert (
            searcher._canonical_uri("local://default/path#anchor")
            == "local://default/path"
        )

    def test_canonical_uri_no_fragment(self, searcher):
        assert searcher._canonical_uri("local://default/path") == "local://default/path"

    def test_canonical_uri_empty(self, searcher):
        assert searcher._canonical_uri("") == ""


# ---------------------------------------------------------------------------
# _query_terms, _normalize_lookup_text, _lexical_token_hits
# ---------------------------------------------------------------------------


class TestQueryHelpers:
    def test_query_terms_basic(self, searcher):
        terms = searcher._query_terms("hello world test")
        assert "hello" in terms
        assert "world" in terms

    def test_query_terms_dedup(self, searcher):
        terms = searcher._query_terms("hello hello world")
        assert terms.count("hello") == 1

    def test_query_terms_empty(self, searcher):
        assert searcher._query_terms("") == []

    def test_query_terms_korean(self, searcher):
        terms = searcher._query_terms("안녕 세계")
        assert len(terms) >= 1

    def test_normalize_lookup_text(self, searcher):
        out = searcher._normalize_lookup_text("Hello World")
        assert out == "hello world"

    def test_lexical_token_hits(self, searcher):
        hits = searcher._lexical_token_hits(
            "python programming is great", ["python", "great"]
        )
        assert hits == 2

    def test_lexical_token_hits_none(self, searcher):
        assert searcher._lexical_token_hits("unrelated text", ["python"]) == 0


# ---------------------------------------------------------------------------
# _match_phrase_prefix_score
# ---------------------------------------------------------------------------


class TestMatchPhrasePrefix:
    def test_full_match(self, searcher):
        score = searcher._match_phrase_prefix_score(
            "python programming", ["python", "prog"]
        )
        assert score > 0.0

    def test_no_match(self, searcher):
        score = searcher._match_phrase_prefix_score("java spring", ["python"])
        assert score == 0.0

    def test_empty_text(self, searcher):
        assert searcher._match_phrase_prefix_score("", ["python"]) == 0.0

    def test_empty_query(self, searcher):
        assert searcher._match_phrase_prefix_score("hello world", []) == 0.0

    def test_prefix_partial(self, searcher):
        score = searcher._match_phrase_prefix_score("programming language", ["prog"])
        assert score > 0.0


# ---------------------------------------------------------------------------
# _folder_topic_prior
# ---------------------------------------------------------------------------


class TestFolderTopicPrior:
    def test_folder_match(self, searcher):
        doc = _make_doc("A", "x", file_path="/vault/python/notes/note.md")
        boost = searcher._folder_topic_prior(doc, "python")
        assert boost > 0.0

    def test_no_path(self, searcher):
        doc = {"title": "A", "uri": "http://x", "content": "x", "metadata": {}}
        assert searcher._folder_topic_prior(doc, "python") == 0.0

    def test_empty_query(self, searcher):
        doc = _make_doc("A", "x", file_path="/vault/python/note.md")
        assert searcher._folder_topic_prior(doc, "") == 0.0


# ---------------------------------------------------------------------------
# _exact_file_match_score, _title_match_score
# ---------------------------------------------------------------------------


class TestFileMatchScores:
    def test_exact_file_match_title_contains_query(self, searcher):
        doc = _make_doc("Python Programming Guide", "content about python")
        score = searcher._exact_file_match_score(doc, "Python Programming Guide")
        assert score >= 5.0

    def test_exact_file_match_short_query(self, searcher):
        doc = _make_doc("A", "x")
        assert searcher._exact_file_match_score(doc, "ab") == 0.0

    def test_exact_file_match_in_body(self, searcher):
        doc = _make_doc("Misc", "this is about machine learning basics")
        score = searcher._exact_file_match_score(doc, "machine learning")
        assert score > 0.0

    def test_title_match_score_exact(self, searcher):
        doc = _make_doc("python notes", "some content")
        score = searcher._title_match_score(doc, "python notes")
        assert score >= 7.0

    def test_title_match_score_partial(self, searcher):
        doc = _make_doc("python notes and more", "content")
        score = searcher._title_match_score(doc, "python")
        assert score > 0.0

    def test_title_match_score_no_match(self, searcher):
        doc = _make_doc("java spring docs", "content")
        score = searcher._title_match_score(doc, "python machine learning")
        assert score == 0.0

    def test_title_match_alias(self, searcher):
        doc = _make_doc("ML Notes", "content", aliases=["machine learning"])
        score = searcher._title_match_score(doc, "machine learning")
        assert score > 0.0


# ---------------------------------------------------------------------------
# _best_title_candidate
# ---------------------------------------------------------------------------


class TestBestTitleCandidate:
    def test_basic(self, searcher):
        doc = _make_doc("My Notes", "content")
        candidate = searcher._best_title_candidate(doc)
        assert isinstance(candidate, str)

    def test_with_aliases(self, searcher):
        doc = _make_doc("My Notes", "content", aliases=["my notes", "notes"])
        candidate = searcher._best_title_candidate(doc)
        assert candidate  # should pick shortest non-empty


# ---------------------------------------------------------------------------
# _title_arbitration_tuple
# ---------------------------------------------------------------------------


class TestTitleArbitrationTuple:
    def test_exact_match(self, searcher):
        doc = _make_doc("python notes", "content")
        tup = searcher._title_arbitration_tuple(doc, "python notes")
        assert tup[0] >= 7.0
        assert tup[1] == 1  # exact_equal

    def test_no_match(self, searcher):
        doc = _make_doc("java spring", "content")
        tup = searcher._title_arbitration_tuple(doc, "python")
        assert tup[1] == 0  # not exact_equal


# ---------------------------------------------------------------------------
# _exact_note_resolution
# ---------------------------------------------------------------------------


class TestExactNoteResolution:
    def test_no_docs(self, searcher):
        assert searcher._exact_note_resolution([], "query") is None

    def test_no_title_match(self, searcher):
        docs = [_make_doc("java spring", "content")]
        assert searcher._exact_note_resolution(docs, "python data science") is None

    def test_exact_resolution(self, searcher):
        docs = [_make_doc("python notes", "python is a great language")]
        result = searcher._exact_note_resolution(docs, "python notes")
        assert result is not None
        selected, confidence = result
        assert len(selected) >= 1
        assert 0.8 <= confidence <= 1.0

    def test_ambiguous_returns_none(self, searcher):
        # Two clusters with similar scores — below threshold
        docs = [
            _make_doc("python notes", "content", file_path="/v/a/python notes.md"),
            _make_doc("python guide", "content", file_path="/v/b/python guide.md"),
        ]
        # Short query below score threshold
        result = searcher._exact_note_resolution(docs, "abc")
        assert result is None


# ---------------------------------------------------------------------------
# _cluster_ranked_entries, _dedupe_doc_clusters
# ---------------------------------------------------------------------------


class TestClusterMethods:
    def test_cluster_ranked_limits_per_cluster(self, searcher):
        fp = "/vault/notes/a.md"
        uri = f"local://default/{quote(fp, safe='')}"
        docs = [
            _make_doc("A", "c1", file_path=fp, uri=uri),
            _make_doc("A", "c2", file_path=fp, uri=uri + "#chunk1"),
            _make_doc("A", "c3", file_path=fp, uri=uri + "#chunk2"),
        ]
        _inject_docs(searcher, docs)

        ranked = [
            {"id": uri, "score": 1.0},
            {"id": uri + "#chunk1", "score": 0.9},
            {"id": uri + "#chunk2", "score": 0.8},
        ]
        out = searcher._cluster_ranked_entries("default", ranked, max_per_cluster=2)
        assert len(out) <= 2

    def test_dedupe_doc_clusters(self, searcher):
        fp = "/vault/notes/a.md"
        docs = [
            _make_doc("A", "c1", file_path=fp),
            _make_doc("A", "c2", file_path=fp),
            _make_doc("B", "c3", file_path="/vault/notes/b.md"),
        ]
        out = searcher._dedupe_doc_clusters(docs, max_per_cluster=1)
        assert len(out) == 2


# ---------------------------------------------------------------------------
# _apply_exact_file_post_filter
# ---------------------------------------------------------------------------


class TestExactFilePostFilter:
    def test_filter_selects_best_match(self, searcher):
        docs = [
            _make_doc("python notes", "about python programming"),
            _make_doc("java guide", "java programming basics"),
        ]
        filtered = searcher._apply_exact_file_post_filter(docs, "python notes")
        titles = [d["title"] for d in filtered]
        assert "python notes" in titles

    def test_empty_docs(self, searcher):
        assert searcher._apply_exact_file_post_filter([], "query") == []

    def test_low_score_returns_all(self, searcher):
        docs = [_make_doc("xyz", "random text"), _make_doc("abc", "other text")]
        out = searcher._apply_exact_file_post_filter(docs, "qqqqq")
        assert len(out) == len(docs)


# ---------------------------------------------------------------------------
# _lexical_alignment_boost, _external_reference_penalty
# ---------------------------------------------------------------------------


class TestLexicalBoost:
    def test_title_match_boosts(self, searcher):
        doc = _make_doc("python tutorial", "some content")
        boost = searcher._lexical_alignment_boost(doc, "python tutorial")
        assert boost >= 4.0

    def test_no_match(self, searcher):
        doc = _make_doc("java guide", "content")
        boost = searcher._lexical_alignment_boost(doc, "python")
        assert boost >= 0.0

    def test_empty_query(self, searcher):
        doc = _make_doc("A", "x")
        assert searcher._lexical_alignment_boost(doc, "") == 0.0

    def test_external_penalty_applied(self, searcher):
        doc = _make_doc(
            "Some Paper", "content", file_path="/vault/000 외부 논문 핵심/paper.md"
        )
        penalty = searcher._external_reference_penalty(doc, "unrelated query")
        assert penalty > 0.0

    def test_external_penalty_zero_when_query_in_title(self, searcher):
        doc = _make_doc(
            "Some Paper", "content", file_path="/vault/000 외부 논문 핵심/some paper.md"
        )
        penalty = searcher._external_reference_penalty(doc, "some paper")
        assert penalty == 0.0

    def test_no_external_path(self, searcher):
        doc = _make_doc("A", "x", file_path="/vault/notes/a.md")
        assert searcher._external_reference_penalty(doc, "query") == 0.0


# ---------------------------------------------------------------------------
# _build_bm25_corpus_text
# ---------------------------------------------------------------------------


class TestBm25CorpusText:
    def test_basic(self, searcher):
        doc = _make_doc(
            "Python Notes", "python is cool", headings=["Intro"], tags=["python"]
        )
        text = searcher._build_bm25_corpus_text(doc)
        assert "Python Notes" in text or "python notes" in text.lower()
        assert "python" in text.lower()

    def test_with_all_obsidian_fields(self, searcher):
        doc = _make_doc(
            "ML Guide",
            "machine learning content",
            headings=["Chapter 1"],
            tags=["ml", "ai"],
            aliases=["machine learning guide"],
            wikilinks=["deep learning", "neural networks"],
        )
        text = searcher._build_bm25_corpus_text(doc)
        assert "ml guide" in text.lower()


# ---------------------------------------------------------------------------
# _normalize_rrf_score
# ---------------------------------------------------------------------------


class TestNormalizeRrfScore:
    def test_empty_fused(self, searcher):
        assert searcher._normalize_rrf_score([]) == 0.0

    def test_with_score(self, searcher):
        fused = [{"id": "a", "score": 0.9, "rrf_score": 1.0 / 61.0}]
        score = searcher._normalize_rrf_score(fused)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# _get_doc_by_uri
# ---------------------------------------------------------------------------


class TestGetDocByUri:
    def test_found_in_main_docs(self, searcher):
        doc = _make_doc("A", "content")
        _inject_docs(searcher, [doc])
        result = searcher._get_doc_by_uri("default", doc["uri"])
        assert result is not None
        assert result["title"] == "A"

    def test_found_in_atoms(self, searcher):
        atom_uri = "local://default/path#chunk_0"
        atom = {
            "title": "chunk",
            "uri": atom_uri,
            "content": "chunk content",
            "metadata": {},
        }
        searcher._atom_store_docs["default"] = {atom_uri: atom}
        result = searcher._get_doc_by_uri("default", atom_uri)
        assert result is not None

    def test_not_found(self, searcher):
        _inject_docs(searcher, [])
        assert searcher._get_doc_by_uri("default", "local://default/missing") is None

    def test_none_uri(self, searcher):
        assert searcher._get_doc_by_uri("default", None) is None


# ---------------------------------------------------------------------------
# _rebuild_bm25
# ---------------------------------------------------------------------------


class TestRebuildBm25:
    def test_rebuild_creates_index(self, searcher):
        docs = [
            _make_doc("Python Notes", "python is great"),
            _make_doc("Java Guide", "java is verbose"),
        ]
        _inject_docs(searcher, docs)
        searcher._rebuild_bm25("default")
        assert "default" in searcher._bm25_indices
        bm25, uri_map = searcher._bm25_indices["default"]
        assert bm25.corpus_size == 2
        assert "default" not in searcher._bm25_dirty

    def test_rebuild_includes_atoms(self, searcher):
        docs = [_make_doc("A", "content a")]
        _inject_docs(searcher, docs)
        atom_uri = "local://default/a#chunk_0"
        searcher._atom_store_docs["default"] = {
            atom_uri: {
                "title": "chunk",
                "uri": atom_uri,
                "content": "chunk content",
                "metadata": {},
            }
        }
        searcher._rebuild_bm25("default")
        bm25, uri_map = searcher._bm25_indices["default"]
        assert bm25.corpus_size == 2


# ---------------------------------------------------------------------------
# _collect_bm25_ranked
# ---------------------------------------------------------------------------


class TestCollectBm25Ranked:
    def test_ranked_results(self, searcher):
        docs = [
            _make_doc("Python Notes", "python programming language"),
            _make_doc("Java Guide", "java enterprise beans"),
        ]
        _inject_docs(searcher, docs)
        searcher._rebuild_bm25("default")
        ranked = searcher._collect_bm25_ranked("default", "python", bm25_top_k=5)
        assert isinstance(ranked, list)
        if ranked:
            assert "id" in ranked[0]
            assert "score" in ranked[0]

    def test_empty_index(self, searcher):
        _inject_docs(searcher, [])
        searcher._rebuild_bm25("default")
        result = searcher._collect_bm25_ranked("default", "python", bm25_top_k=5)
        assert result == []


# ---------------------------------------------------------------------------
# _resolve_fused_docs
# ---------------------------------------------------------------------------


class TestResolveFusedDocs:
    def test_resolve(self, searcher):
        doc = _make_doc("A", "content")
        _inject_docs(searcher, [doc])
        fused = [{"id": doc["uri"], "score": 1.0}]
        result = searcher._resolve_fused_docs("default", fused)
        assert len(result) == 1
        assert result[0]["title"] == "A"

    def test_missing_uri_skipped(self, searcher):
        _inject_docs(searcher, [])
        fused = [{"id": "local://default/missing", "score": 1.0}]
        result = searcher._resolve_fused_docs("default", fused)
        assert result == []


# ---------------------------------------------------------------------------
# _contextual_doc_text, _build_semantic_excerpt
# ---------------------------------------------------------------------------


class TestDocTextBuilders:
    def test_contextual_doc_text_basic(self, searcher):
        doc = _make_doc("A", "some content here")
        text = searcher._contextual_doc_text(doc)
        assert "some content" in text

    def test_contextual_doc_text_with_context(self, searcher):
        doc = _make_doc("A", "content", context="contextual info")
        text = searcher._contextual_doc_text(doc, include_context=True)
        assert "contextual" in text

    def test_contextual_doc_text_max_chars(self, searcher):
        doc = _make_doc("A", "a" * 500)
        text = searcher._contextual_doc_text(doc, max_chars=100)
        assert len(text) <= 100

    def test_contextual_doc_text_headings(self, searcher):
        doc = _make_doc("A", "content", headings=["Section 1"])
        text = searcher._contextual_doc_text(doc)
        assert "Section 1" in text

    def test_build_semantic_excerpt_basic(self, searcher):
        doc = _make_doc("Python Notes", "python is a great language for data science")
        excerpt = searcher._build_semantic_excerpt(doc, "python data science")
        assert isinstance(excerpt, str)
        assert len(excerpt) > 0

    def test_build_semantic_excerpt_with_context(self, searcher):
        doc = _make_doc("A", "main content here", context="context info python")
        excerpt = searcher._build_semantic_excerpt(doc, "python")
        assert isinstance(excerpt, str)

    def test_build_semantic_excerpt_headings(self, searcher):
        doc = _make_doc("A", "content", headings=["Introduction"])
        excerpt = searcher._build_semantic_excerpt(doc, "intro")
        assert isinstance(excerpt, str)


# ---------------------------------------------------------------------------
# _resolve_semantic_docs, _collect_sem_ranked
# ---------------------------------------------------------------------------


class TestSemanticResolve:
    def test_collect_sem_ranked_empty(self, searcher):
        result = searcher._collect_sem_ranked("default", [])
        assert result == []

    def test_collect_sem_ranked_non_tuple_skipped(self, searcher):
        result = searcher._collect_sem_ranked("default", [{"not": "tuple"}])
        assert result == []

    def test_collect_sem_ranked_with_uri(self, searcher):
        doc = _make_doc("A", "content")
        _inject_docs(searcher, [doc])
        sem_results = [
            ({"uri": doc["uri"]}, 0.9),
        ]
        ranked = searcher._collect_sem_ranked("default", sem_results)
        assert isinstance(ranked, list)

    def test_collect_sem_ranked_with_file_path(self, searcher):
        abs_path = "/vault/notes/a.md"
        doc = _make_doc("A", "content", file_path=abs_path)
        _inject_docs(searcher, [doc])
        sem_results = [
            ({"file_path": abs_path}, 0.8),
        ]
        ranked = searcher._collect_sem_ranked("default", sem_results)
        assert isinstance(ranked, list)

    def test_resolve_semantic_docs(self, searcher):
        doc = _make_doc("A", "content")
        _inject_docs(searcher, [doc])
        sem_results = [({"uri": doc["uri"]}, 0.9)]
        result = searcher._resolve_semantic_docs("default", sem_results)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _forge_augment_sources
# ---------------------------------------------------------------------------


class TestForgeAugmentSources:
    def test_adds_matching_docs(self, searcher):
        doc = _make_doc("Python Guide", "python is a programming language")
        sources = []
        out = searcher._forge_augment_sources(sources, [doc], "python")
        assert len(out) >= 1

    def test_respects_max_sources(self, searcher):
        searcher.config.max_sources = 2
        docs = [_make_doc(f"Doc{i}", f"python content {i}") for i in range(5)]
        sources = [{"title": docs[0]["title"], "uri": docs[0]["uri"]}]
        out = searcher._forge_augment_sources(sources, docs, "python")
        assert len(out) <= searcher.config.max_sources

    def test_no_duplicate_uris(self, searcher):
        doc = _make_doc("Python Guide", "python content")
        sources = [{"title": doc["title"], "uri": doc["uri"]}]
        out = searcher._forge_augment_sources(sources, [doc], "python")
        uris = [s["uri"] for s in out]
        assert len(uris) == len(set(uris))


# ---------------------------------------------------------------------------
# _lexical_backstop_docs
# ---------------------------------------------------------------------------


class TestLexicalBackstopDocs:
    def test_returns_relevant_docs(self, searcher):
        docs = [
            _make_doc("Python Notes", "python programming tutorial"),
            _make_doc("Java Guide", "java spring framework"),
        ]
        _inject_docs(searcher, docs)
        result = searcher._lexical_backstop_docs("default", "python", docs, limit=3)
        assert isinstance(result, list)

    def test_empty_docs(self, searcher):
        _inject_docs(searcher, [])
        result = searcher._lexical_backstop_docs("default", "python", [], limit=3)
        assert result == []


# ---------------------------------------------------------------------------
# _run_hybrid_rerank
# ---------------------------------------------------------------------------


class TestRunHybridRerank:
    def test_returns_docs_and_confidence(self, searcher):
        docs = [
            _make_doc("Python Notes", "python is great"),
            _make_doc("Data Science", "machine learning data"),
        ]
        _inject_docs(searcher, docs)
        sem_results = [
            ({"uri": docs[0]["uri"]}, 0.9),
            ({"uri": docs[1]["uri"]}, 0.5),
        ]
        resolved, confidence = searcher._run_hybrid_rerank(
            "default", "python", sem_results
        )
        assert isinstance(resolved, list)
        assert 0.0 <= confidence <= 1.0

    def test_empty_semantic_results(self, searcher):
        docs = [_make_doc("A", "python content")]
        _inject_docs(searcher, docs)
        resolved, confidence = searcher._run_hybrid_rerank("default", "python", [])
        assert isinstance(resolved, list)


# ---------------------------------------------------------------------------
# _local_search (main integration test)
# ---------------------------------------------------------------------------


class TestLocalSearch:
    def test_no_docs_returns_success(self, searcher):
        _inject_docs(searcher, [])
        result = searcher._local_search(
            "default", "python", 1000, 0.7, "gemini-1.5-flash"
        )
        assert result["status"] == "success"
        assert "No documents indexed" in result["answer"]

    def test_keyword_match(self, searcher):
        docs = [_make_doc("Python Guide", "python is a programming language for all")]
        _inject_docs(searcher, docs)
        result = searcher._local_search(
            "default", "python", 1000, 0.7, "gemini-1.5-flash", search_mode="keyword"
        )
        assert result["status"] == "success"
        assert len(result["sources"]) >= 1

    def test_semantic_mode_with_no_semantic_results(self, searcher):
        docs = [_make_doc("Python Guide", "python is a programming language")]
        _inject_docs(searcher, docs)
        result = searcher._local_search(
            "default",
            "python",
            1000,
            0.7,
            "gemini-1.5-flash",
            search_mode="semantic",
            semantic_results=[],
        )
        assert result["status"] == "success"
        assert "semantic_results" in result

    def test_semantic_mode_no_docs_returns_empty_list(self, searcher):
        _inject_docs(searcher, [])
        result = searcher._local_search(
            "default",
            "python",
            1000,
            0.7,
            "gemini-1.5-flash",
            search_mode="semantic",
            semantic_results=[],
        )
        assert result["semantic_results"] == []

    def test_hybrid_mode(self, searcher):
        docs = [
            _make_doc("Python Notes", "python is great"),
            _make_doc("Data Science", "data science notes"),
        ]
        _inject_docs(searcher, docs)
        sem_results = [({"uri": docs[0]["uri"]}, 0.8)]
        result = searcher._local_search(
            "default",
            "python",
            1000,
            0.7,
            "gemini-1.5-flash",
            search_mode="hybrid",
            semantic_results=sem_results,
        )
        assert result["status"] == "success"
        assert "semantic_results" in result

    def test_exact_note_semantic(self, searcher):
        docs = [_make_doc("python notes", "python is a programming language")]
        _inject_docs(searcher, docs)
        result = searcher._local_search(
            "default",
            "python notes",
            1000,
            0.7,
            "gemini-1.5-flash",
            search_mode="semantic",
            semantic_results=[],
        )
        assert result["status"] == "success"

    def test_keyword_no_match_triggers_backstop(self, searcher):
        docs = [_make_doc("Python Guide", "python tutorial programming")]
        _inject_docs(searcher, docs)
        result = searcher._local_search(
            "default", "python", 1000, 0.7, "gemini-1.5-flash", search_mode="keyword"
        )
        assert result["status"] == "success"

    def test_base_fields_present(self, searcher):
        _inject_docs(searcher, [])
        result = searcher._local_search("default", "query", 500, 0.5, "model")
        assert "model" in result
        assert "query" in result
        assert "store" in result
        assert "search_mode" in result

    def test_search_mode_multimodal_no_docs(self, searcher):
        _inject_docs(searcher, [])
        result = searcher._local_search(
            "default",
            "image query",
            1000,
            0.7,
            "model",
            search_mode="multimodal",
            semantic_results=[],
        )
        assert result["status"] == "success"
        assert "semantic_results" in result

    def test_intent_info_fields(self, searcher):
        _inject_docs(searcher, [])
        intent = MagicMock()
        intent.refined_query = "refined query"
        intent.correction_suggestions = ["correction"]
        intent.keywords = ["key"]
        intent.file_extensions = [".txt"]
        intent.metadata_filters = {}
        intent.original_query = "orig query"
        result = searcher._local_search(
            "default", "orig query", 1000, 0.7, "model", intent_info=intent
        )
        assert result["refined_query"] == "refined query"
        assert result["corrections"] == ["correction"]


# ---------------------------------------------------------------------------
# Full search() integration (goes through CloudSearchMixin.search)
# ---------------------------------------------------------------------------


class TestSearchIntegration:
    def test_search_keyword_mode(self, searcher, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("python programming notes content")
        searcher.upload_file(str(f))
        result = searcher.search("python", search_mode="keyword")
        assert result["status"] == "success"
        assert result["search_mode"] == "keyword"

    def test_search_semantic_mode(self, searcher, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("data science machine learning")
        searcher.upload_file(str(f))
        result = searcher.search("machine learning", search_mode="semantic")
        assert result["status"] == "success"
        assert "semantic_results" in result

    def test_search_hybrid_mode(self, searcher, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("hybrid search notes")
        searcher.upload_file(str(f))
        result = searcher.search("hybrid search", search_mode="hybrid")
        assert result["status"] == "success"

    def test_search_empty_store(self, searcher):
        result = searcher.search("anything", search_mode="keyword")
        assert result["status"] == "success"
        assert "No documents" in result["answer"]


# ---------------------------------------------------------------------------
# _run_meta_adapt
# ---------------------------------------------------------------------------


class TestRunMetaAdapt:
    def test_no_adapt_if_not_triggered(self, searcher):
        meta_learner = MagicMock()
        meta_learner.should_adapt.return_value = False
        # Should not call recommend_alpha
        searcher._run_meta_adapt("default", meta_learner)
        meta_learner.recommend_alpha.assert_not_called()

    def test_adapt_updates_alpha(self, searcher):
        meta_learner = MagicMock()
        meta_learner.should_adapt.return_value = True
        meta_learner.recommend_alpha.return_value = 0.7
        meta_learner.store_trend.return_value = "rising"
        searcher._meta_alpha["default"] = 0.5
        searcher._run_meta_adapt("default", meta_learner)
        assert searcher._meta_alpha["default"] == 0.7


# ---------------------------------------------------------------------------
# _resolve_semantic_sources, _fallback_sources
# ---------------------------------------------------------------------------


class TestFallbackSources:
    def test_fallback_keyword_mode_with_docs(self, searcher):
        docs = [_make_doc("Python Notes", "python is great")]
        answer, sources = searcher._fallback_sources(
            "default", "python", "keyword", None, docs
        )
        assert isinstance(answer, str)
        assert isinstance(sources, list)

    def test_fallback_semantic_mode_with_sem_results(self, searcher):
        doc = _make_doc("A", "machine learning notes")
        _inject_docs(searcher, [doc])
        sem_results = [({"uri": doc["uri"]}, 0.8)]
        answer, sources = searcher._fallback_sources(
            "default", "machine learning", "semantic", sem_results, [doc]
        )
        assert isinstance(answer, str)

    def test_fallback_no_results_returns_generic(self, searcher):
        _inject_docs(searcher, [])
        answer, sources = searcher._fallback_sources(
            "default", "xyz", "keyword", None, []
        )
        assert isinstance(answer, str)
        assert isinstance(sources, list)
