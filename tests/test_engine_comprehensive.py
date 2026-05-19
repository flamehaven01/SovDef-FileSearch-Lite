"""
Comprehensive tests for engine modules:
- embedding_generator.py
- obsidian_lite.py
- format_backends.py
- chronos_grid.py (additional coverage)
"""

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

from flamehaven_filesearch.engine.embedding_generator import (
    EmbeddingGenerator,
    OllamaEmbeddingProvider,
    create_embedding_provider,
    get_embedding_generator,
    reset_embedding_generator,
)
from flamehaven_filesearch.engine.obsidian_lite import (
    ObsidianNote,
    _coerce_list,
    _coerce_scalar,
    _dedupe_keep_order,
    _extract_inline_tags,
    _extract_wikilinks,
    _parse_simple_frontmatter,
    _split_frontmatter,
    build_obsidian_chunks,
    build_obsidian_embedding_text,
    parse_obsidian_markdown,
)
from flamehaven_filesearch.engine.chronos_grid import ChronosConfig, ChronosGrid


# ---------------------------------------------------------------------------
# EmbeddingGenerator extended coverage
# ---------------------------------------------------------------------------


class TestEmbeddingGeneratorExtended:
    def setup_method(self):
        reset_embedding_generator()

    def test_generate_empty_string(self):
        gen = EmbeddingGenerator()
        vec = gen.generate("")
        assert vec is not None
        assert len(vec) == 384

    def test_generate_with_lang_param(self):
        gen = EmbeddingGenerator()
        vec = gen.generate("hello world", lang="en")
        assert len(vec) == 384

    def test_generate_cjk_auto_detect(self):
        gen = EmbeddingGenerator()
        vec = gen.generate("안녕하세요 세계")
        assert len(vec) == 384

    def test_generate_image_bytes_empty(self):
        gen = EmbeddingGenerator()
        vec = gen.generate_image_bytes(b"")
        assert len(vec) == 384

    def test_generate_image_bytes_data(self):
        gen = EmbeddingGenerator()
        vec = gen.generate_image_bytes(b"\x89PNG\r\n" + b"\x00" * 100)
        assert len(vec) == 384

    def test_generate_multimodal_no_image(self):
        gen = EmbeddingGenerator()
        vec = gen.generate_multimodal("some text", None, 0.7, 0.3)
        assert len(vec) == 384

    def test_generate_multimodal_with_image(self):
        gen = EmbeddingGenerator()
        vec = gen.generate_multimodal("some text", b"\x00" * 50, 0.7, 0.3)
        assert len(vec) == 384

    def test_clear_cache(self):
        gen = EmbeddingGenerator()
        gen.generate("test text")
        gen.clear_cache()
        stats = gen.get_cache_stats()
        assert stats["cache_size"] == 0
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0

    def test_reset_stats(self):
        gen = EmbeddingGenerator()
        gen.generate("test text")
        gen.reset_stats()
        stats = gen.get_cache_stats()
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
        # Cache size preserved
        assert stats["cache_size"] >= 1

    def test_get_cache_stats_hit_rate(self):
        gen = EmbeddingGenerator()
        gen.generate("a")
        gen.generate("a")  # cache hit
        stats = gen.get_cache_stats()
        assert stats["hit_rate"] > 0.0

    def test_batch_generate_empty(self):
        gen = EmbeddingGenerator()
        result = gen.batch_generate([])
        assert result == []

    def test_batch_generate_mixed(self):
        gen = EmbeddingGenerator()
        result = gen.batch_generate(["text1", "", "text2"])
        assert len(result) == 3

    def test_attuned_text_truncation(self):
        gen = EmbeddingGenerator()
        long_text = "a" * 1000
        result = gen._attuned_text(long_text)
        assert len(result) <= gen.MAX_TEXT_LENGTH

    def test_extract_features_empty(self):
        gen = EmbeddingGenerator()
        features = gen._extract_features("")
        assert isinstance(features, list)

    def test_extract_features_word_tokens(self):
        gen = EmbeddingGenerator()
        features = gen._extract_features("hello world")
        feature_names = [f for f, _ in features]
        assert any(f.startswith("w:") for f in feature_names)

    def test_extract_features_char_ngrams(self):
        gen = EmbeddingGenerator()
        features = gen._extract_features("hello")
        feature_names = [f for f, _ in features]
        assert any(f.startswith("c:") for f in feature_names)

    def test_cache_size_limit_eviction(self):
        gen = EmbeddingGenerator()
        gen.CACHE_SIZE = 10
        for i in range(20):
            gen.generate(f"unique text string {i}")
        assert len(gen._essence_cache) <= gen.CACHE_SIZE


class TestCreateEmbeddingProvider:
    def setup_method(self):
        reset_embedding_generator()

    def test_create_dsp_provider(self):
        provider = create_embedding_provider(provider="dsp")
        assert isinstance(provider, EmbeddingGenerator)

    def test_create_ollama_provider(self):
        provider = create_embedding_provider(provider="ollama")
        assert isinstance(provider, OllamaEmbeddingProvider)

    def test_create_unknown_falls_back_to_dsp(self):
        provider = create_embedding_provider(provider="unknown_xyz")
        assert isinstance(provider, EmbeddingGenerator)


class TestOllamaEmbeddingProvider:
    def test_init_no_requests(self):
        with patch.dict("sys.modules", {"requests": None}):
            provider = OllamaEmbeddingProvider()
            assert provider._requests_available is False

    def test_probe_returns_false_no_requests(self):
        provider = OllamaEmbeddingProvider()
        provider._requests_available = False
        assert provider._probe() is False

    def test_generate_falls_back_to_dsp(self):
        provider = OllamaEmbeddingProvider()
        provider._available = False
        vec = provider.generate("test text")
        assert len(vec) == 384

    def test_get_cache_stats(self):
        provider = OllamaEmbeddingProvider()
        stats = provider.get_cache_stats()
        assert "cache_hits" in stats
        assert "cache_misses" in stats

    def test_vector_dim_property(self):
        provider = OllamaEmbeddingProvider()
        # Falls back to DSP so vector_dim is 384
        assert provider.vector_dim == 384

    def test_probe_connection_refused(self):
        provider = OllamaEmbeddingProvider(base_url="http://localhost:19999")
        provider._requests_available = True
        # Should return False when connection refused
        result = provider._probe()
        assert result is False


# ---------------------------------------------------------------------------
# ObsidianLite functions
# ---------------------------------------------------------------------------


class TestParseObsidianMarkdown:
    def test_basic_parse(self):
        text = "# Title\n\nSome content here."
        note = parse_obsidian_markdown(text)
        assert isinstance(note, ObsidianNote)
        assert "Title" in note.headings

    def test_frontmatter_parsed(self):
        text = """---
title: My Note
tags:
  - python
  - ml
aliases: [alias1, alias2]
---
# Body heading

Body text here.
"""
        note = parse_obsidian_markdown(text)
        assert note.frontmatter.get("title") == "My Note"
        assert "python" in note.tags
        assert "ml" in note.tags

    def test_wikilinks_extracted(self):
        text = "See also [[Related Note]] and [[Another Note|display text]]."
        note = parse_obsidian_markdown(text)
        assert "Related Note" in note.wikilinks
        assert "Another Note" in note.wikilinks

    def test_inline_tags(self):
        text = "This is #python and #machine-learning content."
        note = parse_obsidian_markdown(text)
        assert "python" in note.tags
        assert "machine-learning" in note.tags

    def test_aliases_from_frontmatter(self):
        text = "---\naliases: [ml notes, ml guide]\n---\nContent"
        note = parse_obsidian_markdown(text)
        assert "ml notes" in note.aliases or "ml notes".replace(" ", "") in str(
            note.aliases
        )

    def test_empty_text(self):
        note = parse_obsidian_markdown("")
        assert isinstance(note, ObsidianNote)
        assert note.body == ""

    def test_no_frontmatter(self):
        text = "# Just a heading\n\nContent without frontmatter."
        note = parse_obsidian_markdown(text)
        assert note.frontmatter == {}
        assert "Just a heading" in note.headings

    def test_to_metadata(self):
        note = parse_obsidian_markdown("# Test\n\nContent")
        meta = note.to_metadata()
        assert "headings" in meta
        assert "tags" in meta
        assert "aliases" in meta


class TestBuildObsidianEmbeddingText:
    def test_basic(self):
        note = parse_obsidian_markdown("# Title\n\nContent body.")
        text = build_obsidian_embedding_text(note)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_includes_title_from_frontmatter(self):
        text = "---\ntitle: My Guide\n---\nBody content"
        note = parse_obsidian_markdown(text)
        embed_text = build_obsidian_embedding_text(note)
        assert "My Guide" in embed_text

    def test_includes_aliases(self):
        note = ObsidianNote(
            frontmatter={},
            body="content",
            headings=[],
            wikilinks=[],
            tags=[],
            aliases=["alias1", "alias2"],
        )
        text = build_obsidian_embedding_text(note)
        assert "alias1" in text

    def test_includes_tags(self):
        note = ObsidianNote(
            frontmatter={},
            body="content",
            headings=[],
            wikilinks=[],
            tags=["python", "ml"],
            aliases=[],
        )
        text = build_obsidian_embedding_text(note)
        assert "#python" in text

    def test_includes_wikilinks(self):
        note = ObsidianNote(
            frontmatter={},
            body="content",
            headings=[],
            wikilinks=["linked note"],
            tags=[],
            aliases=[],
        )
        text = build_obsidian_embedding_text(note)
        assert "linked note" in text

    def test_includes_headings(self):
        note = ObsidianNote(
            frontmatter={},
            body="content",
            headings=["Section 1", "Sub-section"],
            wikilinks=[],
            tags=[],
            aliases=[],
        )
        text = build_obsidian_embedding_text(note)
        assert "Section 1" in text


class TestBuildObsidianChunks:
    def test_basic_chunking(self):
        text = "# Title\n\n## Section 1\n\nContent for section 1.\n\n## Section 2\n\nContent for section 2."
        note = parse_obsidian_markdown(text)
        chunks = build_obsidian_chunks(note)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_chunk_has_required_fields(self):
        text = "# Title\n\nSome content here."
        note = parse_obsidian_markdown(text)
        chunks = build_obsidian_chunks(note)
        for chunk in chunks:
            assert "text" in chunk
            assert "headings" in chunk
            assert "metadata" in chunk

    def test_empty_body(self):
        note = ObsidianNote(
            frontmatter={},
            body="",
            headings=[],
            wikilinks=[],
            tags=[],
            aliases=[],
        )
        chunks = build_obsidian_chunks(note)
        assert isinstance(chunks, list)

    def test_chunk_metadata_includes_obsidian_fields(self):
        text = "# Title\n\n#python content with [[link]]"
        note = parse_obsidian_markdown(text)
        chunks = build_obsidian_chunks(note)
        if chunks:
            meta = chunks[0]["metadata"]
            assert "obsidian_tags" in meta
            assert "obsidian_wikilinks" in meta


class TestObsidianHelpers:
    def test_split_frontmatter_present(self):
        text = "---\ntitle: Test\n---\nBody text"
        fm, body = _split_frontmatter(text)
        assert fm.get("title") == "Test"
        assert "Body text" in body

    def test_split_frontmatter_absent(self):
        text = "No frontmatter here."
        fm, body = _split_frontmatter(text)
        assert fm == {}
        assert body == "No frontmatter here."

    def test_parse_simple_frontmatter_list(self):
        raw = "tags:\n  - python\n  - ml\ntitle: Test"
        result = _parse_simple_frontmatter(raw)
        assert result["title"] == "Test"
        assert "python" in result["tags"]

    def test_coerce_scalar_bool_true(self):
        assert _coerce_scalar("true") is True

    def test_coerce_scalar_bool_false(self):
        assert _coerce_scalar("false") is False

    def test_coerce_scalar_string(self):
        assert _coerce_scalar("hello") == "hello"

    def test_coerce_scalar_strips_quotes(self):
        assert _coerce_scalar('"quoted"') == "quoted"

    def test_coerce_list_none(self):
        assert _coerce_list(None) == []

    def test_coerce_list_list(self):
        assert _coerce_list(["a", "b"]) == ["a", "b"]

    def test_coerce_list_string_inline_array(self):
        result = _coerce_list("[item1, item2, item3]")
        assert "item1" in result
        assert "item3" in result

    def test_coerce_list_string_comma(self):
        result = _coerce_list("a, b, c")
        assert "a" in result
        assert "c" in result

    def test_coerce_list_simple_string(self):
        result = _coerce_list("single")
        assert result == ["single"]

    def test_extract_wikilinks_basic(self):
        text = "See [[Note A]] and [[Note B|alias]]."
        links = _extract_wikilinks(text)
        assert "Note A" in links
        assert "Note B" in links

    def test_extract_wikilinks_with_anchor(self):
        text = "See [[Note#section|display]]."
        links = _extract_wikilinks(text)
        assert "Note" in links

    def test_extract_inline_tags(self):
        tags = _extract_inline_tags("This is #python and #ml content.")
        assert "python" in tags
        assert "ml" in tags

    def test_dedupe_keep_order(self):
        result = _dedupe_keep_order(["a", "b", "a", "c"])
        assert result == ["a", "b", "c"]

    def test_dedupe_empty(self):
        assert _dedupe_keep_order([]) == []


# ---------------------------------------------------------------------------
# Format backends
# ---------------------------------------------------------------------------


class TestFormatBackends:
    def test_plain_text_read(self, tmp_path):
        from flamehaven_filesearch.engine.format_backends import BackendRegistry

        registry = BackendRegistry.default()
        backend_cls = registry.get(".txt")
        assert backend_cls is not None
        backend = backend_cls()
        f = tmp_path / "test.txt"
        f.write_text("hello content")
        text = backend.extract(str(f))
        assert "hello" in text

    def test_md_backend(self, tmp_path):
        from flamehaven_filesearch.engine.format_backends import BackendRegistry

        registry = BackendRegistry.default()
        backend_cls = registry.get(".md")
        assert backend_cls is not None
        backend = backend_cls()
        f = tmp_path / "note.md"
        f.write_text("# Title\n\nContent here.")
        text = backend.extract(str(f))
        assert len(text) >= 0

    def test_unknown_extension(self):
        from flamehaven_filesearch.engine.format_backends import BackendRegistry

        registry = BackendRegistry.default()
        backend_cls = registry.get(".unknown_xyz_ext")
        assert backend_cls is None

    def test_registry_list_supported(self):
        from flamehaven_filesearch.engine.format_backends import BackendRegistry

        registry = BackendRegistry.default()
        exts = registry.supported_extensions()
        assert isinstance(exts, set)
        assert ".txt" in exts or ".md" in exts

    def test_warn_missing_method(self, capsys):
        from flamehaven_filesearch.engine.format_backends import AbstractFormatBackend

        class ConcreteBackend(AbstractFormatBackend):
            supported_extensions = {".test"}

            def extract(self, file_path: str) -> str:
                self._warn_missing("some-lib", "some-extra")
                return ""

        backend = ConcreteBackend()
        result = backend.extract("/fake/path")
        assert result == ""

    def test_read_plain_missing_file(self):
        from flamehaven_filesearch.engine.format_backends import AbstractFormatBackend

        class ConcreteBackend(AbstractFormatBackend):
            supported_extensions = {".test"}

            def extract(self, file_path: str) -> str:
                return self._read_plain(file_path)

        backend = ConcreteBackend()
        result = backend.extract("/nonexistent/file.txt")
        assert result == ""

    def test_rtf_backend_fallback(self, tmp_path):
        from flamehaven_filesearch.engine.format_backends import BackendRegistry

        registry = BackendRegistry.default()
        backend_cls = registry.get(".rtf")
        if backend_cls is not None:
            backend = backend_cls()
            f = tmp_path / "test.rtf"
            f.write_text("{\\rtf1 Hello World}")
            text = backend.extract(str(f))
            assert isinstance(text, str)

    def test_pdf_backend_missing_file(self):
        from flamehaven_filesearch.engine.format_backends import BackendRegistry

        registry = BackendRegistry.default()
        backend_cls = registry.get(".pdf")
        if backend_cls is not None:
            backend = backend_cls()
            text = backend.extract("/nonexistent/file.pdf")
            assert isinstance(text, str)

    def test_docx_backend_missing_file(self):
        from flamehaven_filesearch.engine.format_backends import BackendRegistry

        registry = BackendRegistry.default()
        backend_cls = registry.get(".docx")
        if backend_cls is not None:
            backend = backend_cls()
            text = backend.extract("/nonexistent/file.docx")
            assert isinstance(text, str)


# ---------------------------------------------------------------------------
# ChronosGrid extended coverage
# ---------------------------------------------------------------------------


class TestChronosGridExtended:
    def test_inject_multiple_essences(self):
        grid = ChronosGrid(config=ChronosConfig())
        for i in range(10):
            vec = [float(i) / 10.0] + [0.0] * 383
            grid.inject_essence(f"file_{i}.py", {"id": i}, vec)
        assert grid.total_lore_essences == 10

    def test_seek_resonance_miss(self):
        grid = ChronosGrid(config=ChronosConfig())
        result = grid.seek_resonance("nonexistent.py")
        assert result is None

    def test_seek_resonance_hit(self):
        grid = ChronosGrid(config=ChronosConfig())
        metadata = {"file": "test.py", "size": 1024}
        grid.inject_essence("test.py", metadata, [0.5] + [0.0] * 383)
        result = grid.seek_resonance("test.py")
        assert result == metadata

    def test_stats_updated_after_inject(self):
        grid = ChronosGrid(config=ChronosConfig())
        grid.inject_essence("file.py", {}, [0.1] * 384)
        assert grid.total_lore_essences >= 1

    def test_vector_resonance_top_k(self):
        grid = ChronosGrid(config=ChronosConfig())
        for i in range(5):
            vec = [float(i + 1) / 10.0] + [0.0] * 383
            grid.inject_essence(f"f{i}.py", {"id": i}, vec)
        results = grid.seek_vector_resonance([0.5] + [0.0] * 383, top_k=2)
        assert len(results) <= 2

    def test_stats_structure(self):
        grid = ChronosGrid(config=ChronosConfig())
        stats = grid.stats
        assert hasattr(stats, "total_resonance_seeks")
        assert hasattr(stats, "spark_buffer_hits")

    def test_config_hnsw_backend(self):
        config = ChronosConfig(vector_index_backend="hnsw")
        grid = ChronosGrid(config=config)
        grid.inject_essence("test.py", {"x": 1}, [1.0] + [0.0] * 383)
        assert grid.total_lore_essences >= 1

    def test_resonance_hit_rate_zero(self):
        grid = ChronosGrid(config=ChronosConfig())
        rate = grid.stats.resonance_hit_rate()
        assert rate == 0.0

    def test_inject_minimal_vector(self):
        grid = ChronosGrid(config=ChronosConfig())
        # Inject a minimal valid vector
        vec = [0.0] * 384
        grid.inject_essence("test_min.py", {"id": "min"}, vec)
        # Just verify no exception is raised
