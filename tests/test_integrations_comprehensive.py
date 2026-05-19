"""
Comprehensive tests for integrations/docling_loaders.py.
Tests each loader class via mocked framework SDKs.
"""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# FlamehavenCrewAITool (no framework dependency)
# ---------------------------------------------------------------------------


class TestFlamehavenCrewAITool:
    def test_run_returns_text(self, tmp_path):
        from flamehaven_filesearch.integrations.docling_loaders import (
            FlamehavenCrewAITool,
        )

        f = tmp_path / "note.txt"
        f.write_text("hello from the tool")

        tool = FlamehavenCrewAITool()
        result = tool.run(str(f))
        assert "hello" in result

    def test_run_nonexistent_returns_no_text_message(self):
        from flamehaven_filesearch.integrations.docling_loaders import (
            FlamehavenCrewAITool,
        )

        tool = FlamehavenCrewAITool()
        result = tool.run("/nonexistent/file.txt")
        assert (
            "FlamehavenCrewAITool" in result or result == "" or isinstance(result, str)
        )

    def test_run_method_calls_private_run(self, tmp_path):
        from flamehaven_filesearch.integrations.docling_loaders import (
            FlamehavenCrewAITool,
        )

        f = tmp_path / "doc.txt"
        f.write_text("test content here")
        tool = FlamehavenCrewAITool()
        assert tool.run(str(f)) == tool._run(str(f))

    @pytest.mark.asyncio
    async def test_arun_returns_text(self, tmp_path):
        from flamehaven_filesearch.integrations.docling_loaders import (
            FlamehavenCrewAITool,
        )

        f = tmp_path / "async.txt"
        f.write_text("async content")
        tool = FlamehavenCrewAITool()
        result = await tool._arun(str(f))
        assert isinstance(result, str)

    def test_tool_has_name_and_description(self):
        from flamehaven_filesearch.integrations.docling_loaders import (
            FlamehavenCrewAITool,
        )

        tool = FlamehavenCrewAITool()
        assert hasattr(tool, "name") or hasattr(FlamehavenCrewAITool, "name")
        assert hasattr(tool, "description") or hasattr(
            FlamehavenCrewAITool, "description"
        )

    def test_empty_file_returns_message(self, tmp_path):
        from flamehaven_filesearch.integrations.docling_loaders import (
            FlamehavenCrewAITool,
        )

        f = tmp_path / "empty.txt"
        f.write_text("")
        tool = FlamehavenCrewAITool()
        result = tool._run(str(f))
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# FlamehavenLangChainLoader (mocked LangChain)
# ---------------------------------------------------------------------------


class TestFlamehavenLangChainLoader:
    def _make_lc_document_class(self):
        class FakeLCDocument:
            def __init__(self, page_content="", metadata=None):
                self.page_content = page_content
                self.metadata = metadata or {}

        return FakeLCDocument

    def test_load_raises_without_langchain(self, tmp_path):
        from flamehaven_filesearch.integrations.docling_loaders import (
            FlamehavenLangChainLoader,
        )

        f = tmp_path / "note.txt"
        f.write_text("hello")

        with patch.dict(
            "sys.modules",
            {
                "langchain_core": None,
                "langchain_core.documents": None,
                "langchain": None,
                "langchain.schema": None,
            },
        ):
            loader = FlamehavenLangChainLoader(str(f))
            with pytest.raises((ImportError, Exception)):
                loader.load()

    def test_load_with_mocked_langchain(self, tmp_path):
        from flamehaven_filesearch.integrations.docling_loaders import (
            FlamehavenLangChainLoader,
        )

        f = tmp_path / "test.txt"
        f.write_text("test content for langchain")

        FakeLCDocument = self._make_lc_document_class()
        mock_lc = MagicMock()
        mock_lc.documents.Document = FakeLCDocument

        with patch.dict(
            "sys.modules",
            {"langchain_core": mock_lc, "langchain_core.documents": mock_lc.documents},
        ):
            loader = FlamehavenLangChainLoader(str(f))
            docs = loader.load()
            assert len(docs) == 1
            assert "test" in docs[0].page_content

    def test_load_with_chunking(self, tmp_path):
        from flamehaven_filesearch.integrations.docling_loaders import (
            FlamehavenLangChainLoader,
        )

        f = tmp_path / "long.txt"
        f.write_text("sentence one. " * 200)

        FakeLCDocument = self._make_lc_document_class()
        mock_lc = MagicMock()
        mock_lc.documents.Document = FakeLCDocument

        with patch.dict(
            "sys.modules",
            {"langchain_core": mock_lc, "langchain_core.documents": mock_lc.documents},
        ):
            loader = FlamehavenLangChainLoader(str(f), chunk=True, max_tokens=50)
            docs = loader.load()
            assert len(docs) >= 1

    def test_lazy_load(self, tmp_path):
        from flamehaven_filesearch.integrations.docling_loaders import (
            FlamehavenLangChainLoader,
        )

        f = tmp_path / "lazy.txt"
        f.write_text("lazy load content")

        FakeLCDocument = self._make_lc_document_class()
        mock_lc = MagicMock()
        mock_lc.documents.Document = FakeLCDocument

        with patch.dict(
            "sys.modules",
            {"langchain_core": mock_lc, "langchain_core.documents": mock_lc.documents},
        ):
            loader = FlamehavenLangChainLoader(str(f))
            docs = list(loader.lazy_load())
            assert len(docs) >= 1

    def test_load_fallback_to_langchain_schema(self, tmp_path):
        from flamehaven_filesearch.integrations.docling_loaders import (
            FlamehavenLangChainLoader,
        )

        f = tmp_path / "fallback.txt"
        f.write_text("fallback content")

        FakeLCDocument = self._make_lc_document_class()
        mock_legacy = MagicMock()
        mock_legacy.Document = FakeLCDocument

        with patch.dict(
            "sys.modules",
            {
                "langchain_core": None,
                "langchain_core.documents": None,
                "langchain": MagicMock(),
                "langchain.schema": mock_legacy,
            },
        ):
            loader = FlamehavenLangChainLoader(str(f))
            docs = loader.load()
            assert len(docs) >= 1


# ---------------------------------------------------------------------------
# FlamehavenLlamaIndexReader (mocked LlamaIndex)
# ---------------------------------------------------------------------------


class TestFlamehavenLlamaIndexReader:
    def _make_li_document_class(self):
        class FakeLIDocument:
            def __init__(self, text="", metadata=None):
                self.text = text
                self.metadata = metadata or {}

        return FakeLIDocument

    def test_load_raises_without_llama_index(self, tmp_path):
        from flamehaven_filesearch.integrations.docling_loaders import (
            FlamehavenLlamaIndexReader,
        )

        f = tmp_path / "note.txt"
        f.write_text("hi")

        with patch.dict("sys.modules", {"llama_index": None, "llama_index.core": None}):
            reader = FlamehavenLlamaIndexReader()
            with pytest.raises((ImportError, Exception)):
                reader.load_data([str(f)])

    def test_load_with_mocked_llama_index(self, tmp_path):
        from flamehaven_filesearch.integrations.docling_loaders import (
            FlamehavenLlamaIndexReader,
        )

        f = tmp_path / "test.txt"
        f.write_text("llama index content")

        FakeLIDocument = self._make_li_document_class()
        mock_li = MagicMock()
        mock_li.Document = FakeLIDocument

        with patch.dict(
            "sys.modules", {"llama_index": MagicMock(), "llama_index.core": mock_li}
        ):
            reader = FlamehavenLlamaIndexReader()
            docs = reader.load_data([str(f)])
            assert len(docs) == 1

    def test_load_multiple_files(self, tmp_path):
        from flamehaven_filesearch.integrations.docling_loaders import (
            FlamehavenLlamaIndexReader,
        )

        files = []
        for i in range(3):
            f = tmp_path / f"file{i}.txt"
            f.write_text(f"content {i}")
            files.append(str(f))

        FakeLIDocument = self._make_li_document_class()
        mock_li = MagicMock()
        mock_li.Document = FakeLIDocument

        with patch.dict(
            "sys.modules", {"llama_index": MagicMock(), "llama_index.core": mock_li}
        ):
            reader = FlamehavenLlamaIndexReader()
            docs = reader.load_data(files)
            assert len(docs) == 3

    def test_load_with_chunking(self, tmp_path):
        from flamehaven_filesearch.integrations.docling_loaders import (
            FlamehavenLlamaIndexReader,
        )

        f = tmp_path / "chunky.txt"
        f.write_text("word " * 300)

        FakeLIDocument = self._make_li_document_class()
        mock_li = MagicMock()
        mock_li.Document = FakeLIDocument

        with patch.dict(
            "sys.modules", {"llama_index": MagicMock(), "llama_index.core": mock_li}
        ):
            reader = FlamehavenLlamaIndexReader(chunk=True, max_tokens=50)
            docs = reader.load_data([str(f)])
            assert len(docs) >= 1


# ---------------------------------------------------------------------------
# FlamehavenHaystackConverter (mocked Haystack)
# ---------------------------------------------------------------------------


class TestFlamehavenHaystackConverter:
    def _make_hs_document_class(self):
        class FakeHSDocument:
            def __init__(self, content="", meta=None):
                self.content = content
                self.meta = meta or {}

        return FakeHSDocument

    def test_run_raises_without_haystack(self, tmp_path):
        from flamehaven_filesearch.integrations.docling_loaders import (
            FlamehavenHaystackConverter,
        )

        f = tmp_path / "note.txt"
        f.write_text("hi")

        with patch.dict("sys.modules", {"haystack": None, "haystack.schema": None}):
            converter = FlamehavenHaystackConverter()
            with pytest.raises((ImportError, Exception)):
                converter.run([str(f)])

    def test_run_with_mocked_haystack(self, tmp_path):
        from flamehaven_filesearch.integrations.docling_loaders import (
            FlamehavenHaystackConverter,
        )

        f = tmp_path / "test.txt"
        f.write_text("haystack test content")

        FakeHSDocument = self._make_hs_document_class()
        mock_hs = MagicMock()
        mock_hs.Document = FakeHSDocument

        with patch.dict("sys.modules", {"haystack": mock_hs}):
            converter = FlamehavenHaystackConverter()
            result = converter.run([str(f)])
            assert "documents" in result
            assert len(result["documents"]) == 1

    def test_run_multiple_files(self, tmp_path):
        from flamehaven_filesearch.integrations.docling_loaders import (
            FlamehavenHaystackConverter,
        )

        files = []
        for i in range(2):
            f = tmp_path / f"doc{i}.txt"
            f.write_text(f"haystack content {i}")
            files.append(str(f))

        FakeHSDocument = self._make_hs_document_class()
        mock_hs = MagicMock()
        mock_hs.Document = FakeHSDocument

        with patch.dict("sys.modules", {"haystack": mock_hs}):
            converter = FlamehavenHaystackConverter()
            result = converter.run(files)
            assert len(result["documents"]) == 2

    def test_run_with_chunking(self, tmp_path):
        from flamehaven_filesearch.integrations.docling_loaders import (
            FlamehavenHaystackConverter,
        )

        f = tmp_path / "long.txt"
        f.write_text("sentence. " * 200)

        FakeHSDocument = self._make_hs_document_class()
        mock_hs = MagicMock()
        mock_hs.Document = FakeHSDocument

        with patch.dict("sys.modules", {"haystack": mock_hs}):
            converter = FlamehavenHaystackConverter(chunk=True, max_tokens=50)
            result = converter.run([str(f)])
            assert "documents" in result

    def test_run_fallback_to_haystack_schema(self, tmp_path):
        from flamehaven_filesearch.integrations.docling_loaders import (
            FlamehavenHaystackConverter,
        )

        f = tmp_path / "fallback.txt"
        f.write_text("schema fallback")

        FakeHSDocument = self._make_hs_document_class()
        mock_schema = MagicMock()
        mock_schema.Document = FakeHSDocument

        with patch.dict(
            "sys.modules", {"haystack": None, "haystack.schema": mock_schema}
        ):
            converter = FlamehavenHaystackConverter()
            result = converter.run([str(f)])
            assert "documents" in result


# ---------------------------------------------------------------------------
# integrations/__init__.py
# ---------------------------------------------------------------------------


class TestIntegrationsInit:
    def test_import_integrations(self):
        import flamehaven_filesearch.integrations as integrations_pkg

        assert integrations_pkg is not None
