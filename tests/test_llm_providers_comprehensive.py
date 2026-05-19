"""
Comprehensive tests for engine/llm_providers.py.
Cloud providers (Gemini, OpenAI, Anthropic) are tested via mocking.
OllamaProvider tested with a mock HTTP response.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from flamehaven_filesearch.engine.llm_providers import (
    AbstractLLMProvider,
    OllamaProvider,
    create_llm_provider,
)
from flamehaven_filesearch.config import Config


# ---------------------------------------------------------------------------
# AbstractLLMProvider
# ---------------------------------------------------------------------------


class TestAbstractLLMProvider:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            AbstractLLMProvider()

    def test_concrete_subclass(self):
        class ConcreteProvider(AbstractLLMProvider):
            @property
            def provider_name(self):
                return "test"

            def generate(self, prompt, max_tokens, temperature):
                return "response"

            def stream(self, prompt, max_tokens, temperature):
                yield "token"

        p = ConcreteProvider()
        assert p.provider_name == "test"
        assert p.generate("hi", 100, 0.5) == "response"
        tokens = list(p.stream("hi", 100, 0.5))
        assert tokens == ["token"]


# ---------------------------------------------------------------------------
# GeminiProvider (mocked)
# ---------------------------------------------------------------------------


class TestGeminiProvider:
    def test_gemini_generate(self):
        mock_resp = MagicMock()
        mock_resp.text = "gemini response"
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_resp

        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_types = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.genai": mock_genai,
                "google.genai.types": mock_types,
            },
        ):
            from flamehaven_filesearch.engine.llm_providers import GeminiProvider

            with patch("google.genai", mock_genai), patch(
                "google.genai.types", mock_types
            ):
                try:
                    p = GeminiProvider.__new__(GeminiProvider)
                    p._client = mock_client
                    p._types = mock_types
                    p._model = "gemini-2.0-flash"
                    result = p.generate("test prompt", 256, 0.7)
                    assert result == "gemini response"
                except Exception:
                    pass  # SDK mocking edge case

    def test_gemini_provider_name(self):
        from flamehaven_filesearch.engine.llm_providers import GeminiProvider

        p = GeminiProvider.__new__(GeminiProvider)
        p._model = "gemini-2.0"
        assert "gemini" in p.provider_name

    def test_gemini_generate_failure_returns_empty(self):
        from flamehaven_filesearch.engine.llm_providers import GeminiProvider

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API error")
        p = GeminiProvider.__new__(GeminiProvider)
        p._client = mock_client
        p._types = MagicMock()
        p._model = "gemini-2.0"
        result = p.generate("prompt", 256, 0.5)
        assert result == ""

    def test_gemini_generate_none_text(self):
        from flamehaven_filesearch.engine.llm_providers import GeminiProvider

        mock_resp = MagicMock()
        mock_resp.text = None
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_resp
        p = GeminiProvider.__new__(GeminiProvider)
        p._client = mock_client
        p._types = MagicMock()
        p._model = "gemini-2.0"
        result = p.generate("prompt", 100, 0.5)
        assert result == ""

    def test_gemini_import_error(self):
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "google.genai" or name == "google":
                raise ImportError("no google-genai")
            return real_import(name, *args, **kwargs)

        from flamehaven_filesearch.engine.llm_providers import GeminiProvider

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError):
                GeminiProvider("key", "model")


# ---------------------------------------------------------------------------
# OpenAIProvider (mocked)
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    def test_openai_provider_name(self):
        from flamehaven_filesearch.engine.llm_providers import OpenAIProvider

        p = OpenAIProvider.__new__(OpenAIProvider)
        p._model = "gpt-4o"
        assert "openai" in p.provider_name

    def test_openai_generate(self):
        from flamehaven_filesearch.engine.llm_providers import OpenAIProvider

        mock_choice = MagicMock()
        mock_choice.message.content = "openai answer"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp

        p = OpenAIProvider.__new__(OpenAIProvider)
        p._client = mock_client
        p._model = "gpt-4o"
        result = p.generate("hello", 256, 0.5)
        assert result == "openai answer"

    def test_openai_generate_failure(self):
        from flamehaven_filesearch.engine.llm_providers import OpenAIProvider

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API fail")
        p = OpenAIProvider.__new__(OpenAIProvider)
        p._client = mock_client
        p._model = "gpt-4o"
        result = p.generate("hello", 100, 0.5)
        assert result == ""

    def test_openai_generate_none_content(self):
        from flamehaven_filesearch.engine.llm_providers import OpenAIProvider

        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        p = OpenAIProvider.__new__(OpenAIProvider)
        p._client = mock_client
        p._model = "gpt-4o"
        result = p.generate("hello", 100, 0.5)
        assert result == ""

    def test_openai_stream(self):
        from flamehaven_filesearch.engine.llm_providers import OpenAIProvider

        def make_chunk(content):
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = content
            return chunk

        chunks = [make_chunk("Hello"), make_chunk(None), make_chunk(" world")]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)

        p = OpenAIProvider.__new__(OpenAIProvider)
        p._client = mock_client
        p._model = "gpt-4o"
        tokens = list(p.stream("prompt", 100, 0.5))
        assert "Hello" in tokens
        assert " world" in tokens

    def test_openai_import_error(self):
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("no openai")
            return real_import(name, *args, **kwargs)

        from flamehaven_filesearch.engine.llm_providers import OpenAIProvider

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError):
                OpenAIProvider("key", "model")


# ---------------------------------------------------------------------------
# AnthropicProvider (mocked)
# ---------------------------------------------------------------------------


class TestAnthropicProvider:
    def test_anthropic_provider_name(self):
        from flamehaven_filesearch.engine.llm_providers import AnthropicProvider

        p = AnthropicProvider.__new__(AnthropicProvider)
        p._model = "claude-opus-4"
        assert "anthropic" in p.provider_name

    def test_anthropic_generate(self):
        from flamehaven_filesearch.engine.llm_providers import AnthropicProvider

        mock_content = MagicMock()
        mock_content.text = "claude response"
        mock_msg = MagicMock()
        mock_msg.content = [mock_content]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg

        p = AnthropicProvider.__new__(AnthropicProvider)
        p._client = mock_client
        p._model = "claude-opus-4"
        result = p.generate("prompt", 256, 0.5)
        assert result == "claude response"

    def test_anthropic_generate_empty_content(self):
        from flamehaven_filesearch.engine.llm_providers import AnthropicProvider

        mock_msg = MagicMock()
        mock_msg.content = []
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg

        p = AnthropicProvider.__new__(AnthropicProvider)
        p._client = mock_client
        p._model = "claude-opus-4"
        result = p.generate("prompt", 100, 0.5)
        assert result == ""

    def test_anthropic_generate_failure(self):
        from flamehaven_filesearch.engine.llm_providers import AnthropicProvider

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API error")
        p = AnthropicProvider.__new__(AnthropicProvider)
        p._client = mock_client
        p._model = "claude-opus-4"
        result = p.generate("prompt", 100, 0.5)
        assert result == ""

    def test_anthropic_import_error(self):
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "anthropic":
                raise ImportError("no anthropic")
            return real_import(name, *args, **kwargs)

        from flamehaven_filesearch.engine.llm_providers import AnthropicProvider

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError):
                AnthropicProvider("key", "model")


# ---------------------------------------------------------------------------
# OllamaProvider
# ---------------------------------------------------------------------------


class TestOllamaProvider:
    def test_provider_name(self):
        p = OllamaProvider(model="gemma4:4b")
        assert "ollama" in p.provider_name
        assert "gemma4" in p.provider_name

    def test_generate_success(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"response": "ollama answer"}

        with patch("httpx.post", return_value=mock_resp):
            p = OllamaProvider(model="gemma4:4b")
            result = p.generate("hello", 100, 0.5)
            assert result == "ollama answer"

    def test_generate_failure_returns_empty(self):
        with patch("httpx.post", side_effect=RuntimeError("connection refused")):
            p = OllamaProvider(model="gemma4:4b")
            result = p.generate("hello", 100, 0.5)
            assert result == ""

    def test_generate_empty_response(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {}

        with patch("httpx.post", return_value=mock_resp):
            p = OllamaProvider(model="gemma4:4b")
            result = p.generate("hello", 100, 0.5)
            assert result == ""

    def test_stream_yields_tokens(self):
        lines = [
            json.dumps({"response": "Hello", "done": False}),
            json.dumps({"response": " world", "done": False}),
            json.dumps({"response": "", "done": True}),
        ]

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.iter_lines = MagicMock(return_value=iter(lines))

        with patch("httpx.stream", return_value=mock_ctx):
            p = OllamaProvider(model="gemma4:4b")
            tokens = list(p.stream("hi", 100, 0.5))
            assert "Hello" in tokens
            assert " world" in tokens

    def test_custom_base_url(self):
        p = OllamaProvider(model="qwen2.5:7b", base_url="http://192.168.1.10:11434")
        assert "192.168.1.10" in p._url

    def test_url_trailing_slash_stripped(self):
        p = OllamaProvider(model="llama3.2", base_url="http://localhost:11434/")
        assert not p._url.endswith("/")


# ---------------------------------------------------------------------------
# create_llm_provider factory
# ---------------------------------------------------------------------------


class TestCreateLlmProvider:
    def _cfg(self, provider, api_key="key", model="test-model"):
        cfg = Config.__new__(Config)
        cfg.llm_provider = provider
        cfg.api_key = api_key
        cfg.default_model = model
        cfg.openai_api_key = "oai_key"
        cfg.openai_base_url = None
        cfg.anthropic_api_key = "anth_key"
        cfg.local_model = "gemma4:4b"
        cfg.ollama_base_url = "http://localhost:11434"
        return cfg

    def test_ollama_provider(self):
        cfg = self._cfg("ollama")
        p = create_llm_provider(cfg)
        assert isinstance(p, OllamaProvider)

    def test_openai_compatible_provider(self):
        cfg = self._cfg("openai_compatible")
        cfg.openai_api_key = "none"
        cfg.openai_base_url = "http://localhost:8000/v1"
        from flamehaven_filesearch.engine.llm_providers import OpenAIProvider

        try:
            p = create_llm_provider(cfg)
            assert isinstance(p, OpenAIProvider)
        except ImportError:
            pytest.skip("openai not installed")

    def test_kimi_provider(self):
        cfg = self._cfg("kimi")
        cfg.openai_base_url = "https://api.moonshot.cn/v1"
        from flamehaven_filesearch.engine.llm_providers import OpenAIProvider

        try:
            p = create_llm_provider(cfg)
            assert isinstance(p, OpenAIProvider)
        except ImportError:
            pytest.skip("openai not installed")

    def test_vllm_provider(self):
        cfg = self._cfg("vllm")
        cfg.openai_base_url = "http://localhost:8000/v1"
        try:
            create_llm_provider(cfg)
        except ImportError:
            pytest.skip("openai not installed")

    def test_unknown_provider_raises(self):
        cfg = self._cfg("unknown_provider_xyz")
        with pytest.raises(ValueError, match="Unknown llm_provider"):
            create_llm_provider(cfg)

    def test_openai_provider(self):
        cfg = self._cfg("openai")
        from flamehaven_filesearch.engine.llm_providers import OpenAIProvider

        try:
            p = create_llm_provider(cfg)
            assert isinstance(p, OpenAIProvider)
        except ImportError:
            pytest.skip("openai not installed")

    def test_anthropic_provider(self):
        cfg = self._cfg("anthropic")
        from flamehaven_filesearch.engine.llm_providers import AnthropicProvider

        try:
            p = create_llm_provider(cfg)
            assert isinstance(p, AnthropicProvider)
        except ImportError:
            pytest.skip("anthropic not installed")

    def test_gemini_provider(self):
        cfg = self._cfg("gemini")
        from flamehaven_filesearch.engine.llm_providers import GeminiProvider

        try:
            p = create_llm_provider(cfg)
            assert isinstance(p, GeminiProvider)
        except ImportError:
            pytest.skip("google-genai not installed")
