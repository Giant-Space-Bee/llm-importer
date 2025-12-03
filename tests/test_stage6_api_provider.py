"""
Stage 6 Tests: APIProvider (Anthropic Claude API)

What APIProvider does:
- Connect to Anthropic API with API key
- Use Claude Sonnet 4 for structured outputs
- Track RPM/TPM rate limits
- Sleep when limits approached
- is_local = False (parallel execution OK)

These are unit tests with mocked API calls.
Manual testing verifies real API integration.

Run: pytest tests/test_stage6_api_provider.py -v
"""

import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from src.providers import APIProvider, LLMProvider


class TestAPIProviderBasics:
    """APIProvider basic structure and properties."""

    def test_implements_llm_provider(self):
        """Should be an LLMProvider subclass."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider()
            assert isinstance(provider, LLMProvider)

    def test_is_local_false(self):
        """API provider = parallel execution OK."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider()
            assert provider.is_local is False

    def test_requires_api_key(self):
        """Should require ANTHROPIC_API_KEY env var or explicit key."""
        # Clear any existing key
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises((ValueError, RuntimeError)):
                APIProvider()

    def test_accepts_explicit_api_key(self):
        """Should accept API key as constructor arg."""
        provider = APIProvider(api_key="sk-ant-test-key")
        assert provider._api_key == "sk-ant-test-key"

    def test_uses_env_var_api_key(self):
        """Should use ANTHROPIC_API_KEY from environment."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-env-key"}):
            provider = APIProvider()
            assert provider._api_key == "sk-ant-env-key"

    def test_default_model_is_sonnet(self):
        """Should default to Claude Sonnet 4."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider()
            assert "sonnet" in provider.model.lower()


class TestAPIProviderRateLimiting:
    """Rate limiting (RPM/TPM) tracking and enforcement."""

    def test_has_rpm_limit(self):
        """Should have requests per minute limit."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider(rpm=5)
            assert provider.rpm == 5

    def test_has_tpm_limit(self):
        """Should have tokens per minute limit."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider(tpm=20000)
            assert provider.tpm == 20000

    def test_default_rate_limits(self):
        """Should have sensible defaults for Tier 1."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider()
            assert provider.rpm >= 5  # At least Tier 1 limits
            assert provider.tpm >= 20000

    def test_tracks_requests_this_minute(self):
        """Should track how many requests made this minute."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider()
            assert hasattr(provider, "_requests_this_minute")
            assert provider._requests_this_minute == 0

    def test_tracks_tokens_this_minute(self):
        """Should track tokens used this minute."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider()
            assert hasattr(provider, "_tokens_this_minute")
            assert provider._tokens_this_minute == 0

    def test_resets_counters_after_minute(self):
        """Should reset counters when minute rolls over."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider()
            # Simulate past minute
            provider._minute_start = time.time() - 61
            provider._requests_this_minute = 100
            provider._tokens_this_minute = 50000

            provider._maybe_reset_minute()

            assert provider._requests_this_minute == 0
            assert provider._tokens_this_minute == 0

    def test_wait_if_rpm_exceeded(self):
        """Should wait if RPM limit would be exceeded."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider(rpm=5)
            provider._requests_this_minute = 5
            provider._minute_start = time.time()

            # Should return wait time > 0
            wait_time = provider._get_rate_limit_wait()
            assert wait_time > 0

    def test_wait_if_tpm_exceeded(self):
        """Should wait if TPM limit would be exceeded."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider(tpm=20000)
            provider._tokens_this_minute = 20000
            provider._minute_start = time.time()

            # Estimate for next request
            wait_time = provider._get_rate_limit_wait(estimated_tokens=1000)
            assert wait_time > 0

    def test_no_wait_when_under_limits(self):
        """Should not wait when under limits."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider(rpm=5, tpm=20000)
            provider._requests_this_minute = 2
            provider._tokens_this_minute = 5000
            provider._minute_start = time.time()

            wait_time = provider._get_rate_limit_wait()
            assert wait_time == 0


class TestAPIProviderComplete:
    """complete() method - plain text completion."""

    @patch("anthropic.Anthropic")
    def test_complete_returns_text(self, mock_anthropic_class):
        """Should return response text."""
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello, world!")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_client.messages.create.return_value = mock_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider()
            result = provider.complete("Say hello")

        assert result == "Hello, world!"

    @patch("anthropic.Anthropic")
    def test_complete_updates_token_count(self, mock_anthropic_class):
        """Should update tokens_this_minute after call."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_client.messages.create.return_value = mock_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider()
            initial_tokens = provider._tokens_this_minute
            provider.complete("Test prompt")

        assert provider._tokens_this_minute == initial_tokens + 150

    @patch("anthropic.Anthropic")
    def test_complete_updates_request_count(self, mock_anthropic_class):
        """Should increment requests_this_minute after call."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_client.messages.create.return_value = mock_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider()
            initial_requests = provider._requests_this_minute
            provider.complete("Test")

        assert provider._requests_this_minute == initial_requests + 1


class TestAPIProviderStructuredOutput:
    """complete_structured() method - JSON schema output."""

    @patch("anthropic.Anthropic")
    def test_structured_returns_dict(self, mock_anthropic_class):
        """Should return parsed dict from JSON response."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"facts": []}')]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 10
        mock_client.beta.messages.create.return_value = mock_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider()
            result = provider.complete_structured("Extract facts", {"type": "object"})

        assert isinstance(result, dict)
        assert "facts" in result

    @patch("anthropic.Anthropic")
    def test_structured_uses_beta_endpoint(self, mock_anthropic_class):
        """Should use beta.messages.create for structured outputs."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"result": "ok"}')]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_client.beta.messages.create.return_value = mock_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider()
            provider.complete_structured("Test", {"type": "object"})

        # Verify beta endpoint was called
        mock_client.beta.messages.create.assert_called_once()

    @patch("anthropic.Anthropic")
    def test_structured_includes_beta_header(self, mock_anthropic_class):
        """Should include structured-outputs beta header."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"result": "ok"}')]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_client.beta.messages.create.return_value = mock_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider()
            provider.complete_structured("Test", {"type": "object"})

        call_kwargs = mock_client.beta.messages.create.call_args[1]
        assert "betas" in call_kwargs
        assert "structured-outputs-2025-11-13" in call_kwargs["betas"]

    @patch("anthropic.Anthropic")
    def test_structured_passes_schema(self, mock_anthropic_class):
        """Should pass JSON schema in output_format."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"result": "ok"}')]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_client.beta.messages.create.return_value = mock_response

        test_schema = {
            "type": "object",
            "properties": {"facts": {"type": "array"}}
        }

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider()
            provider.complete_structured("Test", test_schema)

        call_kwargs = mock_client.beta.messages.create.call_args[1]
        assert "output_format" in call_kwargs


class TestAPIProviderErrorHandling:
    """Error handling for API failures."""

    @patch("anthropic.Anthropic")
    def test_handles_api_error(self, mock_anthropic_class):
        """Should raise RuntimeError on API failure."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API Error")

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider()
            with pytest.raises(RuntimeError, match="API"):
                provider.complete("Test")

    @patch("anthropic.Anthropic")
    def test_handles_invalid_json(self, mock_anthropic_class):
        """Should raise RuntimeError on invalid JSON response."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="not valid json {")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_client.beta.messages.create.return_value = mock_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider()
            with pytest.raises(RuntimeError, match="JSON"):
                provider.complete_structured("Test", {"type": "object"})
