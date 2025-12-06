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

    @patch("anthropic.Anthropic")
    def test_handles_refusal_stop_reason(self, mock_anthropic_class):
        """Should raise RuntimeError when Claude refuses for safety reasons."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.stop_reason = "refusal"
        mock_response.content = [MagicMock(text="I can't help with that.")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_client.beta.messages.create.return_value = mock_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider()
            with pytest.raises(RuntimeError, match="refused"):
                provider.complete_structured("Test", {"type": "object"})

    @patch("anthropic.Anthropic")
    def test_handles_max_tokens_stop_reason(self, mock_anthropic_class):
        """Should raise RuntimeError when response is truncated at token limit."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.stop_reason = "max_tokens"
        mock_response.content = [MagicMock(text='{"facts": [{"incomplete":')]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 8192
        mock_client.beta.messages.create.return_value = mock_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider()
            with pytest.raises(RuntimeError, match="truncated"):
                provider.complete_structured("Test", {"type": "object"})

    @patch("anthropic.Anthropic")
    def test_handles_empty_content(self, mock_anthropic_class):
        """Should raise RuntimeError when response content is empty."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = []
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 0
        mock_client.beta.messages.create.return_value = mock_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider()
            with pytest.raises(RuntimeError, match="empty"):
                provider.complete_structured("Test", {"type": "object"})


class TestAPIProviderQualityFirstChunking:
    """Quality-first chunking: 8k chunks for better extraction, parallelism fills TPM budget."""

    def test_get_safe_chunk_size_always_quality_optimal(self):
        """Should always return 8192 (quality-optimal size) regardless of TPM."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            # All TPM values should return same quality-optimal chunk size
            for tpm in [30000, 80000, 200000, 500000]:
                provider = APIProvider(tpm=tpm)
                assert provider.get_safe_chunk_size() == 8192

    def test_get_max_concurrent_tier1(self):
        """Tier 1 (30k TPM) should allow 2 concurrent (30k / 11k ≈ 2)."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider(tpm=30000)
            # 30k / (8k + 3k) = 30k / 11k ≈ 2.7 → 2
            assert provider.get_max_concurrent() == 2

    def test_get_max_concurrent_low_tpm(self):
        """Very low TPM (10k) should still allow at least 1 concurrent."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider(tpm=10000)
            # Even with low TPM, minimum is 1
            assert provider.get_max_concurrent() >= 1

    def test_get_max_concurrent_medium_tpm(self):
        """Medium TPM (80k) should allow more parallelism."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider(tpm=80000)
            # 80k / 11k ≈ 7.2, but capped at 5
            assert provider.get_max_concurrent() == 5

    def test_get_max_concurrent_high_tpm(self):
        """High TPM (200k) should cap at 5 concurrent."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider(tpm=200000)
            # 200k / 11k ≈ 18, but capped at 5
            assert provider.get_max_concurrent() == 5

    def test_get_max_concurrent_very_high_tpm(self):
        """Very high TPM should still cap at 5 concurrent."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider(tpm=500000)
            # 500k / 11k ≈ 45, but capped at 5
            assert provider.get_max_concurrent() == 5

    def test_adaptive_methods_exist(self):
        """APIProvider should have get_safe_chunk_size and get_max_concurrent."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider()
            assert hasattr(provider, "get_safe_chunk_size")
            assert hasattr(provider, "get_max_concurrent")
            assert callable(provider.get_safe_chunk_size)
            assert callable(provider.get_max_concurrent)


class TestAPIProviderTokenEstimation:
    """Token estimation for proactive rate limiting."""

    def test_estimate_tokens_exists(self):
        """Should have _estimate_tokens method."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider()
            assert hasattr(provider, "_estimate_tokens")
            assert callable(provider._estimate_tokens)

    def test_estimate_tokens_short_prompt(self):
        """Short prompt should estimate tokens correctly."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider()
            # 40 chars / 4 = 10 tokens + 10000 reserve = 10010
            result = provider._estimate_tokens("a" * 40)
            assert result == 10010

    def test_estimate_tokens_long_prompt(self):
        """Long prompt should scale correctly."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider()
            # 80000 chars / 4 = 20000 tokens + 10000 reserve = 30000
            result = provider._estimate_tokens("a" * 80000)
            assert result == 30000

    def test_estimate_tokens_includes_reserve(self):
        """Estimate should always include output reserve."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider()
            # Even empty prompt should have reserve
            result = provider._estimate_tokens("")
            assert result == 10000  # Just the reserve


class TestAPIProviderRetryBehavior:
    """Retry with exponential backoff on rate limit errors."""

    def test_call_with_retry_exists(self):
        """Should have _call_with_retry method."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider()
            assert hasattr(provider, "_call_with_retry")
            assert callable(provider._call_with_retry)

    def test_call_with_retry_success_first_try(self):
        """Should return result on successful first attempt."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider()
            call_fn = MagicMock(return_value="success")

            result = provider._call_with_retry(call_fn, "test prompt")

            assert result == "success"
            assert call_fn.call_count == 1

    @patch("time.sleep")
    def test_call_with_retry_retries_on_rate_limit(self, mock_sleep):
        """Should retry on RateLimitError."""
        import anthropic

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider()
            # Fail first, succeed second
            call_fn = MagicMock(
                side_effect=[
                    anthropic.RateLimitError(
                        message="rate limited",
                        response=MagicMock(status_code=429),
                        body={}
                    ),
                    "success"
                ]
            )

            result = provider._call_with_retry(call_fn, "test prompt")

            assert result == "success"
            assert call_fn.call_count == 2
            mock_sleep.assert_called()  # Should have slept

    @patch("time.sleep")
    def test_call_with_retry_exponential_backoff(self, mock_sleep):
        """Should use exponential backoff: 5s, 10s, 20s."""
        import anthropic

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider()
            # Fail twice, succeed third
            rate_limit_error = anthropic.RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429),
                body={}
            )
            call_fn = MagicMock(
                side_effect=[rate_limit_error, rate_limit_error, "success"]
            )

            result = provider._call_with_retry(call_fn, "test prompt")

            assert result == "success"
            assert call_fn.call_count == 3
            # Check backoff times (5s, 10s)
            sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
            assert 5 in sleep_calls
            assert 10 in sleep_calls

    @patch("time.sleep")
    def test_call_with_retry_max_retries_exceeded(self, mock_sleep):
        """Should raise RuntimeError after MAX_RETRIES failures."""
        import anthropic

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider()
            rate_limit_error = anthropic.RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429),
                body={}
            )
            call_fn = MagicMock(side_effect=rate_limit_error)

            with pytest.raises(RuntimeError, match="Rate limit exceeded after 3 retries"):
                provider._call_with_retry(call_fn, "test prompt")

            assert call_fn.call_count == 3

    @patch("time.sleep")
    def test_call_with_retry_uses_token_estimation(self, mock_sleep):
        """Should pass estimated tokens to rate limiter."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = APIProvider()
            provider._wait_for_rate_limit = MagicMock()
            call_fn = MagicMock(return_value="success")

            # Long prompt = more estimated tokens
            long_prompt = "a" * 40000  # 10000 tokens + 10000 reserve = 20000

            provider._call_with_retry(call_fn, long_prompt)

            # Should have called wait with estimated tokens
            provider._wait_for_rate_limit.assert_called_with(estimated_tokens=20000)
