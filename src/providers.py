"""
providers.py - LLM provider abstraction

Stage 4:
- LocalProvider for LM Studio (localhost:1234)
- OpenAI-compatible API
- Local = sequential (is_local = True)
- Structured outputs via json_schema response_format

Stage 6:
- APIProvider for Anthropic Claude API
- Claude Sonnet 4 with structured outputs
- Rate limiting (RPM/TPM tracking)
- Parallel execution OK (is_local = False)
"""

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Optional

import anthropic
import httpx


class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Send prompt, get response text."""
        pass

    @abstractmethod
    def complete_structured(self, prompt: str, schema: dict) -> dict:
        """Send prompt with JSON schema, get parsed dict back."""
        pass

    @property
    @abstractmethod
    def is_local(self) -> bool:
        """True = sequential execution, False = parallel OK."""
        pass


class LocalProvider(LLMProvider):
    """
    LM Studio / any OpenAI-compatible local server.

    Default: http://127.0.0.1:1234/v1
    Always sequential (is_local = True) because local LLM = all your RAM/GPU
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:1234/v1",
        timeout: float = None  # No timeout for local LLMs - they can take hours on big chunks
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    @property
    def is_local(self) -> bool:
        return True

    def complete(self, prompt: str) -> str:
        """
        Call local LLM via OpenAI-compatible API.

        Uses /chat/completions endpoint with a single user message.
        """
        url = f"{self.base_url}/chat/completions"

        payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,  # Lower = more deterministic for extraction
            "max_tokens": 8192,  # Plenty of room for extracted facts
        }

        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            # Extract the response text
            choices = data.get("choices", [])
            if not choices:
                return ""

            message = choices[0].get("message", {})
            return message.get("content", "")

        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"LLM API error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            raise RuntimeError(f"LLM connection error: {e}")

    def complete_structured(self, prompt: str, schema: dict) -> dict:
        """
        Call local LLM with structured output (json_schema mode).

        Args:
            prompt: The prompt text
            schema: JSON schema dict (will be wrapped in response_format)

        Returns:
            Parsed dict from LLM response
        """
        url = f"{self.base_url}/chat/completions"

        payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 8192,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction",
                    "strict": True,
                    "schema": schema
                }
            }
        }

        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            choices = data.get("choices", [])
            if not choices:
                return {}

            content = choices[0].get("message", {}).get("content", "")
            if not content:
                return {}

            return json.loads(content)

        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"LLM API error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            raise RuntimeError(f"LLM connection error: {e}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"LLM returned invalid JSON: {e}")

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "_client"):
            self._client.close()


class APIProvider(LLMProvider):
    """
    Anthropic Claude API provider.

    Uses Claude Sonnet 4 for structured outputs.
    Parallel execution OK with rate limiting (RPM/TPM tracking).
    """

    # Tier 1 defaults (conservative)
    DEFAULT_RPM = 5
    DEFAULT_TPM = 20000
    DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
    STRUCTURED_OUTPUTS_BETA = "structured-outputs-2025-11-13"

    def __init__(
        self,
        api_key: Optional[str] = None,
        rpm: Optional[int] = None,
        tpm: Optional[int] = None,
        model: Optional[str] = None
    ):
        # Get API key from arg or environment
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError(
                "API key required. Set ANTHROPIC_API_KEY env var or pass api_key argument."
            )

        self.rpm = rpm if rpm is not None else self.DEFAULT_RPM
        self.tpm = tpm if tpm is not None else self.DEFAULT_TPM
        self.model = model or self.DEFAULT_MODEL

        # Rate limiting state
        self._requests_this_minute = 0
        self._tokens_this_minute = 0
        self._minute_start = time.time()

        # Initialize Anthropic client
        self._client = anthropic.Anthropic(api_key=self._api_key)

    @property
    def is_local(self) -> bool:
        return False

    def _maybe_reset_minute(self) -> None:
        """Reset counters if minute has rolled over."""
        now = time.time()
        if now - self._minute_start >= 60:
            self._requests_this_minute = 0
            self._tokens_this_minute = 0
            self._minute_start = now

    def _get_rate_limit_wait(self, estimated_tokens: int = 0) -> float:
        """
        Calculate how long to wait before next request.

        Returns 0 if under limits, otherwise seconds to wait.
        """
        self._maybe_reset_minute()

        # Check RPM
        if self._requests_this_minute >= self.rpm:
            elapsed = time.time() - self._minute_start
            return max(0, 60 - elapsed + 0.1)  # Wait until next minute + buffer

        # Check TPM
        if self._tokens_this_minute + estimated_tokens > self.tpm:
            elapsed = time.time() - self._minute_start
            return max(0, 60 - elapsed + 0.1)

        return 0

    def _wait_for_rate_limit(self, estimated_tokens: int = 0) -> None:
        """Wait if necessary to respect rate limits."""
        wait_time = self._get_rate_limit_wait(estimated_tokens)
        if wait_time > 0:
            time.sleep(wait_time)
            self._maybe_reset_minute()

    def _update_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Update rate limit counters after a request."""
        self._requests_this_minute += 1
        self._tokens_this_minute += input_tokens + output_tokens

    def complete(self, prompt: str) -> str:
        """
        Send prompt to Claude API, get response text.

        Respects rate limits automatically.
        """
        self._wait_for_rate_limit()

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=8192,
                messages=[{"role": "user", "content": prompt}]
            )

            self._update_usage(
                response.usage.input_tokens,
                response.usage.output_tokens
            )

            return response.content[0].text

        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")

    def complete_structured(self, prompt: str, schema: dict) -> dict:
        """
        Send prompt with JSON schema, get parsed dict back.

        Uses Anthropic's structured outputs beta.
        """
        self._wait_for_rate_limit()

        try:
            response = self._client.beta.messages.create(
                model=self.model,
                max_tokens=8192,
                betas=[self.STRUCTURED_OUTPUTS_BETA],
                messages=[{"role": "user", "content": prompt}],
                output_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction",
                        "strict": True,
                        "schema": schema
                    }
                }
            )

            self._update_usage(
                response.usage.input_tokens,
                response.usage.output_tokens
            )

            content = response.content[0].text
            return json.loads(content)

        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON from Claude API: {e}")
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")
