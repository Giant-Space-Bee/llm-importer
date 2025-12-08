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
import sys
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional, TypeVar

T = TypeVar("T")

import anthropic
import httpx

# Import centralized constants
from src.config import (
    # LLM parameters
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    # Rate limiting
    SECONDS_PER_MINUTE,
    RATE_LIMIT_BUFFER,
    DEFAULT_TPM,
    DEFAULT_RPM,
    MAX_CONCURRENT_REQUESTS,
    # Token budgets
    OUTPUT_RESERVE,
    OUTPUT_ESTIMATE,
    # Chunk sizes
    MIN_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    QUALITY_CHUNK_SIZE,
    # Retry
    MAX_RETRIES,
    INITIAL_BACKOFF_SECONDS,
    # Local LLM
    LOCAL_LLM_BASE_URL,
    DEFAULT_LOCAL_MODEL,
    LOCAL_LLM_REPETITION_PENALTY,
    LOCAL_LLM_TOP_P,
    # Model
    DEFAULT_MODEL,
    STRUCTURED_OUTPUTS_BETA,
)

# Re-export for backward compatibility
MAX_CONCURRENT = MAX_CONCURRENT_REQUESTS
INITIAL_BACKOFF = INITIAL_BACKOFF_SECONDS


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
        base_url: str = LOCAL_LLM_BASE_URL,
        timeout: float = None,  # No timeout for local LLMs - they can take hours on big chunks
        model: Optional[str] = None
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.model = model or DEFAULT_LOCAL_MODEL
        self._client = httpx.Client(timeout=timeout)

    @property
    def is_local(self) -> bool:
        return True

    def _make_request(self, url: str, payload: dict) -> dict:
        """Make request with retry logic and connection reset."""
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    # Reset client connection
                    self._client.close()
                    self._client = httpx.Client(timeout=self.timeout)
                    wait_time = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                    print(
                        f"[retry] Connection reset, attempt {attempt + 2}/{MAX_RETRIES} "
                        f"in {wait_time}s",
                        file=sys.stderr
                    )
                    time.sleep(wait_time)

        raise RuntimeError(f"LLM connection failed after {MAX_RETRIES} attempts: {last_error}")

    def _log_json_error(self, content: str, error: json.JSONDecodeError) -> None:
        """Dump malformed JSON to error log for debugging."""
        from datetime import datetime
        from pathlib import Path

        error_dir = Path("output/json_errors")
        error_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_file = error_dir / f"json_error_{timestamp}.txt"

        with open(error_file, "w") as f:
            f.write(f"=== JSON Parse Error ===\n")
            f.write(f"Time: {datetime.now().isoformat()}\n")
            f.write(f"Error: {error}\n")
            f.write(f"Position: line {error.lineno}, col {error.colno}, char {error.pos}\n")
            f.write(f"\n=== Raw Content ({len(content)} chars) ===\n")
            f.write(content)
            f.write(f"\n\n=== Context around error (chars {max(0, error.pos-100)}:{error.pos+100}) ===\n")
            f.write(content[max(0, error.pos-100):error.pos+100])

        print(f"[error] Malformed JSON dumped to: {error_file}", file=sys.stderr)

    def complete(self, prompt: str) -> str:
        """
        Call local LLM via OpenAI-compatible API.

        Uses /chat/completions endpoint with a single user message.
        """
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "top_p": LOCAL_LLM_TOP_P,
            "repetition_penalty": LOCAL_LLM_REPETITION_PENALTY,
        }

        data = self._make_request(url, payload)

        # Extract the response text
        choices = data.get("choices", [])
        if not choices:
            return ""

        message = choices[0].get("message", {})
        return message.get("content", "")

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
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "top_p": LOCAL_LLM_TOP_P,
            "repetition_penalty": LOCAL_LLM_REPETITION_PENALTY,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction",
                    "strict": True,
                    "schema": schema
                }
            }
        }

        data = self._make_request(url, payload)

        choices = data.get("choices", [])
        if not choices:
            return {}

        content = choices[0].get("message", {}).get("content", "")
        if not content:
            return {}

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # Dump bad JSON to error log for debugging
            self._log_json_error(content, e)
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

        self.rpm = rpm if rpm is not None else DEFAULT_RPM
        self.tpm = tpm if tpm is not None else DEFAULT_TPM
        self.model = model or DEFAULT_MODEL

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
        if now - self._minute_start >= SECONDS_PER_MINUTE:
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
            return max(0, SECONDS_PER_MINUTE - elapsed + RATE_LIMIT_BUFFER)  # Wait until next minute + buffer

        # Check TPM
        if self._tokens_this_minute + estimated_tokens > self.tpm:
            elapsed = time.time() - self._minute_start
            return max(0, SECONDS_PER_MINUTE - elapsed + RATE_LIMIT_BUFFER)

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

    def _estimate_tokens(self, prompt: str) -> int:
        """
        Rough token estimate for rate limit pre-checking.

        Uses ~4 chars per token heuristic plus output reserve.
        Conservative: better to overestimate than underestimate.
        """
        return len(prompt) // 4 + OUTPUT_RESERVE

    def _call_with_retry(
        self, call_fn: Callable[[], T], prompt: str
    ) -> T:
        """
        Execute API call with proactive rate limiting and retry on 429.

        Layer 1: Pre-estimate tokens and wait if needed (proactive)
        Layer 2: Retry with exponential backoff on rate limit (reactive)

        Args:
            call_fn: Zero-arg callable that makes the actual API call
            prompt: The prompt text (used for token estimation)

        Returns:
            Result from call_fn

        Raises:
            RuntimeError: After MAX_RETRIES failed attempts
        """
        estimated = self._estimate_tokens(prompt)

        for attempt in range(MAX_RETRIES):
            self._wait_for_rate_limit(estimated_tokens=estimated)

            try:
                return call_fn()
            except anthropic.RateLimitError as e:
                if attempt == MAX_RETRIES - 1:
                    raise RuntimeError(
                        f"Rate limit exceeded after {MAX_RETRIES} retries: {e}"
                    )

                backoff = INITIAL_BACKOFF * (2 ** attempt)  # 5s, 10s, 20s
                print(
                    f"[Rate limit hit, waiting {backoff}s before retry {attempt + 2}]",
                    file=sys.stderr,
                )
                time.sleep(backoff)
                self._maybe_reset_minute()

        # Should never reach here, but satisfy type checker
        raise RuntimeError("Retry loop exited unexpectedly")

    def complete(self, prompt: str) -> str:
        """
        Send prompt to Claude API, get response text.

        Uses proactive rate limiting and retry on 429.
        """
        def _do_call() -> str:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=DEFAULT_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}]
            )
            self._update_usage(
                response.usage.input_tokens,
                response.usage.output_tokens
            )
            return response.content[0].text

        try:
            return self._call_with_retry(_do_call, prompt)
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise  # Already wrapped
            raise RuntimeError(f"Anthropic API error: {e}")

    def complete_structured(self, prompt: str, schema: dict) -> dict:
        """
        Send prompt with JSON schema, get parsed dict back.

        Uses Anthropic's structured outputs beta with retry on 429.
        """
        def _do_call() -> dict:
            response = self._client.beta.messages.create(
                model=self.model,
                max_tokens=DEFAULT_MAX_TOKENS,
                betas=[STRUCTURED_OUTPUTS_BETA],
                messages=[{"role": "user", "content": prompt}],
                output_format={
                    "type": "json_schema",
                    "schema": schema
                }
            )
            self._update_usage(
                response.usage.input_tokens,
                response.usage.output_tokens
            )

            # Check stop_reason BEFORE parsing - these bypass schema guarantees
            if response.stop_reason == "refusal":
                raise RuntimeError(
                    "Claude refused request for safety reasons. "
                    "Check extraction prompt for policy violations."
                )

            if response.stop_reason == "max_tokens":
                raise RuntimeError(
                    f"Response truncated at {DEFAULT_MAX_TOKENS} tokens. "
                    "JSON is incomplete. Reduce input size or increase max_tokens."
                )

            # Validate response content
            if not response.content:
                raise RuntimeError("Claude API returned empty content array")

            content_block = response.content[0]

            # Check content block type (could be ToolUseBlock, etc.)
            if not hasattr(content_block, "text"):
                raise RuntimeError(
                    f"Unexpected content block type: {type(content_block).__name__}"
                )

            content = content_block.text

            # Final validation before parsing
            if not content or not content.strip():
                raise RuntimeError("Claude returned empty text content")

            return json.loads(content)

        try:
            return self._call_with_retry(_do_call, prompt)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON from Claude API: {e}")
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise  # Already wrapped
            raise RuntimeError(f"Anthropic API error: {e}")

    def get_safe_chunk_size(self) -> int:
        """
        Return quality-optimal chunk size for extraction.

        Philosophy: Smaller chunks = better LLM recall (avoid "lost in middle")
        Fixed at 8k regardless of TPM - parallelism handles throughput.

        Returns:
            8192 tokens (quality-optimal size)
        """
        return QUALITY_CHUNK_SIZE

    def get_max_concurrent(self) -> int:
        """
        Calculate parallelism that fits within TPM budget.

        Formula: concurrent = tpm / (chunk_size + output_estimate)
        Uses OUTPUT_ESTIMATE (3k) not OUTPUT_RESERVE (10k) for realistic calc.

        Examples:
            TPM 30k → 30000 / 11000 = 2 concurrent
            TPM 80k → 80000 / 11000 = 5 concurrent (capped)
        """
        chunk_size = self.get_safe_chunk_size()
        tokens_per_request = chunk_size + OUTPUT_ESTIMATE  # ~11k per request
        concurrent = max(1, self.tpm // tokens_per_request)
        return min(concurrent, MAX_CONCURRENT_REQUESTS)
