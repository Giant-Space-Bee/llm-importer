"""
providers.py - LLM provider abstraction

Stage 4:
- LocalProvider for LM Studio (localhost:1234)
- OpenAI-compatible API
- Local = sequential (is_local = True)
- Structured outputs via json_schema response_format
"""

import json
from abc import ABC, abstractmethod
from typing import Optional

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
    Cloud API provider (Claude, OpenAI, etc.)

    Parallel execution with rate limiting.
    Not implemented in Stage 4 - coming later.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        rpm: Optional[int] = None,
        tpm: Optional[int] = None
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.rpm = rpm
        self.tpm = tpm
        raise NotImplementedError("APIProvider coming in a later stage")

    @property
    def is_local(self) -> bool:
        return False

    def complete(self, prompt: str) -> str:
        raise NotImplementedError("APIProvider coming in a later stage")

    def complete_structured(self, prompt: str, schema: dict) -> dict:
        raise NotImplementedError("APIProvider coming in a later stage")
