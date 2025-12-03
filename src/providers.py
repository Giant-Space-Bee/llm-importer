"""
providers.py - LLM provider abstraction

Local = sequential (one LLM = all your RAM/GPU)
API = parallel with rate limiting (RPM/TPM)
"""

from abc import ABC, abstractmethod
from typing import Optional


class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Send prompt, get response."""
        pass

    @property
    @abstractmethod
    def is_local(self) -> bool:
        """True = sequential execution, False = parallel OK."""
        pass


class LocalProvider(LLMProvider):
    """
    LM Studio / any OpenAI-compatible local server.

    Default: http://localhost:1234/v1
    Always sequential (is_local = True)
    """

    def __init__(self, base_url: str = "http://localhost:1234/v1"):
        # TODO: Store base_url, create httpx client
        pass

    @property
    def is_local(self) -> bool:
        return True

    def complete(self, prompt: str) -> str:
        """Call local LLM via OpenAI-compatible API."""
        # TODO: POST to /chat/completions
        # {
        #   "messages": [{"role": "user", "content": prompt}],
        #   "temperature": 0.3
        # }
        pass


class APIProvider(LLMProvider):
    """
    Claude, OpenAI, etc.

    Supports rate limiting via RPM (requests/min) and TPM (tokens/min).
    Parallel execution (is_local = False)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        rpm: Optional[int] = None,
        tpm: Optional[int] = None
    ):
        # TODO: Store config, create rate limiter
        # If RPM provided, calculate max concurrent requests
        # If TPM provided, track token usage
        pass

    @property
    def is_local(self) -> bool:
        return False

    def complete(self, prompt: str) -> str:
        """Call API with rate limiting."""
        # TODO: Check rate limits, wait if needed, then call
        pass

    async def complete_async(self, prompt: str) -> str:
        """Async version for parallel execution."""
        # TODO: Same as complete but async
        pass
