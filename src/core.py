"""
core.py - BatchedLLMTask + token counting

The universal pattern for all LLM work in this project.
"""

import time
from abc import ABC, abstractmethod
from typing import List, Any, Callable

# TODO: import tiktoken for token counting


class UserCancelledError(Exception):
    """User cancelled at depth warning prompt."""
    pass


class BatchedLLMTask:
    """
    Universal LLM task runner with auto-batching.

    Like merge-sort for LLM work:
    - Small job -> 1 LLM call
    - Medium job -> N batches -> 1 merge call
    - Huge job -> batch -> merge -> still too big -> batch again -> done

    Usage:
        task = BatchedLLMTask(provider, max_tokens=65536)
        result = task.run(items, prompt_template)
    """

    def __init__(self, provider, max_tokens: int = 65536, warn_depth: int = 5, retries: int = 3):
        # TODO: Store provider, max_tokens, warn_depth, retries
        # provider: LLMProvider instance
        # max_tokens: 65536 default (2^16)
        # warn_depth: warn user at 5, 10, 15... (never breaks, user decides)
        # retries: exponential backoff retry count
        pass

    def run(self, items: List[Any], prompt_template: str, depth: int = 0) -> Any:
        """
        Run task, auto-batch if too big, recursively merge.

        Steps:
        1. Check depth, warn at warn_depth intervals
        2. If fits in one call -> do it
        3. If too big -> split -> process each -> merge results
        4. Merged results might ALSO be too big -> recurse!
        """
        # TODO: Implement the merge-sort style batching
        # - _fits(items) to check token count
        # - _split(items) to batch
        # - _call_with_retry(prompt) for LLM call
        # - _flatten(results) to merge
        pass

    def _fits(self, items: List[Any]) -> bool:
        """Check if items fit in max_tokens."""
        # TODO: Use tiktoken to count tokens
        # Return True if total tokens < max_tokens
        pass

    def _split(self, items: List[Any]) -> List[List[Any]]:
        """Split items into batches that each fit in max_tokens."""
        # TODO: Greedy bin-packing
        pass

    def _format(self, items: List[Any]) -> str:
        """Format items for LLM prompt."""
        # TODO: JSON serialize items
        pass

    def _flatten(self, results: List[Any]) -> List[Any]:
        """Flatten nested results from batch processing."""
        # TODO: Concat all result arrays
        pass

    def _call_with_retry(self, prompt: str) -> Any:
        """Call LLM with exponential backoff retry."""
        # TODO:
        # for attempt in range(retries):
        #     try: return provider.complete(prompt)
        #     except: sleep(2 ** attempt)
        pass

    def _prompt_continue(self) -> bool:
        """Ask user if they want to continue at depth warning."""
        # TODO: input("Continue? [Y/n]: ")
        pass


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens using tiktoken."""
    # TODO: tiktoken.encoding_for_model(model).encode(text)
    pass
