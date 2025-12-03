"""
Stage 6 Tests: Chunk Processor (Sequential & Parallel)

What the processor does:
- Process all chunks through extraction + verification
- Sequential mode for local LLM (is_local=True)
- Parallel mode for API (is_local=False) with rate limiting
- Return all verified facts

Run: pytest tests/test_stage6_processor.py -v
"""

import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import pytest

from src.processor import (
    process_chunks_sequential,
    process_chunks_parallel,
    process_all_chunks,
)
from src.providers import LLMProvider
from src.chunker import Chunk
from src.parser import Conversation


class MockProvider(LLMProvider):
    """Mock provider for testing."""

    def __init__(self, is_local: bool = False):
        self._is_local = is_local
        self.call_count = 0

    @property
    def is_local(self) -> bool:
        return self._is_local

    def complete(self, prompt: str) -> str:
        self.call_count += 1
        return "response"

    def complete_structured(self, prompt: str, schema: dict) -> dict:
        self.call_count += 1
        return {"facts": []}


def make_test_chunk(chunk_id: int, token_count: int = 1000) -> Chunk:
    """Create a test chunk."""
    return Chunk(
        id=chunk_id,
        conversations=[
            Conversation(
                id=f"conv-{chunk_id}",
                title=f"Test Conversation {chunk_id}",
                create_time=1700000000.0,
                messages=[f"Test message {chunk_id}"]
            )
        ],
        token_count=token_count
    )


class TestProcessChunksSequential:
    """Sequential processing for local LLMs."""

    def test_processes_all_chunks_in_order(self):
        """Should process each chunk one at a time."""
        provider = MockProvider(is_local=True)
        chunks = [make_test_chunk(i) for i in range(3)]

        with patch('src.processor.extract_and_verify_chunk') as mock_extract:
            mock_extract.return_value = []
            result = process_chunks_sequential(chunks, provider)

        assert mock_extract.call_count == 3

    def test_returns_combined_facts(self):
        """Should return facts from all chunks combined."""
        provider = MockProvider(is_local=True)
        chunks = [make_test_chunk(i) for i in range(2)]

        fake_fact = {"fact": "test", "category": "personal"}

        with patch('src.processor.extract_and_verify_chunk') as mock_extract:
            mock_extract.side_effect = [
                [fake_fact],
                [fake_fact, fake_fact]
            ]
            result = process_chunks_sequential(chunks, provider)

        assert len(result) == 3

    def test_empty_chunks_returns_empty(self):
        """Should handle empty chunk list."""
        provider = MockProvider(is_local=True)
        result = process_chunks_sequential([], provider)
        assert result == []


class TestProcessChunksParallel:
    """Parallel processing for API providers."""

    @pytest.mark.asyncio
    async def test_processes_chunks_concurrently(self):
        """Should process multiple chunks at once."""
        provider = MockProvider(is_local=False)
        chunks = [make_test_chunk(i) for i in range(5)]

        with patch('src.processor.extract_and_verify_chunk_async') as mock_extract:
            mock_extract.return_value = []
            result = await process_chunks_parallel(chunks, provider)

        assert mock_extract.call_count == 5

    @pytest.mark.asyncio
    async def test_respects_concurrency_limit(self):
        """Should not exceed max concurrent requests."""
        provider = MockProvider(is_local=False)
        chunks = [make_test_chunk(i) for i in range(10)]

        concurrent_count = 0
        max_concurrent = 0

        async def track_concurrency(*args, **kwargs):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.01)  # Simulate work
            concurrent_count -= 1
            return []

        with patch('src.processor.extract_and_verify_chunk_async', side_effect=track_concurrency):
            result = await process_chunks_parallel(chunks, provider, max_concurrent=3)

        assert max_concurrent <= 3

    @pytest.mark.asyncio
    async def test_returns_combined_facts(self):
        """Should combine facts from all parallel chunks."""
        provider = MockProvider(is_local=False)
        chunks = [make_test_chunk(i) for i in range(3)]

        fake_fact = {"fact": "test", "category": "personal"}

        call_count = 0

        async def mock_extract(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return [fake_fact] * call_count  # 1, 2, 3 facts

        with patch('src.processor.extract_and_verify_chunk_async', side_effect=mock_extract):
            result = await process_chunks_parallel(chunks, provider)

        assert len(result) == 6  # 1 + 2 + 3


class TestProcessAllChunks:
    """Main entry point that chooses sequential vs parallel."""

    def test_uses_sequential_for_local(self):
        """Should use sequential processing for local provider."""
        provider = MockProvider(is_local=True)
        chunks = [make_test_chunk(0)]

        with patch('src.processor.process_chunks_sequential') as mock_seq:
            mock_seq.return_value = []
            process_all_chunks(chunks, provider)

        mock_seq.assert_called_once()

    def test_uses_parallel_for_api(self):
        """Should use parallel processing for API provider."""
        provider = MockProvider(is_local=False)
        chunks = [make_test_chunk(0)]

        with patch('src.processor.process_chunks_parallel') as mock_par:
            # Mock the coroutine
            async def async_return():
                return []
            mock_par.return_value = async_return()
            process_all_chunks(chunks, provider)

        mock_par.assert_called_once()

    def test_returns_facts_list(self):
        """Should return list of verified facts."""
        provider = MockProvider(is_local=True)
        chunks = [make_test_chunk(0)]

        fake_fact = {"fact": "test", "category": "personal"}

        with patch('src.processor.extract_and_verify_chunk') as mock_extract:
            mock_extract.return_value = [fake_fact]
            result = process_all_chunks(chunks, provider)

        assert isinstance(result, list)
        assert len(result) == 1
