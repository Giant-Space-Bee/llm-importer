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

        async def mock_extract(chunk, *args, **kwargs):
            # Return facts based on chunk id
            return [fake_fact] * (chunk.id + 1)  # 1, 2, 3 facts

        with patch('src.processor.extract_and_verify_chunk_async', side_effect=mock_extract):
            result = await process_chunks_parallel(chunks, provider)

        assert len(result.facts) == 6  # 1 + 2 + 3


class TestProcessChunksParallelFailure:
    """Tests for partial failure handling in parallel processing."""

    @pytest.mark.asyncio
    async def test_continues_after_chunk_failure(self):
        """Should continue processing other chunks even if one fails."""
        provider = MockProvider(is_local=False)
        chunks = [make_test_chunk(i) for i in range(5)]

        call_count = 0

        async def mock_extract(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 3:  # Third chunk fails
                raise ValueError("Simulated API error")
            return [{"fact": f"fact-{call_count}", "category": "personal"}]

        with patch('src.processor.extract_and_verify_chunk_async', side_effect=mock_extract):
            result = await process_chunks_parallel(chunks, provider)

        # All 5 chunks should have been attempted
        assert call_count == 5
        # 4 successful chunks should have results
        assert len(result.completed_indices) == 4
        assert len(result.facts) == 4
        # 1 chunk should be in failed_indices
        assert len(result.failed_indices) == 1
        assert 2 in result.failed_indices  # 0-indexed, 3rd chunk

    @pytest.mark.asyncio
    async def test_checkpoint_callback_called_per_chunk(self):
        """Should call checkpoint callback after each successful chunk."""
        provider = MockProvider(is_local=False)
        chunks = [make_test_chunk(i) for i in range(3)]

        checkpoint_calls = []

        def on_checkpoint(completed_indices, facts):
            checkpoint_calls.append((completed_indices.copy(), len(facts)))

        async def mock_extract(*args, **kwargs):
            return [{"fact": "test", "category": "personal"}]

        with patch('src.processor.extract_and_verify_chunk_async', side_effect=mock_extract):
            result = await process_chunks_parallel(
                chunks, provider,
                on_chunk_complete=on_checkpoint
            )

        # Should have been called 3 times (once per chunk)
        assert len(checkpoint_calls) == 3
        # Final call should have all 3 chunks and 3 facts
        final_indices, final_count = checkpoint_calls[-1]
        assert len(final_indices) == 3
        assert final_count == 3

    @pytest.mark.asyncio
    async def test_checkpoint_saved_even_on_partial_failure(self):
        """Checkpoint should contain results from successful chunks even if some fail."""
        provider = MockProvider(is_local=False)
        chunks = [make_test_chunk(i) for i in range(5)]

        checkpoint_calls = []

        def on_checkpoint(completed_indices, facts):
            checkpoint_calls.append((completed_indices.copy(), list(facts)))

        async def mock_extract(chunk, *args, **kwargs):
            # Use chunk id to determine failure (chunk 2 fails)
            if chunk.id == 2:
                raise ValueError("Simulated error")
            return [{"fact": f"fact-{chunk.id}", "category": "personal"}]

        with patch('src.processor.extract_and_verify_chunk_async', side_effect=mock_extract):
            result = await process_chunks_parallel(
                chunks, provider,
                on_chunk_complete=on_checkpoint
            )

        # Should have 4 successful checkpoint calls (chunk 2 failed)
        assert len(checkpoint_calls) == 4
        # Result should have 4 facts from successful chunks
        assert len(result.facts) == 4
        assert len(result.failed_indices) == 1
        assert 2 in result.failed_indices


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

        with patch('src.processor.process_chunks_parallel', new_callable=AsyncMock) as mock_par:
            from src.processor import ParallelResult
            mock_par.return_value = ParallelResult(
                facts=[],
                completed_indices=set(),
                failed_indices=[],
                errors={}
            )
            process_all_chunks(chunks, provider)

        mock_par.assert_called_once()

    def test_returns_parallel_result(self):
        """Should return ParallelResult with facts."""
        from src.processor import ParallelResult
        provider = MockProvider(is_local=True)
        chunks = [make_test_chunk(0)]

        fake_fact = {"fact": "test", "category": "personal"}

        with patch('src.processor.extract_and_verify_chunk') as mock_extract:
            mock_extract.return_value = [fake_fact]
            result = process_all_chunks(chunks, provider)

        assert isinstance(result, ParallelResult)
        assert len(result.facts) == 1
