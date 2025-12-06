"""
processor.py - Chunk processing orchestration

Stage 6:
- Process all chunks through extraction + verification
- Sequential mode for local LLM (is_local=True)
- Parallel mode for API (is_local=False) with rate limiting
- Return all verified facts
"""

import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional, Set
from concurrent.futures import ThreadPoolExecutor

from src.chunker import Chunk
from src.extractor import extract_chunk, ExtractedFact
from src.verifier import verify_all
from src.providers import LLMProvider


# Type alias for checkpoint callback
# Called with (completed_indices, all_facts_so_far) after each chunk
CheckpointCallback = Callable[[Set[int], List[ExtractedFact]], None]


@dataclass
class ParallelResult:
    """Result from parallel chunk processing."""
    facts: List[ExtractedFact]
    completed_indices: Set[int]
    failed_indices: List[int]
    errors: Dict[int, Exception]


def extract_and_verify_chunk(
    chunk: Chunk,
    provider: LLMProvider,
    conversations_by_id: Dict[str, Any] = None
) -> List[ExtractedFact]:
    """
    Extract facts from chunk and verify them.

    Args:
        chunk: The chunk to process
        provider: LLM provider for extraction
        conversations_by_id: Dict mapping convo_id -> raw conversation dict.
                            MUST be raw dicts from load_conversations(), NOT
                            parsed Conversation objects. The verifier needs
                            raw format to extract text via conversation_to_text().
                            Supports both ChatGPT (mapping) and Claude (chat_messages).
                            If None, skips verification.

    Returns:
        List of verified ExtractedFact objects
    """
    # Extract facts from chunk
    extracted = extract_chunk(chunk, provider)

    if not extracted:
        return []

    # If no conversations dict provided, skip verification
    # (used in testing or when verification happens later)
    if conversations_by_id is None:
        return extracted

    # Verify facts against source conversations
    verified, _ = verify_all(extracted, conversations_by_id)
    return verified


async def extract_and_verify_chunk_async(
    chunk: Chunk,
    provider: LLMProvider,
    conversations_by_id: Dict[str, Any] = None,
    executor: ThreadPoolExecutor = None
) -> List[ExtractedFact]:
    """
    Async wrapper for extract_and_verify_chunk.

    Runs the synchronous extraction in a thread pool to avoid blocking.
    """
    loop = asyncio.get_event_loop()

    if executor is None:
        executor = ThreadPoolExecutor(max_workers=1)

    return await loop.run_in_executor(
        executor,
        lambda: extract_and_verify_chunk(chunk, provider, conversations_by_id)
    )


def process_chunks_sequential(
    chunks: List[Chunk],
    provider: LLMProvider,
    conversations_by_id: Dict[str, Any] = None
) -> List[ExtractedFact]:
    """
    Process chunks one at a time (for local LLMs).

    Args:
        chunks: List of chunks to process
        provider: LLM provider (should have is_local=True)
        conversations_by_id: Raw conversation dicts from load_conversations()
                            keyed by conversation ID. Required for verification.

    Returns:
        Combined list of verified facts from all chunks
    """
    all_facts = []

    for chunk in chunks:
        facts = extract_and_verify_chunk(chunk, provider, conversations_by_id)
        all_facts.extend(facts)

    return all_facts


async def process_chunks_parallel(
    chunks: List[Chunk],
    provider: LLMProvider,
    conversations_by_id: Dict[str, Any] = None,
    max_concurrent: int = 5,
    chunk_indices: List[int] = None,
    on_chunk_complete: CheckpointCallback = None,
    existing_facts: List[ExtractedFact] = None
) -> ParallelResult:
    """
    Process chunks concurrently (for API providers).

    Respects rate limits via semaphore to cap concurrent requests.
    Saves checkpoint after each chunk completes (if callback provided).
    Continues processing even if some chunks fail.

    Args:
        chunks: List of chunks to process
        provider: LLM provider (should have is_local=False)
        conversations_by_id: Raw conversation dicts from load_conversations()
                            keyed by conversation ID. Required for verification.
        max_concurrent: Maximum concurrent requests (default 5 for Tier 1)
        chunk_indices: Indices of chunks being processed (for checkpoint tracking).
                      If None, uses 0..len(chunks)-1.
        on_chunk_complete: Callback called after each chunk completes with
                          (completed_indices, all_facts). Used for checkpointing.
        existing_facts: Facts from previous run to include in checkpoint.

    Returns:
        ParallelResult with facts, completed indices, and any failures
    """
    if not chunks:
        return ParallelResult(
            facts=[],
            completed_indices=set(),
            failed_indices=[],
            errors={}
        )

    # Default indices if not provided
    if chunk_indices is None:
        chunk_indices = list(range(len(chunks)))

    semaphore = asyncio.Semaphore(max_concurrent)
    executor = ThreadPoolExecutor(max_workers=max_concurrent)

    # Shared state protected by lock
    lock = asyncio.Lock()
    all_facts: List[ExtractedFact] = list(existing_facts) if existing_facts else []
    completed_indices: Set[int] = set()

    async def process_with_checkpoint(idx: int, chunk: Chunk) -> List[ExtractedFact]:
        async with semaphore:
            facts = await extract_and_verify_chunk_async(
                chunk, provider, conversations_by_id, executor
            )

            # Update shared state under lock
            async with lock:
                all_facts.extend(facts)
                completed_indices.add(idx)

                # Call checkpoint callback if provided
                if on_chunk_complete:
                    on_chunk_complete(completed_indices.copy(), list(all_facts))

            return facts

    # Process all chunks concurrently with return_exceptions=True
    # This ensures one failure doesn't kill all other chunks
    results = await asyncio.gather(
        *[process_with_checkpoint(idx, chunk)
          for idx, chunk in zip(chunk_indices, chunks)],
        return_exceptions=True
    )

    executor.shutdown(wait=True)

    # Separate successes from failures
    failed_indices = []
    errors = {}
    for idx, result in zip(chunk_indices, results):
        if isinstance(result, Exception):
            failed_indices.append(idx)
            errors[idx] = result

    return ParallelResult(
        facts=all_facts,
        completed_indices=completed_indices,
        failed_indices=failed_indices,
        errors=errors
    )


def process_all_chunks(
    chunks: List[Chunk],
    provider: LLMProvider,
    conversations_by_id: Dict[str, Any] = None,
    max_concurrent: int = 5,
    chunk_indices: List[int] = None,
    on_chunk_complete: CheckpointCallback = None,
    existing_facts: List[ExtractedFact] = None
) -> ParallelResult:
    """
    Process all chunks using appropriate strategy based on provider type.

    Automatically chooses:
    - Sequential for local providers (is_local=True)
    - Parallel for API providers (is_local=False)

    Args:
        chunks: List of chunks to process
        provider: LLM provider
        conversations_by_id: Raw conversation dicts from load_conversations()
                            keyed by conversation ID. Required for verification.
                            Example: {c['id']: c for c in load_conversations(path)}
        max_concurrent: Max concurrent requests for parallel mode
        chunk_indices: Indices of chunks being processed (for checkpoint tracking)
        on_chunk_complete: Callback for checkpointing (parallel mode only)
        existing_facts: Facts from previous run to include

    Returns:
        ParallelResult with facts, completed indices, and any failures
    """
    if chunk_indices is None:
        chunk_indices = list(range(len(chunks)))

    if provider.is_local:
        # Sequential mode doesn't use this function for checkpointing
        # (handled in phase_extract directly)
        facts = process_chunks_sequential(chunks, provider, conversations_by_id)
        return ParallelResult(
            facts=facts,
            completed_indices=set(chunk_indices),
            failed_indices=[],
            errors={}
        )
    else:
        # Run async parallel processing
        return asyncio.run(
            process_chunks_parallel(
                chunks, provider, conversations_by_id, max_concurrent,
                chunk_indices, on_chunk_complete, existing_facts
            )
        )
