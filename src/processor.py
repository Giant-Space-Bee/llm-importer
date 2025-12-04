"""
processor.py - Chunk processing orchestration

Stage 6:
- Process all chunks through extraction + verification
- Sequential mode for local LLM (is_local=True)
- Parallel mode for API (is_local=False) with rate limiting
- Return all verified facts
"""

import asyncio
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from src.chunker import Chunk
from src.extractor import extract_chunk, ExtractedFact
from src.verifier import verify_all
from src.providers import LLMProvider


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
    max_concurrent: int = 5
) -> List[ExtractedFact]:
    """
    Process chunks concurrently (for API providers).

    Respects rate limits via semaphore to cap concurrent requests.

    Args:
        chunks: List of chunks to process
        provider: LLM provider (should have is_local=False)
        conversations_by_id: Raw conversation dicts from load_conversations()
                            keyed by conversation ID. Required for verification.
        max_concurrent: Maximum concurrent requests (default 5 for Tier 1)

    Returns:
        Combined list of verified facts from all chunks
    """
    if not chunks:
        return []

    semaphore = asyncio.Semaphore(max_concurrent)
    executor = ThreadPoolExecutor(max_workers=max_concurrent)

    async def process_with_semaphore(chunk: Chunk) -> List[ExtractedFact]:
        async with semaphore:
            return await extract_and_verify_chunk_async(
                chunk, provider, conversations_by_id, executor
            )

    # Process all chunks concurrently (semaphore limits concurrency)
    results = await asyncio.gather(
        *[process_with_semaphore(chunk) for chunk in chunks]
    )

    executor.shutdown(wait=False)

    # Flatten results
    all_facts = []
    for facts in results:
        all_facts.extend(facts)

    return all_facts


def process_all_chunks(
    chunks: List[Chunk],
    provider: LLMProvider,
    conversations_by_id: Dict[str, Any] = None,
    max_concurrent: int = 5
) -> List[ExtractedFact]:
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

    Returns:
        Combined list of verified facts from all chunks
    """
    if provider.is_local:
        return process_chunks_sequential(chunks, provider, conversations_by_id)
    else:
        # Run async parallel processing
        return asyncio.run(
            process_chunks_parallel(
                chunks, provider, conversations_by_id, max_concurrent
            )
        )
