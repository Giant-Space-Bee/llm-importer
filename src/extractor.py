"""
extractor.py - LLM fact extraction

Local = sequential, API = parallel with rate limits.
Uses BatchedLLMTask for auto-batching.
Outputs facts with source_quote for hallucination detection.
"""

from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path

# from .core import BatchedLLMTask
# from .providers import LLMProvider
# from .chunker import Chunk


@dataclass
class ExtractedFact:
    """
    A single extracted fact with provenance.

    source_quote is VERBATIM text from the conversation.
    Used for hallucination detection (string match).
    """
    fact: str  # "Lives in Victoria, BC, Canada"
    category: str  # personal|professional|family|preferences|interests|personality
    source_convo_id: str  # UUID of source conversation
    source_timestamp: float  # Unix timestamp (for "prefer newer" in dedup)
    source_quote: str  # VERBATIM quote from conversation


def extract_chunk(chunk, provider) -> List[ExtractedFact]:
    """
    Extract facts from a single chunk.

    Uses extraction prompt from prompts/extraction.txt
    """
    # TODO:
    # 1. Load prompt template
    # 2. Format chunk content
    # 3. Call provider.complete(prompt + content)
    # 4. Parse JSON response into ExtractedFact objects
    # 5. Handle invalid JSON (retry with "please return valid JSON")
    pass


def extract_all(
    chunks: List,  # List[Chunk]
    provider,  # LLMProvider
    checkpoint_dir: Optional[Path] = None
) -> List[ExtractedFact]:
    """
    Extract facts from all chunks.

    Execution mode based on provider:
    - provider.is_local = True -> sequential
    - provider.is_local = False -> parallel with rate limiting

    Checkpointing: saves progress to checkpoint_dir, resumes on failure.
    """
    # TODO:
    # 1. Check for existing checkpoints
    # 2. If local: sequential loop with progress bar
    # 3. If API: asyncio.gather with rate limiting
    # 4. Save checkpoint after each chunk
    # 5. Return all facts
    pass


def load_extraction_prompt() -> str:
    """Load the extraction prompt template."""
    # TODO: Read prompts/extraction.txt
    pass
