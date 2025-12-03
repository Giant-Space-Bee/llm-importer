"""
deduplicator.py - Semantic deduplication

Uses LLM to find duplicates: "Lives in Victoria" = "Victoria BC resident"
Runs AFTER aggregate (needs full picture).

Prefer: newer > older, specific > vague, frequent > rare
"""

from typing import List

# from .core import BatchedLLMTask
# from .aggregator import AggregatedFact


def deduplicate(
    facts: List,  # List[AggregatedFact]
    provider  # LLMProvider
) -> List:  # List[ExtractedFact] - deduplicated
    """
    Semantic deduplication using LLM.

    Uses BatchedLLMTask for auto-batching if too many facts.

    Resolution rules (encoded in prompt):
    1. More recent > older (use source_timestamp)
    2. More specific > vague
    3. Higher frequency > rare
    4. Explicit statement > inference
    """
    # TODO:
    # 1. Load dedup prompt template
    # 2. Use BatchedLLMTask to process
    # 3. Parse response into deduplicated facts
    # 4. Return unique facts
    pass


def load_dedup_prompt() -> str:
    """Load the deduplication prompt template."""
    # TODO: Read prompts/dedup.txt
    pass
