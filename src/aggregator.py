"""
aggregator.py - Combine verified facts

NO LLM. Just concat and count frequency.
Runs AFTER verifier, BEFORE deduplicator.
"""

from typing import List, Dict
from dataclasses import dataclass
from collections import Counter

# from .extractor import ExtractedFact


@dataclass
class AggregatedFact:
    """Fact with frequency count (same fact from multiple chunks = stronger signal)."""
    fact: object  # ExtractedFact
    frequency: int  # How many times this exact fact appeared


def aggregate(facts: List) -> List[AggregatedFact]:
    """
    Combine all verified facts, count frequency.

    Same fact extracted from multiple chunks = higher frequency = stronger signal.
    Used by deduplicator: prefer frequent > rare.
    """
    # TODO:
    # 1. Group by fact text (exact match)
    # 2. Count occurrences
    # 3. Keep the newest instance (highest timestamp) as canonical
    # 4. Return list with frequency counts
    pass


def group_by_category(facts: List[AggregatedFact]) -> Dict[str, List[AggregatedFact]]:
    """Group facts by category for deduplication."""
    # TODO: {category: [facts...]}
    pass
