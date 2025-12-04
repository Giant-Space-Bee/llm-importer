"""
aggregator.py - Combine verified facts

NO LLM. Just concat and count frequency.
Runs AFTER verifier, BEFORE deduplicator.
"""

from typing import List, Dict, Any, Union
from dataclasses import dataclass
from collections import defaultdict

from src.extractor import ExtractedFact
from src.verifier import normalize_text


# Type alias: facts can be dicts or ExtractedFact dataclasses
Fact = Union[Dict[str, Any], ExtractedFact]


@dataclass
class AggregatedFact:
    """Fact with frequency count (same fact from multiple chunks = stronger signal)."""

    fact: ExtractedFact  # The canonical fact (newest instance)
    frequency: int  # How many times this exact fact appeared


def get_fact_attr(fact: Fact, attr: str) -> Any:
    """Get attribute from fact, whether dict or dataclass."""
    if isinstance(fact, dict):
        return fact[attr]
    return getattr(fact, attr)


def to_extracted_fact(fact: Fact) -> ExtractedFact:
    """Convert a fact (dict or dataclass) to ExtractedFact."""
    if isinstance(fact, ExtractedFact):
        return fact
    return ExtractedFact(
        fact=fact["fact"],
        category=fact["category"],
        source_convo_id=fact["source_convo_id"],
        source_timestamp=fact["source_timestamp"],
        source_quote=fact["source_quote"],
    )


def aggregate(facts: List[Fact]) -> List[AggregatedFact]:
    """
    Combine all verified facts, count frequency.

    Same fact extracted from multiple chunks = higher frequency = stronger signal.
    Used by deduplicator: prefer frequent > rare.

    Grouping is case-insensitive and whitespace-normalized, but the canonical
    fact preserves original text from the newest instance.

    Args:
        facts: List of verified facts (dicts or ExtractedFact objects)

    Returns:
        List of AggregatedFact, each with frequency count
    """
    if not facts:
        return []

    # Group facts by normalized text
    # Key: normalized fact text
    # Value: list of (original fact, timestamp)
    groups: Dict[str, List[Fact]] = defaultdict(list)

    for fact in facts:
        fact_text = get_fact_attr(fact, "fact")
        normalized = normalize_text(fact_text)
        groups[normalized].append(fact)

    # For each group: count frequency, keep newest as canonical
    result: List[AggregatedFact] = []

    for normalized_text, group in groups.items():
        frequency = len(group)

        # Find newest fact (highest timestamp)
        newest = max(group, key=lambda f: get_fact_attr(f, "source_timestamp"))

        # Convert to ExtractedFact if needed
        canonical = to_extracted_fact(newest)

        result.append(AggregatedFact(fact=canonical, frequency=frequency))

    return result


def group_by_category(facts: List[AggregatedFact]) -> Dict[str, List[AggregatedFact]]:
    """
    Group aggregated facts by category for deduplication.

    Args:
        facts: List of AggregatedFact objects

    Returns:
        Dict mapping category -> list of facts in that category
    """
    if not facts:
        return {}

    result: Dict[str, List[AggregatedFact]] = defaultdict(list)

    for agg_fact in facts:
        category = agg_fact.fact.category
        result[category].append(agg_fact)

    return dict(result)  # Convert from defaultdict to regular dict
