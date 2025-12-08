"""
deduplicator.py - Semantic deduplication

Two-phase hybrid approach:
- Phase 1: Within-category dedup (timestamp-batched, merge-sort)
- Phase 2: Cross-category merge-sort (catches miscategorized duplicates)

Uses LLM to find duplicates: "Lives in Victoria" = "Victoria BC resident"
Runs AFTER aggregate (needs full picture).

Prefer: newer > older, specific > vague, frequent > rare
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from src.aggregator import AggregatedFact, group_by_category
from src.extractor import CATEGORIES
from src.providers import LLMProvider

# Default max facts per LLM call
DEFAULT_MAX_BATCH = 50


@dataclass
class DeduplicatedFact:
    """
    A deduplicated fact - just fact text and category.

    Provenance info is merged. We keep a human-readable period string.
    """

    fact: str
    category: str
    period: str = "" # e.g. "2023", "2021-2024", "Oct 2025"


# JSON schema for structured output deduplication
DEDUP_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string"},
                    "category": {"type": "string", "enum": CATEGORIES},
                    "period": {"type": "string", "description": "Time range for this fact, e.g. '2023-2025' or 'May 2024'"}
                },
                "required": ["fact", "category", "period"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["facts"],
    "additionalProperties": False,
}


def load_dedup_prompt() -> str:
    """Load the deduplication prompt template from file."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "dedup.txt"
    return prompt_path.read_text()


def build_dedup_prompt(
    facts: Union[List[AggregatedFact], List[DeduplicatedFact]],
) -> str:
    """
    Build prompt from template + facts as JSON.

    Handles both AggregatedFact (Phase 1) and DeduplicatedFact (Phase 2).
    """
    template = load_dedup_prompt()

    # Convert facts to JSON-serializable format
    facts_json = []
    from datetime import datetime

    def fmt_ts(ts: float) -> str:
        # Use full date for precision (YYYY-MM-DD)
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")

    for f in facts:
        if isinstance(f, AggregatedFact):
            facts_json.append(
                {
                    "fact": f.fact.fact,
                    "category": f.fact.category,
                    "frequency": f.frequency,
                    "period": f"{fmt_ts(f.min_timestamp)} to {fmt_ts(f.max_timestamp)}",
                }
            )
        else:
            # DeduplicatedFact - already has period string
            facts_json.append(
                {
                    "fact": f.fact,
                    "category": f.category,
                    "period": f.period,
                }
            )

    return template + json.dumps(facts_json, indent=2)


def dedup_batch(
    facts: Union[List[AggregatedFact], List[DeduplicatedFact]],
    provider: LLMProvider,
) -> List[DeduplicatedFact]:
    """
    Single LLM call to dedup a batch of facts.

    Batch must be <= max_batch size.

    Args:
        facts: List of facts to deduplicate
        provider: LLM provider to use

    Returns:
        List of deduplicated facts
    """
    if not facts:
        return []

    prompt = build_dedup_prompt(facts)
    result = provider.complete_structured(prompt, DEDUP_SCHEMA)

    # Convert response to DeduplicatedFact objects
    deduped = []
    for item in result.get("facts", []):
        try:
            df = DeduplicatedFact(
                fact=str(item["fact"]),
                category=str(item["category"]),
                period=str(item.get("period", "")),
            )
            deduped.append(df)
        except (KeyError, ValueError, TypeError):
            # Skip malformed facts (shouldn't happen with schema enforcement)
            continue

    return deduped


def deduplicate_category(
    facts: List[AggregatedFact],
    category: str,
    provider: LLMProvider,
    max_batch: int = DEFAULT_MAX_BATCH,
) -> List[DeduplicatedFact]:
    """
    Dedup within one category using timestamp-based batching.

    Merge-sort style if multiple batches needed:
    1. Sort facts by source_timestamp (oldest -> newest)
    2. Split into batches of max_batch
    3. Dedup each batch via LLM
    4. If multiple batches: merge and dedup again

    Args:
        facts: List of AggregatedFact objects in this category
        category: The category name (for logging, not enforced)
        provider: LLM provider to use
        max_batch: Maximum facts per LLM call

    Returns:
        List of DeduplicatedFact objects
    """
    if not facts:
        return []

    # Sort by timestamp (oldest first)
    sorted_facts = sorted(facts, key=lambda f: f.fact.source_timestamp)

    # Single batch: just dedup
    if len(sorted_facts) <= max_batch:
        return dedup_batch(sorted_facts, provider)

    # Multiple batches: split, dedup each, then merge-sort
    batches: List[List[AggregatedFact]] = []
    for i in range(0, len(sorted_facts), max_batch):
        batches.append(sorted_facts[i : i + max_batch])

    # Dedup each batch
    batch_results: List[List[DeduplicatedFact]] = []
    for batch in batches:
        deduped = dedup_batch(batch, provider)
        batch_results.append(deduped)

    # Merge-sort style: combine batches pairwise until one remains
    while len(batch_results) > 1:
        new_results: List[List[DeduplicatedFact]] = []
        for i in range(0, len(batch_results), 2):
            if i + 1 < len(batch_results):
                # Merge two batches
                combined = batch_results[i] + batch_results[i + 1]
                if len(combined) <= max_batch:
                    merged = dedup_batch(combined, provider)
                else:
                    # Recursively dedup if still too large
                    merged = deduplicate_merge_sort(combined, provider, max_batch)
                new_results.append(merged)
            else:
                # Odd batch out - carry forward
                new_results.append(batch_results[i])
        batch_results = new_results

    return batch_results[0] if batch_results else []


def deduplicate_merge_sort(
    facts: List[DeduplicatedFact],
    provider: LLMProvider,
    max_batch: int = DEFAULT_MAX_BATCH,
) -> List[DeduplicatedFact]:
    """
    Cross-category merge-sort dedup.

    Recursive split-dedup-merge for Phase 2.

    Args:
        facts: List of DeduplicatedFact objects from Phase 1
        provider: LLM provider to use
        max_batch: Maximum facts per LLM call

    Returns:
        Final deduplicated facts
    """
    if not facts:
        return []

    # Base case: fits in one batch
    if len(facts) <= max_batch:
        return dedup_batch(facts, provider)

    # Split in half
    mid = len(facts) // 2
    left = facts[:mid]
    right = facts[mid:]

    # Recursively dedup each half
    left_deduped = deduplicate_merge_sort(left, provider, max_batch)
    right_deduped = deduplicate_merge_sort(right, provider, max_batch)

    # Merge and dedup
    combined = left_deduped + right_deduped
    if len(combined) <= max_batch:
        return dedup_batch(combined, provider)

    # Still too large after dedup - process in sliding window batches
    # This handles pathological cases where LLM doesn't reduce the count
    result = combined
    while len(result) > max_batch:
        # Take first two batches worth, dedup them, continue
        to_process = result[:max_batch]
        remaining = result[max_batch:]
        deduped = dedup_batch(to_process, provider)
        result = deduped + remaining

        # Safety: if no reduction, we're done (LLM won't merge these)
        if len(result) >= len(combined):
            break
        combined = result

    return result


def deduplicate(
    facts: List[AggregatedFact],
    provider: LLMProvider,
    max_batch: int = DEFAULT_MAX_BATCH,
) -> List[DeduplicatedFact]:
    """
    Main entry point. Two-phase dedup.

    Phase 1: Within-category dedup (parallel by category)
    Phase 2: Cross-category merge-sort

    Args:
        facts: List of AggregatedFact objects from aggregator
        provider: LLM provider to use
        max_batch: Maximum facts per LLM call (default 50)

    Returns:
        List of DeduplicatedFact objects - final deduplicated facts
    """
    if not facts:
        return []

    # Phase 1: Within-category dedup
    # Group facts by category
    by_category = group_by_category(facts)

    # Dedup each category
    # Note: Could parallelize here for API providers, but keeping simple for now
    phase1_results: List[DeduplicatedFact] = []
    for category, category_facts in by_category.items():
        deduped = deduplicate_category(category_facts, category, provider, max_batch)
        phase1_results.extend(deduped)

    # Phase 2: Cross-category merge-sort
    if len(phase1_results) <= max_batch:
        # Small enough for single pass
        return dedup_batch(phase1_results, provider)
    else:
        # Need merge-sort for cross-category
        return deduplicate_merge_sort(phase1_results, provider, max_batch)
