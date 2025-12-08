"""
Internal checkpoint helper for building standardized checkpoint dictionaries.

This module is internal to the phases package (note the leading underscore).
It extracts the duplicated checkpoint-building logic from multiple phases.
"""

from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Union

from src.extractor import ExtractedFact
from src.deduplicator import DeduplicatedFact
from src.distiller import DistilledProfile


def build_checkpoint_dict(
    source_file_hash: str,
    completed_chunks: Iterable[int],
    verified_facts: List[Union[ExtractedFact, Dict[str, Any]]],
    dedup_completed: bool = False,
    deduplicated_facts: Optional[List[Union[DeduplicatedFact, Dict[str, Any]]]] = None,
    distill_completed: bool = False,
    distilled_profile: Optional[DistilledProfile] = None,
) -> Dict[str, Any]:
    """Build standardized checkpoint dictionary.

    This function centralizes the checkpoint structure to ensure consistency
    across all phases that save checkpoints (extract, deduplicate, distill).

    Args:
        source_file_hash: SHA256 hash of the source input file.
        completed_chunks: Indices of completed chunks.
        verified_facts: List of verified facts (ExtractedFact or dict).
        dedup_completed: Whether deduplication phase completed.
        deduplicated_facts: List of deduplicated facts (if dedup completed).
        distill_completed: Whether distillation phase completed.
        distilled_profile: Final distilled profile (if distill completed).

    Returns:
        Dictionary ready to pass to save_checkpoint().

    Example:
        >>> checkpoint = build_checkpoint_dict(
        ...     source_file_hash=hash_file(input_file),
        ...     completed_chunks=[0, 1, 2],
        ...     verified_facts=facts,
        ... )
        >>> save_checkpoint(checkpoint_path, checkpoint)
    """
    result: Dict[str, Any] = {
        "source_file_hash": source_file_hash,
        "completed_chunks": sorted(completed_chunks),
        "verified_facts": [
            asdict(f) if isinstance(f, ExtractedFact) else f
            for f in verified_facts
        ],
    }

    if dedup_completed:
        result["dedup_completed"] = True
        result["deduplicated_facts"] = [
            asdict(f) if isinstance(f, DeduplicatedFact) else f
            for f in (deduplicated_facts or [])
        ]

    if distill_completed:
        result["distill_completed"] = True
        if distilled_profile:
            result["distilled_profile"] = asdict(distilled_profile)

    return result
