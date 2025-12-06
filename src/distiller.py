"""
distiller.py - Final compression to memory profile (Stage 9)

Uses LLM to compress deduplicated facts into coherent profile.
Outputs both memory-profile.md and memory-profile.json.

Supports "trusted baseline" pattern:
- Claude exports include memories.json (pre-synthesized profile)
- ChatGPT exports may have user_editable_context (custom instructions)
- These are passed as trusted_context, which the distiller merges
  with extracted facts rather than starting from scratch.

Example:
    >>> from src.distiller import distill, DistilledProfile
    >>> profile = distill(facts, provider, trusted_context="existing profile")
    >>> print(profile.name)
    'Landon'
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from src.deduplicator import DeduplicatedFact
from src.providers import LLMProvider


# Categories for the memory profile
CATEGORIES = ["personal", "professional", "family", "preferences", "interests", "personality"]


# JSON schema for structured output from LLM
DISTILL_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "User's name extracted from facts, or 'User' if unknown"
        },
        "categories": {
            "type": "object",
            "properties": {
                "personal": {"type": "array", "items": {"type": "string"}},
                "professional": {"type": "array", "items": {"type": "string"}},
                "family": {"type": "array", "items": {"type": "string"}},
                "preferences": {"type": "array", "items": {"type": "string"}},
                "interests": {"type": "array", "items": {"type": "string"}},
                "personality": {"type": "array", "items": {"type": "string"}},
            },
            "required": CATEGORIES,
            "additionalProperties": False,
        }
    },
    "required": ["name", "categories"],
    "additionalProperties": False,
}


@dataclass
class DistilledProfile:
    """Final memory profile ready for output.

    Attributes:
        name: User's name (extracted or 'User')
        categories: Dict mapping category names to fact lists
        generated: ISO timestamp of when profile was created
        source: Description of source data (e.g., "ChatGPT export (450 conversations)")
        fact_count: Number of input facts (for engineering data)

    Example:
        >>> profile = DistilledProfile(
        ...     name="Landon",
        ...     categories={"personal": ["Lives in Victoria, BC"]},
        ...     generated="2025-12-05T22:00:00",
        ...     source="ChatGPT export (450 conversations)",
        ...     fact_count=85
        ... )
    """
    name: str
    categories: Dict[str, List[str]]
    generated: str
    source: str
    fact_count: int


def load_distill_prompt() -> str:
    """Load the distillation prompt template from file.

    Returns:
        Prompt template string.

    Example:
        >>> template = load_distill_prompt()
        >>> print(template[:50])
        'You are creating a memory profile...'
    """
    prompt_path = Path(__file__).parent.parent / "prompts" / "distill.txt"
    return prompt_path.read_text()


def build_distill_prompt(
    facts: List[DeduplicatedFact],
    trusted_context: Optional[str] = None
) -> str:
    """Build the full distill prompt with facts as JSON.

    If trusted_context is provided, includes it with instructions to
    preserve existing phrasing and only add new information.

    Args:
        facts: Deduplicated facts to synthesize into profile.
        trusted_context: Optional existing profile to merge with.

    Returns:
        Full prompt string ready for LLM.

    Example:
        >>> prompt = build_distill_prompt(facts, trusted_context="User lives in BC")
        >>> "TRUSTED BASELINE" in prompt
        True
    """
    template = load_distill_prompt()

    facts_json = json.dumps(
        [{"fact": f.fact, "category": f.category} for f in facts],
        indent=2
    )

    if trusted_context:
        # Merge mode: preserve trusted, add new
        return (
            template +
            "\n\nTRUSTED BASELINE (preserve this phrasing, only ADD new information):\n" +
            trusted_context +
            "\n\nNEW FACTS TO MERGE:\n" +
            facts_json
        )
    else:
        return template + facts_json


def distill(
    facts: List[DeduplicatedFact],
    provider: LLMProvider,
    trusted_context: Optional[str] = None,
    source_info: str = "",
) -> DistilledProfile:
    """Compress deduplicated facts into final memory profile using LLM.

    Takes a list of deduplicated facts and uses the LLM to organize them
    into a coherent, categorized profile. If trusted_context is provided
    (from Claude memories or ChatGPT custom instructions), preserves that
    phrasing and only adds new information.

    Args:
        facts: Deduplicated facts from the deduplicator stage.
        provider: LLMProvider for making LLM calls.
        trusted_context: Optional pre-existing profile prose to merge with.
        source_info: Description of source data for metadata.

    Returns:
        DistilledProfile containing organized facts and metadata.

    Raises:
        RuntimeError: If LLM call fails.

    Example:
        >>> profile = distill(facts, provider, source_info="ChatGPT (450 convos)")
        >>> print(profile.name)
        'Landon'
        >>> print(len(profile.categories['personal']))
        15
    """
    if not facts:
        # Return empty profile for empty input
        return DistilledProfile(
            name="User",
            categories={cat: [] for cat in CATEGORIES},
            generated=datetime.now().isoformat(),
            source=source_info or "Unknown",
            fact_count=0,
        )

    prompt = build_distill_prompt(facts, trusted_context)
    result = provider.complete_structured(prompt, DISTILL_SCHEMA)

    # Ensure all categories exist (LLM might omit empty ones)
    categories = result.get("categories", {})
    for cat in CATEGORIES:
        if cat not in categories:
            categories[cat] = []

    return DistilledProfile(
        name=result.get("name", "User"),
        categories=categories,
        generated=datetime.now().isoformat(),
        source=source_info or "Unknown",
        fact_count=len(facts),
    )


def write_markdown(profile: DistilledProfile, output_dir: Path) -> Path:
    """Write human-readable markdown profile.

    Creates a formatted markdown document with the profile organized
    by category. Includes metadata header with generation timestamp
    and source information.

    Args:
        profile: The distilled profile to write.
        output_dir: Directory to write the file to.

    Returns:
        Path to the written markdown file.

    Example:
        >>> md_path = write_markdown(profile, Path("output"))
        >>> print(md_path)
        output/memory-profile.md
    """
    md_path = output_dir / "memory-profile.md"

    lines = [f"# Memory Profile for {profile.name}\n\n"]
    lines.append(f"> Generated: {profile.generated}\n")
    lines.append(f"> Source: {profile.source}\n\n")

    for category in CATEGORIES:
        facts = profile.categories.get(category, [])
        if facts:
            lines.append(f"## {category.title()}\n\n")
            for fact in facts:
                lines.append(f"- {fact}\n")
            lines.append("\n")

    md_path.write_text("".join(lines))
    return md_path


def write_json(profile: DistilledProfile, output_dir: Path) -> Path:
    """Write machine-readable JSON profile.

    Creates a JSON file suitable for importing into other AI systems.
    Structure matches the expected format for memory imports.

    Args:
        profile: The distilled profile to write.
        output_dir: Directory to write the file to.

    Returns:
        Path to the written JSON file.

    Example:
        >>> json_path = write_json(profile, Path("output"))
        >>> print(json_path)
        output/memory-profile.json
    """
    json_path = output_dir / "memory-profile.json"

    data = {
        "name": profile.name,
        "generated": profile.generated,
        "source": profile.source,
        "categories": profile.categories,
    }

    json_path.write_text(json.dumps(data, indent=2))
    return json_path
