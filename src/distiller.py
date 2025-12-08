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
FACT_WITH_PERIOD = {
    "type": "object",
    "properties": {
        "fact": {"type": "string"},
        "period": {"type": "string", "description": "Time range or date context, e.g. '2023-2025'"}
    },
    "required": ["fact", "period"],
    "additionalProperties": False,
}

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
                "personal": {"type": "array", "items": FACT_WITH_PERIOD},
                "professional": {"type": "array", "items": FACT_WITH_PERIOD},
                "family": {"type": "array", "items": FACT_WITH_PERIOD},
                "preferences": {"type": "array", "items": FACT_WITH_PERIOD},
                "interests": {"type": "array", "items": FACT_WITH_PERIOD},
                "personality": {"type": "array", "items": FACT_WITH_PERIOD},
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
        categories: Dict mapping category names to list of {fact, period} dicts
        generated: ISO timestamp of when profile was created
        source: Description of source data (e.g., "ChatGPT export (450 conversations)")
        fact_count: Number of input facts (for engineering data)
    """
    name: str
    categories: Dict[str, List[Dict[str, str]]]
    generated: str
    source: str
    fact_count: int


def load_distill_prompt() -> str:
    """Load the distillation prompt template from file."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "distill.txt"
    return prompt_path.read_text()


def build_distill_prompt(
    facts: List[DeduplicatedFact],
    trusted_context: Optional[str] = None
) -> str:
    """Build the full distill prompt with facts as JSON."""
    template = load_distill_prompt()

    facts_json = json.dumps(
        [{"fact": f.fact, "category": f.category, "period": f.period} for f in facts],
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
    """Compress deduplicated facts into final memory profile using LLM."""
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
    """
    md_path = output_dir / "memory-profile.md"

    lines = [f"# Memory Profile for {profile.name}\n\n"]
    lines.append(f"> Generated: {profile.generated}\n")
    lines.append(f"> Source: {profile.source}\n\n")

    for category in CATEGORIES:
        facts = profile.categories.get(category, [])
        if facts:
            lines.append(f"## {category.title()}\n\n")
            for item in facts:
                # Handle both dict (new) and string (legacy/fallback)
                if isinstance(item, dict):
                    fact_text = item.get("fact", "")
                    period = item.get("period", "")
                    if period:
                        lines.append(f"- {fact_text} *({period})*\n")
                    else:
                        lines.append(f"- {fact_text}\n")
                else:
                    lines.append(f"- {item}\n")
            lines.append("\n")

    md_path.write_text("".join(lines))
    return md_path


def write_json(profile: DistilledProfile, output_dir: Path) -> Path:
    """Write machine-readable JSON profile."""
    json_path = output_dir / "memory-profile.json"

    data = {
        "name": profile.name,
        "generated": profile.generated,
        "source": profile.source,
        "categories": profile.categories,
    }

    json_path.write_text(json.dumps(data, indent=2))
    return json_path
