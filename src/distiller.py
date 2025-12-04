"""
distiller.py - Final compression to memory profile

Uses LLM to compress deduplicated facts into coherent profile.
Outputs both memory-profile.md and memory-profile.json.

Supports "trusted baseline" pattern:
- Claude exports include memories.json (pre-synthesized profile)
- ChatGPT exports may have user_editable_context (custom instructions)
- These are passed as trusted_context, which the distiller merges
  with extracted facts rather than starting from scratch.
"""

from typing import List, Dict, Optional
from pathlib import Path

# from .core import BatchedLLMTask


def distill(
    facts: List,  # Deduplicated ExtractedFacts
    provider,  # LLMProvider
    output_dir: Path,
    trusted_context: Optional[str] = None  # Pre-existing trusted profile (Claude memories, etc.)
) -> tuple[Path, Path]:
    """
    Compress facts into final memory profile.

    If trusted_context is provided (e.g., from Claude memories.json),
    the distiller merges extracted facts with the existing profile
    rather than creating from scratch. This preserves user-curated
    content while supplementing with newly extracted details.

    Args:
        facts: Deduplicated ExtractedFacts from aggregator
        provider: LLMProvider for LLM calls
        output_dir: Where to write output files
        trusted_context: Optional pre-existing profile prose to merge with

    Returns:
        (path_to_md, path_to_json)
    """
    # TODO (Stage 9):
    # 1. Load distill prompt template
    # 2. If trusted_context provided:
    #    - Use merge prompt: "Here's a trusted profile. Add only NEW info from facts."
    #    - Prefer trusted_context phrasing where overlapping
    # 3. Else:
    #    - Use standard prompt: "Create profile from these facts."
    # 4. Use BatchedLLMTask to process
    # 5. Parse response into structured profile
    # 6. Write memory-profile.md
    # 7. Write memory-profile.json
    # 8. Return paths
    pass


def write_markdown(profile: Dict, output_dir: Path) -> Path:
    """
    Write human-readable markdown profile.

    Format:
    # Memory Profile for {name}

    ## Personal
    - Lives in Victoria, BC, Canada
    - Age 35
    ...
    """
    # TODO: Generate markdown from profile dict
    pass


def write_json(profile: Dict, output_dir: Path) -> Path:
    """
    Write machine-readable JSON profile.

    Format:
    {
      "name": "Landon",
      "generated": "2025-12-02T23:00:00",
      "source": "ChatGPT export (450 conversations)",
      "categories": {
        "personal": ["fact1", "fact2"],
        ...
      }
    }
    """
    # TODO: json.dump profile dict
    pass


def load_distill_prompt() -> str:
    """Load the distillation prompt template."""
    # TODO: Read prompts/distill.txt
    pass
