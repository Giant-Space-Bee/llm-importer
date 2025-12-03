"""
distiller.py - Final compression to memory profile

Uses LLM to compress deduplicated facts into coherent profile.
Outputs both memory-profile.md and memory-profile.json.
"""

from typing import List, Dict
from pathlib import Path

# from .core import BatchedLLMTask


def distill(
    facts: List,  # Deduplicated ExtractedFacts
    provider,  # LLMProvider
    output_dir: Path
) -> tuple[Path, Path]:
    """
    Compress facts into final memory profile.

    Uses BatchedLLMTask for auto-batching.

    Returns:
        (path_to_md, path_to_json)
    """
    # TODO:
    # 1. Load distill prompt template
    # 2. Use BatchedLLMTask to process
    # 3. Parse response into structured profile
    # 4. Write memory-profile.md
    # 5. Write memory-profile.json
    # 6. Return paths
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
