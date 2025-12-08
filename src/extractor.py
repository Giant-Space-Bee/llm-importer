"""
extractor.py - LLM fact extraction

Stage 4:
- Extract facts from conversation chunks
- Parse LLM JSON response
- Return ExtractedFact objects with source_quote for verification
"""

from typing import List
from dataclasses import dataclass
from pathlib import Path

from src.providers import LLMProvider
from src.chunker import Chunk, format_chunk_for_extraction

# Valid fact categories for personal growth/journaling platform
CATEGORIES = [
    "identity",       # Who: name, age, location, background, self-description
    "values",         # What matters: beliefs, principles, priorities
    "emotions",       # Feeling patterns: recurring emotions, triggers, coping
    "relationships",  # People: family, friends, partners, pets, dynamics
    "growth",         # Goals: aspirations, dreams, what they're building toward
    "history",        # Past: life events, milestones, formative experiences
    "practices",      # Habits: routines, self-care, meditation, therapy, journaling
    "shadows",        # Depths: patterns to transform, fears, blocks, inner conflicts
]

# JSON schema for structured output extraction
# Note: additionalProperties: false required for Anthropic structured outputs
FACT_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": CATEGORIES
                    },
                    "source_convo_id": {"type": "string"},
                    "source_timestamp": {"type": "number"},
                    "source_quote": {"type": "string"}
                },
                "required": ["fact", "category", "source_convo_id", "source_timestamp", "source_quote"],
                "additionalProperties": False
            }
        }
    },
    "required": ["facts"],
    "additionalProperties": False
}


@dataclass
class ExtractedFact:
    """
    A single extracted fact with provenance.

    source_quote is VERBATIM text from the conversation.
    Used for hallucination detection via string matching.
    """
    fact: str
    category: str
    source_convo_id: str
    source_timestamp: float
    source_quote: str


def load_extraction_prompt(is_local: bool = False) -> str:
    """
    Load the extraction prompt template for the specified provider type.

    Args:
        is_local: True for local LLM (explicit prompt), False for API (nuanced prompt)

    Returns:
        The prompt template text

    Raises:
        FileNotFoundError: If the required prompt file doesn't exist
    """
    prompts_dir = Path(__file__).parent.parent / "prompts"
    filename = "extraction-local.txt" if is_local else "extraction-api.txt"
    prompt_path = prompts_dir / filename

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Missing prompt file: {prompt_path}\n"
            f"Provider-specific prompts are required. No fallback."
        )

    return prompt_path.read_text()


def build_extraction_prompt(conversation_content: str, is_local: bool = False) -> str:
    """
    Build the full extraction prompt.

    Combines the provider-specific prompt template with the conversation content.

    Args:
        conversation_content: Formatted conversation text to extract from
        is_local: True for local LLM prompt, False for API prompt
    """
    template = load_extraction_prompt(is_local=is_local)
    return template + conversation_content


def extract_chunk(chunk: Chunk, provider: LLMProvider) -> List[ExtractedFact]:
    """
    Extract facts from a single chunk using LLM with structured output.

    Args:
        chunk: The Chunk containing conversations
        provider: LLM provider to use

    Returns:
        List of ExtractedFact objects
    """
    # Format the chunk content
    content = format_chunk_for_extraction(chunk)

    # Build the full prompt (use provider-specific prompt)
    prompt = build_extraction_prompt(content, is_local=provider.is_local)

    # Call the LLM with structured output
    result = provider.complete_structured(prompt, FACT_SCHEMA)

    # Convert to ExtractedFact objects
    facts = []
    for item in result.get("facts", []):
        try:
            fact = ExtractedFact(
                fact=str(item["fact"]),
                category=str(item["category"]),
                source_convo_id=str(item["source_convo_id"]),
                source_timestamp=float(item["source_timestamp"]),
                source_quote=str(item["source_quote"])
            )
            facts.append(fact)
        except (KeyError, ValueError, TypeError):
            # Skip malformed facts (shouldn't happen with schema enforcement)
            continue

    return facts
