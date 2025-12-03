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

# Valid fact categories
CATEGORIES = [
    "personal",
    "professional",
    "family",
    "preferences",
    "interests",
    "personality"
]

# JSON schema for structured output extraction
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
                "required": ["fact", "category", "source_convo_id", "source_timestamp", "source_quote"]
            }
        }
    },
    "required": ["facts"]
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


def load_extraction_prompt() -> str:
    """Load the extraction prompt template from file."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "extraction.txt"
    return prompt_path.read_text()


def build_extraction_prompt(conversation_content: str) -> str:
    """
    Build the full extraction prompt.

    Combines the prompt template with the conversation content.
    """
    template = load_extraction_prompt()
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

    # Build the full prompt
    prompt = build_extraction_prompt(content)

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
