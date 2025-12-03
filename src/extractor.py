"""
extractor.py - LLM fact extraction

Stage 4:
- Extract facts from conversation chunks
- Parse LLM JSON response
- Return ExtractedFact objects with source_quote for verification
"""

import json
import re
from typing import List, Optional
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


def parse_extraction_response(response: str) -> List[ExtractedFact]:
    """
    Parse LLM response into ExtractedFact objects.

    Handles:
    - Clean JSON arrays
    - JSON wrapped in ```json code blocks
    - Hermes 4 <think>...</think> tags before JSON
    - Invalid JSON (returns empty list)
    - Malformed facts (skips them)
    """
    if not response:
        return []

    # Strip <think>...</think> tags (Hermes 4 reasoning)
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

    # Extract JSON from markdown code blocks
    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
    if code_block_match:
        response = code_block_match.group(1)

    # Clean up whitespace
    response = response.strip()

    # Try to parse JSON
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        # Try to find a JSON array anywhere in the response
        array_match = re.search(r'\[[\s\S]*\]', response)
        if array_match:
            try:
                data = json.loads(array_match.group())
            except json.JSONDecodeError:
                return []
        else:
            return []

    # Must be a list
    if not isinstance(data, list):
        return []

    # Parse each fact
    facts = []
    for item in data:
        if not isinstance(item, dict):
            continue

        # Check required fields
        required = ["fact", "category", "source_convo_id", "source_timestamp", "source_quote"]
        if not all(key in item for key in required):
            continue

        try:
            fact = ExtractedFact(
                fact=str(item["fact"]),
                category=str(item["category"]),
                source_convo_id=str(item["source_convo_id"]),
                source_timestamp=float(item["source_timestamp"]),
                source_quote=str(item["source_quote"])
            )
            facts.append(fact)
        except (ValueError, TypeError):
            # Skip malformed facts
            continue

    return facts


def extract_chunk(chunk: Chunk, provider: LLMProvider) -> List[ExtractedFact]:
    """
    Extract facts from a single chunk using LLM.

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

    # Call the LLM
    response = provider.complete(prompt)

    # Parse the response
    return parse_extraction_response(response)
