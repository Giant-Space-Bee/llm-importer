"""
verifier.py - Hallucination detection via string matching

NO LLM NEEDED. Pure text search.
Match source_quote back to original conversation.
No match = hallucinated = discard.
"""

from typing import List, Tuple
from dataclasses import dataclass

# from .extractor import ExtractedFact
# from .parser import Conversation


@dataclass
class VerificationResult:
    """Result of verifying a fact."""
    fact: object  # ExtractedFact
    verified: bool
    match_location: str  # Where the quote was found (or "NOT FOUND")


def verify_fact(fact: dict, conversation_text: str) -> VerificationResult:
    """
    Verify a single fact by finding its source_quote in conversation text.

    Args:
        fact: Dict with source_quote field
        conversation_text: Full conversation text (ALL messages, not just user)

    Returns:
        VerificationResult with verified=True if quote found
    """
    source_quote = fact.get("source_quote", "")

    # Empty quote = automatic fail
    if not source_quote or not source_quote.strip():
        return VerificationResult(
            fact=fact,
            verified=False,
            match_location="NOT FOUND - empty quote"
        )

    # Normalize both for matching
    normalized_quote = normalize_text(source_quote)
    normalized_convo = normalize_text(conversation_text)

    # Check if quote exists in conversation
    if normalized_quote in normalized_convo:
        return VerificationResult(
            fact=fact,
            verified=True,
            match_location="FOUND"
        )
    else:
        return VerificationResult(
            fact=fact,
            verified=False,
            match_location="NOT FOUND"
        )


def verify_all(
    facts: List[dict],
    conversations: dict
) -> Tuple[List[dict], List[dict]]:
    """
    Verify all facts, separate into verified and discarded.

    Args:
        facts: List of fact dicts with source_quote and source_convo_id
        conversations: Dict mapping convo_id -> full conversation text

    Returns:
        - verified: facts where source_quote was found
        - discarded: facts where source_quote was NOT found (hallucinations)
    """
    verified = []
    discarded = []

    for fact in facts:
        convo_id = fact.get("source_convo_id", "")
        convo_text = conversations.get(convo_id, "")

        # If conversation not found, fail verification
        if not convo_text:
            discarded.append(fact)
            log_hallucination(fact, "conversation not found")
            continue

        result = verify_fact(fact, convo_text)

        if result.verified:
            verified.append(fact)
        else:
            discarded.append(fact)
            log_hallucination(fact, result.match_location)

    return verified, discarded


def normalize_text(text: str) -> str:
    """
    Normalize text for quote matching.

    Handles:
    - Lowercase
    - Collapse whitespace (spaces, tabs, newlines → single space)
    - Curly quotes → straight quotes (LLMs often return these)
    - Em/en dashes → hyphens
    - Ellipsis → three dots
    """
    import re

    if not text:
        return ""

    # Unicode replacements: curly quotes, dashes, ellipsis
    replacements = {
        '\u2018': "'",   # Left single quote '
        '\u2019': "'",   # Right single quote ' (apostrophe)
        '\u201C': '"',   # Left double quote "
        '\u201D': '"',   # Right double quote "
        '\u2014': '-',   # Em dash —
        '\u2013': '-',   # En dash –
        '\u2026': '...', # Ellipsis …
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Lowercase
    text = text.lower()

    # Collapse whitespace to single space
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing
    text = text.strip()

    return text


def log_hallucination(fact: dict, reason: str = "") -> None:
    """
    Log a discarded hallucination for debugging.

    Args:
        fact: The fact that failed verification
        reason: Why it failed (e.g., "NOT FOUND", "conversation not found")
    """
    # For now, just print to stderr. Later can write to file.
    import sys
    quote = fact.get("source_quote", "")[:50]
    fact_text = fact.get("fact", "")[:50]
    print(f"[HALLUCINATION] {reason}: '{quote}...' -> '{fact_text}...'", file=sys.stderr)
