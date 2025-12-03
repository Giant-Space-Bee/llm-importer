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


def verify_fact(fact, conversations: dict) -> VerificationResult:
    """
    Verify a single fact by finding its source_quote.

    Args:
        fact: ExtractedFact with source_quote and source_convo_id
        conversations: Dict mapping convo_id -> conversation text

    Returns:
        VerificationResult with verified=True if quote found
    """
    # TODO:
    # 1. Get conversation text by source_convo_id
    # 2. Search for source_quote in conversation
    # 3. Case-insensitive, normalize whitespace
    # 4. Return result
    pass


def verify_all(
    facts: List,  # List[ExtractedFact]
    conversations: dict
) -> Tuple[List, List]:  # (verified, discarded)
    """
    Verify all facts, separate into verified and discarded.

    Returns:
        - verified: facts where source_quote was found
        - discarded: facts where source_quote was NOT found (hallucinations)
    """
    # TODO:
    # verified = []
    # discarded = []
    # for fact in facts:
    #     result = verify_fact(fact, conversations)
    #     if result.verified:
    #         verified.append(fact)
    #     else:
    #         discarded.append(fact)
    #         log_hallucination(fact)
    # return verified, discarded
    pass


def normalize_text(text: str) -> str:
    """Normalize text for matching (lowercase, collapse whitespace)."""
    # TODO: text.lower().strip(), re.sub(r'\s+', ' ', text)
    pass


def log_hallucination(fact) -> None:
    """Log a discarded hallucination for debugging."""
    # TODO: Write to output/hallucinations.log
    pass
