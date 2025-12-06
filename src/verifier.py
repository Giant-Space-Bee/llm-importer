"""
verifier.py - Hallucination detection via string matching

NO LLM NEEDED. Pure text search.
Match source_quote back to original conversation.
No match = hallucinated = discard.
"""

from typing import List, Tuple, Union, Any, Dict, TYPE_CHECKING
from dataclasses import dataclass

from src.parser import flatten_tree

if TYPE_CHECKING:
    from src.extractor import ExtractedFact

# Type alias for facts - can be dict or ExtractedFact dataclass
Fact = Union[Dict[str, Any], "ExtractedFact"]


def get_fact_attr(fact: Fact, attr: str, default: Any = "") -> Any:
    """Get attribute from fact (works with both dict and ExtractedFact dataclass)."""
    if isinstance(fact, dict):
        return fact.get(attr, default)
    else:
        return getattr(fact, attr, default)


def conversation_to_text(conversation: Any) -> str:
    """
    Convert a conversation (dict or text) to a searchable text string.

    Handles:
    - Raw ChatGPT conversation dict with 'mapping' field -> flatten and join message texts
    - Raw Claude conversation dict with 'chat_messages' field -> join message texts
    - Already-text string -> return as-is

    Returns:
        Concatenated text of all messages in the conversation
    """
    from src.parser import _parse_claude_message

    # Already a string
    if isinstance(conversation, str):
        return conversation

    # ChatGPT: Raw conversation dict with mapping tree
    if isinstance(conversation, dict) and "mapping" in conversation:
        messages = flatten_tree(conversation["mapping"])
        return "\n".join(msg.content for msg in messages if msg.content)

    # Claude: Raw conversation dict with chat_messages array
    if isinstance(conversation, dict) and "chat_messages" in conversation:
        texts = []
        for msg_data in conversation.get("chat_messages", []):
            msg = _parse_claude_message(msg_data)
            if msg and msg.content:
                texts.append(msg.content)
        return "\n".join(texts)

    # Unknown format
    return ""


@dataclass
class VerificationResult:
    """Result of verifying a fact."""
    fact: Fact  # ExtractedFact or dict
    verified: bool
    match_location: str  # Where the quote was found (or "NOT FOUND")


def verify_fact(fact: Fact, conversation_text: str) -> VerificationResult:
    """
    Verify a single fact by finding its source_quote in conversation text.

    Args:
        fact: Dict or ExtractedFact with source_quote field
        conversation_text: Full conversation text (ALL messages, not just user)

    Returns:
        VerificationResult with verified=True if quote found
    """
    source_quote = get_fact_attr(fact, "source_quote", "")

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
    facts: List[Fact],
    conversations: Dict[str, Any]
) -> Tuple[List[Fact], List[Fact], List[str]]:
    """
    Verify all facts, separate into verified and discarded.

    Args:
        facts: List of fact dicts or ExtractedFact objects with source_quote and source_convo_id
        conversations: Dict mapping convo_id -> raw conversation dict (ChatGPT or Claude format)

    Returns:
        Tuple of:
        - verified: facts where source_quote was found
        - discarded: facts where source_quote was NOT found (hallucinations)
        - hallucination_logs: log strings for each hallucination (for file output)
    """
    verified = []
    discarded = []
    hallucination_logs = []

    for fact in facts:
        convo_id = get_fact_attr(fact, "source_convo_id", "")
        convo_raw = conversations.get(convo_id, "")

        # Convert to text (handles both raw dict and already-text)
        convo_text = conversation_to_text(convo_raw) if convo_raw else ""

        # If conversation not found, fail verification
        if not convo_text:
            discarded.append(fact)
            hallucination_logs.append(format_hallucination(fact, "conversation not found"))
            continue

        result = verify_fact(fact, convo_text)

        if result.verified:
            verified.append(fact)
        else:
            discarded.append(fact)
            hallucination_logs.append(format_hallucination(fact, result.match_location))

    return verified, discarded, hallucination_logs


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


def format_hallucination(fact: Fact, reason: str = "") -> str:
    """
    Format a hallucination log entry.

    Args:
        fact: The fact (dict or ExtractedFact) that failed verification
        reason: Why it failed (e.g., "NOT FOUND", "conversation not found")

    Returns:
        Formatted log string for the hallucination
    """
    quote = get_fact_attr(fact, "source_quote", "")[:100]
    fact_text = get_fact_attr(fact, "fact", "")[:100]
    convo_id = get_fact_attr(fact, "source_convo_id", "unknown")
    return f"[{reason}] Fact: '{fact_text}...'\n  Quote: '{quote}...'\n  Conversation: {convo_id}"


def log_hallucination(fact: Fact, reason: str = "") -> None:
    """
    Log a discarded hallucination for debugging (deprecated).

    Use format_hallucination() instead and collect logs for file output.

    Args:
        fact: The fact (dict or ExtractedFact) that failed verification
        reason: Why it failed (e.g., "NOT FOUND", "conversation not found")
    """
    import sys
    print(format_hallucination(fact, reason), file=sys.stderr)
