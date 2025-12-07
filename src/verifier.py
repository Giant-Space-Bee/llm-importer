"""
verifier.py - Hallucination detection via string matching

NO LLM NEEDED. Pure text search.
Match source_quote back to original conversation.
No match = hallucinated = discard.

Verification strategy:
1. Exact substring match (after normalization)
2. Fuzzy match fallback (handles typos, minor paraphrasing)
3. Search all conversations (handles wrong conversation ID)
"""

from typing import List, Tuple, Union, Any, Dict, TYPE_CHECKING, Optional
from dataclasses import dataclass
from difflib import SequenceMatcher

from src.parser import flatten_tree

if TYPE_CHECKING:
    from src.extractor import ExtractedFact

# Fuzzy matching threshold (0.0 to 1.0)
# 0.85 = 85% similar - catches typos like "Rpd" vs "Rod" but rejects paraphrasing
FUZZY_MATCH_THRESHOLD = 0.85

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


def verify_fact(fact: Fact, conversation_text: str, use_fuzzy: bool = True) -> VerificationResult:
    """
    Verify a single fact by finding its source_quote in conversation text.

    Verification strategy:
    1. Try exact substring match (after normalization)
    2. If use_fuzzy=True, try fuzzy match as fallback

    Args:
        fact: Dict or ExtractedFact with source_quote field
        conversation_text: Full conversation text (ALL messages, not just user)
        use_fuzzy: Whether to try fuzzy matching if exact match fails

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

    # Strategy 1: Exact substring match
    if normalized_quote in normalized_convo:
        return VerificationResult(
            fact=fact,
            verified=True,
            match_location="FOUND (exact)"
        )

    # Strategy 2: Fuzzy match fallback
    if use_fuzzy:
        fuzzy_result = fuzzy_find_in_text(normalized_quote, normalized_convo)
        if fuzzy_result:
            # Calculate actual ratio for logging
            ratio = SequenceMatcher(None, normalized_quote, fuzzy_result).ratio()
            log_fuzzy_match(fact, fuzzy_result, ratio)
            return VerificationResult(
                fact=fact,
                verified=True,
                match_location=f"FOUND (fuzzy {ratio:.0%})"
            )

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

    Uses two-pass verification:
    1. Try exact match in specified conversation
    2. If not found, search ALL conversations (fallback for wrong conversation ID)

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

        # Try exact match in specified conversation first
        if convo_text:
            result = verify_fact(fact, convo_text)
            if result.verified:
                verified.append(fact)
                continue

        # Fallback: search ALL conversations for the quote
        # This catches cases where LLM recorded wrong conversation ID
        source_quote = get_fact_attr(fact, "source_quote", "")
        if source_quote and source_quote.strip():
            correct_id = _find_quote_in_conversations(source_quote, conversations)
            if correct_id:
                # Found in different conversation - fix the ID and verify
                _update_fact_convo_id(fact, correct_id)
                log_id_correction(fact, convo_id, correct_id)
                verified.append(fact)
                continue

        # Not found anywhere - this is a true hallucination
        discarded.append(fact)
        if not convo_text:
            hallucination_logs.append(format_hallucination(fact, "conversation not found"))
        else:
            hallucination_logs.append(format_hallucination(fact, "NOT FOUND"))

    return verified, discarded, hallucination_logs


def _find_quote_in_conversations(
    source_quote: str,
    conversations: Dict[str, Any],
    use_fuzzy: bool = True
) -> str:
    """
    Search all conversations for a quote, return conversation ID if found.

    Uses two-pass search:
    1. Try exact match in all conversations
    2. If use_fuzzy=True, try fuzzy match in all conversations

    Args:
        source_quote: The quote to search for
        conversations: Dict mapping convo_id -> raw conversation dict
        use_fuzzy: Whether to try fuzzy matching if exact match fails

    Returns:
        Conversation ID where quote was found, or empty string if not found
    """
    normalized_quote = normalize_text(source_quote)

    # Pass 1: Exact match
    for convo_id, convo_raw in conversations.items():
        convo_text = conversation_to_text(convo_raw)
        if convo_text:
            normalized_convo = normalize_text(convo_text)
            if normalized_quote in normalized_convo:
                return convo_id

    # Pass 2: Fuzzy match
    if use_fuzzy:
        for convo_id, convo_raw in conversations.items():
            convo_text = conversation_to_text(convo_raw)
            if convo_text:
                normalized_convo = normalize_text(convo_text)
                if fuzzy_find_in_text(normalized_quote, normalized_convo):
                    return convo_id

    return ""


def _update_fact_convo_id(fact: Fact, new_convo_id: str) -> None:
    """Update the conversation ID on a fact (handles both dict and dataclass)."""
    if isinstance(fact, dict):
        fact["source_convo_id"] = new_convo_id
    else:
        # Dataclass - need to use object.__setattr__ if frozen
        try:
            fact.source_convo_id = new_convo_id
        except AttributeError:
            object.__setattr__(fact, "source_convo_id", new_convo_id)


def log_id_correction(fact: Fact, old_id: str, new_id: str) -> None:
    """Log when we correct a conversation ID."""
    import sys
    quote = get_fact_attr(fact, "source_quote", "")[:40]
    print(
        f"[ID CORRECTED] '{quote}...' moved from {old_id[:12]}... to {new_id[:12]}...",
        file=sys.stderr
    )


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


def fuzzy_find_in_text(quote: str, text: str, threshold: float = FUZZY_MATCH_THRESHOLD) -> Optional[str]:
    """
    Find a fuzzy match for quote within text.

    Uses sliding window approach: check each window of text that's roughly
    the same length as the quote, find the best match above threshold.

    Args:
        quote: The normalized quote to search for
        text: The normalized text to search in
        threshold: Minimum similarity ratio (0.0 to 1.0)

    Returns:
        The matching substring from text if found, None otherwise
    """
    if not quote or not text:
        return None

    quote_len = len(quote)
    text_len = len(text)

    if quote_len > text_len:
        return None

    # For short quotes, require higher similarity to avoid false positives
    if quote_len < 20:
        threshold = max(threshold, 0.90)

    best_match = None
    best_ratio = threshold  # Only accept matches above threshold

    # Sliding window: check windows of varying sizes around quote length
    # This handles cases where quote has extra/missing words
    for window_size in range(max(10, quote_len - 10), min(text_len + 1, quote_len + 20)):
        for start in range(0, text_len - window_size + 1, 5):  # Step by 5 for efficiency
            window = text[start:start + window_size]
            ratio = SequenceMatcher(None, quote, window).ratio()

            if ratio > best_ratio:
                best_ratio = ratio
                best_match = window

    return best_match


def log_fuzzy_match(fact: Fact, matched_text: str, ratio: float) -> None:
    """Log when we verify via fuzzy match (for monitoring)."""
    import sys
    quote = get_fact_attr(fact, "source_quote", "")[:40]
    print(
        f"[FUZZY MATCH] {ratio:.0%} similarity: '{quote}...' matched '{matched_text[:40]}...'",
        file=sys.stderr
    )


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
