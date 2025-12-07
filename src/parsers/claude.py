"""
parsers/claude.py - Claude export parser

Handles Claude conversations.json format:
- Flat array with 'uuid' and 'chat_messages' fields
- ISO 8601 timestamps
- sender: "human" / "assistant"
"""

import sys
from datetime import datetime
from typing import List, Dict, Any, Optional

from src.parsers.types import Message, Conversation


def parse_iso_timestamp(iso_str: str, fallback_timestamp: float | None = None) -> float:
    """Convert ISO timestamp string to Unix timestamp.

    Args:
        iso_str: ISO 8601 timestamp string
        fallback_timestamp: If provided, use this on parse failure instead of current time
    """
    try:
        # Handle ISO format with microseconds and Z suffix
        if iso_str.endswith("Z"):
            iso_str = iso_str[:-1] + "+00:00"
        dt = datetime.fromisoformat(iso_str)
        return dt.timestamp()
    except (ValueError, AttributeError):
        if fallback_timestamp is not None:
            print(f"[warning] Malformed timestamp '{iso_str}', using conversation time", file=sys.stderr)
            return fallback_timestamp
        print(f"[warning] Malformed timestamp '{iso_str}', using current time", file=sys.stderr)
        return datetime.now().timestamp()


def parse_claude_message(msg_data: Dict[str, Any], fallback_timestamp: float | None = None) -> Optional[Message]:
    """Parse a Claude message dict into a Message object."""
    if msg_data is None:
        return None

    # Map Claude's "human" to our "user" role
    sender = msg_data.get("sender", "unknown")
    role = "user" if sender == "human" else sender

    # Get text from content array or top-level text field
    content = ""
    content_parts = msg_data.get("content", [])
    if content_parts:
        # Join all text parts
        content = "".join(
            part.get("text", "") for part in content_parts
            if isinstance(part, dict)
        )
    # Fallback to top-level text
    if not content:
        content = msg_data.get("text", "")

    # Parse ISO timestamp (fall back to conversation time if malformed)
    timestamp = parse_iso_timestamp(msg_data.get("created_at", ""), fallback_timestamp)

    return Message(
        id=msg_data.get("uuid", ""),
        role=role,
        content=content,
        timestamp=timestamp,
        is_hidden=False  # Claude exports don't have hidden messages
    )


def parse_claude_conversation(convo_data: Dict[str, Any]) -> Conversation:
    """Parse a single Claude conversation into our Conversation format."""
    # Parse conversation timestamp first (used as fallback for malformed message timestamps)
    create_time = parse_iso_timestamp(convo_data.get("created_at", ""))

    messages = []
    for msg_data in convo_data.get("chat_messages", []):
        msg = parse_claude_message(msg_data, fallback_timestamp=create_time)
        if msg is not None:
            messages.append(msg)

    return Conversation(
        id=convo_data.get("uuid", ""),
        title=convo_data.get("name", "") or "Untitled",
        create_time=create_time,
        messages=messages
    )


def parse_claude_conversations(raw: List[Dict[str, Any]]) -> List[Conversation]:
    """Parse all Claude conversations."""
    return [parse_claude_conversation(convo) for convo in raw]
