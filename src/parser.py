"""
parser.py - Parse LLM conversation exports (ChatGPT, Claude)

This is the public API facade. Implementation details are in src/parsers/:
- parsers/types.py: Shared dataclasses
- parsers/chatgpt.py: ChatGPT-specific parsing
- parsers/claude.py: Claude-specific parsing
- parsers/memories.py: Trusted baseline handling

Usage:
    from src.parser import parse_all, load_conversations, detect_export_type
    conversations, user_profile = parse_all("conversations.json")

Adding a new export format:
    1. Create parsers/newformat.py with parse_newformat_conversations()
    2. Add detection logic to detect_export_type()
    3. Register in _PARSERS dict: _PARSERS["newformat"] = (parser_func, profile_extractor)
    4. Update ExportType in parsers/types.py

Backward compatibility:
    _parse_claude_message and _parse_iso_timestamp are aliased for verifier.py
    and tests. Do not remove without updating those consumers.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable

# Re-export types from parsers package
from src.parsers.types import (
    ExportType,
    Message,
    Conversation,
    UserProfile,
    ConversationStats,
)

# Re-export ChatGPT parsing functions
from src.parsers.chatgpt import (
    flatten_tree,
    parse_message,
    extract_user_profile,
    get_conversation_stats,
    parse_chatgpt_conversations,
)

# Re-export Claude parsing functions
from src.parsers.claude import (
    parse_iso_timestamp,
    parse_claude_message,
    parse_claude_conversation,
    parse_claude_conversations,
)

# Re-export memories/profile functions
from src.parsers.memories import (
    ClaudeMemories,
    load_claude_memories,
    format_memories_for_distiller,
    format_user_profile_for_distiller,
)

# Type alias for parser registry entries
# Each entry is (parser_func, profile_extractor_or_None)
ParserEntry = Tuple[
    Callable[[List[Dict[str, Any]]], List[Conversation]],
    Optional[Callable[[List[Dict[str, Any]]], Optional[UserProfile]]]
]

# Parser registry: maps export_type -> (parser_func, profile_extractor)
# To add a new format: add entry here after implementing parser module
_PARSERS: Dict[str, ParserEntry] = {
    "chatgpt": (parse_chatgpt_conversations, extract_user_profile),
    "claude": (parse_claude_conversations, None),
}

# Backward compatibility aliases (used by verifier.py and tests)
_parse_iso_timestamp = parse_iso_timestamp
_parse_claude_message = parse_claude_message


def detect_export_type(data: List[Dict[str, Any]]) -> ExportType:
    """
    Auto-detect which LLM service created this export.

    Detection logic:
    - ChatGPT: has 'mapping' field with tree structure
    - Claude: has 'uuid' and 'chat_messages' fields
    - Unknown: neither pattern matches

    Args:
        data: Parsed JSON data (list of conversation objects)

    Returns:
        "chatgpt", "claude", or "unknown"
    """
    if not data or not isinstance(data, list):
        return "unknown"

    # Check first conversation for distinctive fields
    first = data[0]

    # ChatGPT: has 'mapping' field (tree structure)
    if "mapping" in first and isinstance(first.get("mapping"), dict):
        return "chatgpt"

    # Claude: has 'uuid' and 'chat_messages' (flat array)
    if "uuid" in first and "chat_messages" in first:
        return "claude"

    return "unknown"


def load_conversations(path: str) -> List[Dict[str, Any]]:
    """Load raw conversations from JSON file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def filter_user_messages(messages: List[Message]) -> List[Message]:
    """Filter to just user messages (not hidden)."""
    return [m for m in messages if m.role == "user" and not m.is_hidden]


def get_stats_from_parsed(
    conversations: List[Conversation],
    user_profile: Optional[UserProfile]
) -> ConversationStats:
    """Calculate stats from already-parsed conversations."""
    total_messages = 0
    user_messages = 0
    total_chars = 0

    for convo in conversations:
        total_messages += len(convo.messages)
        user_messages += len([m for m in convo.messages if m.role == "user" and not m.is_hidden])
        for msg in convo.messages:
            total_chars += len(msg.content)

    return ConversationStats(
        total_conversations=len(conversations),
        total_messages=total_messages,
        user_messages=user_messages,
        total_chars=total_chars,
        has_user_profile=user_profile is not None
    )


def parse_all(path: str) -> Tuple[List[Conversation], Optional[UserProfile]]:
    """
    Main entry point: parse everything.

    Auto-detects export type (ChatGPT vs Claude) and uses the appropriate
    parser from the registry.

    Returns:
        - List of linearized Conversations
        - UserProfile if found (ChatGPT custom instructions only)

    Raises:
        ValueError: If export type is unknown/unsupported
    """
    raw = load_conversations(path)
    export_type = detect_export_type(raw)

    if export_type not in _PARSERS:
        supported = ", ".join(_PARSERS.keys())
        raise ValueError(f"Unknown export type. Expected one of: {supported}")

    parser, profile_extractor = _PARSERS[export_type]
    conversations = parser(raw)
    user_profile = profile_extractor(raw) if profile_extractor else None

    return conversations, user_profile


# Export all public symbols
__all__ = [
    # Types
    "ExportType",
    "Message",
    "Conversation",
    "UserProfile",
    "ConversationStats",
    # Core functions
    "detect_export_type",
    "load_conversations",
    "filter_user_messages",
    "get_stats_from_parsed",
    "parse_all",
    # ChatGPT
    "flatten_tree",
    "parse_message",
    "extract_user_profile",
    "get_conversation_stats",
    "parse_chatgpt_conversations",
    # Claude
    "parse_iso_timestamp",
    "parse_claude_message",
    "parse_claude_conversation",
    "parse_claude_conversations",
    # Memories
    "ClaudeMemories",
    "load_claude_memories",
    "format_memories_for_distiller",
    "format_user_profile_for_distiller",
    # Backward compat aliases (verifier.py, tests)
    "_parse_iso_timestamp",
    "_parse_claude_message",
]
