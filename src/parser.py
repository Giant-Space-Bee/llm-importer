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
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

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

# Backward compatibility aliases for internal functions
# Tests import these with underscore prefix
_parse_message = parse_message
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

    Auto-detects export type (ChatGPT vs Claude) and uses the appropriate parser.

    Returns:
        - List of linearized Conversations
        - UserProfile if found (ChatGPT custom instructions only)
    """
    raw = load_conversations(path)
    export_type = detect_export_type(raw)

    if export_type == "chatgpt":
        conversations = parse_chatgpt_conversations(raw)
        user_profile = extract_user_profile(raw)
    elif export_type == "claude":
        conversations = parse_claude_conversations(raw)
        user_profile = None  # Claude doesn't have user_editable_context
    else:
        raise ValueError("Unknown export type. Expected ChatGPT or Claude format.")

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
    # Backward compat aliases
    "_parse_message",
    "_parse_iso_timestamp",
    "_parse_claude_message",
]
