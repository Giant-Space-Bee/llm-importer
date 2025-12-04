"""
parsers - Format-specific parsing modules

This package contains parsers for different LLM export formats:
- chatgpt: ChatGPT conversations.json (tree structure)
- claude: Claude conversations.json (flat structure)
- memories: Claude memories.json and ChatGPT user profile (trusted baselines)
- types: Shared dataclasses (Message, Conversation, etc.)
"""

from src.parsers.types import (
    ExportType,
    Message,
    Conversation,
    UserProfile,
    ConversationStats,
)
from src.parsers.chatgpt import (
    flatten_tree,
    parse_message,
    extract_user_profile,
    get_conversation_stats,
    parse_chatgpt_conversations,
)
from src.parsers.claude import (
    parse_iso_timestamp,
    parse_claude_message,
    parse_claude_conversation,
    parse_claude_conversations,
)
from src.parsers.memories import (
    ClaudeMemories,
    load_claude_memories,
    format_memories_for_distiller,
    format_user_profile_for_distiller,
)

__all__ = [
    # Types
    "ExportType",
    "Message",
    "Conversation",
    "UserProfile",
    "ConversationStats",
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
]
