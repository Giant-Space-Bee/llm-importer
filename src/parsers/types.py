"""
parsers/types.py - Shared types for all parsers

These dataclasses are used across ChatGPT, Claude, and future formats.
"""

from dataclasses import dataclass
from typing import List, Optional, Literal


ExportType = Literal["chatgpt", "claude", "unknown"]


@dataclass
class Message:
    """A single message from a conversation."""
    id: str
    role: str  # "user" | "assistant" | "system" | "tool"
    content: str
    timestamp: Optional[float]
    is_hidden: bool


@dataclass
class Conversation:
    """A linearized conversation with metadata."""
    id: str
    title: str
    create_time: float
    messages: List[Message]


@dataclass
class UserProfile:
    """Extracted from user_editable_context (ChatGPT custom instructions)."""
    user_profile: str
    user_instructions: str


@dataclass
class ConversationStats:
    """Aggregate statistics about the conversations."""
    total_conversations: int
    total_messages: int
    user_messages: int
    total_chars: int
    has_user_profile: bool
