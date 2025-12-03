"""
parser.py - Parse ChatGPT conversations.json

Tree structure -> Linear conversations -> User messages
Also extracts user_editable_context (free wins!)
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


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
    user_profile: str  # "Preferred name: Landon\nRole: Head of AI..."
    user_instructions: str  # "Follow the instructions below..."


def load_conversations(path: str) -> List[Dict[str, Any]]:
    """Load raw conversations from JSON file."""
    # TODO: json.load, handle large file (55MB+)
    pass


def extract_user_profile(conversations: List[Dict]) -> Optional[UserProfile]:
    """
    Extract user_editable_context from conversations.

    Found in ~80% of conversations as hidden user messages.
    content_type: "user_editable_context"
    """
    # TODO: Search for first user_editable_context message
    # Return UserProfile or None if not found
    pass


def flatten_tree(mapping: Dict[str, Any]) -> List[Message]:
    """
    Convert tree structure to linear message list.

    The mapping is a tree (supports branching for edits/regenerations).
    We follow children[0] to get the "main" conversation path.
    """
    # TODO:
    # 1. Find root node (parent is None)
    # 2. Walk tree following first child
    # 3. Build Message objects for each node with message != None
    # 4. Handle null timestamps (system messages)
    pass


def filter_user_messages(messages: List[Message]) -> List[Message]:
    """Filter to just user messages (not hidden)."""
    # TODO: [m for m in messages if m.role == "user" and not m.is_hidden]
    pass


def parse_all(path: str) -> tuple[List[Conversation], Optional[UserProfile]]:
    """
    Main entry point: parse everything.

    Returns:
        - List of linearized Conversations
        - UserProfile if found (free wins from custom instructions)
    """
    # TODO:
    # 1. load_conversations(path)
    # 2. extract_user_profile(raw)
    # 3. For each convo: flatten_tree(mapping) -> Conversation
    # 4. Return (conversations, user_profile)
    pass
