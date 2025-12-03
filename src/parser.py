"""
parser.py - Parse ChatGPT conversations.json

Stage 2:
- Load JSON file
- Flatten tree structure to linear messages
- Filter to user messages
- Extract user_editable_context (free wins!)
- Calculate stats
"""

import json
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass
from pathlib import Path


ExportType = Literal["chatgpt", "claude", "unknown"]


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


def load_conversations(path: str) -> List[Dict[str, Any]]:
    """Load raw conversations from JSON file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def flatten_tree(mapping: Dict[str, Any]) -> List[Message]:
    """
    Convert tree structure to linear message list.

    The mapping is a tree supporting branching (edits/regenerations).
    We follow children[0] to get the "main" conversation path.
    """
    messages = []

    # Find root node (parent is None)
    root_id = None
    for node_id, node in mapping.items():
        if node.get("parent") is None:
            root_id = node_id
            break

    if root_id is None:
        return messages

    # Walk tree following first child
    current_id = root_id
    while current_id:
        node = mapping.get(current_id)
        if node is None:
            break

        # Extract message if present
        msg_data = node.get("message")
        if msg_data is not None:
            message = _parse_message(msg_data)
            if message is not None:
                messages.append(message)

        # Follow first child
        children = node.get("children", [])
        current_id = children[0] if children else None

    return messages


def _parse_message(msg_data: Dict[str, Any]) -> Optional[Message]:
    """Parse a message dict into a Message object."""
    if msg_data is None:
        return None

    author = msg_data.get("author", {})
    role = author.get("role", "unknown")

    content_data = msg_data.get("content", {})
    content_type = content_data.get("content_type", "")

    # Extract text content
    if content_type == "text":
        parts = content_data.get("parts", [])
        content = "".join(str(p) for p in parts if isinstance(p, str))
    elif content_type == "user_editable_context":
        # This is custom instructions, mark as hidden
        content = content_data.get("user_profile", "") + "\n" + content_data.get("user_instructions", "")
    elif content_type == "multimodal_text":
        parts = content_data.get("parts", [])
        content = "".join(str(p) for p in parts if isinstance(p, str))
    elif content_type == "code":
        content = content_data.get("text", "")
    else:
        content = str(content_data)

    # Check if hidden
    metadata = msg_data.get("metadata", {})
    is_hidden = metadata.get("is_visually_hidden_from_conversation", False)

    # user_editable_context is always hidden
    if content_type == "user_editable_context":
        is_hidden = True

    return Message(
        id=msg_data.get("id", ""),
        role=role,
        content=content,
        timestamp=msg_data.get("create_time"),
        is_hidden=is_hidden
    )


def filter_user_messages(messages: List[Message]) -> List[Message]:
    """Filter to just user messages (not hidden)."""
    return [m for m in messages if m.role == "user" and not m.is_hidden]


def extract_user_profile(conversations: List[Dict[str, Any]]) -> Optional[UserProfile]:
    """
    Extract user_editable_context from conversations.

    Found in ~80% of conversations as hidden user messages.
    Returns the first one found (they're usually identical).
    """
    for convo in conversations:
        mapping = convo.get("mapping", {})
        for node in mapping.values():
            msg_data = node.get("message")
            if msg_data is None:
                continue

            content_data = msg_data.get("content", {})
            if content_data.get("content_type") == "user_editable_context":
                return UserProfile(
                    user_profile=content_data.get("user_profile", ""),
                    user_instructions=content_data.get("user_instructions", "")
                )

    return None


def get_conversation_stats(conversations: List[Dict[str, Any]]) -> ConversationStats:
    """Calculate aggregate statistics about the conversations."""
    total_messages = 0
    user_messages = 0
    total_chars = 0

    for convo in conversations:
        mapping = convo.get("mapping", {})
        messages = flatten_tree(mapping)
        total_messages += len(messages)

        user_msgs = filter_user_messages(messages)
        user_messages += len(user_msgs)

        for msg in messages:
            total_chars += len(msg.content)

    has_profile = extract_user_profile(conversations) is not None

    return ConversationStats(
        total_conversations=len(conversations),
        total_messages=total_messages,
        user_messages=user_messages,
        total_chars=total_chars,
        has_user_profile=has_profile
    )


def get_stats_from_parsed(conversations: List[Conversation], user_profile: Optional[UserProfile]) -> ConversationStats:
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


def parse_all(path: str) -> tuple[List[Conversation], Optional[UserProfile]]:
    """
    Main entry point: parse everything.

    Returns:
        - List of linearized Conversations
        - UserProfile if found (free wins from custom instructions)
    """
    raw = load_conversations(path)
    user_profile = extract_user_profile(raw)

    conversations = []
    for convo_data in raw:
        mapping = convo_data.get("mapping", {})
        messages = flatten_tree(mapping)

        convo = Conversation(
            id=convo_data.get("id", ""),
            title=convo_data.get("title", "Untitled"),
            create_time=convo_data.get("create_time", 0.0),
            messages=messages
        )
        conversations.append(convo)

    return conversations, user_profile
