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


# --- Claude export parsing ---

def _parse_iso_timestamp(iso_str: str) -> float:
    """Convert ISO timestamp string to Unix timestamp."""
    from datetime import datetime
    try:
        # Handle ISO format with microseconds and Z suffix
        if iso_str.endswith("Z"):
            iso_str = iso_str[:-1] + "+00:00"
        dt = datetime.fromisoformat(iso_str)
        return dt.timestamp()
    except (ValueError, AttributeError):
        return 0.0


def _parse_claude_message(msg_data: Dict[str, Any]) -> Optional[Message]:
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

    # Parse ISO timestamp
    timestamp = _parse_iso_timestamp(msg_data.get("created_at", ""))

    return Message(
        id=msg_data.get("uuid", ""),
        role=role,
        content=content,
        timestamp=timestamp,
        is_hidden=False  # Claude exports don't have hidden messages
    )


def parse_claude_conversation(convo_data: Dict[str, Any]) -> Conversation:
    """Parse a single Claude conversation into our Conversation format."""
    messages = []
    for msg_data in convo_data.get("chat_messages", []):
        msg = _parse_claude_message(msg_data)
        if msg is not None:
            messages.append(msg)

    # Parse create_time from ISO format
    create_time = _parse_iso_timestamp(convo_data.get("created_at", ""))

    return Conversation(
        id=convo_data.get("uuid", ""),
        title=convo_data.get("name", "") or "Untitled",
        create_time=create_time,
        messages=messages
    )


def parse_claude_conversations(raw: List[Dict[str, Any]]) -> List[Conversation]:
    """Parse all Claude conversations."""
    return [parse_claude_conversation(convo) for convo in raw]


# --- Claude memories.json parsing ---

@dataclass
class ClaudeMemories:
    """
    Parsed Claude memories.json content.

    Claude exports include a memories.json with pre-synthesized user profile.
    This is a "trusted baseline" - already curated by Claude, not raw data.
    """
    conversations_memory: str  # Main biography prose (markdown)
    project_memories: Dict[str, str]  # UUID -> prose per project
    account_uuid: str


def load_claude_memories(folder_path: str) -> Optional[ClaudeMemories]:
    """
    Load memories.json from a Claude export folder.

    Claude exports are folders with structure:
        data-{timestamp}-batch-{N}/
        ├── conversations.json
        ├── memories.json      ← this file
        ├── projects.json
        └── users.json

    Args:
        folder_path: Path to the Claude export folder

    Returns:
        ClaudeMemories if found and valid, None otherwise
    """
    memories_path = Path(folder_path) / "memories.json"

    if not memories_path.exists():
        return None

    try:
        with open(memories_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # memories.json is an array with a single object
        if not data or not isinstance(data, list) or len(data) == 0:
            return None

        mem_obj = data[0]

        # Extract fields with defaults for missing data
        conversations_memory = mem_obj.get("conversations_memory", "")
        project_memories_raw = mem_obj.get("project_memories", {})
        account_uuid = mem_obj.get("account_uuid", "")

        # Flatten project_memories: each value is a dict with prose fields
        # We'll concatenate all prose fields into a single string per project
        project_memories: Dict[str, str] = {}
        if isinstance(project_memories_raw, dict):
            for uuid, mem_data in project_memories_raw.items():
                if isinstance(mem_data, str):
                    # Already a string
                    project_memories[uuid] = mem_data
                elif isinstance(mem_data, dict):
                    # Concatenate all string values (the prose sections)
                    sections = []
                    for key, value in mem_data.items():
                        if isinstance(value, str) and key != "uuid":
                            sections.append(f"**{key}**\n\n{value}")
                    project_memories[uuid] = "\n\n".join(sections)

        # Check if we have any actual content
        if not conversations_memory and not project_memories:
            return None

        return ClaudeMemories(
            conversations_memory=conversations_memory,
            project_memories=project_memories,
            account_uuid=account_uuid
        )

    except (json.JSONDecodeError, KeyError, TypeError):
        # Malformed file - return None, caller can proceed without memories
        return None


def format_memories_for_distiller(memories: ClaudeMemories) -> str:
    """
    Convert ClaudeMemories to prose string for distiller context.

    Combines conversations_memory and project_memories into a single
    formatted string that the distiller can use as trusted context.

    Args:
        memories: Parsed ClaudeMemories object

    Returns:
        Formatted prose string
    """
    sections = []

    # Add conversations memory (the main user profile)
    if memories.conversations_memory:
        sections.append(memories.conversations_memory)

    # Add project memories if present
    if memories.project_memories:
        sections.append("\n---\n\n## Project-Specific Context\n")
        for uuid, prose in memories.project_memories.items():
            # Use a separator between projects
            sections.append(f"### Project {uuid[:8]}...\n\n{prose}")

    return "\n\n".join(sections)


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


def parse_chatgpt_conversations(raw: List[Dict[str, Any]]) -> List[Conversation]:
    """Parse all ChatGPT conversations."""
    conversations = []
    for convo_data in raw:
        mapping = convo_data.get("mapping", {})
        messages = flatten_tree(mapping)

        convo = Conversation(
            id=convo_data.get("id", ""),
            title=convo_data.get("title", "Untitled"),
            create_time=convo_data.get("create_time", 0.0) or 0.0,
            messages=messages
        )
        conversations.append(convo)
    return conversations


def parse_all(path: str) -> tuple[List[Conversation], Optional[UserProfile]]:
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
        raise ValueError(f"Unknown export type. Expected ChatGPT or Claude format.")

    return conversations, user_profile
