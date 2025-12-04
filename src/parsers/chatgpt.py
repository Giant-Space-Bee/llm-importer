"""
parsers/chatgpt.py - ChatGPT export parser

Handles ChatGPT conversations.json format:
- Tree structure with 'mapping' field
- Supports branching (edits/regenerations)
- user_editable_context for custom instructions
"""

from typing import List, Dict, Any, Optional

from src.parsers.types import Message, Conversation, UserProfile, ConversationStats


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
            message = parse_message(msg_data)
            if message is not None:
                messages.append(message)

        # Follow first child
        children = node.get("children", [])
        current_id = children[0] if children else None

    return messages


def parse_message(msg_data: Dict[str, Any]) -> Optional[Message]:
    """Parse a ChatGPT message dict into a Message object."""
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
    """Calculate aggregate statistics about ChatGPT conversations."""
    total_messages = 0
    user_messages = 0
    total_chars = 0

    for convo in conversations:
        mapping = convo.get("mapping", {})
        messages = flatten_tree(mapping)
        total_messages += len(messages)

        user_msgs = [m for m in messages if m.role == "user" and not m.is_hidden]
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
