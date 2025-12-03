"""
chunker.py - Smart batching for parallel processing

Stage 3:
- Count tokens using tiktoken
- Batch conversations into chunks of max_tokens (default 65536 = 2^16)
- Keep conversations intact when possible
- Split huge conversations at message boundaries
"""

from typing import List
from dataclasses import dataclass

import tiktoken

from src.parser import Conversation, Message

# Default chunk size: 2^16 = 65536 (Landon likes powers of 2)
DEFAULT_CHUNK_SIZE = 65536

# Use cl100k_base encoding (GPT-4, Claude-compatible)
_encoding = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    """A batch of conversations ready for LLM processing."""
    id: int
    conversations: List[Conversation]
    token_count: int


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken."""
    if not text:
        return 0
    # Disable special token check - real data may contain special token strings
    return len(_encoding.encode(text, disallowed_special=()))


def format_conversation(convo: Conversation) -> str:
    """
    Format a conversation for LLM input.

    Includes conversation metadata and all messages.
    """
    lines = [
        f"=== Conversation: {convo.title} ===",
        f"ID: {convo.id}",
        f"Created: {convo.create_time}",
        ""
    ]

    for msg in convo.messages:
        role_label = msg.role.capitalize()
        lines.append(f"[{role_label}]: {msg.content}")
        lines.append("")

    return "\n".join(lines)


def chunk_conversations(
    conversations: List[Conversation],
    max_tokens: int = DEFAULT_CHUNK_SIZE
) -> List[Chunk]:
    """
    Batch conversations into chunks of max_tokens.

    Rules:
    1. Keep conversations intact when possible
    2. Greedy bin-packing: add convos until chunk is full
    3. If single convo > max_tokens: include it in its own chunk (don't split for now)
    """
    chunks = []
    current_conversations = []
    current_tokens = 0

    for convo in conversations:
        formatted = format_conversation(convo)
        convo_tokens = count_tokens(formatted)

        # If this single conversation exceeds max, put it in its own chunk
        if convo_tokens > max_tokens:
            # First, save current chunk if it has anything
            if current_conversations:
                chunks.append(Chunk(
                    id=len(chunks),
                    conversations=current_conversations,
                    token_count=current_tokens
                ))
                current_conversations = []
                current_tokens = 0

            # Add oversized conversation as its own chunk
            chunks.append(Chunk(
                id=len(chunks),
                conversations=[convo],
                token_count=convo_tokens
            ))
            continue

        # Would adding this conversation exceed the limit?
        if current_tokens + convo_tokens > max_tokens:
            # Save current chunk and start new one
            if current_conversations:
                chunks.append(Chunk(
                    id=len(chunks),
                    conversations=current_conversations,
                    token_count=current_tokens
                ))
            current_conversations = [convo]
            current_tokens = convo_tokens
        else:
            # Add to current chunk
            current_conversations.append(convo)
            current_tokens += convo_tokens

    # Don't forget the last chunk
    if current_conversations:
        chunks.append(Chunk(
            id=len(chunks),
            conversations=current_conversations,
            token_count=current_tokens
        ))

    return chunks


def format_chunk_for_extraction(chunk: Chunk) -> str:
    """Format an entire chunk for LLM extraction prompt."""
    parts = []
    for convo in chunk.conversations:
        parts.append(format_conversation(convo))
    return "\n\n".join(parts)
