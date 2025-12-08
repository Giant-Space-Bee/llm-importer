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

from src.parser import Conversation

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
        if msg.timestamp:
            # Format timestamp as YYYY-MM-DD
            from datetime import datetime
            date_str = datetime.fromtimestamp(msg.timestamp).strftime('%Y-%m-%d')
            lines.append(f"[{role_label} ({date_str})]: {msg.content}")
        else:
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


# ============================================================================
# Phase 2: Smart Splitting for Oversized Content
# ============================================================================

import re
import sys
from src.parsers.types import Message


def count_conversation_tokens(convo: Conversation) -> int:
    """Count tokens for a conversation when formatted."""
    return count_tokens(format_conversation(convo))


def _merge_parts_to_limit(
    parts: List[str],
    max_tokens: int,
    separator: str
) -> List[str]:
    """
    Merge parts back together up to token limit.

    Takes split parts (paragraphs or sentences) and re-combines them
    into chunks that fit within max_tokens.
    """
    result = []
    current: List[str] = []
    current_tokens = 0

    for part in parts:
        part_tokens = count_tokens(part)

        # If single part exceeds limit, it goes in its own chunk
        if part_tokens > max_tokens:
            if current:
                result.append(separator.join(current))
                current = []
                current_tokens = 0
            result.append(part)
            continue

        # Would adding this part exceed the limit?
        if current_tokens + part_tokens > max_tokens and current:
            result.append(separator.join(current))
            current = []
            current_tokens = 0

        current.append(part)
        current_tokens += part_tokens

    if current:
        result.append(separator.join(current))

    return result


def split_message_text(text: str, max_tokens: int) -> List[str]:
    """
    Split giant message text at natural boundaries.

    Priority:
    1. Paragraph breaks (\\n\\n) - best quality, natural topic boundaries
    2. Sentence breaks (. ! ?) - good quality, complete thoughts
    3. Hard token cut - last resort, with warning

    Args:
        text: The message text to split
        max_tokens: Maximum tokens per part

    Returns:
        List of text parts, each under max_tokens
    """
    if count_tokens(text) <= max_tokens:
        return [text]

    # Priority 1: Split on paragraph breaks
    paragraphs = text.split('\n\n')
    if len(paragraphs) > 1:
        merged = _merge_parts_to_limit(paragraphs, max_tokens, '\n\n')
        if all(count_tokens(p) <= max_tokens for p in merged):
            return merged

    # Priority 2: Split on sentence breaks
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) > 1:
        merged = _merge_parts_to_limit(sentences, max_tokens, ' ')
        if all(count_tokens(s) <= max_tokens for s in merged):
            return merged

    # Priority 3: Hard cut by tokens (last resort)
    print(
        f"[warning] Splitting message mid-text at {max_tokens} tokens",
        file=sys.stderr
    )
    result = []
    # Rough approximation: 4 chars per token
    chars_per_chunk = max_tokens * 4
    for i in range(0, len(text), chars_per_chunk):
        chunk = text[i:i + chars_per_chunk]
        # Trim to actual token limit if over
        while count_tokens(chunk) > max_tokens and len(chunk) > 100:
            chunk = chunk[:-100]  # Remove 100 chars at a time
        result.append(chunk)

    return result


def split_conversation(
    convo: Conversation,
    max_tokens: int
) -> List[Conversation]:
    """
    Split an oversized conversation into smaller parts.

    Strategy:
    1. Try splitting at message boundaries (preferred)
    2. If single message exceeds limit, split that message's text

    Args:
        convo: The conversation to split
        max_tokens: Maximum tokens per resulting conversation

    Returns:
        List of Conversation objects, each under max_tokens
    """
    # Check if splitting is needed
    total_tokens = count_conversation_tokens(convo)
    if total_tokens <= max_tokens:
        return [convo]

    # Overhead for conversation metadata (title, id, etc.)
    # Estimate ~50 tokens for header
    header_overhead = 50
    effective_max = max_tokens - header_overhead

    result = []
    current_messages: List[Message] = []
    current_tokens = 0
    part_num = 1

    for msg in convo.messages:
        msg_tokens = count_tokens(msg.content)

        # If single message exceeds limit, split its text
        if msg_tokens > effective_max:
            # First, save current batch if any
            if current_messages:
                result.append(_create_sub_conversation(
                    convo, current_messages, part_num
                ))
                part_num += 1
                current_messages = []
                current_tokens = 0

            # Split the giant message
            text_parts = split_message_text(msg.content, effective_max)
            for i, text_part in enumerate(text_parts):
                sub_msg = Message(
                    id=f"{msg.id}_part{i+1}",
                    role=msg.role,
                    content=text_part,
                    timestamp=msg.timestamp,
                    is_hidden=msg.is_hidden
                )
                result.append(_create_sub_conversation(
                    convo, [sub_msg], part_num
                ))
                part_num += 1
            continue

        # Would adding this message exceed the limit?
        if current_tokens + msg_tokens > effective_max and current_messages:
            result.append(_create_sub_conversation(
                convo, current_messages, part_num
            ))
            part_num += 1
            current_messages = []
            current_tokens = 0

        current_messages.append(msg)
        current_tokens += msg_tokens

    # Don't forget the last batch
    if current_messages:
        result.append(_create_sub_conversation(
            convo, current_messages, part_num
        ))

    return result


def _create_sub_conversation(
    original: Conversation,
    messages: List[Message],
    part_num: int
) -> Conversation:
    """Create a sub-conversation preserving original ID for verification.

    IMPORTANT: We keep the original ID so the verifier can look up the
    full conversation and find the source_quote. Split parts are just
    for chunking - verification needs the original conversation.
    """
    return Conversation(
        id=original.id,  # Keep original ID for verification lookup
        title=f"{original.title} (part {part_num})",
        create_time=original.create_time,
        messages=messages
    )


def prepare_conversations(
    conversations: List[Conversation],
    max_tokens: int
) -> List[Conversation]:
    """
    Split oversized conversations, return all ready for chunking.

    This is the main entry point for Phase 2 splitting.
    Called before chunk_conversations() to ensure no single
    conversation exceeds the chunk limit.

    Args:
        conversations: List of conversations to prepare
        max_tokens: Maximum tokens per conversation

    Returns:
        List of conversations, all under max_tokens
    """
    prepared = []
    oversized_count = 0

    for convo in conversations:
        convo_tokens = count_conversation_tokens(convo)

        if convo_tokens > max_tokens:
            oversized_count += 1
            parts = split_conversation(convo, max_tokens)
            prepared.extend(parts)
            print(
                f"[split] Conversation '{convo.title[:30]}...' "
                f"({convo_tokens:,} tokens) → {len(parts)} parts",
                file=sys.stderr
            )
        else:
            prepared.append(convo)

    if oversized_count > 0:
        print(
            f"[split] {oversized_count} oversized conversations split",
            file=sys.stderr
        )

    return prepared
