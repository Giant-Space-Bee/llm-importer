"""
chunker.py - Smart batching for parallel processing

Keep conversations intact, batch to 65536 tokens (configurable 2^n).
Split huge convos at message boundaries.
"""

from typing import List
from dataclasses import dataclass

# from .parser import Conversation
# from .core import count_tokens


@dataclass
class Chunk:
    """A batch of conversations ready for LLM processing."""
    id: int
    conversations: List  # List[Conversation]
    token_count: int


def chunk_conversations(
    conversations: List,  # List[Conversation]
    max_tokens: int = 65536  # 2^16 default
) -> List[Chunk]:
    """
    Batch conversations into chunks of max_tokens.

    Rules:
    1. Keep conversations intact when possible
    2. Greedy bin-packing: add convos until chunk is full
    3. If single convo > max_tokens: split at message boundaries
    """
    # TODO:
    # chunks = []
    # current_chunk = []
    # current_tokens = 0
    #
    # for convo in conversations:
    #     convo_tokens = count_tokens(format_convo(convo))
    #
    #     if convo_tokens > max_tokens:
    #         # Split huge convo
    #         split_chunks = split_conversation(convo, max_tokens)
    #         chunks.extend(split_chunks)
    #     elif current_tokens + convo_tokens > max_tokens:
    #         # Start new chunk
    #         chunks.append(Chunk(len(chunks), current_chunk, current_tokens))
    #         current_chunk = [convo]
    #         current_tokens = convo_tokens
    #     else:
    #         # Add to current chunk
    #         current_chunk.append(convo)
    #         current_tokens += convo_tokens
    #
    # # Don't forget last chunk
    # if current_chunk:
    #     chunks.append(Chunk(len(chunks), current_chunk, current_tokens))
    #
    # return chunks
    pass


def split_conversation(conversation, max_tokens: int) -> List[Chunk]:
    """
    Split a single huge conversation at message boundaries.

    Used when one conversation exceeds max_tokens.
    """
    # TODO: Split messages into groups that fit in max_tokens
    # Each group becomes its own Chunk
    pass


def format_for_extraction(chunk: Chunk) -> str:
    """Format chunk as text for LLM extraction prompt."""
    # TODO: Render conversations as readable text
    # Include convo ID, title, timestamps, messages
    pass
