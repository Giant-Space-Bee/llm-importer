"""
Stage 3 Tests: Chunker

What Stage 3 does:
- Count tokens using tiktoken
- Batch conversations into chunks of max_tokens (default 65536 = 2^16)
- Keep conversations intact when possible
- Split huge conversations at message boundaries

Run: pytest tests/test_stage3_chunker.py -v
"""

import pytest

# These imports WILL FAIL until we implement - that's TDD!
from src.chunker import (
    count_tokens,
    chunk_conversations,
    format_conversation,
    Chunk,
    DEFAULT_CHUNK_SIZE,
)
from src.parser import Conversation, Message, load_conversations, parse_all


class TestCountTokens:
    """count_tokens() uses tiktoken for accurate counts."""

    def test_counts_simple_text(self):
        """Should return reasonable token count for simple text."""
        result = count_tokens("Hello, world!")
        # "Hello, world!" is about 4 tokens
        assert result >= 3
        assert result <= 6

    def test_counts_longer_text(self):
        """Token count should scale with text length."""
        short = count_tokens("Hi")
        long = count_tokens("This is a much longer piece of text that should have more tokens.")
        assert long > short

    def test_empty_string(self):
        """Empty string should return 0 tokens."""
        result = count_tokens("")
        assert result == 0

    def test_unicode(self):
        """Should handle unicode characters."""
        result = count_tokens("Hello! How are you?")
        assert result > 0


class TestDefaultChunkSize:
    """Default chunk size should be 2^16 = 65536."""

    def test_default_is_power_of_two(self):
        """Landon likes powers of 2."""
        assert DEFAULT_CHUNK_SIZE == 65536
        assert DEFAULT_CHUNK_SIZE == 2 ** 16


class TestFormatConversation:
    """format_conversation() renders a conversation for LLM input."""

    def test_includes_metadata(self):
        """Should include conversation ID and title."""
        convo = Conversation(
            id="test-123",
            title="Test Conversation",
            create_time=1700000000.0,
            messages=[]
        )
        result = format_conversation(convo)
        assert "test-123" in result
        assert "Test Conversation" in result

    def test_includes_messages(self):
        """Should include message content with role labels."""
        convo = Conversation(
            id="test",
            title="Test",
            create_time=1700000000.0,
            messages=[
                Message(id="1", role="user", content="Hello", timestamp=1.0, is_hidden=False),
                Message(id="2", role="assistant", content="Hi there", timestamp=2.0, is_hidden=False),
            ]
        )
        result = format_conversation(convo)
        assert "Hello" in result
        assert "Hi there" in result
        assert "user" in result.lower() or "User" in result


class TestChunk:
    """Chunk dataclass structure."""

    def test_has_required_fields(self):
        """Chunk must have id, conversations, token_count."""
        chunk = Chunk(
            id=0,
            conversations=[],
            token_count=1000
        )
        assert chunk.id == 0
        assert chunk.conversations == []
        assert chunk.token_count == 1000


class TestChunkConversations:
    """chunk_conversations() batches conversations into token-limited chunks."""

    def test_single_small_conversation(self):
        """One small conversation should produce one chunk."""
        convos = [Conversation(
            id="small",
            title="Small Talk",
            create_time=1.0,
            messages=[Message(id="1", role="user", content="Hi", timestamp=1.0, is_hidden=False)]
        )]

        chunks = chunk_conversations(convos, max_tokens=65536)

        assert len(chunks) == 1
        assert len(chunks[0].conversations) == 1

    def test_batches_small_conversations(self):
        """Multiple small conversations should batch together."""
        convos = [
            Conversation(
                id=f"conv-{i}",
                title=f"Conversation {i}",
                create_time=float(i),
                messages=[Message(id="1", role="user", content="Hello", timestamp=1.0, is_hidden=False)]
            )
            for i in range(10)
        ]

        chunks = chunk_conversations(convos, max_tokens=65536)

        # 10 tiny conversations should fit in 1 chunk
        assert len(chunks) == 1
        assert len(chunks[0].conversations) == 10

    def test_respects_max_tokens(self):
        """Should not exceed max_tokens per chunk."""
        # Create conversations with substantial content
        convos = [
            Conversation(
                id=f"conv-{i}",
                title=f"Conversation {i}",
                create_time=float(i),
                messages=[Message(id="1", role="user", content="x " * 500, timestamp=1.0, is_hidden=False)]
            )
            for i in range(100)
        ]

        chunks = chunk_conversations(convos, max_tokens=1000)

        for chunk in chunks:
            assert chunk.token_count <= 1000 or len(chunk.conversations) == 1

    def test_keeps_conversations_intact(self):
        """Should not split a conversation across chunks unless it's huge."""
        convos = [
            Conversation(
                id="conv-1",
                title="First",
                create_time=1.0,
                messages=[Message(id="1", role="user", content="Hello " * 100, timestamp=1.0, is_hidden=False)]
            ),
            Conversation(
                id="conv-2",
                title="Second",
                create_time=2.0,
                messages=[Message(id="2", role="user", content="World " * 100, timestamp=2.0, is_hidden=False)]
            )
        ]

        chunks = chunk_conversations(convos, max_tokens=50000)

        # Both should be in one chunk (they're small)
        all_convo_ids = []
        for chunk in chunks:
            for c in chunk.conversations:
                all_convo_ids.append(c.id)

        assert "conv-1" in all_convo_ids
        assert "conv-2" in all_convo_ids

    def test_returns_chunk_ids_in_order(self):
        """Chunks should have sequential IDs starting at 0."""
        convos = [
            Conversation(
                id=f"conv-{i}",
                title=f"Conv {i}",
                create_time=float(i),
                messages=[Message(id="1", role="user", content="x " * 200, timestamp=1.0, is_hidden=False)]
            )
            for i in range(50)
        ]

        chunks = chunk_conversations(convos, max_tokens=500)

        for i, chunk in enumerate(chunks):
            assert chunk.id == i


class TestChunkConversationsWithRealData:
    """Test chunking with the real conversations.json file."""

    def test_chunks_real_data(self, conversations_file):
        """Should chunk the real 450 conversations into reasonable batches."""
        convos, _ = parse_all(str(conversations_file))

        chunks = chunk_conversations(convos, max_tokens=65536)

        # Should produce multiple chunks (data is ~1.6M tokens)
        assert len(chunks) > 1

        # Should cover all conversations
        total_convos = sum(len(c.conversations) for c in chunks)
        assert total_convos == 450

        # Each chunk should have token count set
        for chunk in chunks:
            assert chunk.token_count > 0

    def test_chunk_sizes_are_reasonable(self, conversations_file):
        """Most chunks should be close to max_tokens (efficient packing)."""
        convos, _ = parse_all(str(conversations_file))

        chunks = chunk_conversations(convos, max_tokens=65536)

        # At least half the chunks should be at least 50% full
        # (unless they contain a single huge conversation)
        well_packed = sum(
            1 for c in chunks
            if c.token_count >= 32768 or len(c.conversations) == 1
        )
        assert well_packed >= len(chunks) // 2
