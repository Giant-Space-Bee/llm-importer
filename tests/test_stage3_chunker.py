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


class TestChunkConversationsWithFixtures:
    """Test chunking with fixture files."""

    def test_chunks_chatgpt_fixture(self, chatgpt_fixture):
        """Should chunk the ChatGPT fixture."""
        convos, _ = parse_all(str(chatgpt_fixture))

        chunks = chunk_conversations(convos, max_tokens=65536)

        # Small fixture should fit in one chunk
        assert len(chunks) == 1

        # Should cover all conversations
        total_convos = sum(len(c.conversations) for c in chunks)
        assert total_convos == 3

        # Each chunk should have token count set
        for chunk in chunks:
            assert chunk.token_count > 0

    def test_chunks_claude_fixture(self, claude_fixture):
        """Should chunk the Claude fixture."""
        convos, _ = parse_all(str(claude_fixture))

        chunks = chunk_conversations(convos, max_tokens=65536)

        # Small fixture should fit in one chunk
        assert len(chunks) == 1

        # Should cover all conversations
        total_convos = sum(len(c.conversations) for c in chunks)
        assert total_convos == 3

    def test_chunks_into_multiple_batches(self, chatgpt_fixture):
        """Should split conversations into multiple chunks with small token limit."""
        convos, _ = parse_all(str(chatgpt_fixture))

        # Use small limit to force multiple chunks
        chunks = chunk_conversations(convos, max_tokens=100)

        # Should produce multiple chunks
        assert len(chunks) > 1

        # Should cover all conversations
        total_convos = sum(len(c.conversations) for c in chunks)
        assert total_convos == 3

        # Each chunk has token count
        for chunk in chunks:
            assert chunk.token_count > 0

    def test_chunk_token_limits_respected(self, chatgpt_fixture):
        """Chunks should not exceed max_tokens (unless single oversized convo)."""
        convos, _ = parse_all(str(chatgpt_fixture))
        chunks = chunk_conversations(convos, max_tokens=100)

        for chunk in chunks:
            # Either under limit or single conversation
            assert chunk.token_count <= 100 or len(chunk.conversations) == 1


# ============================================================================
# Phase 2: Smart Splitting Tests
# ============================================================================

from src.chunker import (
    split_message_text,
    split_conversation,
    prepare_conversations,
    count_conversation_tokens,
)


class TestSplitMessageText:
    """split_message_text() splits giant message text at natural boundaries."""

    def test_no_split_for_small_text(self):
        """Text under limit should not be split."""
        text = "This is a short message."
        result = split_message_text(text, max_tokens=1000)
        assert result == [text]

    def test_splits_on_paragraph_breaks(self):
        """Should split on paragraph breaks (\\n\\n) first."""
        # Create longer paragraphs to ensure they exceed the token limit
        para1 = "This is a longer first paragraph with more content to ensure it has enough tokens. " * 5
        para2 = "This is a longer second paragraph with different content to ensure it has enough tokens. " * 5
        para3 = "This is a longer third paragraph with even more content to ensure it has enough tokens. " * 5
        text = f"{para1}\n\n{para2}\n\n{para3}"
        result = split_message_text(text, max_tokens=100)  # Force split at 100 tokens
        assert len(result) > 1
        # Each part should be complete paragraphs
        for part in result:
            assert not part.startswith("\n\n")
            assert not part.endswith("\n\n")

    def test_splits_on_sentence_breaks_if_no_paragraphs(self):
        """Should split on sentences if no paragraph breaks."""
        # Create longer sentences to ensure they exceed the token limit
        text = (
            "This is a first sentence with quite a lot of words to make it longer. " * 3 +
            "This is a second sentence with quite a lot of words to make it longer. " * 3 +
            "This is a third sentence with quite a lot of words to make it longer. " * 3 +
            "This is a fourth sentence with quite a lot of words to make it longer. " * 3
        )
        result = split_message_text(text, max_tokens=100)  # Force split at 100 tokens
        assert len(result) > 1

    def test_handles_empty_text(self):
        """Empty text should return empty list or single empty string."""
        result = split_message_text("", max_tokens=1000)
        assert result == [""] or result == []

    def test_merges_small_parts_together(self):
        """Small paragraphs should be merged to fit within limit."""
        text = "A.\n\nB.\n\nC.\n\nD."
        result = split_message_text(text, max_tokens=1000)  # Large limit
        assert len(result) == 1
        assert result[0] == text


class TestSplitConversation:
    """split_conversation() splits oversized conversations at message boundaries."""

    def test_no_split_for_small_conversation(self):
        """Conversation under limit should not be split."""
        convo = Conversation(
            id="small",
            title="Small Talk",
            create_time=1.0,
            messages=[
                Message(id="1", role="user", content="Hi", timestamp=1.0, is_hidden=False)
            ]
        )
        result = split_conversation(convo, max_tokens=10000)
        assert len(result) == 1
        assert result[0].id == "small"

    def test_splits_at_message_boundaries(self):
        """Should split between messages, not mid-message."""
        convo = Conversation(
            id="large",
            title="Long Discussion",
            create_time=1.0,
            messages=[
                Message(id="1", role="user", content="x " * 500, timestamp=1.0, is_hidden=False),
                Message(id="2", role="assistant", content="y " * 500, timestamp=2.0, is_hidden=False),
                Message(id="3", role="user", content="z " * 500, timestamp=3.0, is_hidden=False),
            ]
        )
        result = split_conversation(convo, max_tokens=500)
        assert len(result) > 1
        # Each part should have complete messages
        for part in result:
            for msg in part.messages:
                assert msg.content  # Not truncated

    def test_creates_sub_conversation_ids(self):
        """Split parts should keep original ID for verification lookup."""
        convo = Conversation(
            id="original-123",
            title="Original Title",
            create_time=1.0,
            messages=[
                Message(id="1", role="user", content="x " * 500, timestamp=1.0, is_hidden=False),
                Message(id="2", role="user", content="y " * 500, timestamp=2.0, is_hidden=False),
            ]
        )
        result = split_conversation(convo, max_tokens=500)
        if len(result) > 1:
            # All parts keep original ID so verifier can look up the conversation
            assert result[0].id == "original-123"
            assert result[1].id == "original-123"
            # Parts are numbered in the title for traceability
            assert "(part 1)" in result[0].title
            assert "(part 2)" in result[1].title

    def test_handles_single_giant_message(self):
        """Should handle conversation with single giant message."""
        convo = Conversation(
            id="giant",
            title="Giant Message",
            create_time=1.0,
            messages=[
                Message(
                    id="1",
                    role="user",
                    content="paragraph one.\n\nparagraph two.\n\nparagraph three.\n\nparagraph four.",
                    timestamp=1.0,
                    is_hidden=False
                )
            ]
        )
        result = split_conversation(convo, max_tokens=50)
        # Should split the message text
        assert len(result) >= 1


class TestPrepareConversations:
    """prepare_conversations() splits all oversized conversations."""

    def test_no_changes_for_small_conversations(self):
        """Small conversations should pass through unchanged."""
        convos = [
            Conversation(
                id="small-1",
                title="Small 1",
                create_time=1.0,
                messages=[Message(id="1", role="user", content="Hi", timestamp=1.0, is_hidden=False)]
            ),
            Conversation(
                id="small-2",
                title="Small 2",
                create_time=2.0,
                messages=[Message(id="2", role="user", content="Hello", timestamp=2.0, is_hidden=False)]
            ),
        ]
        result = prepare_conversations(convos, max_tokens=10000)
        assert len(result) == 2
        assert result[0].id == "small-1"
        assert result[1].id == "small-2"

    def test_splits_oversized_conversations(self):
        """Should split conversations that exceed limit."""
        convos = [
            Conversation(
                id="large",
                title="Large Convo",
                create_time=1.0,
                messages=[
                    Message(id="1", role="user", content="x " * 500, timestamp=1.0, is_hidden=False),
                    Message(id="2", role="user", content="y " * 500, timestamp=2.0, is_hidden=False),
                    Message(id="3", role="user", content="z " * 500, timestamp=3.0, is_hidden=False),
                ]
            )
        ]
        result = prepare_conversations(convos, max_tokens=500)
        assert len(result) > 1
        # All parts should trace back to original
        for conv in result:
            assert "large" in conv.id

    def test_preserves_small_while_splitting_large(self):
        """Should preserve small conversations while splitting large ones."""
        convos = [
            Conversation(
                id="small",
                title="Small",
                create_time=1.0,
                messages=[Message(id="1", role="user", content="Hi", timestamp=1.0, is_hidden=False)]
            ),
            Conversation(
                id="large",
                title="Large",
                create_time=2.0,
                messages=[
                    Message(id="2", role="user", content="x " * 500, timestamp=2.0, is_hidden=False),
                    Message(id="3", role="user", content="y " * 500, timestamp=3.0, is_hidden=False),
                ]
            ),
        ]
        result = prepare_conversations(convos, max_tokens=500)
        # Small should be unchanged
        assert result[0].id == "small"
        # Large should be split (parts come after small)
        assert len(result) > 2


class TestCountConversationTokens:
    """count_conversation_tokens() returns token count for formatted conversation."""

    def test_counts_formatted_tokens(self):
        """Should count tokens for the formatted representation."""
        convo = Conversation(
            id="test",
            title="Test",
            create_time=1.0,
            messages=[
                Message(id="1", role="user", content="Hello world", timestamp=1.0, is_hidden=False)
            ]
        )
        result = count_conversation_tokens(convo)
        assert result > 0
        # Should include overhead for metadata
        assert result > count_tokens("Hello world")
