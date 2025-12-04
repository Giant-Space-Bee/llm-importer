"""
Stage 2 Tests: Parser

What Stage 2 does:
- Load conversations.json (55MB)
- Count conversations, total messages, user messages
- Flatten tree structure to linear messages
- Extract user_editable_context (ChatGPT custom instructions - free wins!)
- Filter to user messages only

Run: pytest tests/test_stage2_parser.py -v
"""

import pytest
import json
from pathlib import Path

# These imports WILL FAIL until we implement - that's TDD!
from src.parser import (
    load_conversations,
    flatten_tree,
    filter_user_messages,
    extract_user_profile,
    get_conversation_stats,
    Message,
    Conversation,
    UserProfile,
    ConversationStats,
)


class TestLoadConversations:
    """load_conversations() reads the JSON file."""

    def test_returns_list(self, sample_json_file):
        """Should return a list of conversation dicts."""
        result = load_conversations(str(sample_json_file))
        assert isinstance(result, list)

    def test_loads_sample_file(self, sample_json_file):
        """Should parse our sample JSON."""
        result = load_conversations(str(sample_json_file))
        assert len(result) == 1
        assert result[0]["id"] == "test"

    def test_loads_chatgpt_fixture(self, chatgpt_fixture):
        """Should load the ChatGPT fixture file."""
        result = load_conversations(str(chatgpt_fixture))
        assert len(result) == 3  # Our fixture has 3 conversations

    def test_loads_claude_fixture(self, claude_fixture):
        """Should load the Claude fixture file."""
        result = load_conversations(str(claude_fixture))
        assert len(result) == 3  # Our fixture has 3 conversations

    def test_raises_on_missing_file(self):
        """Should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            load_conversations("/nonexistent/file.json")


class TestFlattenTree:
    """flatten_tree() converts tree mapping to linear message list."""

    def test_simple_tree(self):
        """Should flatten a simple parent->child tree."""
        mapping = {
            "root": {"id": "root", "message": None, "parent": None, "children": ["msg1"]},
            "msg1": {
                "id": "msg1",
                "message": {
                    "id": "msg1",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["Hello"]},
                    "create_time": 1700000000.0,
                    "metadata": {}
                },
                "parent": "root",
                "children": ["msg2"]
            },
            "msg2": {
                "id": "msg2",
                "message": {
                    "id": "msg2",
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["Hi there!"]},
                    "create_time": 1700000001.0,
                    "metadata": {}
                },
                "parent": "msg1",
                "children": []
            }
        }

        messages = flatten_tree(mapping)

        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Hi there!"

    def test_skips_null_messages(self):
        """Root nodes have message=None, should be skipped."""
        mapping = {
            "root": {"id": "root", "message": None, "parent": None, "children": ["msg1"]},
            "msg1": {
                "id": "msg1",
                "message": {
                    "id": "msg1",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["Test"]},
                    "create_time": 1700000000.0,
                    "metadata": {}
                },
                "parent": "root",
                "children": []
            }
        }

        messages = flatten_tree(mapping)
        assert len(messages) == 1

    def test_follows_first_child_on_branch(self):
        """When conversation branches, follow children[0] for main path."""
        mapping = {
            "root": {"id": "root", "message": None, "parent": None, "children": ["msg1"]},
            "msg1": {
                "id": "msg1",
                "message": {
                    "id": "msg1",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["Question"]},
                    "create_time": 1700000000.0,
                    "metadata": {}
                },
                "parent": "root",
                "children": ["branch1", "branch2"]  # Two branches!
            },
            "branch1": {
                "id": "branch1",
                "message": {
                    "id": "branch1",
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["First answer"]},
                    "create_time": 1700000001.0,
                    "metadata": {}
                },
                "parent": "msg1",
                "children": []
            },
            "branch2": {
                "id": "branch2",
                "message": {
                    "id": "branch2",
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["Regenerated answer"]},
                    "create_time": 1700000002.0,
                    "metadata": {}
                },
                "parent": "msg1",
                "children": []
            }
        }

        messages = flatten_tree(mapping)

        # Should only have 2 messages (user + first branch)
        assert len(messages) == 2
        assert messages[1].content == "First answer"


class TestFilterUserMessages:
    """filter_user_messages() keeps only non-hidden user messages."""

    def test_filters_to_user_only(self):
        """Should keep only role=user messages."""
        messages = [
            Message(id="1", role="user", content="Hi", timestamp=1.0, is_hidden=False),
            Message(id="2", role="assistant", content="Hello", timestamp=2.0, is_hidden=False),
            Message(id="3", role="user", content="Thanks", timestamp=3.0, is_hidden=False),
        ]

        result = filter_user_messages(messages)

        assert len(result) == 2
        assert all(m.role == "user" for m in result)

    def test_excludes_hidden_messages(self):
        """Should exclude is_hidden=True messages (like user_editable_context)."""
        messages = [
            Message(id="1", role="user", content="Context", timestamp=1.0, is_hidden=True),
            Message(id="2", role="user", content="Real message", timestamp=2.0, is_hidden=False),
        ]

        result = filter_user_messages(messages)

        assert len(result) == 1
        assert result[0].content == "Real message"


class TestExtractUserProfile:
    """extract_user_profile() finds user_editable_context blocks."""

    def test_extracts_profile(self):
        """Should extract user_profile and user_instructions."""
        conversations = [{
            "id": "test",
            "mapping": {
                "root": {"id": "root", "message": None, "parent": None, "children": ["ctx"]},
                "ctx": {
                    "id": "ctx",
                    "message": {
                        "id": "ctx",
                        "author": {"role": "user"},
                        "content": {
                            "content_type": "user_editable_context",
                            "user_profile": "Name: Landon\nRole: AI Architect",
                            "user_instructions": "Be concise"
                        },
                        "create_time": None,
                        "metadata": {"is_visually_hidden_from_conversation": True}
                    },
                    "parent": "root",
                    "children": []
                }
            }
        }]

        profile = extract_user_profile(conversations)

        assert profile is not None
        assert "Landon" in profile.user_profile
        assert "concise" in profile.user_instructions

    def test_returns_none_if_not_found(self):
        """Should return None if no user_editable_context exists."""
        conversations = [{
            "id": "test",
            "mapping": {
                "root": {"id": "root", "message": None, "parent": None, "children": []}
            }
        }]

        profile = extract_user_profile(conversations)
        assert profile is None


class TestConversationStats:
    """get_conversation_stats() returns aggregate statistics."""

    def test_counts_conversations_fixture(self, chatgpt_fixture):
        """Should count total conversations from fixture."""
        convos = load_conversations(str(chatgpt_fixture))
        stats = get_conversation_stats(convos)

        assert stats.total_conversations == 3

    def test_counts_messages_fixture(self, chatgpt_fixture):
        """Should count total and user messages from fixture."""
        convos = load_conversations(str(chatgpt_fixture))
        stats = get_conversation_stats(convos)

        assert stats.total_messages > 0
        assert stats.user_messages > 0
        assert stats.user_messages <= stats.total_messages
        assert stats.total_chars > 0  # Also verify char counting works

    def test_has_user_profile_flag_fixture(self, chatgpt_fixture):
        """Should indicate if user_editable_context was found in fixture."""
        convos = load_conversations(str(chatgpt_fixture))
        stats = get_conversation_stats(convos)

        # Our fixture has user_editable_context
        assert stats.has_user_profile is True


class TestMessage:
    """Message dataclass structure."""

    def test_has_required_fields(self):
        """Message must have id, role, content, timestamp, is_hidden."""
        msg = Message(
            id="test-id",
            role="user",
            content="Hello world",
            timestamp=1700000000.0,
            is_hidden=False
        )

        assert msg.id == "test-id"
        assert msg.role == "user"
        assert msg.content == "Hello world"
        assert msg.timestamp == 1700000000.0
        assert msg.is_hidden is False


class TestConversation:
    """Conversation dataclass structure."""

    def test_has_required_fields(self):
        """Conversation must have id, title, create_time, messages."""
        convo = Conversation(
            id="convo-id",
            title="Test Conversation",
            create_time=1700000000.0,
            messages=[]
        )

        assert convo.id == "convo-id"
        assert convo.title == "Test Conversation"
        assert convo.create_time == 1700000000.0
        assert convo.messages == []


# --- Stage 6i: Claude export parsing tests ---

from src.parser import (
    detect_export_type,
    parse_all,
    parse_claude_conversations,
    parse_claude_conversation,
    _parse_claude_message,
    _parse_iso_timestamp,
)


class TestDetectExportType:
    """detect_export_type() identifies ChatGPT vs Claude exports."""

    def test_detects_chatgpt(self):
        """Should detect ChatGPT export by 'mapping' field."""
        data = [{"id": "test", "mapping": {"root": {}}}]
        assert detect_export_type(data) == "chatgpt"

    def test_detects_claude(self):
        """Should detect Claude export by 'uuid' and 'chat_messages' fields."""
        data = [{"uuid": "test", "chat_messages": []}]
        assert detect_export_type(data) == "claude"

    def test_returns_unknown_for_empty(self):
        """Should return 'unknown' for empty list."""
        assert detect_export_type([]) == "unknown"

    def test_returns_unknown_for_unrecognized(self):
        """Should return 'unknown' for unrecognized format."""
        data = [{"some_field": "value"}]
        assert detect_export_type(data) == "unknown"


class TestParseIsoTimestamp:
    """_parse_iso_timestamp() converts ISO strings to Unix timestamps."""

    def test_parses_iso_with_z(self):
        """Should handle ISO format with Z suffix."""
        ts = _parse_iso_timestamp("2025-12-01T06:01:43.108834Z")
        assert ts > 0

    def test_parses_iso_with_timezone(self):
        """Should handle ISO format with timezone offset."""
        ts = _parse_iso_timestamp("2025-12-01T06:01:43.108834+00:00")
        assert ts > 0

    def test_returns_zero_for_invalid(self):
        """Should return 0 for invalid strings."""
        assert _parse_iso_timestamp("not a date") == 0.0
        assert _parse_iso_timestamp("") == 0.0


class TestParseClaudeMessage:
    """_parse_claude_message() converts Claude message dicts to Message objects."""

    def test_parses_human_message(self):
        """Should parse human message with content array."""
        msg_data = {
            "uuid": "msg-1",
            "sender": "human",
            "content": [{"type": "text", "text": "Hello Claude"}],
            "created_at": "2025-12-01T06:01:44.304455Z"
        }
        msg = _parse_claude_message(msg_data)

        assert msg is not None
        assert msg.id == "msg-1"
        assert msg.role == "user"  # human -> user
        assert msg.content == "Hello Claude"
        assert msg.timestamp > 0

    def test_parses_assistant_message(self):
        """Should parse assistant message."""
        msg_data = {
            "uuid": "msg-2",
            "sender": "assistant",
            "content": [{"type": "text", "text": "Hello human"}],
            "created_at": "2025-12-01T06:02:00Z"
        }
        msg = _parse_claude_message(msg_data)

        assert msg.role == "assistant"

    def test_uses_top_level_text_fallback(self):
        """Should use top-level text field if content array is empty."""
        msg_data = {
            "uuid": "msg-3",
            "sender": "human",
            "text": "Fallback text",
            "content": [],
            "created_at": "2025-12-01T06:00:00Z"
        }
        msg = _parse_claude_message(msg_data)

        assert msg.content == "Fallback text"


class TestParseClaudeConversation:
    """parse_claude_conversation() converts Claude conversation dicts to Conversation objects."""

    def test_parses_conversation(self):
        """Should parse a complete Claude conversation."""
        convo_data = {
            "uuid": "convo-1",
            "name": "Test Conversation",
            "created_at": "2025-12-01T06:00:00Z",
            "chat_messages": [
                {"uuid": "msg-1", "sender": "human", "content": [{"text": "Hi"}], "created_at": "2025-12-01T06:00:01Z"},
                {"uuid": "msg-2", "sender": "assistant", "content": [{"text": "Hello"}], "created_at": "2025-12-01T06:00:02Z"},
            ]
        }
        convo = parse_claude_conversation(convo_data)

        assert convo.id == "convo-1"
        assert convo.title == "Test Conversation"
        assert len(convo.messages) == 2
        assert convo.messages[0].role == "user"
        assert convo.messages[1].role == "assistant"

    def test_handles_empty_name(self):
        """Should use 'Untitled' for empty name."""
        convo_data = {
            "uuid": "convo-1",
            "name": "",
            "created_at": "2025-12-01T06:00:00Z",
            "chat_messages": []
        }
        convo = parse_claude_conversation(convo_data)

        assert convo.title == "Untitled"


class TestParseAllUnified:
    """parse_all() auto-detects format and parses appropriately."""

    def test_parses_chatgpt_fixture(self, chatgpt_fixture):
        """Should parse ChatGPT fixture with user profile."""
        convos, profile = parse_all(str(chatgpt_fixture))

        assert len(convos) == 3
        assert profile is not None
        assert "Test User" in profile.user_profile

    def test_parses_claude_fixture(self, claude_fixture):
        """Should parse Claude fixture."""
        convos, profile = parse_all(str(claude_fixture))

        assert len(convos) == 3
        assert profile is None  # Claude format doesn't have user_editable_context

    def test_raises_for_unknown_format(self, tmp_path):
        """Should raise ValueError for unknown format."""
        # Create file with unrecognized format
        unknown_file = tmp_path / "unknown.json"
        unknown_file.write_text('[{"weird_field": "value"}]')

        with pytest.raises(ValueError, match="Unknown export type"):
            parse_all(str(unknown_file))
