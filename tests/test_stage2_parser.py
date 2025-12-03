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

    def test_loads_real_conversations(self, conversations_file):
        """Should load the actual 55MB file."""
        result = load_conversations(str(conversations_file))
        # We know from analysis: 450 conversations
        assert len(result) >= 400
        assert len(result) <= 500

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

    def test_counts_conversations(self, conversations_file):
        """Should count total conversations."""
        convos = load_conversations(str(conversations_file))
        stats = get_conversation_stats(convos)

        assert stats.total_conversations >= 400
        assert stats.total_conversations <= 500

    def test_counts_messages(self, conversations_file):
        """Should count total and user messages."""
        convos = load_conversations(str(conversations_file))
        stats = get_conversation_stats(convos)

        # Following main path (children[0]): ~7.5k messages, ~1.7k user messages
        assert stats.total_messages > 5000
        assert stats.user_messages > 1000
        assert stats.user_messages < stats.total_messages

    def test_calculates_total_chars(self, conversations_file):
        """Should calculate total character count."""
        convos = load_conversations(str(conversations_file))
        stats = get_conversation_stats(convos)

        # Following main path: ~6.4M characters
        assert stats.total_chars > 5_000_000

    def test_has_user_profile_flag(self, conversations_file):
        """Should indicate if user_editable_context was found."""
        convos = load_conversations(str(conversations_file))
        stats = get_conversation_stats(convos)

        # We know Landon's export has user_editable_context
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
