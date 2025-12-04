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


# --- Claude memories.json tests ---

from src.parser import (
    load_claude_memories,
    format_memories_for_distiller,
    ClaudeMemories,
)


@pytest.fixture
def claude_memories_fixture():
    """Path to the Claude memories test fixture."""
    return Path(__file__).parent / "fixtures" / "claude_memories.json"


@pytest.fixture
def claude_export_folder(tmp_path, claude_fixture, claude_memories_fixture):
    """Create a mock Claude export folder with conversations and memories."""
    # Copy fixtures to temp folder
    import shutil
    folder = tmp_path / "data-2025-12-03-test"
    folder.mkdir()
    shutil.copy(claude_fixture, folder / "conversations.json")
    shutil.copy(claude_memories_fixture, folder / "memories.json")
    return folder


class TestLoadClaudeMemories:
    """Tests for load_claude_memories() function."""

    def test_loads_valid_memories(self, claude_export_folder):
        """Should load and parse valid memories.json."""
        memories = load_claude_memories(str(claude_export_folder))

        assert memories is not None
        assert isinstance(memories, ClaudeMemories)
        assert "Test User" in memories.conversations_memory
        assert "software developer" in memories.conversations_memory
        assert len(memories.project_memories) == 2
        assert memories.account_uuid == "test-account-uuid-12345"

    def test_parses_project_memories(self, claude_export_folder):
        """Should parse project memories with all fields."""
        memories = load_claude_memories(str(claude_export_folder))

        assert memories is not None
        # Check first project
        project1 = memories.project_memories.get("test-project-uuid-001")
        assert project1 is not None
        assert "document processing" in project1
        assert "Purpose & context" in project1

    def test_returns_none_for_missing_file(self, tmp_path):
        """Should return None if memories.json doesn't exist."""
        empty_folder = tmp_path / "empty"
        empty_folder.mkdir()

        result = load_claude_memories(str(empty_folder))
        assert result is None

    def test_returns_none_for_malformed_json(self, tmp_path):
        """Should return None for invalid JSON."""
        folder = tmp_path / "bad"
        folder.mkdir()
        (folder / "memories.json").write_text("not valid json")

        result = load_claude_memories(str(folder))
        assert result is None

    def test_returns_none_for_empty_array(self, tmp_path):
        """Should return None for empty array."""
        folder = tmp_path / "empty_array"
        folder.mkdir()
        (folder / "memories.json").write_text("[]")

        result = load_claude_memories(str(folder))
        assert result is None

    def test_returns_none_for_empty_content(self, tmp_path):
        """Should return None if both memory fields are empty."""
        folder = tmp_path / "empty_content"
        folder.mkdir()
        (folder / "memories.json").write_text(
            '[{"conversations_memory": "", "project_memories": {}, "account_uuid": "test"}]'
        )

        result = load_claude_memories(str(folder))
        assert result is None

    def test_handles_conversations_only(self, tmp_path):
        """Should work with only conversations_memory (no project memories)."""
        folder = tmp_path / "convos_only"
        folder.mkdir()
        (folder / "memories.json").write_text(
            '[{"conversations_memory": "Some memories here", "project_memories": {}, "account_uuid": "test"}]'
        )

        result = load_claude_memories(str(folder))
        assert result is not None
        assert result.conversations_memory == "Some memories here"
        assert len(result.project_memories) == 0


class TestFormatMemoriesForDistiller:
    """Tests for format_memories_for_distiller() function."""

    def test_formats_full_memories(self, claude_export_folder):
        """Should format both conversations and project memories."""
        memories = load_claude_memories(str(claude_export_folder))
        assert memories is not None

        formatted = format_memories_for_distiller(memories)

        # Should contain conversations memory
        assert "Test User" in formatted
        assert "software developer" in formatted

        # Should contain project memories section
        assert "Project-Specific Context" in formatted
        assert "document processing" in formatted

    def test_formats_conversations_only(self):
        """Should work with only conversations memory."""
        memories = ClaudeMemories(
            conversations_memory="User is a developer.",
            project_memories={},
            account_uuid="test"
        )

        formatted = format_memories_for_distiller(memories)

        assert "User is a developer" in formatted
        assert "Project-Specific Context" not in formatted

    def test_formats_with_multiple_projects(self):
        """Should include all project memories."""
        memories = ClaudeMemories(
            conversations_memory="Main memory",
            project_memories={
                "proj-1": "Project 1 content",
                "proj-2": "Project 2 content",
            },
            account_uuid="test"
        )

        formatted = format_memories_for_distiller(memories)

        assert "Main memory" in formatted
        assert "Project 1 content" in formatted
        assert "Project 2 content" in formatted


# --- ChatGPT user profile as trusted baseline tests ---

from src.parser import format_user_profile_for_distiller


class TestFormatUserProfileForDistiller:
    """Tests for format_user_profile_for_distiller() function."""

    def test_formats_full_profile(self):
        """Should format both user_profile and user_instructions."""
        profile = UserProfile(
            user_profile="I am a software developer specializing in Python.",
            user_instructions="Be concise and use code examples."
        )

        formatted = format_user_profile_for_distiller(profile)

        assert "About the User" in formatted
        assert "software developer" in formatted
        assert "User Preferences" in formatted
        assert "Be concise" in formatted

    def test_formats_profile_only(self):
        """Should work with only user_profile (no instructions)."""
        profile = UserProfile(
            user_profile="I am a teacher.",
            user_instructions=""
        )

        formatted = format_user_profile_for_distiller(profile)

        assert "About the User" in formatted
        assert "teacher" in formatted
        assert "User Preferences" not in formatted  # No instructions section

    def test_formats_instructions_only(self):
        """Should work with only user_instructions (no profile)."""
        profile = UserProfile(
            user_profile="",
            user_instructions="Always explain step by step."
        )

        formatted = format_user_profile_for_distiller(profile)

        assert "About the User" not in formatted  # No profile section
        assert "User Preferences" in formatted
        assert "step by step" in formatted

    def test_empty_profile(self):
        """Should handle empty profile gracefully."""
        profile = UserProfile(
            user_profile="",
            user_instructions=""
        )

        formatted = format_user_profile_for_distiller(profile)

        # Should be empty or minimal
        assert formatted == ""
