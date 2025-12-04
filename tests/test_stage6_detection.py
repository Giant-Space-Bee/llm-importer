"""
Stage 6 Tests: Export Type Detection

What this tests:
- Auto-detect ChatGPT vs Claude vs unknown export formats
- "Coming soon" message for unsupported formats

Run: pytest tests/test_stage6_detection.py -v
"""

import pytest
from typing import List, Dict, Any

# These imports WILL FAIL until we implement - that's TDD!
from src.parser import detect_export_type


class TestDetectExportType:
    """detect_export_type() identifies the LLM service that created the export."""

    def test_detect_chatgpt_export(self):
        """Should detect ChatGPT export by presence of 'mapping' field."""
        chatgpt_data = [
            {
                "id": "abc-123",
                "title": "Test Conversation",
                "create_time": 1700000000.0,
                "mapping": {
                    "root": {"id": "root", "message": None, "parent": None, "children": []}
                },
                "current_node": "root"
            }
        ]

        result = detect_export_type(chatgpt_data)
        assert result == "chatgpt"

    def test_detect_claude_export(self):
        """Should detect Claude export by presence of 'uuid' and 'chat_messages'."""
        claude_data = [
            {
                "uuid": "abc-123-def-456",
                "name": "Test Conversation",
                "created_at": "2025-12-03T15:37:38.537083Z",
                "updated_at": "2025-12-03T15:37:54.552834Z",
                "chat_messages": []
            }
        ]

        result = detect_export_type(claude_data)
        assert result == "claude"

    def test_detect_unknown_export(self):
        """Should return 'unknown' for unrecognized format."""
        unknown_data = [
            {
                "some_field": "value",
                "another_field": 123
            }
        ]

        result = detect_export_type(unknown_data)
        assert result == "unknown"

    def test_detect_empty_list(self):
        """Should return 'unknown' for empty list."""
        result = detect_export_type([])
        assert result == "unknown"

    def test_detect_chatgpt_with_user_editable_context(self):
        """Should detect ChatGPT even with user_editable_context content."""
        chatgpt_data = [
            {
                "id": "abc-123",
                "title": "Test",
                "create_time": 1700000000.0,
                "mapping": {
                    "root": {"id": "root", "message": None, "parent": None, "children": ["ctx"]},
                    "ctx": {
                        "id": "ctx",
                        "message": {
                            "content": {
                                "content_type": "user_editable_context",
                                "user_profile": "Test profile"
                            }
                        },
                        "parent": "root",
                        "children": []
                    }
                }
            }
        ]

        result = detect_export_type(chatgpt_data)
        assert result == "chatgpt"

    def test_detect_claude_with_thinking_content(self):
        """Should detect Claude even with thinking content blocks."""
        claude_data = [
            {
                "uuid": "abc-123",
                "name": "Test",
                "created_at": "2025-12-03T15:37:38Z",
                "chat_messages": [
                    {
                        "uuid": "msg-1",
                        "sender": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "Let me think..."},
                            {"type": "text", "text": "Here's my answer"}
                        ]
                    }
                ]
            }
        ]

        result = detect_export_type(claude_data)
        assert result == "claude"


class TestComingSoonMessage:
    """Tests for the coming soon functionality in main.py."""

    def test_get_coming_soon_message_for_claude(self):
        """Should return friendly message for Claude exports."""
        from src.main import get_coming_soon_message

        message = get_coming_soon_message("claude")

        assert "coming soon" in message.lower() or "Coming soon" in message
        assert "Claude" in message
        assert "ChatGPT" in message  # Should mention what IS supported

    def test_get_coming_soon_message_for_unknown(self):
        """Should return message for unknown formats."""
        from src.main import get_coming_soon_message

        message = get_coming_soon_message("unknown")

        assert "coming soon" in message.lower() or "not recognized" in message.lower()
        assert "ChatGPT" in message  # Should mention what IS supported

    def test_is_supported_export_chatgpt(self):
        """ChatGPT should be supported."""
        from src.main import is_supported_export

        assert is_supported_export("chatgpt") is True

    def test_is_supported_export_claude(self):
        """Claude should be supported."""
        from src.main import is_supported_export

        assert is_supported_export("claude") is True

    def test_is_supported_export_unknown(self):
        """Unknown should NOT be supported."""
        from src.main import is_supported_export

        assert is_supported_export("unknown") is False
