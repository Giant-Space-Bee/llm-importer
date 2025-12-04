"""
Stage 1 Tests: CLI Shell

What Stage 1 does:
- Shows a banner identifying the tool
- Accepts an input file path (default: conversations.json)
- Validates the file exists
- Shows file stats (size in human-readable format)
- Exits cleanly

Stage 6 additions:
- Provider selection (local vs API)
- get_provider() function

Run: pytest tests/test_stage1_cli.py -v
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

# These imports WILL FAIL until we implement - that's TDD!
from src.main import (
    get_banner,
    validate_input_file,
    format_file_size,
    ValidationResult,
    DEFAULT_INPUT_PATH,
    get_provider,
    PROVIDER_LOCAL,
    PROVIDER_API,
)
from src.providers import LocalProvider, APIProvider, LLMProvider


class TestBanner:
    """The banner identifies the tool to the user."""

    def test_contains_project_name(self):
        """User should immediately know what tool they're running."""
        banner = get_banner()
        assert "LLM IMPORTER" in banner.upper()

    def test_contains_purpose(self):
        """User should understand what this tool does."""
        banner = get_banner()
        banner_lower = banner.lower()
        # Should mention extracting memories or similar
        assert "extract" in banner_lower or "memory" in banner_lower or "import" in banner_lower


class TestValidationResult:
    """ValidationResult should be a proper data structure."""

    def test_has_required_fields(self):
        """Result must have valid, size_bytes, and error fields."""
        # Valid file result
        result = ValidationResult(valid=True, size_bytes=1000, error=None)
        assert result.valid is True
        assert result.size_bytes == 1000
        assert result.error is None

        # Invalid file result
        result = ValidationResult(valid=False, size_bytes=0, error="File not found")
        assert result.valid is False
        assert result.error == "File not found"


class TestFileValidation:
    """validate_input_file() checks if file exists and gets stats."""

    def test_valid_file_returns_success(self, sample_json_file):
        """Existing file should return valid=True with size."""
        result = validate_input_file(str(sample_json_file))

        assert result.valid is True
        assert result.error is None
        assert result.size_bytes > 0

    def test_missing_file_returns_error(self):
        """Non-existent file should return valid=False with error message."""
        result = validate_input_file("/fake/path/nonexistent.json")

        assert result.valid is False
        assert result.error is not None
        assert "not found" in result.error.lower() or "does not exist" in result.error.lower()

    def test_empty_path_returns_error(self):
        """Empty string should return an error, not crash."""
        result = validate_input_file("")

        assert result.valid is False
        assert result.error is not None

    def test_fixture_file(self, chatgpt_fixture):
        """Test against the ChatGPT fixture file."""
        result = validate_input_file(str(chatgpt_fixture))

        assert result.valid is True
        assert result.size_bytes > 0


class TestFileSizeFormatting:
    """format_file_size() converts bytes to human-readable strings."""

    def test_bytes_small(self):
        """Small files show bytes."""
        result = format_file_size(500)
        assert "500" in result
        assert "byte" in result.lower()

    def test_kilobytes(self):
        """KB-sized files show KB."""
        result = format_file_size(5_000)
        assert "KB" in result
        # 5000 bytes = 4.88 KB
        assert "4" in result or "5" in result

    def test_megabytes(self):
        """MB-sized files show MB."""
        result = format_file_size(57_913_708)
        assert "MB" in result
        # 57913708 bytes = 55.24 MB
        assert "55" in result

    def test_gigabytes(self):
        """GB-sized files show GB."""
        result = format_file_size(2_500_000_000)
        assert "GB" in result
        # 2.5 billion bytes = 2.33 GB
        assert "2" in result


class TestDefaults:
    """Default configuration values."""

    def test_default_input_path_is_conversations_json(self):
        """Default should be conversations.json (ChatGPT export filename)."""
        assert DEFAULT_INPUT_PATH == "conversations.json"


class TestProviderConstants:
    """Provider type constants."""

    def test_provider_local_constant(self):
        """Should have LOCAL provider constant."""
        assert PROVIDER_LOCAL == "local"

    def test_provider_api_constant(self):
        """Should have API provider constant."""
        assert PROVIDER_API == "api"


class TestGetProvider:
    """get_provider() returns correct provider based on choice."""

    def test_local_returns_local_provider(self):
        """Choosing local should return LocalProvider."""
        provider = get_provider(PROVIDER_LOCAL)
        assert isinstance(provider, LocalProvider)
        assert provider.is_local is True

    def test_api_returns_api_provider(self):
        """Choosing API should return APIProvider."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = get_provider(PROVIDER_API)
            assert isinstance(provider, APIProvider)
            assert provider.is_local is False

    def test_api_requires_api_key(self):
        """API provider should fail without API key."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises((ValueError, RuntimeError)):
                get_provider(PROVIDER_API)

    def test_returns_llm_provider(self):
        """Both provider types should be LLMProvider subclasses."""
        local = get_provider(PROVIDER_LOCAL)
        assert isinstance(local, LLMProvider)

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            api = get_provider(PROVIDER_API)
            assert isinstance(api, LLMProvider)

    def test_invalid_choice_raises(self):
        """Invalid provider choice should raise error."""
        with pytest.raises(ValueError):
            get_provider("invalid")
