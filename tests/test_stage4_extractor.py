"""
Stage 4 Tests: Extractor (Unit Tests Only)

What Stage 4 does:
- Connect to local LLM (localhost:1234)
- Send chunk to LLM with extraction prompt
- Parse JSON response into ExtractedFact objects

These are unit tests - no actual LLM calls.
Manual CLI testing verifies the real LLM integration.

Run: pytest tests/test_stage4_extractor.py -v
"""

import pytest

from src.providers import LocalProvider, LLMProvider
from src.extractor import (
    ExtractedFact,
    build_extraction_prompt,
    CATEGORIES,
)


class TestLocalProvider:
    """LocalProvider connects to LM Studio at localhost:1234."""

    def test_implements_llm_provider(self):
        """Should be an LLMProvider subclass."""
        provider = LocalProvider()
        assert isinstance(provider, LLMProvider)

    def test_is_local_true(self):
        """Local provider = sequential execution."""
        provider = LocalProvider()
        assert provider.is_local is True

    def test_default_url(self):
        """Default should be localhost:1234."""
        provider = LocalProvider()
        assert "127.0.0.1:1234" in provider.base_url or "localhost:1234" in provider.base_url

    def test_custom_url(self):
        """Should accept custom URL."""
        provider = LocalProvider(base_url="http://192.168.1.100:8080/v1")
        assert "192.168.1.100:8080" in provider.base_url


class TestExtractedFact:
    """ExtractedFact dataclass structure."""

    def test_has_required_fields(self):
        """Must have fact, category, source_convo_id, source_timestamp, source_quote."""
        fact = ExtractedFact(
            fact="Lives in Victoria, BC",
            category="personal",
            source_convo_id="abc-123",
            source_timestamp=1700000000.0,
            source_quote="I live in Victoria BC"
        )

        assert fact.fact == "Lives in Victoria, BC"
        assert fact.category == "personal"
        assert fact.source_convo_id == "abc-123"
        assert fact.source_timestamp == 1700000000.0
        assert fact.source_quote == "I live in Victoria BC"

    def test_category_validation(self):
        """Category should be one of the valid categories."""
        valid_categories = ["personal", "professional", "family", "preferences", "interests", "personality"]
        for cat in valid_categories:
            fact = ExtractedFact(
                fact="test",
                category=cat,
                source_convo_id="123",
                source_timestamp=1.0,
                source_quote="test"
            )
            assert fact.category == cat


class TestCategories:
    """Valid fact categories."""

    def test_has_expected_categories(self):
        """Should have all 8 growth-focused categories."""
        assert "identity" in CATEGORIES
        assert "values" in CATEGORIES
        assert "emotions" in CATEGORIES
        assert "relationships" in CATEGORIES
        assert "growth" in CATEGORIES
        assert "history" in CATEGORIES
        assert "practices" in CATEGORIES
        assert "shadows" in CATEGORIES
        assert len(CATEGORIES) == 8


class TestBuildExtractionPrompt:
    """build_extraction_prompt() creates the full prompt for LLM."""

    def test_includes_system_instructions(self):
        """Should include extraction instructions."""
        prompt = build_extraction_prompt("test content")
        assert "extract" in prompt.lower()
        assert "fact" in prompt.lower()

    def test_includes_schema(self):
        """Should include the JSON schema."""
        prompt = build_extraction_prompt("test content")
        assert "source_quote" in prompt
        assert "category" in prompt

    def test_includes_conversation_content(self):
        """Should include the actual conversation content."""
        content = "This is my test conversation content XYZ123"
        prompt = build_extraction_prompt(content)
        assert "XYZ123" in prompt

    def test_emphasizes_verbatim_quote(self):
        """Should emphasize that source_quote must be VERBATIM."""
        prompt = build_extraction_prompt("test")
        prompt_upper = prompt.upper()
        assert "VERBATIM" in prompt_upper or "EXACT" in prompt_upper
