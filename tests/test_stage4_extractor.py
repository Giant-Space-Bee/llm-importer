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
import json

from src.providers import LocalProvider, LLMProvider
from src.extractor import (
    ExtractedFact,
    parse_extraction_response,
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
        """Should have all 6 categories."""
        assert "personal" in CATEGORIES
        assert "professional" in CATEGORIES
        assert "family" in CATEGORIES
        assert "preferences" in CATEGORIES
        assert "interests" in CATEGORIES
        assert "personality" in CATEGORIES
        assert len(CATEGORIES) == 6


class TestParseExtractionResponse:
    """parse_extraction_response() converts LLM JSON to ExtractedFact objects."""

    def test_parses_valid_json_array(self):
        """Should parse a valid JSON array of facts."""
        response = json.dumps([
            {
                "fact": "Lives in Victoria, BC",
                "category": "personal",
                "source_convo_id": "conv-123",
                "source_timestamp": 1700000000.0,
                "source_quote": "I live in Victoria BC"
            },
            {
                "fact": "Works as AI architect",
                "category": "professional",
                "source_convo_id": "conv-456",
                "source_timestamp": 1700000001.0,
                "source_quote": "I'm an AI architect"
            }
        ])

        facts = parse_extraction_response(response)

        assert len(facts) == 2
        assert facts[0].fact == "Lives in Victoria, BC"
        assert facts[0].category == "personal"
        assert facts[1].fact == "Works as AI architect"

    def test_parses_empty_array(self):
        """Empty array = no facts found (valid response)."""
        response = "[]"
        facts = parse_extraction_response(response)
        assert facts == []

    def test_handles_json_in_markdown_code_block(self):
        """LLMs often wrap JSON in ```json blocks."""
        response = """```json
[
    {
        "fact": "Has a dog named Kit",
        "category": "family",
        "source_convo_id": "conv-789",
        "source_timestamp": 1700000002.0,
        "source_quote": "my dog Kit"
    }
]
```"""

        facts = parse_extraction_response(response)
        assert len(facts) == 1
        assert facts[0].fact == "Has a dog named Kit"

    def test_handles_thinking_tags(self):
        """Hermes 4 may include <think>...</think> before JSON."""
        response = """<think>
Let me analyze this conversation for personal facts...
The user mentions living in Victoria.
</think>

[
    {
        "fact": "Lives in Victoria",
        "category": "personal",
        "source_convo_id": "conv-1",
        "source_timestamp": 1700000000.0,
        "source_quote": "I live in Victoria"
    }
]"""

        facts = parse_extraction_response(response)
        assert len(facts) == 1
        assert facts[0].fact == "Lives in Victoria"

    def test_returns_empty_on_invalid_json(self):
        """Invalid JSON should return empty list, not crash."""
        response = "This is not valid JSON at all"
        facts = parse_extraction_response(response)
        assert facts == []

    def test_returns_empty_on_non_array(self):
        """Non-array JSON should return empty list."""
        response = '{"fact": "test"}'  # Object, not array
        facts = parse_extraction_response(response)
        assert facts == []

    def test_skips_malformed_facts(self):
        """Should skip facts missing required fields."""
        response = json.dumps([
            {
                "fact": "Valid fact",
                "category": "personal",
                "source_convo_id": "conv-1",
                "source_timestamp": 1700000000.0,
                "source_quote": "valid quote"
            },
            {
                "fact": "Missing fields"
                # Missing category, source_convo_id, etc.
            }
        ])

        facts = parse_extraction_response(response)
        assert len(facts) == 1
        assert facts[0].fact == "Valid fact"


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
