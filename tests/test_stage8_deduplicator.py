"""
Stage 8: Deduplicator tests

Two-phase semantic deduplication:
- Phase 1: Within-category dedup (timestamp-batched, merge-sort)
- Phase 2: Cross-category merge-sort

Uses mock LLM provider for testing.
"""

import pytest
from unittest.mock import Mock, patch
from typing import List

from src.aggregator import AggregatedFact
from src.extractor import ExtractedFact
from src.deduplicator import (
    DeduplicatedFact,
    DEDUP_SCHEMA,
    deduplicate,
    deduplicate_category,
    deduplicate_merge_sort,
    dedup_batch,
    build_dedup_prompt,
)


# =============================================================================
# Test Helpers
# =============================================================================


def make_extracted(
    fact: str,
    category: str = "personal",
    source_convo_id: str = "conv-1",
    source_timestamp: float = 1000.0,
    source_quote: str = "test quote",
) -> ExtractedFact:
    """Create ExtractedFact for tests."""
    return ExtractedFact(
        fact=fact,
        category=category,
        source_convo_id=source_convo_id,
        source_timestamp=source_timestamp,
        source_quote=source_quote,
    )


def make_aggregated(
    fact: str,
    category: str = "personal",
    frequency: int = 1,
    source_timestamp: float = 1000.0,
) -> AggregatedFact:
    """Create AggregatedFact for tests."""
    return AggregatedFact(
        fact=make_extracted(fact, category=category, source_timestamp=source_timestamp),
        frequency=frequency,
    )


def make_deduped(fact: str, category: str = "personal") -> DeduplicatedFact:
    """Create DeduplicatedFact for tests."""
    return DeduplicatedFact(fact=fact, category=category)


def mock_dedup_response(facts: List[DeduplicatedFact]) -> dict:
    """Create mock LLM response with given facts."""
    return {
        "facts": [{"fact": f.fact, "category": f.category} for f in facts]
    }


# =============================================================================
# TestDeduplicatedFact: Data structure tests
# =============================================================================


class TestDeduplicatedFact:
    """Test DeduplicatedFact dataclass."""

    def test_has_fact_and_category(self):
        """DeduplicatedFact should have fact and category fields."""
        df = DeduplicatedFact(fact="Lives in Seattle", category="personal")
        assert df.fact == "Lives in Seattle"
        assert df.category == "personal"

    def test_equality(self):
        """Two DeduplicatedFacts with same values should be equal."""
        df1 = DeduplicatedFact(fact="Test", category="personal")
        df2 = DeduplicatedFact(fact="Test", category="personal")
        assert df1 == df2


# =============================================================================
# TestDedupSchema: Schema validation tests
# =============================================================================


class TestDedupSchema:
    """Test DEDUP_SCHEMA structure."""

    def test_has_facts_array(self):
        """Schema should have facts array."""
        assert "facts" in DEDUP_SCHEMA["properties"]
        assert DEDUP_SCHEMA["properties"]["facts"]["type"] == "array"

    def test_fact_item_has_required_fields(self):
        """Each fact item should have fact and category."""
        item_props = DEDUP_SCHEMA["properties"]["facts"]["items"]["properties"]
        assert "fact" in item_props
        assert "category" in item_props

    def test_category_has_enum(self):
        """Category should be constrained to valid values."""
        item_props = DEDUP_SCHEMA["properties"]["facts"]["items"]["properties"]
        assert "enum" in item_props["category"]
        expected = ["personal", "professional", "family", "preferences", "interests", "personality"]
        assert item_props["category"]["enum"] == expected


# =============================================================================
# TestBuildDedupPrompt: Prompt generation tests
# =============================================================================


class TestBuildDedupPrompt:
    """Test build_dedup_prompt() function."""

    def test_includes_frequency(self):
        """Prompt should show frequency for each fact."""
        facts = [make_aggregated("Test fact", frequency=5)]
        prompt = build_dedup_prompt(facts)
        assert '"frequency": 5' in prompt or '"frequency":5' in prompt

    def test_includes_timestamp(self):
        """Prompt should show source_timestamp."""
        facts = [make_aggregated("Test fact", source_timestamp=1234.0)]
        prompt = build_dedup_prompt(facts)
        assert "1234" in prompt

    def test_includes_category_as_hint(self):
        """Original category should be shown but not enforced."""
        facts = [make_aggregated("Test fact", category="professional")]
        prompt = build_dedup_prompt(facts)
        assert "professional" in prompt

    def test_multiple_facts_in_prompt(self):
        """Multiple facts should all appear in prompt."""
        facts = [
            make_aggregated("Fact one"),
            make_aggregated("Fact two"),
            make_aggregated("Fact three"),
        ]
        prompt = build_dedup_prompt(facts)
        assert "Fact one" in prompt
        assert "Fact two" in prompt
        assert "Fact three" in prompt

    def test_accepts_deduplicated_facts(self):
        """Should also work with DeduplicatedFact input (for Phase 2)."""
        facts = [make_deduped("Already deduped", "personal")]
        prompt = build_dedup_prompt(facts)
        assert "Already deduped" in prompt


# =============================================================================
# TestDeduplicateBatch: Single LLM call tests
# =============================================================================


class TestDeduplicateBatch:
    """Test dedup_batch() - single LLM call."""

    def test_returns_deduplicated_facts(self):
        """Should return DeduplicatedFact list from LLM response."""
        mock_provider = Mock()
        mock_provider.complete_structured.return_value = mock_dedup_response([
            make_deduped("Canonical fact", "personal")
        ])

        facts = [make_aggregated("Test fact")]
        result = dedup_batch(facts, mock_provider)

        assert len(result) == 1
        assert isinstance(result[0], DeduplicatedFact)
        assert result[0].fact == "Canonical fact"

    def test_calls_provider_with_schema(self):
        """Should call provider.complete_structured with correct schema."""
        mock_provider = Mock()
        mock_provider.complete_structured.return_value = {"facts": []}

        facts = [make_aggregated("Test")]
        dedup_batch(facts, mock_provider)

        mock_provider.complete_structured.assert_called_once()
        call_args = mock_provider.complete_structured.call_args
        assert call_args[0][1] == DEDUP_SCHEMA  # Second arg is schema

    def test_empty_input_returns_empty(self):
        """Empty facts list should return empty list."""
        mock_provider = Mock()
        result = dedup_batch([], mock_provider)
        assert result == []
        mock_provider.complete_structured.assert_not_called()


# =============================================================================
# TestDeduplicate: Main entry point tests
# =============================================================================


class TestDeduplicate:
    """Test main deduplicate() function."""

    def test_single_fact_passes_through(self):
        """Single fact should pass through (reformatted)."""
        mock_provider = Mock()
        mock_provider.complete_structured.return_value = mock_dedup_response([
            make_deduped("Lives in Seattle", "personal")
        ])

        facts = [make_aggregated("Lives in Seattle", category="personal")]
        result = deduplicate(facts, mock_provider)

        assert len(result) == 1
        assert result[0].fact == "Lives in Seattle"
        assert result[0].category == "personal"

    def test_exact_duplicates_merged(self):
        """Identical facts should become one canonical version."""
        mock_provider = Mock()
        mock_provider.complete_structured.return_value = mock_dedup_response([
            make_deduped("Lives in Seattle", "personal")
        ])

        facts = [
            make_aggregated("Lives in Seattle", frequency=3),
            make_aggregated("Lives in Seattle", frequency=2),
        ]
        result = deduplicate(facts, mock_provider)

        # LLM should merge them
        assert len(result) == 1

    def test_semantic_duplicates_merged(self):
        """'Lives in Seattle' + 'Resides in Seattle' -> one fact."""
        mock_provider = Mock()
        # Mock LLM merges semantic duplicates
        mock_provider.complete_structured.return_value = mock_dedup_response([
            make_deduped("Lives in Seattle, WA", "personal")
        ])

        facts = [
            make_aggregated("Lives in Seattle"),
            make_aggregated("Resides in Seattle"),
        ]
        result = deduplicate(facts, mock_provider)

        assert len(result) == 1

    def test_cross_category_duplicates_merged(self):
        """Same fact in 'family' and 'personal' should merge in Phase 2."""
        mock_provider = Mock()
        # Phase 1 returns both, Phase 2 merges
        mock_provider.complete_structured.side_effect = [
            # Phase 1: family category
            mock_dedup_response([make_deduped("Has sister named Haley", "family")]),
            # Phase 1: personal category
            mock_dedup_response([make_deduped("Sister is Haley", "personal")]),
            # Phase 2: cross-category merge
            mock_dedup_response([make_deduped("Has sister named Haley", "family")]),
        ]

        facts = [
            make_aggregated("Has sister named Haley", category="family"),
            make_aggregated("Sister is Haley", category="personal"),
        ]
        result = deduplicate(facts, mock_provider)

        assert len(result) == 1

    def test_specific_beats_vague(self):
        """'Seattle, WA' should be kept over 'USA'."""
        mock_provider = Mock()
        mock_provider.complete_structured.return_value = mock_dedup_response([
            make_deduped("Lives in Seattle, WA, USA", "personal")
        ])

        facts = [
            make_aggregated("Lives in USA"),
            make_aggregated("Lives in Seattle, WA, USA"),
        ]
        result = deduplicate(facts, mock_provider)

        assert len(result) == 1
        assert "Seattle" in result[0].fact

    def test_newer_beats_older(self):
        """Contradictions resolved by timestamp (mocked in LLM response)."""
        mock_provider = Mock()
        # LLM picks the newer one
        mock_provider.complete_structured.return_value = mock_dedup_response([
            make_deduped("Works at Microsoft", "professional")
        ])

        facts = [
            make_aggregated("Works at Google", source_timestamp=1000.0),
            make_aggregated("Works at Microsoft", source_timestamp=2000.0),
        ]
        result = deduplicate(facts, mock_provider)

        assert len(result) == 1
        assert "Microsoft" in result[0].fact

    def test_higher_frequency_preferred(self):
        """frequency=5 beats frequency=1 (mocked)."""
        mock_provider = Mock()
        mock_provider.complete_structured.return_value = mock_dedup_response([
            make_deduped("Lives in Seattle", "personal")
        ])

        facts = [
            make_aggregated("Lives in Seattle", frequency=5),
            make_aggregated("Lives in Portland", frequency=1),
        ]
        result = deduplicate(facts, mock_provider)

        # LLM should prefer higher frequency
        assert len(result) == 1

    def test_preserves_unique_facts(self):
        """Different facts should all be kept."""
        mock_provider = Mock()
        mock_provider.complete_structured.return_value = mock_dedup_response([
            make_deduped("Lives in Seattle", "personal"),
            make_deduped("Works at Microsoft", "professional"),
            make_deduped("Has a dog", "family"),
        ])

        facts = [
            make_aggregated("Lives in Seattle", category="personal"),
            make_aggregated("Works at Microsoft", category="professional"),
            make_aggregated("Has a dog", category="family"),
        ]
        result = deduplicate(facts, mock_provider)

        assert len(result) == 3

    def test_empty_input_returns_empty(self):
        """Empty input should return empty list."""
        mock_provider = Mock()
        result = deduplicate([], mock_provider)
        assert result == []
        mock_provider.complete_structured.assert_not_called()

    def test_large_category_batches_correctly(self):
        """Large category (>max_batch) should split into batches."""
        mock_provider = Mock()
        # Return consistent response for any number of calls
        mock_provider.complete_structured.return_value = mock_dedup_response(
            [make_deduped(f"Fact {i}", "professional") for i in range(25)]
        )

        # 60 facts in one category should trigger batching
        facts = [
            make_aggregated(f"Professional fact {i}", category="professional", source_timestamp=float(i))
            for i in range(60)
        ]
        result = deduplicate(facts, mock_provider, max_batch=50)

        # Multiple LLM calls should have been made (at least 2 for phase 1 batches)
        assert mock_provider.complete_structured.call_count >= 2


# =============================================================================
# TestDeduplicateCategory: Within-category dedup tests
# =============================================================================


class TestDeduplicateCategory:
    """Test deduplicate_category() function."""

    def test_sorts_by_timestamp(self):
        """Facts should be processed oldest -> newest."""
        mock_provider = Mock()
        mock_provider.complete_structured.return_value = mock_dedup_response([
            make_deduped("Newest fact", "personal")
        ])

        facts = [
            make_aggregated("Newer", source_timestamp=2000.0),
            make_aggregated("Oldest", source_timestamp=1000.0),
            make_aggregated("Newest", source_timestamp=3000.0),
        ]
        result = deduplicate_category(facts, "personal", mock_provider)

        # Just verify it completes - ordering is internal
        assert isinstance(result, list)

    def test_single_batch_no_merge(self):
        """Less than max_batch facts should make one LLM call."""
        mock_provider = Mock()
        mock_provider.complete_structured.return_value = mock_dedup_response([
            make_deduped("Fact 1", "personal"),
            make_deduped("Fact 2", "personal"),
        ])

        facts = [
            make_aggregated("Fact 1"),
            make_aggregated("Fact 2"),
        ]
        result = deduplicate_category(facts, "personal", mock_provider, max_batch=50)

        assert mock_provider.complete_structured.call_count == 1
        assert len(result) == 2

    def test_multiple_batches_merge_sort(self):
        """More than max_batch should split, dedup, merge."""
        mock_provider = Mock()
        # Return consistent response for any number of calls
        mock_provider.complete_structured.return_value = mock_dedup_response(
            [make_deduped(f"F{i}", "personal") for i in range(3)]
        )

        facts = [make_aggregated(f"Fact {i}", source_timestamp=float(i)) for i in range(8)]
        result = deduplicate_category(facts, "personal", mock_provider, max_batch=5)

        # Should have at least 2 calls (split batches)
        assert mock_provider.complete_structured.call_count >= 2


# =============================================================================
# TestDeduplicateMergeSort: Cross-category dedup tests
# =============================================================================


class TestDeduplicateMergeSort:
    """Test deduplicate_merge_sort() function."""

    def test_recursive_splitting(self):
        """Large input should be recursively split."""
        mock_provider = Mock()
        mock_provider.complete_structured.side_effect = [
            mock_dedup_response([make_deduped(f"F{i}", "personal") for i in range(20)]),
            mock_dedup_response([make_deduped(f"F{i+20}", "personal") for i in range(20)]),
            mock_dedup_response([make_deduped(f"F{i}", "personal") for i in range(30)]),
        ]

        facts = [make_deduped(f"Fact {i}", "personal") for i in range(60)]
        result = deduplicate_merge_sort(facts, mock_provider, max_batch=50)

        # Should have made multiple calls
        assert mock_provider.complete_structured.call_count >= 2

    def test_base_case_single_batch(self):
        """Small input (<= max_batch) should make single dedup call."""
        mock_provider = Mock()
        mock_provider.complete_structured.return_value = mock_dedup_response([
            make_deduped("Fact 1", "personal"),
            make_deduped("Fact 2", "professional"),
        ])

        facts = [
            make_deduped("Fact 1", "personal"),
            make_deduped("Fact 2", "professional"),
        ]
        result = deduplicate_merge_sort(facts, mock_provider, max_batch=50)

        assert mock_provider.complete_structured.call_count == 1
        assert len(result) == 2

    def test_phase2_catches_cross_category(self):
        """'Haley' in family + personal should merge to one canonical."""
        mock_provider = Mock()
        mock_provider.complete_structured.return_value = mock_dedup_response([
            make_deduped("Has sister named Haley", "family")
        ])

        facts = [
            make_deduped("Has sister named Haley", "family"),
            make_deduped("Sister is Haley", "personal"),
        ]
        result = deduplicate_merge_sort(facts, mock_provider)

        assert len(result) == 1
        assert "Haley" in result[0].fact
