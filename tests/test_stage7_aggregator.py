"""
Stage 7: Aggregator tests

Combine verified facts, count frequency. NO LLM.
"""

import pytest
from src.aggregator import aggregate, group_by_category, AggregatedFact
from src.extractor import ExtractedFact


def make_fact(
    fact: str,
    category: str = "personal",
    source_convo_id: str = "conv-1",
    source_timestamp: float = 1000.0,
    source_quote: str = "test quote",
) -> ExtractedFact:
    """Helper to create ExtractedFact for tests."""
    return ExtractedFact(
        fact=fact,
        category=category,
        source_convo_id=source_convo_id,
        source_timestamp=source_timestamp,
        source_quote=source_quote,
    )


def make_fact_dict(
    fact: str,
    category: str = "personal",
    source_convo_id: str = "conv-1",
    source_timestamp: float = 1000.0,
    source_quote: str = "test quote",
) -> dict:
    """Helper to create fact dict for tests."""
    return {
        "fact": fact,
        "category": category,
        "source_convo_id": source_convo_id,
        "source_timestamp": source_timestamp,
        "source_quote": source_quote,
    }


# =============================================================================
# Stage 7a: aggregate() function
# =============================================================================


class TestAggregate:
    """Test aggregate() function."""

    def test_single_fact_frequency_one(self):
        """Single fact should have frequency=1."""
        facts = [make_fact("User lives in Seattle")]
        result = aggregate(facts)

        assert len(result) == 1
        assert result[0].frequency == 1
        assert result[0].fact.fact == "User lives in Seattle"

    def test_duplicate_facts_frequency_two(self):
        """Exact duplicate facts should merge with frequency=2."""
        facts = [
            make_fact("User lives in Seattle", source_timestamp=1000.0),
            make_fact("User lives in Seattle", source_timestamp=2000.0),
        ]
        result = aggregate(facts)

        assert len(result) == 1
        assert result[0].frequency == 2

    def test_case_insensitive_grouping(self):
        """'I live in Seattle' and 'I Live In Seattle' should group."""
        facts = [
            make_fact("I live in Seattle", source_timestamp=1000.0),
            make_fact("I Live In Seattle", source_timestamp=2000.0),
        ]
        result = aggregate(facts)

        assert len(result) == 1
        assert result[0].frequency == 2

    def test_whitespace_insensitive_grouping(self):
        """Facts with different whitespace should group."""
        facts = [
            make_fact("I  live  in Seattle", source_timestamp=1000.0),
            make_fact("I live in Seattle", source_timestamp=2000.0),
        ]
        result = aggregate(facts)

        assert len(result) == 1
        assert result[0].frequency == 2

    def test_keeps_newest_as_canonical(self):
        """When merging, keep fact with highest source_timestamp."""
        facts = [
            make_fact("User lives in Seattle", source_timestamp=1000.0, source_quote="old quote"),
            make_fact("User lives in Seattle", source_timestamp=3000.0, source_quote="newest quote"),
            make_fact("User lives in Seattle", source_timestamp=2000.0, source_quote="middle quote"),
        ]
        result = aggregate(facts)

        assert len(result) == 1
        assert result[0].frequency == 3
        assert result[0].fact.source_timestamp == 3000.0
        assert result[0].fact.source_quote == "newest quote"

    def test_preserves_original_text(self):
        """Canonical fact should have original text, not normalized."""
        facts = [
            make_fact("I Live In Seattle", source_timestamp=2000.0),  # Newer, should be kept
            make_fact("i live in seattle", source_timestamp=1000.0),
        ]
        result = aggregate(facts)

        assert len(result) == 1
        # Should preserve the original casing from the newest fact
        assert result[0].fact.fact == "I Live In Seattle"

    def test_different_facts_stay_separate(self):
        """Semantically different facts should NOT group."""
        facts = [
            make_fact("User lives in Seattle"),
            make_fact("User works at Microsoft"),
            make_fact("User has a dog"),
        ]
        result = aggregate(facts)

        assert len(result) == 3
        assert all(f.frequency == 1 for f in result)

    def test_empty_input(self):
        """Empty list returns empty list."""
        result = aggregate([])
        assert result == []

    def test_handles_dict_input(self):
        """Should work with dict facts."""
        facts = [
            make_fact_dict("User lives in Seattle"),
            make_fact_dict("User lives in Seattle"),
        ]
        result = aggregate(facts)

        assert len(result) == 1
        assert result[0].frequency == 2

    def test_handles_mixed_dict_and_dataclass(self):
        """Should work with both dict facts and ExtractedFact objects."""
        facts = [
            make_fact("User lives in Seattle", source_timestamp=1000.0),
            make_fact_dict("User lives in Seattle", source_timestamp=2000.0),
        ]
        result = aggregate(facts)

        assert len(result) == 1
        assert result[0].frequency == 2

    def test_unicode_normalization(self):
        """Unicode quotes/dashes should be normalized for grouping."""
        facts = [
            make_fact("I'm here", source_timestamp=1000.0),  # straight apostrophe
            make_fact("I\u2019m here", source_timestamp=2000.0),  # curly apostrophe
        ]
        result = aggregate(facts)

        assert len(result) == 1
        assert result[0].frequency == 2

    def test_multiple_groups(self):
        """Multiple groups with different frequencies."""
        facts = [
            # Group 1: appears 3 times
            make_fact("Lives in Seattle", source_timestamp=1000.0),
            make_fact("Lives in Seattle", source_timestamp=2000.0),
            make_fact("Lives in Seattle", source_timestamp=3000.0),
            # Group 2: appears 2 times
            make_fact("Works at Microsoft", source_timestamp=1500.0),
            make_fact("Works at Microsoft", source_timestamp=2500.0),
            # Group 3: appears 1 time
            make_fact("Has a dog"),
        ]
        result = aggregate(facts)

        assert len(result) == 3

        # Find each group by fact text
        by_fact = {r.fact.fact: r for r in result}

        # Check frequencies (canonical should have original casing from newest)
        seattle = next(r for r in result if "seattle" in r.fact.fact.lower())
        microsoft = next(r for r in result if "microsoft" in r.fact.fact.lower())
        dog = next(r for r in result if "dog" in r.fact.fact.lower())

        assert seattle.frequency == 3
        assert microsoft.frequency == 2
        assert dog.frequency == 1


# =============================================================================
# Stage 7b: group_by_category() function
# =============================================================================


class TestGroupByCategory:
    """Test group_by_category() function."""

    def test_groups_by_category(self):
        """Facts should be organized into category buckets."""
        aggregated = [
            AggregatedFact(make_fact("Lives in Seattle", category="personal"), 2),
            AggregatedFact(make_fact("Works at Microsoft", category="professional"), 1),
            AggregatedFact(make_fact("Likes hiking", category="interests"), 3),
        ]
        result = group_by_category(aggregated)

        assert "personal" in result
        assert "professional" in result
        assert "interests" in result

        assert len(result["personal"]) == 1
        assert len(result["professional"]) == 1
        assert len(result["interests"]) == 1

    def test_multiple_facts_same_category(self):
        """Multiple facts in same category should be in same list."""
        aggregated = [
            AggregatedFact(make_fact("Lives in Seattle", category="personal"), 2),
            AggregatedFact(make_fact("Age is 30", category="personal"), 1),
            AggregatedFact(make_fact("Name is John", category="personal"), 1),
        ]
        result = group_by_category(aggregated)

        assert len(result) == 1
        assert "personal" in result
        assert len(result["personal"]) == 3

    def test_empty_input(self):
        """Empty list returns empty dict."""
        result = group_by_category([])
        assert result == {}

    def test_preserves_frequency(self):
        """Frequency should be preserved in grouped facts."""
        aggregated = [
            AggregatedFact(make_fact("Lives in Seattle", category="personal"), 5),
        ]
        result = group_by_category(aggregated)

        assert result["personal"][0].frequency == 5

    def test_all_categories(self):
        """Should handle all valid categories."""
        categories = ["personal", "professional", "family", "preferences", "interests", "personality"]
        aggregated = [
            AggregatedFact(make_fact(f"Fact for {cat}", category=cat), 1)
            for cat in categories
        ]
        result = group_by_category(aggregated)

        assert set(result.keys()) == set(categories)
