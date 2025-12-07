"""
Stage 5: Verifier tests

5a: Unicode normalization (curly quotes, dashes, whitespace)
5b: Verify against ALL messages (not just user)
"""

import pytest
from src.verifier import normalize_text


# =============================================================================
# Stage 5a: Unicode normalization
# =============================================================================

class TestNormalizeText:
    """Test normalize_text for quote matching."""

    def test_lowercase(self):
        """Should lowercase text."""
        assert normalize_text("Hello World") == "hello world"

    def test_collapse_whitespace(self):
        """Should collapse multiple spaces/tabs/newlines to single space."""
        assert normalize_text("hello   world") == "hello world"
        assert normalize_text("hello\t\tworld") == "hello world"
        assert normalize_text("hello\n\nworld") == "hello world"
        assert normalize_text("  hello  world  ") == "hello world"

    def test_curly_apostrophes_to_straight(self):
        """LLMs often return curly apostrophes - normalize to straight."""
        # Right single quote (most common for apostrophes)
        assert normalize_text("I'm here") == "i'm here"
        assert normalize_text("I'm here") == "i'm here"  # U+2019 '
        # Left single quote
        assert normalize_text("'quoted'") == "'quoted'"  # U+2018 ' and U+2019 '

    def test_curly_double_quotes_to_straight(self):
        """Normalize curly double quotes to straight."""
        assert normalize_text('"hello"') == '"hello"'  # U+201C " and U+201D "
        assert normalize_text('He said "hello"') == 'he said "hello"'

    def test_em_dash_to_hyphen(self):
        """Normalize em dashes and en dashes to regular hyphen."""
        assert normalize_text("hello—world") == "hello-world"  # em dash U+2014
        assert normalize_text("hello–world") == "hello-world"  # en dash U+2013

    def test_ellipsis_to_dots(self):
        """Normalize ellipsis character to three dots."""
        assert normalize_text("wait…what") == "wait...what"  # U+2026 …

    def test_combined_normalizations(self):
        """Test multiple normalizations together."""
        # Using raw chars: ' (U+2019), — (U+2014), … (U+2026), " " (U+201C/D)
        text = "I\u2019m   going\u2014wait\u2026  what\u2019s  \u201cthat\u201d?"
        expected = "i'm going-wait... what's \"that\"?"
        assert normalize_text(text) == expected

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert normalize_text("") == ""

    def test_only_whitespace(self):
        """Whitespace-only should return empty string."""
        assert normalize_text("   \t\n  ") == ""

    def test_preserves_normal_punctuation(self):
        """Regular ASCII punctuation should be preserved."""
        assert normalize_text("hello, world! how are you?") == "hello, world! how are you?"

    def test_unicode_letters_preserved(self):
        """Non-ASCII letters should be preserved (just lowercased)."""
        assert normalize_text("Café résumé") == "café résumé"


# =============================================================================
# Stage 5b: Verify facts against ALL messages
# =============================================================================

class TestVerifyFact:
    """Test verify_fact for single fact verification."""

    def test_quote_found_in_text_returns_verified(self):
        """Quote present in conversation text should verify."""
        from src.verifier import verify_fact

        fact = {
            "fact": "User lives in Seattle",
            "source_quote": "I live in Seattle",
            "source_convo_id": "conv-123",
        }
        convo_text = "User: I live in Seattle\nAssistant: Seattle is lovely!"

        result = verify_fact(fact, convo_text)

        assert result.verified is True
        assert result.fact == fact

    def test_quote_found_in_assistant_message_returns_verified(self):
        """Quote from assistant message should also verify."""
        from src.verifier import verify_fact

        fact = {
            "fact": "Seattle is lovely",
            "source_quote": "Seattle is lovely",
            "source_convo_id": "conv-123",
        }
        convo_text = "User: I live in Seattle\nAssistant: Seattle is lovely!"

        result = verify_fact(fact, convo_text)

        assert result.verified is True

    def test_quote_not_found_returns_not_verified(self):
        """Quote not in conversation should fail verification."""
        from src.verifier import verify_fact

        fact = {
            "fact": "User moved to Portland",
            "source_quote": "I moved to Portland",
            "source_convo_id": "conv-123",
        }
        convo_text = "User: I live in Seattle\nAssistant: Seattle is lovely!"

        result = verify_fact(fact, convo_text)

        assert result.verified is False
        assert "NOT FOUND" in result.match_location

    def test_uses_normalized_matching(self):
        """Should match despite case/whitespace/unicode differences."""
        from src.verifier import verify_fact

        fact = {
            "fact": "User is going somewhere",
            "source_quote": "I'm going",  # Curly apostrophe U+2019
            "source_convo_id": "conv-123",
        }
        # Original has straight apostrophe
        convo_text = "User: I'm going to the store"

        result = verify_fact(fact, convo_text)

        assert result.verified is True

    def test_partial_quote_matches(self):
        """Quote that is substring of message should match."""
        from src.verifier import verify_fact

        fact = {
            "fact": "User has a dog",
            "source_quote": "my dog",
            "source_convo_id": "conv-123",
        }
        convo_text = "I love taking my dog for walks in the park"

        result = verify_fact(fact, convo_text)

        assert result.verified is True

    def test_empty_quote_fails(self):
        """Empty source_quote should fail verification."""
        from src.verifier import verify_fact

        fact = {
            "fact": "Some fact",
            "source_quote": "",
            "source_convo_id": "conv-123",
        }
        convo_text = "User: Hello world"

        result = verify_fact(fact, convo_text)

        assert result.verified is False


class TestVerifyAll:
    """Test verify_all for batch verification."""

    def test_separates_verified_and_discarded(self):
        """Should return two lists: verified and discarded facts."""
        from src.verifier import verify_all

        facts = [
            {"fact": "Fact 1", "source_quote": "I live in Seattle", "source_convo_id": "conv-1"},
            {"fact": "Fact 2", "source_quote": "hallucinated quote", "source_convo_id": "conv-1"},
            {"fact": "Fact 3", "source_quote": "love my job", "source_convo_id": "conv-1"},
        ]
        conversations = {
            "conv-1": "I live in Seattle and love my job"
        }

        verified, discarded = verify_all(facts, conversations)

        assert len(verified) == 2
        assert len(discarded) == 1
        assert discarded[0]["fact"] == "Fact 2"

    def test_handles_multiple_conversations(self):
        """Should look up correct conversation for each fact."""
        from src.verifier import verify_all

        facts = [
            {"fact": "Fact 1", "source_quote": "Seattle", "source_convo_id": "conv-1"},
            {"fact": "Fact 2", "source_quote": "Portland", "source_convo_id": "conv-2"},
        ]
        conversations = {
            "conv-1": "I live in Seattle",
            "conv-2": "I visited Portland",
        }

        verified, discarded = verify_all(facts, conversations)

        assert len(verified) == 2
        assert len(discarded) == 0

    def test_missing_conversation_fails_verification(self):
        """Fact referencing non-existent conversation should fail."""
        from src.verifier import verify_all

        facts = [
            {"fact": "Fact 1", "source_quote": "hello", "source_convo_id": "conv-missing"},
        ]
        conversations = {
            "conv-1": "I live in Seattle",
        }

        verified, discarded = verify_all(facts, conversations)

        assert len(verified) == 0
        assert len(discarded) == 1

    def test_empty_facts_list(self):
        """Empty facts list should return empty lists."""
        from src.verifier import verify_all

        verified, discarded = verify_all([], {"conv-1": "text"})

        assert verified == []
        assert discarded == []

    def test_returns_original_fact_objects(self):
        """Verified/discarded lists should contain original fact dicts."""
        from src.verifier import verify_all

        original_fact = {"fact": "Test", "source_quote": "hello", "source_convo_id": "conv-1"}
        facts = [original_fact]
        conversations = {"conv-1": "hello world"}

        verified, discarded = verify_all(facts, conversations)

        assert verified[0] is original_fact  # Same object reference

    def test_fallback_finds_quote_in_different_conversation(self):
        """Should verify fact if quote found in ANY conversation (wrong ID fix)."""
        from src.verifier import verify_all

        # Fact has wrong conversation ID, but quote exists in conv-2
        facts = [
            {"fact": "User loves Seattle", "source_quote": "I love Seattle", "source_convo_id": "conv-WRONG"},
        ]
        conversations = {
            "conv-1": "Hello there",
            "conv-2": "I love Seattle so much",  # Quote is actually here
            "conv-3": "Goodbye",
        }

        verified, discarded = verify_all(facts, conversations)

        assert len(verified) == 1
        assert len(discarded) == 0
        # Should have corrected the ID
        assert verified[0]["source_convo_id"] == "conv-2"

    def test_fallback_does_not_find_hallucinated_quote(self):
        """Should discard if quote not found in ANY conversation."""
        from src.verifier import verify_all

        facts = [
            {"fact": "User loves Portland", "source_quote": "I love Portland", "source_convo_id": "conv-1"},
        ]
        conversations = {
            "conv-1": "Hello there",
            "conv-2": "I love Seattle",  # Different city
            "conv-3": "Goodbye",
        }

        verified, discarded = verify_all(facts, conversations)

        assert len(verified) == 0
        assert len(discarded) == 1  # True hallucination


class TestFuzzyMatching:
    """Test fuzzy matching for near-miss quotes."""

    def test_typo_in_quote_matches_via_fuzzy(self):
        """Should match when quote has typo like 'Rpd' vs 'Rod'."""
        from src.verifier import verify_fact

        fact = {
            "fact": "Rod wants to build a platform",
            "source_quote": "Rod wants me to build a platform",  # LLM "fixed" Rpd->Rod
            "source_convo_id": "conv-123",
        }
        # Original has typo "Rpd" not "Rod"
        convo_text = "Rpd wants me to build a platform, a software as a service platform"

        result = verify_fact(fact, convo_text)

        assert result.verified is True
        assert "fuzzy" in result.match_location.lower()

    def test_minor_word_difference_matches_via_fuzzy(self):
        """Should match when quote has minor differences."""
        from src.verifier import verify_fact

        fact = {
            "fact": "User has a Duolingo streak",
            "source_quote": "178 day streak on Duolingo",
            "source_convo_id": "conv-123",
        }
        # Original has slightly different phrasing
        convo_text = "I have a 178-day streak on Duolingo now"

        result = verify_fact(fact, convo_text)

        assert result.verified is True

    def test_completely_different_quote_still_fails(self):
        """Fuzzy matching should NOT match completely different text."""
        from src.verifier import verify_fact

        fact = {
            "fact": "User likes cats",
            "source_quote": "I love my three cats",
            "source_convo_id": "conv-123",
        }
        convo_text = "I have two dogs that I walk every day"

        result = verify_fact(fact, convo_text)

        assert result.verified is False

    def test_fuzzy_can_be_disabled(self):
        """Should be able to disable fuzzy matching."""
        from src.verifier import verify_fact

        fact = {
            "fact": "Rod wants to build",
            "source_quote": "Rod wants me to build",
            "source_convo_id": "conv-123",
        }
        convo_text = "Rpd wants me to build a platform"

        # With fuzzy disabled, typo should cause failure
        result = verify_fact(fact, convo_text, use_fuzzy=False)

        assert result.verified is False
