# Stage 8: Deduplicator Design

> **Status:** Planned (not yet implemented)
> **Dec 2025** — Use modern Python 3.12+ patterns. When unsure, look it up.

## What It Does

Takes aggregated facts (with frequency counts) and uses an LLM to merge semantically similar facts into canonical versions.

**Input:** `List[AggregatedFact]` from Stage 7 (aggregator)
**Output:** `List[DeduplicatedFact]` for Stage 9 (distiller)

## Dedup Rules (Priority Order)

1. **More recent > older** — use `source_timestamp`
2. **More specific > vague** — "Victoria, BC, Canada" beats "Canada"
3. **Higher frequency > rare** — facts appearing multiple times are stronger signals
4. **Explicit statement > inference** — direct quotes beat implied facts

## Architecture: Two-Phase Hybrid

Based on real data analysis (226 verified facts from sample):
- Most duplicates are within-category (confirmed)
- Largest category (professional) had 104 facts — too big for one prompt
- Categories are reasonably consistent, but cross-category duplicates exist

### Phase 1: Within-Category Dedup (Timestamp-Batched)

For each category:
1. Sort facts by `source_timestamp` (oldest → newest)
2. Split into batches of `max_batch` (default 50)
3. Dedup each batch via LLM
4. If multiple batches: merge-sort style combine and dedup

**Why timestamp ordering?** Newer facts naturally compared against older → "prefer newer" rule built-in.

```
professional (104 facts)
├─ Sort by timestamp
├─ Split: [batch1: 50 oldest] [batch2: 50 middle] [batch3: 4 newest]
├─ Dedup each: [~30] [~30] [~4]
├─ Merge b1+b2: [60] → dedup → [~40]
└─ Merge +b3: [44] → dedup → [~35 final]

personal (46) ─── single batch → dedup → ~25
family (27) ───── single batch → dedup → ~15
interests (26) ── single batch → dedup → ~15
preferences (14) ─ single batch → dedup → ~10
personality (9) ── single batch → dedup → ~8

Phase 1 output: ~108 facts (down from 226)
Categories can run IN PARALLEL
```

### Phase 2: Cross-Category Merge-Sort

Takes all facts from Phase 1, merge-sort style dedup:
1. If facts > max_batch: split in half
2. Recursively dedup each half
3. Merge and dedup final result

**Catches:** "Haley" in family vs personal, location facts across categories
**LLM can reassign categories** if it finds miscategorization

```
~108 facts from Phase 1
├─ Split: [54] [54]
├─ Dedup each: [~40] [~40]
└─ Merge: [80] → dedup → [~60 final]
```

## Data Structures

```python
@dataclass
class DeduplicatedFact:
    fact: str      # The canonical fact text
    category: str  # LLM-assigned category (one of 6)
```

## Function Signatures

```python
def deduplicate(
    facts: List[AggregatedFact],
    provider: LLMProvider,
    max_batch: int = 50
) -> List[DeduplicatedFact]:
    """
    Main entry point. Two-phase dedup.

    Phase 1: Within-category dedup (parallel by category)
    Phase 2: Cross-category merge-sort
    """

def deduplicate_category(
    facts: List[AggregatedFact],
    category: str,
    provider: LLMProvider,
    max_batch: int = 50
) -> List[DeduplicatedFact]:
    """
    Dedup within one category using timestamp-based batching.
    Merge-sort style if multiple batches needed.
    """

def deduplicate_merge_sort(
    facts: List[DeduplicatedFact],
    provider: LLMProvider,
    max_batch: int = 50
) -> List[DeduplicatedFact]:
    """
    Cross-category merge-sort dedup.
    Recursive split-dedup-merge.
    """

def dedup_batch(
    facts: List[AggregatedFact] | List[DeduplicatedFact],
    provider: LLMProvider
) -> List[DeduplicatedFact]:
    """
    Single LLM call to dedup a batch of facts.
    Batch must be <= max_batch size.
    """

def build_dedup_prompt(facts: List) -> str:
    """Build prompt from template + facts as JSON."""
```

## Schema for Structured Output

```python
DEDUP_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": ["personal", "professional", "family",
                                 "preferences", "interests", "personality"]
                    }
                },
                "required": ["fact", "category"],
                "additionalProperties": False
            }
        }
    },
    "required": ["facts"],
    "additionalProperties": False
}
```

## LLM Call Estimates

| Dataset Size | Phase 1 Calls | Phase 2 Calls | Total |
|--------------|---------------|---------------|-------|
| 226 facts (sample) | ~10 | ~4 | ~14 |
| 500 facts | ~15 | ~6 | ~21 |
| 1000 facts | ~25-30 | ~8-10 | ~35-40 |

Phase 1 categories can run in parallel → wall-clock time much lower.

## Test Cases

```python
class TestDeduplicate:
    def test_single_fact_passes_through(self):
        """1 fact in = 1 fact out (reformatted)."""

    def test_exact_duplicates_merged(self):
        """Identical facts → one canonical version."""

    def test_semantic_duplicates_merged(self):
        """'Lives in Seattle' + 'Resides in Seattle' → one fact."""

    def test_cross_category_duplicates_merged(self):
        """Same fact in 'family' and 'personal' → merged in Phase 2."""

    def test_specific_beats_vague(self):
        """'Seattle, WA' kept over 'USA'."""

    def test_newer_beats_older(self):
        """Contradictions resolved by timestamp."""

    def test_higher_frequency_preferred(self):
        """frequency=5 beats frequency=1."""

    def test_preserves_unique_facts(self):
        """Different facts all kept."""

    def test_empty_input_returns_empty(self):
        """[] → []"""

    def test_large_category_batches_correctly(self):
        """104 facts → split into batches, merge-sort."""

    def test_phase2_catches_cross_category(self):
        """'Haley' in family + personal → one canonical."""

class TestDeduplicateCategory:
    def test_sorts_by_timestamp(self):
        """Facts processed oldest → newest."""

    def test_single_batch_no_merge(self):
        """< max_batch → one LLM call."""

    def test_multiple_batches_merge_sort(self):
        """> max_batch → split, dedup, merge."""

class TestDeduplicateMergeSort:
    def test_recursive_splitting(self):
        """Large input recursively split."""

    def test_base_case_single_batch(self):
        """<= max_batch → single dedup call."""

class TestBuildDedupPrompt:
    def test_includes_frequency(self):
        """Prompt shows frequency for each fact."""

    def test_includes_timestamp(self):
        """Prompt shows source_timestamp."""

    def test_includes_category_as_hint(self):
        """Original category shown but not enforced."""
```

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/deduplicator.py` | Replace stub with full implementation |
| `prompts/dedup.txt` | Update for two-phase approach |
| `tests/test_stage8_deduplicator.py` | New test file |
| `TODO.md` | Mark Stage 8 complete when done |

## Implementation Order (TDD)

1. `git checkout -b stage8-deduplicator`
2. Write tests (mock LLM provider)
3. Confirm tests fail
4. Implement `DeduplicatedFact` dataclass + `DEDUP_SCHEMA`
5. Implement `build_dedup_prompt()`
6. Implement `dedup_batch()` (single LLM call)
7. Implement `deduplicate_category()` with merge-sort
8. Implement `deduplicate_merge_sort()` for Phase 2
9. Implement main `deduplicate()` orchestrator
10. Run tests, confirm pass
11. Update `TODO.md`
12. Commit & merge

## Edge Cases

- **Empty input:** Return `[]`
- **Single fact:** Pass through (reformat to `DeduplicatedFact`)
- **All unique:** All returned
- **All duplicates:** Single fact returned
- **Large category:** Timestamp-batched merge-sort handles it
- **LLM error:** Exception bubbles up (caller handles)

## Future Work: Embedding Optimization

When ready, add embedding pre-clustering to Phase 2:
1. Generate embeddings for all Phase 1 output facts
2. Cluster by cosine similarity (threshold ~0.85)
3. Only send clusters to LLM for merge decision
4. Isolated facts (no similar matches) pass through unchanged
5. Reduces Phase 2 calls from ~10 to ~3

**Dependencies needed:** sentence-transformers or OpenAI ada embeddings
