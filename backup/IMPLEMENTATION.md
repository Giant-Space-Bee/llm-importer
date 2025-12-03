# LLM Importer - Full Implementation Plan

> **Status**: Ready to build
> **Last Updated**: 2025-12-02 23:02 PST

## Vision (Landon's Words)
> "Divide and conquer the huge conversation history and have parallel LLMs extract valuable information, then aggregate it and deduplicate then compress/distill and save as strings for client memories"

---

## Data Analysis Summary
- **450 conversations** over 352 days (Dec 2024 - Dec 2025)
- **17M characters** (~4.3M tokens, ~6,830 pages)
- **6,208 user messages** (the extraction gold)
- **367 user_editable_context blocks** (instant profile baseline - ChatGPT custom instructions)
- Conversation sizes: 2-1056 messages (median: 13)
- File: `conversations.json` (55MB)

---

## Architecture: The Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 0: FREE WINS                                             │
│  user_editable_context → Instant profile baseline               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: PARSE & FLATTEN                                       │
│  Tree structure → Linear conversations → User messages          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: CHUNK                                                 │
│  450 convos → ~65 chunks of 65536 tokens (configurable)         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3: EXTRACT (LLM)                                         │
│  Local: sequential | API: parallel with rate limits             │
│  → facts[] with source_quote for each                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 4: VERIFY (no LLM)                                       │
│  String-match source_quote → original conversation              │
│  No match = hallucination = discard                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 5: AGGREGATE (no LLM)                                    │
│  Concat verified facts, count frequency                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 6: DEDUPLICATE (LLM)                                     │
│  Semantic dedup: "Lives in Victoria" = "Victoria BC resident"   │
│  Prefer: newer > older, specific > vague, frequent > rare       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 7: DISTILL (LLM)                                         │
│  Compress to final profile                                      │
│  → memory-profile.md + memory-profile.json                      │
└─────────────────────────────────────────────────────────────────┘
```

**LLM steps:** Extract (3), Dedup (6), Distill (7) - all use BatchedLLMTask
**Pure code steps:** Parse (1), Chunk (2), Verify (4), Aggregate (5)

---

## Fact Schema

```json
{
  "fact": "Lives in Victoria, BC, Canada",
  "category": "personal",           // personal|professional|family|preferences|interests|personality
  "source_convo_id": "uuid-xxx",    // which conversation
  "source_timestamp": 1764694069,   // when (for "prefer newer" in dedup)
  "source_quote": "I live in Victoria BC Canada"  // VERBATIM text from conversation
}
```

**Why `source_quote`?** Hallucination guard. String-match back to original conversation. No match = hallucinated = discard. No LLM needed for verification.

---

## Core Pattern: BatchedLLMTask

```python
class BatchedLLMTask:
    """Universal LLM task runner with auto-batching. Merge-sort for LLM work."""

    def __init__(self, llm, max_tokens=65536, warn_depth=5, retries=3):
        self.llm = llm
        self.max_tokens = max_tokens
        self.warn_depth = warn_depth  # Warn user at intervals, don't break
        self.retries = retries

    def run(self, items, prompt, depth=0):
        """Run task, auto-batch if too big, recursively merge."""

        # Warn at warn_depth, 2x warn_depth, 3x warn_depth, etc.
        if depth > 0 and depth % self.warn_depth == 0:
            print(f"⚠️  Recursion depth {depth} - this is a large job")
            if not self._prompt_continue():
                raise UserCancelledError("User cancelled at depth warning")

        # Fits in one call? Do it.
        if self._fits(items):
            return self._call_with_retry(prompt + self._format(items))

        # Too big: split → process each → merge results
        batches = self._split(items)
        results = [self.run(batch, prompt, depth+1) for batch in batches]

        # Combined results might ALSO be too big - recurse!
        return self.run(self._flatten(results), prompt, depth+1)

    def _prompt_continue(self):
        """Ask user if they want to continue."""
        response = input("Continue? [Y/n]: ").strip().lower()
        return response in ('', 'y', 'yes')

    def _call_with_retry(self, prompt):
        """Call LLM with exponential backoff retry."""
        for attempt in range(self.retries):
            try:
                return self.llm.complete(prompt)
            except Exception as e:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)  # 1s, 2s, 4s
```

**How it works:**
- Small job → 1 LLM call
- Medium job → N batches → 1 merge call
- Huge job → N batches → merge → still too big → batch again → merge again → done
- `warn_depth=5` warns user at depth 5, 10, 15... (never breaks, user decides)

---

## Provider Architecture

```python
class LLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str: ...

    @property
    @abstractmethod
    def is_local(self) -> bool: ...  # Determines sequential vs parallel

class LocalProvider(LLMProvider):
    """LM Studio / any OpenAI-compatible local server"""
    def __init__(self, base_url: str = "http://localhost:1234/v1"): ...
    is_local = True  # → Sequential execution

class APIProvider(LLMProvider):
    """Claude, OpenAI, etc."""
    def __init__(self, base_url: str, api_key: str, rpm: int = None, tpm: int = None): ...
    is_local = False  # → Parallel execution with rate limiting
```

---

## CLI Design

```
$ python -m llm_importer

╔═══════════════════════════════════════════════════════════════╗
║                    🦊 LLM IMPORTER                            ║
║              Extract your AI memories                          ║
╚═══════════════════════════════════════════════════════════════╝

[1] Input file: conversations.json (450 convos, 17M chars)

[2] Extraction depth:
    ○ Quick   - User messages only
    ● Medium  - User + conversation context
    ○ Full    - Complete conversations

[3] LLM Provider:
    ● Local (http://localhost:1234/v1)
    ○ API (enter base URL + key)

    [If API selected]:
    → Base URL: _______________
    → API Key: _______________
    → RPM limit (optional): ___ → auto-calculates parallel workers
    → TPM limit (optional): ___

[4] Chunk size: [65536] tokens (default 2^16)

[5] Output format:
    ☑ Markdown (memory-profile.md)
    ☑ JSON (memory-profile.json)

[Enter] Start  |  [C] Compare depths  |  [Q] Quit
```

**Execution modes:**
- **Local** → Sequential (your machine can only run 1 LLM at a time)
- **API** → Parallel, respects RPM/TPM limits if provided

---

## Final Output Formats

### memory-profile.md (human-readable)
```markdown
# Memory Profile for Landon

## Personal
- Lives in Victoria, BC, Canada
- Age 35
- Drives a 2015 Nissan Leaf SL

## Family
- Girlfriend: Haley
- Daughter: Lily (born Feb 5, 2025)
- Dogs: Kit (14, looks like a fox) and Jupiter (2, giant Jack Russell)

## Professional
- Head of AI at EvolveWell (LA startup, remote)
- Freelance AI architect at Coding Fox Corp (Nov 2025)

## Preferences
- Prefers concise responses without fluff
...
```

### memory-profile.json (for importing into AI systems)
```json
{
  "name": "Landon",
  "generated": "2025-12-02T23:00:00",
  "source": "ChatGPT export (450 conversations)",
  "categories": {
    "personal": ["Lives in Victoria, BC, Canada", "Age 35", ...],
    "family": ["Girlfriend: Haley", "Daughter: Lily (born Feb 5, 2025)", ...],
    "professional": [...],
    "preferences": [...],
    "interests": [...],
    "personality": [...]
  }
}
```

**Use case:** Copy-paste `categories` into Claude memory, system prompt, or any AI's context.

---

## File Structure

```
llm-importer/
├── conversations.json      # Input (already here, 55MB)
├── PLAN.md                 # Vision doc
├── IMPLEMENTATION.md       # THIS FILE - full technical plan
├── docs/
│   └── chatgpt-export-format.md  # ChatGPT JSON structure reference
├── src/
│   ├── __init__.py
│   ├── core.py            # BatchedLLMTask + token counting
│   ├── providers.py       # LLMProvider, LocalProvider, APIProvider
│   ├── parser.py          # Parse & extract from JSON
│   ├── chunker.py         # Smart batching
│   ├── extractor.py       # LLM extraction (uses BatchedLLMTask)
│   ├── verifier.py        # Kill hallucinations (pure string matching)
│   ├── aggregator.py      # Combine facts (pure code)
│   ├── deduplicator.py    # Remove duplicates (uses BatchedLLMTask)
│   ├── distiller.py       # Final compression (uses BatchedLLMTask)
│   └── main.py            # CLI entry point
├── prompts/
│   ├── extraction.txt     # Prompt for fact extraction
│   ├── dedup.txt          # Prompt for deduplication
│   └── distill.txt        # Prompt for final compression
├── output/
│   ├── user-profile.json  # Extracted from user_editable_context (free win)
│   ├── checkpoints/       # Resume from failure
│   ├── extracted/         # Raw extraction results per chunk
│   ├── verified/          # After hallucination check
│   ├── memory-profile.md  # FINAL OUTPUT (markdown)
│   └── memory-profile.json # FINAL OUTPUT (structured)
└── requirements.txt
```

---

## Build Order

1. `core.py` + `providers.py` - foundation (BatchedLLMTask, LLM providers)
2. `parser.py` - get data out of JSON
3. `chunker.py` - split for processing
4. `extractor.py` - first LLM step (test with 1 chunk!)
5. `verifier.py` - hallucination detection
6. `aggregator.py` - simple concat
7. `deduplicator.py` - LLM dedup
8. `distiller.py` - LLM final pass
9. `main.py` - wire it all together with CLI
10. End-to-end test on small subset (5 convos)
11. Full run

---

## Edge Cases to Handle

| Edge Case | Solution |
|-----------|----------|
| No `user_editable_context` in export | Skip Phase 0, proceed with extraction |
| Conversation with 0 user messages | Skip it |
| LLM returns invalid JSON | Retry with "please return valid JSON" |
| LLM returns empty response | Retry, then skip chunk and log |
| Export from different ChatGPT version | Parser handles gracefully (check fields exist) |
| Huge single conversation (>65536 tokens) | Split at message boundaries |

---

## Deferred to v2

- **Compare depths mode** - run all 3 extraction depths and diff results
- **Claude export support** - different format, needs new parser
- **Streaming progress to web UI** - terminal-only for v1
- **Incremental updates** - re-run on new export, only process new convos

---

## Key Decisions Made

| Decision | Choice | Why |
|----------|--------|-----|
| Chunk size | 65536 (2^16) | Fits in 128K context with room for prompt + output |
| Local execution | Sequential | One LLM = all your RAM/GPU |
| API execution | Parallel with RPM/TPM | Respect rate limits |
| Dedup timing | AFTER aggregate | Need full picture for semantic matching |
| Hallucination check | source_quote string match | No LLM needed, pure text search |
| Max depth | warn_depth (warns, doesn't break) | Large exports shouldn't crash |
| Output | Both md + json | Human review + programmatic use |

---

## Summary

```
INPUT: conversations.json (ChatGPT export)
                ↓
         [Parser] → user_profile baseline + flattened messages
                ↓
         [Chunker] → ~65 chunks of 65536 tokens
                ↓
         [Extractor] → LLM extracts facts[] + source_quote each     ← LLM
                ↓
         [Verifier] → string-match quotes, kill hallucinations      ← NO LLM
                ↓
         [Aggregator] → concat verified facts, count frequency      ← NO LLM
                ↓
         [Deduplicator] → LLM semantic dedup, resolve conflicts     ← LLM
                ↓
         [Distiller] → LLM compress to final profile                ← LLM
                ↓
OUTPUT: memory-profile.md + memory-profile.json
```

**Core insights:**
1. `BatchedLLMTask` handles all LLM work with auto-batching and retries
2. `source_quote` enables hallucination detection without LLM (string match)
3. Final output = simple string arrays by category, ready to paste into any AI
