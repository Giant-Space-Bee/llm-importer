# LLM Importer

Extract memories from ChatGPT export → `memory-profile.md` + `.json` for new AI.
No vector DBs, no Mem0, no KV stores. Just LLMs + string matching.

## Pipeline
```
conversations.json → Parse → Chunk(65536) → Extract(LLM) → Verify → Aggregate → Dedup(LLM) → Distill(LLM) → Output
```

| Phase | LLM? | What |
|-------|------|------|
| 0 Free wins | No | Pull `user_editable_context` as baseline profile |
| 1 Parse | No | Tree → linear messages, filter to user messages |
| 2 Chunk | No | Split to 65536 token batches, keep convos intact |
| 3 Extract | Yes | Get facts[] with source_quote each |
| 4 Verify | No | String-match source_quote → original; no match = discard |
| 5 Aggregate | No | Concat verified facts, count frequency |
| 6 Dedup | Yes | Semantic dedup; prefer newer > specific > frequent |
| 7 Distill | Yes | Compress to final categorized profile |

## Export Formats

### ChatGPT Export
Single `conversations.json` file.

**Conversation:**
| Field | Type | Notes |
|-------|------|-------|
| id | UUID | Primary key |
| title | string | |
| create_time / update_time | float | Unix timestamp |
| mapping | object | Tree of messages (supports branching) |
| current_node | UUID | Active branch tip |

**mapping:** `{ "uuid": { id, message, parent, children[] } }` — root has `message: null`

**Message:**
| Field | Notes |
|-------|-------|
| author.role | `user` / `assistant` / `system` / `tool` |
| author.name | Tool name if role=tool |
| content.content_type | See below |
| metadata.is_visually_hidden_from_conversation | Hidden from UI |

**Content types:**
| Type | Structure |
|------|-----------|
| text | `{ parts: ["text"] }` |
| user_editable_context | `{ user_profile, user_instructions }` — custom instructions, ~80% of convos, always hidden |
| multimodal_text | `{ parts: [image_asset_pointer, "text"] }` |
| code | `{ language, text }` |
| execution_output | `{ text }` — from tool |

**Free wins:** `user_editable_context` only (custom instructions, no memories)

### Claude Export
Folder `data-{timestamp}-batch-{N}/` with 4 files:
- `conversations.json` — flat message arrays
- `memories.json` — **rich memories!** conversation + project memories
- `projects.json` — project definitions with docs
- `users.json` — user profile

| Field | Type | Notes |
|-------|------|-------|
| uuid | UUID | Primary key (not `id`) |
| name | string | Title (not `title`) |
| created_at / updated_at | ISO 8601 | String timestamps (not float) |
| chat_messages | array | Flat list (not tree) |

**Message:** `sender` = `human`/`assistant`, `content[]` array with `type` = `text`/`thinking`

**Free wins:** `memories.json` contains `conversations_memory` + `project_memories` — much richer than ChatGPT!

### Auto-Detection Logic
```
has "mapping" field → ChatGPT
has "uuid" + "chat_messages" → Claude
otherwise → unknown
```

## Key Patterns

**BatchedLLMTask:**
- Fits in one call? Do it
- Too big? Split → process each → merge results
- Merged results too big? Recurse
- Like merge-sort for LLM work

**warn_depth (not max_depth):**
- Warns user at depth 5, 10, 15... (intervals of warn_depth)
- Never breaks automatically — user decides to continue or cancel
- Large exports shouldn't crash

**source_quote:**
- Every extracted fact includes verbatim quote from original
- Verify via string match (no LLM needed)
- No match = hallucinated = discard
- **Verifier scope:** Search ALL messages in conversation (not just user messages) — quotes might reference assistant context the user was responding to
- **Unicode normalization:** Normalize curly quotes → straight quotes before matching (LLM may return different apostrophe characters)

**Execution modes:**
- Local provider (LM Studio) → sequential (one LLM = all resources)
- API provider → parallel with RPM/TPM rate limiting

**Provider selection & orchestration (Stage 6):**
- CLI prompts user to choose: local LLM or API
- If local: `provider.is_local = True` → process chunks sequentially, one at a time
- If API: `provider.is_local = False` → process chunks in parallel, respecting `rpm`/`tpm` limits
- Rate limiting: track requests/tokens per minute, sleep when limits approached
- Checkpointing: save progress after each chunk so crashes don't lose work

**Chunk size:** 65536 (2^16) — fits in 128K context with room for prompt + output

**Dedup timing:** After aggregate, not during extraction — need full picture for semantic matching

**No hard timeouts:** Warn + pause between batches. User approves continuation.

## Fact Schema
```
{ fact, category, source_convo_id, source_timestamp, source_quote }
```
Categories: personal, professional, family, preferences, interests, personality

## Output
- `memory-profile.md` — human readable, categorized
- `memory-profile.json` — `{ categories: { personal: [...], ... } }` for importing

## Build Order
1. core.py + providers.py (BatchedLLMTask, token counting, providers)
2. parser.py (load JSON, extract profile, flatten tree, filter user msgs)
3. chunker.py (batch to token limit, keep convos intact)
4. extractor.py (LLM extraction with checkpointing)
5. verifier.py (string match source_quote)
6. aggregator.py (concat + frequency count)
7. deduplicator.py (LLM semantic dedup)
8. distiller.py (LLM final compression)
9. main.py (CLI)

## Before Building Each LLM Step

1. Probe with tiny input (1-2 messages)
2. Check: thinking tokens? structured output? actual format returned?
3. Understand before scaling
4. Small prototype first, then full module

## Stage 6 API Provider Decision

**Model**: Claude Sonnet 4 (`claude-sonnet-4-5-20250929`) - no override
- Haiku 4.5 lacks structured outputs (as of Dec 2025)
- Cost: $3/$15 per MTok (input/output)
- Structured outputs via beta header: `structured-outputs-2025-11-13`
- Rate limits (Tier 1): ~5 RPM, ~20k TPM

**API Request Format** (Anthropic SDK):
```python
from anthropic import Anthropic, transform_schema

client = Anthropic()
response = client.beta.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=8192,
    betas=["structured-outputs-2025-11-13"],
    messages=[{"role": "user", "content": prompt}],
    output_format={
        "type": "json_schema",
        "schema": transform_schema(PydanticModel)
    }
)
```

## Stage 4 LLM Findings (Hermes 4 70B via LM Studio)

**Tested 2025-12-03:**

| Finding | Implication |
|---------|-------------|
| No `<think>` tags by default | Don't need to strip them unless explicitly prompted to think |
| `json_schema` mode works | Use `response_format: {type: "json_schema", json_schema: {...}}` for guaranteed valid JSON |
| Fast on small input (~2 sec) | 70B on 61k tokens will be slow - test with smaller chunks first |
| Clean JSON output | Parser handles it fine |
| Categories not matching ours | Must enforce our category list in the schema |

**Structured output request format:**
```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "facts",
      "strict": true,
      "schema": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "fact": {"type": "string"},
            "category": {"type": "string", "enum": ["personal","professional","family","preferences","interests","personality"]},
            "source_convo_id": {"type": "string"},
            "source_timestamp": {"type": "number"},
            "source_quote": {"type": "string"}
          },
          "required": ["fact", "category", "source_convo_id", "source_timestamp", "source_quote"]
        }
      }
    }
  }
}
```

**Recommendations:**
1. Use structured outputs (`json_schema`) for guaranteed valid JSON
2. Test with smaller chunks first (4096 = 2^12) before going to 65536
3. Enforce our category enum in the schema
4. Chunk sizes show as ~61k because convos don't pack perfectly to exactly 65536

## Data Stats
450 convos, 17M chars (~4.3M tokens), 6,208 user messages, 367 user_editable_context blocks

## Stage 5 Verifier Findings (2025-12-03)

**Implementation:**
- `normalize_text()`: Handles curly quotes, em/en dashes, ellipsis, whitespace, case
- `verify_fact()`: Normalized substring match against full conversation
- `verify_all()`: Batch verification, returns (verified, discarded) lists

**Unicode normalizations:**
| From | To |
|------|-----|
| `'` `'` (U+2018, U+2019) | `'` (straight) |
| `"` `"` (U+201C, U+201D) | `"` (straight) |
| `—` (U+2014 em dash) | `-` |
| `–` (U+2013 en dash) | `-` |
| `…` (U+2026 ellipsis) | `...` |

**Integration test results:**
- 3 facts extracted from 4k chunk
- 3/3 verified (100%)
- 0 hallucinations

**Key design decisions:**
1. Verify against ALL messages (`flatten_tree`), not just user messages
2. Normalize both quote and conversation before matching
3. Log hallucinations to stderr for debugging

---

## Project Ideas

**Warn-and-pause instead of timeout:**
For local LLM calls, never use hard timeouts. A 12-min inference shouldn't die to a 10-min timeout.
Instead: after N seconds of silence, print warning ("this is taking a while..."), pause between batches, user approves continuation.
Generalizable to any long-running local LLM work.
