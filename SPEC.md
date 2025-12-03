# LLM Importer Spec

> Extract memories from ChatGPT exports → portable profile for any AI

**Core insight:** "Divide and conquer the huge conversation history with parallel LLMs, aggregate, deduplicate, compress/distill to strings"

---

## Data (conversations.json)

- 450 convos, 17M chars (~4.3M tokens), 6,208 user messages
- 367 `user_editable_context` blocks (instant baseline)
- File: 55MB

---

## Pipeline

```
conversations.json
  → [Parse] user_profile + flattened messages
  → [Chunk] ~65 batches @ 65536 tokens
  → [Extract] LLM → facts[] with source_quote          ← LLM
  → [Verify] string-match quotes, kill hallucinations  ← no LLM
  → [Aggregate] concat verified facts, count freq      ← no LLM
  → [Dedupe] LLM semantic dedup, resolve conflicts     ← LLM
  → [Distill] LLM compress to final profile            ← LLM
  → memory-profile.md + memory-profile.json
```

---

## Fact Schema

```json
{"fact": "Lives in Victoria, BC", "category": "personal|professional|family|preferences|interests|personality",
 "source_convo_id": "uuid", "source_timestamp": 1764694069, "source_quote": "I live in Victoria BC"}
```

`source_quote` = verbatim text for hallucination check (string match, no LLM)

---

## BatchedLLMTask Pattern

Auto-batching merge-sort for LLM work:
- Small job → 1 call
- Big job → split → process batches → merge results (recurse if still too big)
- `warn_depth=5` warns at 5,10,15... (never breaks)
- Retries with exponential backoff

---

## Providers

| Type | Execution | Config |
|------|-----------|--------|
| Local | Sequential | `http://localhost:1234/v1` |
| API | Parallel | base_url + api_key + RPM/TPM limits |

---

## Key Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Chunk size | 65536 (2^16) | Fits 128K context |
| Dedup timing | After aggregate | Need full picture |
| Hallucination check | source_quote string match | No LLM needed |
| Output | md + json | Human + programmatic |
| No vector DBs/Mem0 | Just LLMs + strings | Simple > complex |

---

## CLI

```
$ python -m llm_importer
[1] Input: conversations.json (450 convos)
[2] Depth: Quick|Medium|Full
[3] Provider: Local|API (if API: url, key, RPM, TPM)
[4] Chunk size: [65536]
[5] Output: md, json
```

---

## Output Format

**memory-profile.json:**
```json
{"name": "...", "generated": "...", "source": "ChatGPT export (450 convos)",
 "categories": {"personal": [...], "family": [...], "professional": [...], "preferences": [...], "interests": [...], "personality": [...]}}
```

---

## ChatGPT JSON Structure

**Conversation:** `{id, title, create_time, update_time, mapping, current_node, default_model_slug, ...}`

**mapping:** Tree of nodes supporting branches (edits/regenerations)
```
{node_id: {id, message, parent, children[]}}
```

**message:** `{id, author: {role: user|assistant|system|tool}, content, create_time, status, metadata}`

**Content types:**
| Type | Structure |
|------|-----------|
| `text` | `{content_type: "text", parts: ["..."]}` |
| `user_editable_context` | `{content_type: "user_editable_context", user_profile: "...", user_instructions: "..."}` |
| `multimodal_text` | `{parts: [{content_type: "image_asset_pointer", asset_pointer, width, height}, "text"]}` |
| `code` | `{content_type: "code", language, text}` |
| `execution_output` | `{content_type: "execution_output", text}` |
| `tether_quote` | `{url, domain, text, title}` |
| `thoughts` | `{content_type: "thoughts", parts: [...]}` (o1/thinking) |

**Hidden messages:** `metadata.is_visually_hidden_from_conversation: true`

**Typical sequence:** root(null) → system(hidden) → user_editable_context(hidden) → user → assistant → ...

---

## File Structure

```
src/
  core.py          # BatchedLLMTask + token counting
  providers.py     # LLMProvider, LocalProvider, APIProvider
  parser.py        # JSON parsing, tree flattening
  chunker.py       # Smart batching
  extractor.py     # LLM extraction
  verifier.py      # Hallucination detection (string match)
  aggregator.py    # Combine facts
  deduplicator.py  # LLM semantic dedup
  distiller.py     # LLM final compression
  main.py          # CLI
prompts/
  extraction.txt, dedup.txt, distill.txt
output/
  checkpoints/, extracted/, verified/, memory-profile.*
```

---

## Edge Cases

| Case | Handle |
|------|--------|
| No user_editable_context | Skip Phase 0 |
| 0 user messages in convo | Skip convo |
| Invalid JSON from LLM | Retry with "return valid JSON" |
| Empty LLM response | Retry, then skip + log |
| Huge single convo (>65536 tokens) | Split at message boundaries |

---

## v2 (Deferred)

- Compare depths mode
- Claude export support
- Web UI streaming
- Incremental updates (only new convos)
