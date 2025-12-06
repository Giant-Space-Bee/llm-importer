# LLM Importer

> **Dec 2025** — Use modern Python 3.12+ patterns. No deprecated APIs. When unsure, look it up.

> **CLAUDE.md rules:** Only essential, always-needed context lives here. No stats, no historical findings, no "completed" lists. If it goes stale, it doesn't belong. Put reference material in `docs/`.

Extract memories from ChatGPT export → `memory-profile.md` + `.json` for new AI.
No vector DBs, no Mem0, no KV stores. Just LLMs + string matching.

## Quick CLI Commands

```bash
# REQUIRED: Set API key first
export ANTHROPIC_API_KEY=sk-ant-...

# Basic usage (API mode, recommended)
python -m src.main path/to/conversations.json --provider api

# Demo mode - quick test with first chunk only
python -m src.main path/to/conversations.json --demo --provider api

# Resume an interrupted run
python -m src.main path/to/conversations.json --resume --provider api

# Local LLM mode (requires LM Studio at localhost:1234)
python -m src.main path/to/conversations.json --provider local

# Custom rate limit (tokens per minute)
python -m src.main path/to/conversations.json --provider api --tpm 40000
```

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
| 8 Output | No | Write `memory-profile.md` + `.json` to `output/` |

## Export Formats

### ChatGPT Export
Single `conversations.json` file.

**Conversation:** `id`, `title`, `create_time`/`update_time` (float), `mapping` (tree), `current_node`

**mapping:** `{ "uuid": { id, message, parent, children[] } }` — root has `message: null`

**Message:** `author.role` (`user`/`assistant`/`system`/`tool`), `content.content_type`, `metadata.is_visually_hidden_from_conversation`

**Content types:** `text` (`parts[]`), `user_editable_context` (`user_profile`, `user_instructions`), `multimodal_text`, `code`, `execution_output`

**Free wins:** `user_editable_context` = custom instructions (~80% of convos, always hidden)

### Claude Export
Folder `data-{timestamp}-batch-{N}/` with: `conversations.json`, `memories.json`, `projects.json`, `users.json`

**Conversation:** `uuid` (not `id`), `name` (not `title`), `created_at`/`updated_at` (ISO 8601), `chat_messages` (flat array, not tree)

**Message:** `sender` = `human`/`assistant`, `content[]` with `type` = `text`/`thinking`

**Free wins:** `memories.json` has `conversations_memory` + `project_memories` — richer than ChatGPT!

### Auto-Detection
```
has "mapping" field → ChatGPT
has "uuid" + "chat_messages" → Claude
otherwise → unknown
```

## Key Patterns

**Trusted baseline:** Claude `memories.json` and ChatGPT `user_editable_context` are "free wins" — already curated. Pass to distiller as `trusted_context`, merge with extracted facts. Priority: Claude memories > ChatGPT profile.

**Chunk-based processing:** Chunker splits convos into 65536-token batches upfront. Sequential for local LLM, parallel for API (semaphore-based RPM/TPM). Checkpointing after each chunk.

**source_quote verification:** Every fact has verbatim quote. Verify via string match (no LLM). No match = hallucinated = discard. Searches ALL messages (not just user). Unicode-normalized (curly quotes → straight).

**warn_depth:** Warns at depth 5, 10, 15... Never breaks automatically — user decides.

**No hard timeouts:** Warn + pause between batches. User approves continuation.

**Dedup timing:** After aggregate, not during extraction — need full picture.

## Fact Schema
```
{ fact, category, source_convo_id, source_timestamp, source_quote }
```
Categories: personal, professional, family, preferences, interests, personality

## Output

After successful run, files are written to `output/`:
- `memory-profile.md` — human readable, categorized profile
- `memory-profile.json` — `{ name, generated, source, categories: {...} }` for importing

Progress checkpointed after each stage. Use `--resume` to continue interrupted runs.

## Reference Docs
- `docs/development-notes.md` — Stage findings, API formats, historical decisions
