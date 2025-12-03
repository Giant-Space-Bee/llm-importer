# Context for Claude (Read This First!)

> **Created**: 2025-12-02 23:02 PST
> **Purpose**: Everything you need to continue building this project

---

## Who is the User?

**Landon Brown**
- Freelance AI architect @ Coding Fox Corp (started Nov 2025)
- Head of AI @ EvolveWell (LA startup, remote)
- Lives in Victoria, BC, Canada
- Using Claude Code CLI + iTerm2 (new to both as of Nov 25, 2025)
- Prefers concise responses, no fluff
- Likes 2^n numbers (he's autistic and appreciates that pattern)

---

## What Are We Building?

**LLM Importer** - A tool to help people switch AI assistants by extracting memories/context from ChatGPT exports and making them available for a new AI.

**The core insight (Landon's words):**
> "Cant we just divide and conquer the huge conversation history and have parallel llms extract valuable information, then agreegate it and deduplicate then compress/distill and save as strings for client memories?"

**Yes. That's exactly what we're building.**

---

## Key Files to Read

1. **`IMPLEMENTATION.md`** - The full technical plan with all details
2. **`docs/chatgpt-export-format.md`** - ChatGPT JSON structure reference
3. **`PLAN.md`** - High-level vision (less technical)
4. **`conversations.json`** - The actual 55MB ChatGPT export to process

---

## Technical Setup

- **LLM Provider**: LM Studio on `http://localhost:1234/v1` (OpenAI-compatible API)
- **Chunk size**: 65536 tokens (2^16) - configurable
- **Local = Sequential**, API = Parallel with rate limiting

---

## Current Status

**Planning complete. Ready to build.**

Build order:
1. `core.py` + `providers.py` - foundation
2. `parser.py` - get data out of JSON
3. `chunker.py` - split for processing
4. `extractor.py` - first LLM step
5. `verifier.py` - hallucination detection
6. `aggregator.py` - concat facts
7. `deduplicator.py` - LLM dedup
8. `distiller.py` - LLM final pass
9. `main.py` - CLI
10. Test end-to-end

---

## Key Decisions (Don't Change Without Asking)

| Decision | Choice |
|----------|--------|
| No vector DBs | Just LLMs + string matching |
| No Mem0 | Rolled our own |
| source_quote | Verbatim text for hallucination check |
| warn_depth not max_depth | Warns user, doesn't crash |
| Dedup AFTER aggregate | Need full picture |
| 65536 chunk size | 2^16, Landon likes powers of 2 |

---

## How to Start

```bash
cd /Users/landonbrown/AI_Projects/llm-importer
# Read IMPLEMENTATION.md first
# Then start with core.py + providers.py
```

---

## User Preferences (from CLAUDE.md)

- Always call Time MCP before starting tasks
- Use TodoWrite for 3+ step tasks
- Read files before modifying
- Wait for "go ahead" before writing code
- Commit after each working increment
- No emojis unless asked
- Keep responses concise

---

## What Success Looks Like

User runs:
```bash
python -m llm_importer
```

Gets back:
- `memory-profile.md` - Human readable
- `memory-profile.json` - For importing into Claude/other AI

Profile contains categorized facts about the user extracted from their ChatGPT history.
