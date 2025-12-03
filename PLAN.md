# LLM Importer - Project Plan

**Created**: December 2, 2025
**Author**: Landon Brown

---

## Project Vision

**LLM Importer** helps people switch between AI assistants (ChatGPT → Claude, etc.) by extracting memories and context from their conversation history and making it available to their new AI.

No vendor lock-in. Your conversations = your data.

---

## The Core Insight

> "Anything they said to ChatGPT that was in the memory was also in the chat logs."

ChatGPT's "memories" didn't appear out of thin air—they came FROM your conversations. That means:

- **`conversations.json` IS the source of truth**
- No need to ask ChatGPT for its memories (we already have the raw data)
- No need for vector databases, KV stores, or third-party tools
- We just need to extract what matters

---

## Architecture

### The Simple Pipeline

```
conversations.json (huge file, 50MB+)
    ↓
1. Chunk into N batches (manageable pieces)
    ↓
2. Parallel LLM extraction
   - Send each batch to an LLM
   - Extract: facts, preferences, context, patterns
   - Each LLM processes independently
    ↓
3. Aggregate raw facts
   - Collect all extracted information
   - Merge into single dataset
    ↓
4. Deduplicate + resolve conflicts
   - Remove redundant information
   - Handle contradictions (newest wins? most frequent?)
   - Consolidate similar facts
    ↓
5. Compress/distill into profile
   - Final LLM pass: turn facts into coherent narrative
   - Prioritize: what's most useful for a new AI to know?
    ↓
6. Output: memory-profile.md
   - Simple markdown file
   - Plain strings (no embeddings, no vectors)
   - Ready to paste into Claude/other AI
```

---

## Key Principles

### 1. Simple > Complex
- One-time extraction (not ongoing sync)
- No infrastructure to maintain
- Just code + LLM API calls

### 2. No Over-Engineering
**We will NOT use:**
- Vector databases
- KV stores
- Mem0 or similar services
- Complex embeddings
- Ongoing memory sync systems

**We WILL use:**
- LLMs for extraction (what they're good at)
- Basic file I/O
- Parallel processing for speed
- Simple deduplication logic

### 3. Divide and Conquer
- Huge file? Split it up.
- Multiple chunks? Process in parallel.
- Let each LLM do what it does best: read and extract.

### 4. Trust the Process
- LLMs are excellent at finding patterns in text
- Deduplication doesn't need to be perfect—close enough works
- The goal: useful context for a new AI, not a perfect database

---

## Success Criteria

**This project succeeds when:**
1. A user can run a command on their `conversations.json`
2. Get back a clean `memory-profile.md` in minutes
3. Paste that profile into Claude (or any AI) and continue where they left off

**Failure looks like:**
- Requiring external services or databases
- Complex setup/configuration
- Over-engineered solutions to simple problems
- Lost in the weeds of "perfect" deduplication

---

## Implementation Notes

### Phase 1: Proof of Concept
- Single-threaded extraction on small sample
- Verify LLMs can extract useful info
- Test deduplication logic

### Phase 2: Scale Up
- Chunking strategy (by conversation? by token count?)
- Parallel LLM calls (how many? rate limits?)
- Aggregation approach

### Phase 3: Polish
- Handle edge cases (empty conversations, huge messages)
- Optimize for cost (token efficiency)
- CLI interface for easy use

---

## Why This Works

Your insight cuts through the complexity:

> "I dunno I just feel like we could do this ourselves and not involve KV or Vector stuff. Cant we just divide and conquer the huge conversation history and have parallel llms extract valuable information, then agreegate it and deduplicate then compress/distill and save as strings for client memories?"

**Yes. We absolutely can.**

The tech industry loves to overcomplicate things. But sometimes the simplest approach is the right one:
- LLMs read text → that's what they do best
- Parallel processing → that's what computers do best
- String output → that's what's easiest to use

No vector embeddings needed. No complex infrastructure. Just smart extraction, simple aggregation, and clean output.

---

## Next Steps

1. **Explore**: Read sample from conversations.json, understand structure
2. **Prototype**: Single-chunk extraction, see what LLM produces
3. **Validate**: Is the extracted info useful? What's missing?
4. **Scale**: Chunking + parallel extraction
5. **Polish**: Deduplication + compression logic
6. **Ship**: CLI tool anyone can run

---

*Keep it simple. Make it work. Help people own their data.*
