# LLM Importer - Build Progress

## Completed
- [x] Stage 1: CLI Shell (12 tests)
- [x] Stage 2: Parser (17 tests)
- [x] Stage 3: Chunker (15 tests)
- [x] Stage 4: Extractor (11 tests)
  - 4a: Update provider to use structured outputs (json_schema)
  - 4b: Add category enum to schema
  - 4c: Test extraction with small chunk (4096 tokens)
  - 4d: Test extraction with full chunk (65536 tokens) - verified via 4c
  - 4e: Removed dead code (`parse_extraction_response`) from extractor.py
  - 4f: Consolidated double file loading in main.py
  - 4g: Documented verifier scope decision in CLAUDE.md

**4c Results (2025-12-03):** 5 facts extracted, quotes 3-5 words, 4/5 verified. One failed due to curly apostrophe (Unicode normalization needed in verifier).

- [x] Stage 5: Verifier (22 tests)
  - 5a: Unicode normalization (curly quotes → straight, em dashes → hyphens, etc.)
  - 5b: Verify against ALL messages (not just user) via `flatten_tree`
  - Integration test: 3/3 facts verified (100%)
- [x] Stage 6: Full extraction loop (65 tests)
  - [x] 6a: Research Anthropic API - Sonnet 4, $3/$15 MTok, structured outputs beta
  - [x] 6b: APIProvider with rate limiting (24 tests) - RPM/TPM tracking, auto-wait
  - [x] 6c: CLI provider selection (7 tests) - get_provider() with local/API choice
  - [x] 6d: Export type detection (11 tests) - ChatGPT/Claude/auto-detect + coming soon
  - [x] 6e: Sequential processing for local - included in 6f processor
  - [x] 6f: Parallel processing for API (9 tests) - process_all_chunks with semaphore
  - [x] 6g: Checkpointing (14 tests) - save/resume after each chunk
  - [x] 6h: Integration test with full pipeline - 14 facts verified from 1 chunk via Claude API

- [x] Stage 6i: Claude export parsing (13 new tests)
  - [x] Auto-detect export type in parse_all() (mapping=ChatGPT, chat_messages=Claude)
  - [x] Add parse_claude() for Claude's flat message array format
  - [x] Update parse_chatgpt() (rename from current parse_all logic)
  - [x] Tests for both export types (13 tests added)
  - [x] Verifier handles both raw conversation formats

## Pending
- [ ] Stage 7: Aggregator (combine + count)
- [ ] Stage 8: Deduplicator (LLM semantic dedup)
- [ ] Stage 9: Distiller (final output)

## Test Count
147 tests passing
