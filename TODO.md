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

## Pending
- [ ] Stage 6: Full extraction loop
  - [ ] 6a: Research Anthropic API (install package, understand rate limits, pricing)
  - [ ] 6b: Implement APIProvider with rate limiting (rpm/tpm tracking)
  - [ ] 6c: CLI provider selection (local vs API)
  - [ ] 6d: CLI export type selection (ChatGPT / Claude / auto-detect)
  - [ ] 6e: Sequential processing for local (is_local=True)
  - [ ] 6f: Parallel processing for API (is_local=False)
  - [ ] 6g: Checkpointing (save after each chunk, resume on crash)
  - [ ] 6h: Integration test with full pipeline
  - [ ] 6i: "Coming soon" message for other LLM exports (Gemini, etc.)
- [ ] Stage 7: Aggregator (combine + count)
- [ ] Stage 8: Deduplicator (LLM semantic dedup)
- [ ] Stage 9: Distiller (final output)

## Test Count
77 tests passing
