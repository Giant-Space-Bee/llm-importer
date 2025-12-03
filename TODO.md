# Build Tasks

## Done
- [x] Document ChatGPT format, architecture, fact schema, CLI design, BatchedLLMTask pattern

## Build Order

| # | Module | Status | Key Functions |
|---|--------|--------|---------------|
| 1 | core.py | pending | BatchedLLMTask, token counting (tiktoken) |
| 2 | providers.py | pending | LLMProvider ABC, LocalProvider, APIProvider (RPM/TPM) |
| 3 | parser.py | pending | load_conversations, extract_user_profile, flatten_tree, filter_user_messages |
| 4 | chunker.py | pending | batch to 65536 tokens, keep convos intact, split huge ones |
| 5 | extractor.py | pending | BatchedLLMTask, checkpointing, output fact schema |
| 6 | verifier.py | pending | string match source_quote → original, discard misses |
| 7 | aggregator.py | pending | concat verified facts, count frequency |
| 8 | deduplicator.py | pending | BatchedLLMTask, semantic dedup, prefer newer/specific/frequent |
| 9 | distiller.py | pending | BatchedLLMTask → memory-profile.md + .json |
| 10 | main.py | pending | CLI menu, provider selection, progress display |
| 11 | test | pending | e2e with 5 convos, then full 450 |
