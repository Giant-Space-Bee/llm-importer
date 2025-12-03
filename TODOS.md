# Build Todos

> **Status**: Ready to start building
> **Last Updated**: 2025-12-02 23:02 PST

## Completed
- [x] Document ChatGPT export format (`docs/chatgpt-export-format.md`)
- [x] Plan the architecture
- [x] Define fact schema with source_quote for hallucination detection
- [x] Design CLI interface
- [x] Define BatchedLLMTask pattern
- [x] Decide on dedup timing (after aggregate)

## Next Up

### 1. Set up project structure
- [ ] Create `src/__init__.py`
- [ ] Create `requirements.txt`
- [ ] Verify `src/`, `prompts/`, `output/` directories exist

### 2. Build core.py + providers.py (Foundation)
- [ ] Implement `BatchedLLMTask` class
- [ ] Implement token counting (use tiktoken)
- [ ] Implement `LLMProvider` ABC
- [ ] Implement `LocalProvider` (localhost:1234)
- [ ] Implement `APIProvider` (with RPM/TPM rate limiting)

### 3. Build parser.py
- [ ] `load_conversations(path)` - load JSON
- [ ] `extract_user_profile(conversations)` - get user_editable_context
- [ ] `flatten_tree(mapping)` - tree → linear messages
- [ ] `filter_user_messages(messages)` - just user content

### 4. Build chunker.py
- [ ] Smart batching to 65536 tokens (configurable)
- [ ] Keep conversations intact
- [ ] Split huge convos at message boundaries

### 5. Build extractor.py
- [ ] Use BatchedLLMTask
- [ ] Local = sequential, API = parallel
- [ ] Checkpointing for resume
- [ ] Output fact schema with source_quote

### 6. Build verifier.py
- [ ] String match source_quote → original conversation
- [ ] No match = discard + log

### 7. Build aggregator.py
- [ ] Concat all verified facts
- [ ] Count frequency

### 8. Build deduplicator.py
- [ ] Use BatchedLLMTask
- [ ] Semantic dedup
- [ ] Prefer: newer, specific, frequent

### 9. Build distiller.py
- [ ] Use BatchedLLMTask
- [ ] Output memory-profile.md
- [ ] Output memory-profile.json

### 10. Build main.py (CLI)
- [ ] Interactive menu with options
- [ ] Extraction depth selection
- [ ] Provider selection (local/API)
- [ ] Chunk size config
- [ ] Output format selection
- [ ] Progress display

### 11. Test
- [ ] End-to-end with 5 conversations
- [ ] Full run on 450 conversations

---

## Start Here

```bash
cd /Users/landonbrown/AI_Projects/llm-importer

# Read the docs first:
# - CONTEXT.md (quick overview)
# - IMPLEMENTATION.md (full technical plan)
# - docs/chatgpt-export-format.md (JSON structure)

# Then build in order, starting with core.py
```
