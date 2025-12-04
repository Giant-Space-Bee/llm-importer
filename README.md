# LLM Importer

Import your AI memories and context from ChatGPT to Claude and other LLMs.

## The Problem

You've spent months/years building a relationship with your AI. It knows your preferences, your projects, your communication style. But switching to a new LLM means starting from scratch.

**LLM Importer solves this.**

## How It Works

1. **Export** your data from ChatGPT or Claude
2. **Parse** conversations and extract meaningful facts with LLM
3. **Verify** extracted facts against source (catches hallucinations)
4. **Generate** a categorized memory profile

## Features

- [x] Parse ChatGPT `conversations.json` export (tree structure)
- [x] Parse Claude export (flat message arrays)
- [x] Auto-detect export type
- [x] LLM-powered fact extraction with structured outputs
- [x] Hallucination detection via source quote verification
- [x] Unicode normalization for quote matching
- [x] Checkpointing (resume interrupted runs)
- [x] Local LLM support (LM Studio) - sequential processing
- [x] API support (Claude Sonnet) - parallel with rate limiting
- [ ] Semantic deduplication (Stage 8)
- [ ] Final profile distillation (Stage 9)

## Pipeline

```
conversations.json → Parse → Chunk → Extract(LLM) → Verify → Aggregate → Dedup(LLM) → Distill(LLM) → Output
```

## Tech Stack

- Python 3.12+
- Local LLM via LM Studio (Hermes 4 70B tested)
- Claude API with structured outputs
- No vector DBs, no Mem0 - just LLMs + string matching

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run with local LLM (LM Studio must be running)
python -m src.main export_data/your-export/conversations.json

# Run tests
pytest tests/ -v
```

## Status

**Active Development** - Core pipeline complete (Stages 1-7), semantic deduplication in progress.

- 186 tests passing
- 11 source modules
- Supports both ChatGPT and Claude exports

## License

MIT
