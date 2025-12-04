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

### Prerequisites

```bash
python --version  # Must be 3.12+
pip install -r requirements.txt
```

### Get Your Export Data

- **ChatGPT:** Settings → Data Controls → Export Data → Download `conversations.json`
- **Claude:** Settings → Export Data → Download folder with `conversations.json` + `memories.json`

### Run Extraction

**Option 1: Local LLM (requires [LM Studio](https://lmstudio.ai))**

```bash
# Start LM Studio with Hermes 4 70B at http://127.0.0.1:1234
python -m src.main path/to/conversations.json --provider local
```

**Option 2: Anthropic API**

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m src.main path/to/conversations.json --provider api
```

**Option 3: Demo Mode (test with first chunk only)**

```bash
python -m src.main path/to/conversations.json --demo --provider local
```

### CLI Reference

```bash
python -m src.main --help                    # Show all options
python -m src.main input.json --demo         # Test mode (4k tokens, 1 chunk)
python -m src.main input.json --resume       # Resume interrupted run
python -m src.main input.json --provider api # Use Claude API (parallel)
```

### Run Tests

```bash
pytest tests/ -v
```

## Status

**Active Development** - Core pipeline complete (Stages 1-7), semantic deduplication in progress.

- 186 tests passing
- 11 source modules
- Supports both ChatGPT and Claude exports

## License

MIT
