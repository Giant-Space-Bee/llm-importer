# LLM Importer

Extract memories from ChatGPT/Claude exports → generate a portable memory profile for any AI.

## Quick CLI Reference

```bash
# REQUIRED: Set API key first
export ANTHROPIC_API_KEY=sk-ant-...

# Basic usage (API mode, recommended)
python -m src.main path/to/conversations.json --provider api

# Demo mode - quick test with first chunk only
python -m src.main path/to/conversations.json --demo --provider api

# Resume an interrupted run
python -m src.main path/to/conversations.json --resume --provider api

# Local LLM mode (requires LM Studio running at localhost:1234)
python -m src.main path/to/conversations.json --provider local

# Custom rate limit (tokens per minute) for API
python -m src.main path/to/conversations.json --provider api --tpm 40000
```

### CLI Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `input` | Yes | Path to `conversations.json` file |
| `--provider {local,api}` | Yes | `api` = Anthropic API (parallel), `local` = LM Studio (sequential) |
| `--demo` | No | Process only first chunk (4k tokens) for quick testing |
| `--resume` | No | Continue from checkpoint if interrupted |
| `--tpm N` | No | Tokens per minute limit for API (default: 20000) |

## Input Files

### ChatGPT Export
1. Go to ChatGPT → Settings → Data Controls → Export Data
2. Wait for email, download zip
3. Use `conversations.json` from the zip

```bash
python -m src.main ~/Downloads/chatgpt-export/conversations.json --provider api
```

### Claude Export
1. Go to Claude → Settings → Export Data
2. Download folder (named `data-YYYY-MM-DD-...`)
3. Use `conversations.json` from that folder

```bash
# The tool auto-detects Claude format and finds memories.json in same folder
python -m src.main ~/Downloads/data-2025-12-03-batch-0000/conversations.json --provider api
```

**Auto-detection:** The tool automatically detects whether input is ChatGPT or Claude format:
- ChatGPT: Has `mapping` field (tree structure)
- Claude: Has `uuid` + `chat_messages` fields (flat array)

## What Happens When You Run It

1. **Parse** - Loads conversations, extracts user messages, finds custom instructions/memories
2. **Chunk** - Splits into batches (4k demo, 8k API, 65k local)
3. **Extract** - LLM extracts facts with source quotes
4. **Verify** - String-matches quotes against originals (catches hallucinations)
5. **Aggregate** - Combines verified facts, counts frequency

Progress is checkpointed after each chunk. If interrupted, use `--resume` to continue.

## Prerequisites

```bash
# Python 3.12+ required
python --version

# Install dependencies
pip install -r requirements.txt

# Set API key (for --provider api)
export ANTHROPIC_API_KEY=sk-ant-...
```

For local LLM mode: Start [LM Studio](https://lmstudio.ai) with a model at `http://127.0.0.1:1234`

## Common Issues

**"File not found"** - Provide full path to `conversations.json`
```bash
# Wrong
python -m src.main conversations.json

# Right
python -m src.main /full/path/to/conversations.json
```

**"API key missing"** - Set the environment variable
```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...
```

**"Connection refused" (local mode)** - LM Studio not running
```bash
# Start LM Studio and load a model first, then:
python -m src.main ... --provider local
```

**Checkpoint exists warning** - Previous run was interrupted
```bash
# Continue where you left off:
python -m src.main ... --resume

# Or start fresh (checkpoint warning is just informational)
python -m src.main ...
```

## Output

Currently outputs aggregated facts to console. Final stages (dedup + distill to `memory-profile.md`) in development.

## Development

```bash
# Run tests
pytest tests/ -v

# Project structure
src/
├── main.py          # CLI entry point
├── cli/             # CLI modules (phases, display, validation)
├── parser.py        # Conversation parsing
├── chunker.py       # Token-based chunking
├── extractor.py     # LLM fact extraction
├── verifier.py      # Quote verification
├── aggregator.py    # Fact aggregation
└── providers.py     # LLM provider abstraction
```

## License

MIT
