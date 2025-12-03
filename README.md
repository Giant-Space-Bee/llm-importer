# LLM Importer

Import your AI memories and context from ChatGPT to Claude and other LLMs.

## The Problem

You've spent months/years building a relationship with your AI. It knows your preferences, your projects, your communication style. But switching to a new LLM means starting from scratch.

**LLM Importer solves this.**

## How It Works

1. **Export** your data from ChatGPT (Settings → Data Controls → Export)
2. **Parse** conversations and extract meaningful context
3. **Generate** a "memory profile" with your preferences, facts, and personality
4. **Import** into Claude, Gemini, or any LLM that accepts context

## Features (Planned)

- [ ] Parse ChatGPT `conversations.json` export
- [ ] LLM-powered fact/preference extraction
- [ ] Personality analysis (Big Five traits)
- [ ] Generate Claude-compatible memory format
- [ ] Generate system prompts for any LLM
- [ ] Privacy-first: all processing local

## Tech Stack

- Python 3.11+
- Mem0 for memory extraction
- LLM APIs (Claude/GPT-4) for intelligent parsing

## Status

🚧 **Early Development** - Research phase complete, building MVP

## License

MIT
