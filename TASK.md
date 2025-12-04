# Task: Parser Refactor + ChatGPT Trusted Baseline

## Context

You're working in a worktree (`feature/parser-refactor`) while another Claude works on the deduplicator in the main worktree.

**Read CLAUDE.md first** — it has all project context.

## Your Tasks

### Task 1: Refactor parser.py (484 lines → modular)

**Goal:** Split the monolithic `parser.py` into focused modules.

**Target structure:**
```
src/
├── parser.py              # Facade: parse_all(), detect_export_type()
└── parsers/
    ├── __init__.py
    ├── chatgpt.py         # ChatGPT-specific: flatten_tree, parse_chatgpt_conversations
    ├── claude.py          # Claude-specific: parse_claude_conversations, _parse_claude_message
    └── memories.py        # Claude memories: ClaudeMemories, load_claude_memories, format_memories_for_distiller
```

**Rules:**
- Keep `parser.py` as the public API (facade pattern)
- Move format-specific code to submodules
- Shared types (`Message`, `Conversation`, `UserProfile`) stay in `parser.py`
- All 186 tests must pass after refactor
- No behavior changes — pure refactor

**Steps:**
1. Create `src/parsers/` directory
2. Move ChatGPT parsing to `parsers/chatgpt.py`
3. Move Claude parsing to `parsers/claude.py`
4. Move memories handling to `parsers/memories.py`
5. Update imports in `parser.py` to re-export from submodules
6. Run tests, fix any import issues
7. Commit

### Task 2: ChatGPT user_editable_context as trusted baseline

**Goal:** Pass ChatGPT's custom instructions to the distiller as `trusted_context`, same pattern as Claude memories.

**Current state:**
- `UserProfile` is parsed in `parser.py:extract_user_profile()`
- It's displayed in `main.py` but NOT passed to distiller
- `distiller.py` already has `trusted_context` param (from Claude memories work)

**What to do:**
1. In `main.py`, after extracting `user_profile`:
   - Format it as prose for distiller (similar to `format_memories_for_distiller`)
   - Store as `trusted_context` if no Claude memories exist
2. Add helper `format_user_profile_for_distiller(profile: UserProfile) -> str`
3. Priority: Claude memories > ChatGPT user_profile (Claude is richer)
4. Add tests
5. Commit

## Commit Style

```
Short summary (imperative mood)

Longer explanation if needed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

## When Done

1. Ensure all 186+ tests pass
2. Commit each task separately
3. Let user know you're done — they'll merge or review
