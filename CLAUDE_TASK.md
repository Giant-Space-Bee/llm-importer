# Claude #2 Task: Export Type Detection

You are working in a git worktree on branch `feature/export-detection`.

## Coordination
- **Claude #1** (main repo): API Provider + rate limiting + parallel processing
- **Claude #2** (you): Export type detection + "coming soon" messages
- **Claude #3** (wt3): Checkpointing system

## Your Tasks (TDD approach)

### 6d: CLI Export Type Detection
**File**: `src/parser.py` (add detection function) + `src/main.py` (CLI integration)

Detect which LLM service exported the file:
- **ChatGPT**: Has `mapping` with tree structure, `user_editable_context` content type
- **Claude**: Different structure (research what Claude exports look like)
- **Auto-detect**: Sniff the JSON structure and return detected type

**Tests first** in `tests/test_parser.py`:
```python
def test_detect_chatgpt_export():
    """Should detect ChatGPT export format."""
    pass

def test_detect_claude_export():
    """Should detect Claude export format (research needed)."""
    pass

def test_detect_unknown_export():
    """Should return 'unknown' for unrecognized format."""
    pass
```

### 6i: Coming Soon Message
**File**: `src/main.py`

When user selects an export type we don't support yet:
- Show friendly "Coming soon!" message
- List what IS supported (ChatGPT)
- Exit gracefully

**Tests first** in `tests/test_main.py`:
```python
def test_coming_soon_message_for_gemini():
    """Should show coming soon for unsupported exports."""
    pass
```

## When Done
1. Run `pytest` - all tests pass
2. Commit to your branch
3. Tell your human to merge or wait for others

## Read First
- `CLAUDE.md` - project context
- `src/parser.py` - existing parser code
- `src/main.py` - CLI structure
- `TODO.md` - overall progress

## Don't Touch
- `src/providers.py` - Claude #1 is working on this
- Checkpointing logic - Claude #3 is handling this
