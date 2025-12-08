# Task: Refactor phases.py → phases/ Package

## What You're Doing

**Read `docs/refactor-phases-plan.md` first** — it has the full plan with rationale.

You're splitting a 766-line god file (`src/cli/phases.py`) into a proper package structure. This is a pure refactor — no behavior changes.

## Quick Summary

**From:**
```
src/cli/phases.py  (766 lines, 8 phases + 2 helpers crammed together)
```

**To:**
```
src/cli/phases/
├── __init__.py          # Re-exports (backward compatible)
├── _checkpoint.py       # Shared checkpoint builder (DRY)
├── parse.py
├── provider.py
├── chunk.py
├── resume.py
├── extract.py           # Biggest one - includes helpers
├── aggregate.py
├── deduplicate.py
├── distill.py
└── output.py
```

## Why This Matters

1. **DRY:** Checkpoint-saving logic is duplicated 4 times — extract it once
2. **Isolation:** Test/change one phase without touching others
3. **Findability:** `phase_parse` → `parse.py` (matches mental model)

## Execution Order

The plan has 14 commits. Key sequence:

1. Create `phases/` package structure (no behavior change)
2. Extract `_checkpoint.py` helper (DRY the 4 duplications)
3. Move phases one at a time: `aggregate` → `output` → `resume` → `provider` → `chunk` → `parse` → `deduplicate` → `distill` → `extract`
4. Update `src/cli/__init__.py` exports
5. Delete old `phases.py`
6. Update docs

## Rules

- **Run tests after every move** — all must pass
- **One commit per phase move** — easy rollback
- **Zero API changes** — existing imports must still work
- **Preserve docstrings and type hints** — don't simplify

## Tests

```bash
python -m pytest tests/ -v
```

## When Done

1. All tests pass
2. `src/cli/phases.py` is deleted
3. Commit history is clean (14 small commits, not 1 giant one)
4. Let user know — they'll review and merge
