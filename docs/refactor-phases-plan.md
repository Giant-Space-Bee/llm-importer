# Refactoring Plan: phases.py → phases/ Package

> **Created:** 2025-12-07
> **Status:** Completed
> **Branch:** `refactor/phases-package`

## Problem Statement

`src/cli/phases.py` is a 766-line god file that violates single responsibility:
- Contains ALL 8 pipeline phases + 2 helper functions
- 29% of the entire CLI package (766/2634 lines)
- High import surface: 29 imports from 14 modules
- Inconsistent abstraction levels: some phases delegate, others inline substantial logic
- Checkpoint save logic duplicated across 3 phases
- Phase sizes vary 10x (16 lines to 125 lines)

**Symptoms of the god file:**
1. Difficult to test individual phases in isolation
2. Changes to one phase risk regressions in others
3. High cognitive load when working on the file
4. Import bloat (imports only needed by specific phases)

---

## Current State Analysis

### Line Counts by Phase

| Phase | Lines | Complexity |
|-------|-------|------------|
| `phase_parse` | 96 | Medium - validation, export detection, trusted context |
| `phase_select_provider` | 16 | Trivial - wrapper |
| `phase_chunk` | 69 | Low - chunking config + demo mode |
| `phase_check_resume` | 34 | Low - checkpoint lookup |
| `process_sequential_with_checkpoints` | 62 | Medium - helper for local LLM |
| `_write_hallucination_log` | 25 | Low - file I/O helper |
| `phase_extract` | 125 | **HIGH** - branching, callbacks, orchestration |
| `phase_aggregate` | 32 | Trivial - wrapper around aggregator |
| `phase_deduplicate` | 92 | Medium - checkpoint resume + save |
| `phase_distill` | 96 | Medium - checkpoint resume + save |
| `phase_output` | 41 | Low - file writing |

**Total:** ~766 lines (excluding blank lines and imports)

### Duplication Identified

**Checkpoint saving** appears in 3 places with similar patterns:
1. `phase_extract:424-438` - saves after each API chunk
2. `phase_extract:327-336` - saves after each local chunk
3. `phase_deduplicate:607-619` - saves after dedup
4. `phase_distill:704-718` - saves after distill

Each manually constructs:
```python
save_checkpoint(checkpoint_path, {
    "source_file_hash": hash_file(ctx.input_file),
    "completed_chunks": sorted(completed),
    "verified_facts": [asdict(f) if isinstance(f, ExtractedFact) else f for f in facts],
    # ... more fields
})
```

---

## Proposed Architecture

### Target Structure

```
src/cli/
├── phases/                  # NEW: Package for phase functions
│   ├── __init__.py          # Re-exports all public phase functions
│   ├── parse.py             # phase_parse
│   ├── provider.py          # phase_select_provider (tiny, could merge)
│   ├── chunk.py             # phase_chunk
│   ├── resume.py            # phase_check_resume
│   ├── extract.py           # phase_extract + helpers
│   ├── aggregate.py         # phase_aggregate
│   ├── deduplicate.py       # phase_deduplicate
│   ├── distill.py           # phase_distill
│   ├── output.py            # phase_output
│   └── _checkpoint.py       # Shared checkpoint builder (internal)
├── __init__.py              # Updated to import from phases/
├── checkpoints.py           # Unchanged (path generation)
├── display.py               # Unchanged (console output)
├── providers.py             # Unchanged (provider selection)
├── types.py                 # Unchanged (PipelineContext)
└── validation.py            # Unchanged (input validation)
```

### Design Decisions

**D1: One file per phase** (not grouped by concern)
- Matches mental model: `phase_parse` lives in `parse.py`
- Easy to find: grep for phase name → find file
- Isolated testing: can test one phase without importing others
- Parallel development: different contributors can work on different phases

**D2: Merge trivial phases?**
- `phase_select_provider` (16 lines) could merge into `chunk.py` since they're called consecutively
- Decision: Keep separate for now. Easy to merge later, hard to split later.

**D3: Extract checkpoint helper**
- Create `_checkpoint.py` with `build_checkpoint_dict()` helper
- Reduces duplication, ensures consistent checkpoint structure
- Leading underscore signals internal-only module

**D4: Keep helpers with their phases**
- `_write_hallucination_log` stays in `extract.py` (only used there)
- `process_sequential_with_checkpoints` stays in `extract.py` (only used there)

---

## Implementation Steps

### Step 1: Create package structure (no behavior change)

1. Create `src/cli/phases/` directory
2. Create `src/cli/phases/__init__.py` that imports from parent's `phases.py`
3. Verify tests pass (zero behavior change)

**Commit:** `refactor(cli): Create phases/ package structure`

### Step 2: Extract checkpoint helper

1. Create `src/cli/phases/_checkpoint.py`
2. Add `build_checkpoint_dict()` function:
   ```python
   def build_checkpoint_dict(
       ctx: PipelineContext,
       completed_chunks: Iterable[int],
       verified_facts: List[ExtractedFact],
       dedup_completed: bool = False,
       deduplicated_facts: List[DeduplicatedFact] = None,
       distill_completed: bool = False,
       distilled_profile: DistilledProfile = None,
   ) -> Dict[str, Any]:
       """Build standardized checkpoint dictionary."""
   ```
3. Update existing checkpoint saves to use helper (still in phases.py)
4. Verify tests pass

**Commit:** `refactor(cli): Extract checkpoint builder helper`

### Step 3: Extract simple phases first

Move one phase at a time, simplest to most complex:

1. **aggregate.py** (32 lines, trivial)
2. **output.py** (41 lines, low complexity)
3. **resume.py** (34 lines, low complexity)
4. **provider.py** (16 lines, trivial)
5. **chunk.py** (69 lines, medium)
6. **parse.py** (96 lines, medium)
7. **deduplicate.py** (92 lines, medium)
8. **distill.py** (96 lines, medium)
9. **extract.py** (212 lines with helpers, highest complexity)

For each phase:
1. Create new file with function + its imports
2. Update `phases/__init__.py` to import from new location
3. Remove from `phases.py`
4. Run tests
5. Commit

**Commits:** One per phase, e.g., `refactor(cli): Move phase_aggregate to phases/aggregate.py`

### Step 4: Update package exports

1. Update `src/cli/__init__.py` to import from `phases/` package
2. Keep backward-compatible exports (same public API)
3. Verify all tests pass

**Commit:** `refactor(cli): Update CLI package to use phases/ package`

### Step 5: Delete old phases.py

1. Remove `src/cli/phases.py`
2. Final test run
3. Update any imports in main.py that reference old location

**Commit:** `refactor(cli): Remove legacy phases.py`

### Step 6: Documentation

1. Update CLAUDE.md if any references to file structure
2. Update `docs/development-notes.md` if exists

**Commit:** `docs: Update for phases/ package refactor`

---

## Risk Mitigation

### Testing Strategy

- **Before each move:** All tests pass
- **After each move:** All tests pass
- **Focus areas:**
  - `tests/test_cli/test_phases.py` (if exists)
  - `tests/test_main.py` (integration tests)
  - Any tests importing from `src.cli`

### Backward Compatibility

All existing imports continue to work:
```python
# These all still work:
from src.cli import phase_parse, phase_extract
from src.cli.phases import phase_parse, phase_extract
```

### Rollback Plan

Each step is a separate commit. If issues arise:
```bash
git revert <commit>
```

---

## Success Criteria

1. **Zero test failures** after each step
2. **Identical behavior** - no functional changes
3. **Cleaner imports** - each phase file only imports what it needs
4. **Testable isolation** - can import one phase without importing all
5. **Reduced phases.py** - file is deleted at end
6. **Documentation** - updated for new structure

---

## Estimated Effort

| Step | Estimated Commits |
|------|-------------------|
| Package structure | 1 |
| Checkpoint helper | 1 |
| Move 9 phases | 9 |
| Update exports | 1 |
| Delete old file | 1 |
| Documentation | 1 |
| **Total** | **14 commits** |

---

## Alternatives Considered

### A: Group by Concern

```
phases/
├── io.py          # parse, chunk, output
├── extraction.py  # extract + helpers
├── refinement.py  # aggregate, deduplicate, distill
├── lifecycle.py   # select_provider, check_resume
```

**Rejected because:**
- Groupings are subjective
- Harder to find specific phase
- Still mixing responsibilities within files

### B: Extract Only Big Phases

Keep simple phases in `phases.py`, extract only:
- `extraction.py` (extract + helpers)
- `refinement.py` (deduplicate + distill)

**Rejected because:**
- Inconsistent organization
- Leaves technical debt for later
- "Just the big ones" is a slippery slope

### C: Keep As-Is with Better Organization

Reorder functions, add section comments, improve docs.

**Rejected because:**
- Doesn't solve the core problem
- Still 766 lines in one file
- Still can't test phases in isolation

---

## Open Questions

1. **Merge `phase_select_provider` into `chunk.py`?**
   - Pro: They're always called consecutively
   - Con: Separate concerns, might want to test independently
   - **Recommendation:** Keep separate, can merge later if warranted

2. **Move `process_sequential_with_checkpoints` to processor.py?**
   - It's extraction-specific but could live with other chunk processors
   - **Recommendation:** Keep in `extract.py` for now (only caller)

3. **Should `_checkpoint.py` be in phases/ or cli/?**
   - Only used by phases
   - **Recommendation:** Keep in `phases/` as internal module

---

## Appendix: Import Analysis

### Current phases.py imports

```python
# Standard library
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

# Core modules (used by multiple phases)
from src.parser import ...  # 7 imports - used by parse
from src.chunker import ...  # 3 imports - used by chunk
from src.providers import ...  # 2 imports - used by chunk, extract
from src.checkpoint import ...  # 4 imports - used by extract, dedup, distill

# Core modules (used by single phase)
from src.extractor import ExtractedFact  # extract only
from src.processor import ...  # 4 imports - extract only
from src.aggregator import ...  # 2 imports - aggregate only
from src.deduplicator import ...  # 2 imports - dedup only
from src.distiller import ...  # 4 imports - distill only

# CLI modules
from src.cli.types import PipelineContext  # all phases
from src.cli.display import ...  # 8 imports - various phases
from src.cli.validation import find_memories_json  # parse only
from src.cli.providers import select_provider  # provider only
from src.cli.checkpoints import ...  # 2 imports - extract, dedup, distill
```

### After refactor: example extract.py imports

```python
import time
from dataclasses import asdict
from pathlib import Path
from typing import List

from src.extractor import ExtractedFact
from src.processor import extract_and_verify_chunk, process_all_chunks, ParallelResult, ChunkResult
from src.checkpoint import hash_file, save_checkpoint

from src.cli.types import PipelineContext
from src.cli.display import format_elapsed_time, show_phase_header, show_phase_complete
from src.cli.checkpoints import get_checkpoint_path
from src.cli.phases._checkpoint import build_checkpoint_dict
```

Much cleaner - only what's needed for extraction.
