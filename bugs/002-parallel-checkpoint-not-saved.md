# Bug 002: Checkpoint Not Saved During Parallel Processing

## Summary
When using `--provider api`, no checkpoint is saved if ANY chunk fails. All extraction work is lost on error.

## Severity
**Critical** - User loses all API credits and extraction work on any single chunk failure.

## Location
`src/cli/phases.py:336-361` - `phase_extract()` parallel processing path

## Reproduction
```bash
python -m src.main "export_data/data-2025-12-03-23-45-23-batch-0000/conversations.json" --provider api
# Wait for some chunks to process
# If JSON error occurs (Bug 001), no checkpoint is saved
```

## Analysis

### Sequential mode (CORRECT)
```python
# src/cli/phases.py:269-298
for chunk_idx in remaining_indices:
    # Process chunk
    new_facts = extract_and_verify_chunk(chunk, provider, conversations_by_id)
    all_facts.extend(new_facts)

    # Save checkpoint AFTER EACH CHUNK
    completed.add(chunk_idx)
    save_checkpoint(checkpoint_path, {...})  # <-- Saved per chunk!
```

### Parallel mode (BROKEN)
```python
# src/cli/phases.py:336-361
# Process ALL chunks at once
new_facts = process_all_chunks(ctx.chunks, provider, ctx.raw_conversations_by_id)

# Save checkpoint ONLY AFTER ALL COMPLETE
save_checkpoint(checkpoint_path, {...})  # <-- Never reached if ANY chunk fails
```

### The problem
1. Parallel processing uses `asyncio.gather()` to process all chunks concurrently
2. If ANY chunk throws an exception (JSON error, rate limit exhausted, etc.), the gather fails
3. Exception propagates up to `phase_extract()`
4. Code never reaches the `save_checkpoint()` call (line 353-361)
5. User loses ALL work, even from successfully completed chunks

## Impact
With 19 chunks and ~$0.10-0.20 per chunk, a failure after 15 chunks means:
- 15 chunks worth of API credits wasted (~$1.50-3.00)
- 15 chunks worth of verified facts lost
- Must restart from scratch

## Suggested Fix

### Option A: Save checkpoint per completed chunk (like sequential mode)
```python
async def process_chunks_parallel_with_checkpoints(
    chunks, provider, conversations_by_id, max_concurrent,
    checkpoint_path, file_hash
):
    semaphore = asyncio.Semaphore(max_concurrent)
    all_facts = []
    completed_chunks = set()
    lock = asyncio.Lock()

    async def process_with_checkpoint(idx, chunk):
        async with semaphore:
            facts = await extract_and_verify_chunk_async(chunk, provider, conversations_by_id)

            async with lock:
                all_facts.extend(facts)
                completed_chunks.add(idx)
                save_checkpoint(checkpoint_path, {
                    "source_file_hash": file_hash,
                    "completed_chunks": sorted(completed_chunks),
                    "verified_facts": [asdict(f) for f in all_facts]
                })

            return facts

    # Use return_exceptions=True to continue even if some fail
    results = await asyncio.gather(
        *[process_with_checkpoint(i, chunk) for i, chunk in enumerate(chunks)],
        return_exceptions=True
    )

    # Report failures but don't lose successful work
    failures = [(i, r) for i, r in enumerate(results) if isinstance(r, Exception)]
    if failures:
        print(f"Warning: {len(failures)} chunks failed, {len(completed_chunks)} succeeded")
        for i, exc in failures:
            print(f"  Chunk {i}: {exc}")

    return all_facts
```

### Option B: Wrap parallel processing in try/finally
```python
# In phase_extract(), wrap the parallel call:
try:
    new_facts = process_all_chunks(ctx.chunks, provider, ctx.raw_conversations_by_id)
except Exception as e:
    # Save whatever we have before re-raising
    # (Need to track completed chunks in process_all_chunks)
    console.print(f"[red]Error during extraction: {e}[/red]")
    console.print("[yellow]Saving partial checkpoint...[/yellow]")
    # ... save partial results
    raise
```

## Recommended Approach
**Option A** is better because:
1. Checkpoint saved as soon as each chunk completes
2. Uses `return_exceptions=True` so other chunks continue even if one fails
3. Reports failures without losing successful work
4. Matches behavior of sequential mode

## Related
- Bug 001: API JSON parsing error (triggers this checkpoint loss)
