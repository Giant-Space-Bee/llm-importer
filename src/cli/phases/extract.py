"""
Extract phase: Run LLM extraction on conversation chunks.

This phase processes conversation chunks through the LLM to extract facts.
Uses sequential processing for local LLM or parallel for API providers.
Includes checkpoint support for crash recovery and hallucination logging.
"""

import time
from pathlib import Path
from typing import Any, List

from src.extractor import ExtractedFact
from src.processor import extract_and_verify_chunk, process_all_chunks, ParallelResult, ChunkResult
from src.checkpoint import hash_file, save_checkpoint
from src.cli.types import PipelineContext
from src.cli.display import format_elapsed_time, show_phase_header, show_phase_complete
from src.cli.checkpoints import get_checkpoint_path
from src.cli.phases._checkpoint import build_checkpoint_dict


def process_sequential_with_checkpoints(
    ctx: PipelineContext,
) -> tuple[List[ExtractedFact], List[str]]:
    """Process chunks sequentially with per-chunk checkpointing.

    Used for local LLM where each chunk takes minutes.
    Saves progress after each chunk so crashes don't lose work.

    Args:
        ctx: Pipeline context with chunks, provider, and checkpoint state.

    Returns:
        Tuple of (all verified facts, hallucination logs).

    Example:
        >>> facts, hallucinations = process_sequential_with_checkpoints(ctx)
        >>> print(f"Extracted {len(facts)} total facts")
    """
    console = ctx.console
    chunks = ctx.chunks
    remaining_indices = ctx.remaining_indices
    provider = ctx.provider
    conversations_by_id = ctx.raw_conversations_by_id
    input_file = ctx.input_file
    existing_facts = ctx.existing_facts

    checkpoint_path = get_checkpoint_path(input_file)
    all_facts: List[Any] = list(existing_facts)
    all_hallucinations: List[str] = []
    completed = set(range(len(chunks))) - set(remaining_indices)

    for chunk_idx in remaining_indices:
        chunk = chunks[chunk_idx]
        start_time = time.time()

        console.print(
            f"\n[cyan]Processing chunk {chunk_idx + 1}/{len(chunks)}[/cyan] "
            f"({chunk.token_count:,} tokens, {len(chunk.conversations)} convos)"
        )

        # Extract and verify
        result: ChunkResult = extract_and_verify_chunk(chunk, provider, conversations_by_id)
        all_facts.extend(result.facts)
        all_hallucinations.extend(result.hallucination_logs)

        elapsed = time.time() - start_time
        console.print(
            f"  [green]+{len(result.facts)} verified facts[/green] "
            f"({len(all_facts)} total) [{format_elapsed_time(elapsed)}]"
        )

        # Save checkpoint
        completed.add(chunk_idx)
        save_checkpoint(checkpoint_path, build_checkpoint_dict(
            source_file_hash=hash_file(input_file),
            completed_chunks=completed,
            verified_facts=all_facts,
        ))

    return all_facts, all_hallucinations  # type: ignore


def _write_hallucination_log(hallucination_logs: List[str]) -> Path:
    """Write hallucination logs to output directory with timestamp.

    Args:
        hallucination_logs: List of formatted hallucination log entries.

    Returns:
        Path to the written log file.
    """
    from datetime import datetime

    output_dir = Path.cwd() / "output"
    output_dir.mkdir(exist_ok=True)

    # Timestamped filename for comparison across runs
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = output_dir / f"hallucinations-{timestamp}.log"

    with open(log_path, "w") as f:
        f.write("# Potential Hallucinations Detected\n")
        f.write("# These facts had source_quote values that couldn't be found in the original conversations.\n")
        f.write("# This may indicate LLM hallucination or quote normalization issues.\n\n")
        for entry in hallucination_logs:
            f.write(entry + "\n\n")

    return log_path


def phase_extract(ctx: PipelineContext) -> PipelineContext:
    """Run extraction on all remaining chunks.

    Uses sequential processing for local LLM or parallel for API.
    Writes hallucination logs to output/hallucinations.log.

    Args:
        ctx: Pipeline context with chunks and provider.

    Returns:
        Updated context with verified_facts and hallucination_count.

    Raises:
        KeyboardInterrupt: If user interrupts processing.

    Example:
        >>> ctx = phase_extract(ctx)
        >>> print(f"Extracted {len(ctx.verified_facts)} facts")
    """
    console = ctx.console
    provider = ctx.provider
    start_time = time.time()

    show_phase_header(console, 3, "EXTRACTION")

    if not ctx.remaining_indices:
        # All chunks already processed
        ctx.verified_facts = [
            ExtractedFact(**f) if isinstance(f, dict) else f
            for f in ctx.existing_facts
        ]
        console.print("[green]All chunks already processed![/green]")
        return ctx

    hallucination_logs: List[str] = []

    if provider.is_local:
        # Local LLM: sequential with per-chunk checkpointing
        verified_facts, hallucination_logs = process_sequential_with_checkpoints(ctx)
    else:
        # API: parallel processing with per-chunk checkpointing
        checkpoint_path = get_checkpoint_path(ctx.input_file)
        file_hash = hash_file(ctx.input_file)

        # Get only the chunks we need to process
        chunks_to_process = [ctx.chunks[i] for i in ctx.remaining_indices]

        console.print(
            f"Processing {len(chunks_to_process)} chunks in parallel "
            f"({ctx.max_concurrent} concurrent)..."
        )

        extraction_start_time = time.time()

        # Checkpoint callback - saves after each chunk completes
        def on_chunk_complete(completed_indices, all_facts):
            save_checkpoint(checkpoint_path, build_checkpoint_dict(
                source_file_hash=file_hash,
                completed_chunks=completed_indices,
                verified_facts=all_facts,
            ))
            # Progress feedback
            elapsed = time.time() - extraction_start_time
            console.print(
                f"  [dim]Chunk {len(completed_indices)}/{len(chunks_to_process)} complete, "
                f"{len(all_facts)} facts ({elapsed:.0f}s elapsed)[/dim]"
            )

        # Process with checkpointing
        result: ParallelResult = process_all_chunks(
            chunks_to_process,
            provider,
            ctx.raw_conversations_by_id,
            chunk_indices=ctx.remaining_indices,
            on_chunk_complete=on_chunk_complete,
            existing_facts=list(ctx.existing_facts)
        )

        hallucination_logs = result.hallucination_logs

        # Report results
        if result.failed_indices:
            console.print(
                f"  [yellow]Warning: {len(result.failed_indices)} chunks failed, "
                f"{len(result.completed_indices)} succeeded[/yellow]"
            )
            for idx in result.failed_indices:
                error = result.errors.get(idx, "Unknown error")
                console.print(f"    [red]Chunk {idx + 1}: {error}[/red]")
            console.print(
                "[dim]Successful chunks have been checkpointed. "
                "Re-run with --resume to retry failed chunks.[/dim]"
            )

        verified_facts = result.facts

    elapsed = time.time() - start_time

    # Store hallucination count in context
    ctx.hallucination_count = len(hallucination_logs)

    # Write hallucination log if any detected
    if hallucination_logs:
        log_path = _write_hallucination_log(hallucination_logs)
        console.print(
            f"[yellow]{len(hallucination_logs)} potential hallucinations[/yellow] "
            f"(see {log_path})"
        )

    # Summary
    new_facts = len(verified_facts) - len(ctx.existing_facts)
    console.print(
        f"[green]✓ {new_facts} facts extracted[/green]"
    )
    show_phase_complete(console, elapsed)

    ctx.verified_facts = verified_facts
    ctx.phase_timings["extract"] = elapsed

    # Free memory - raw conversations no longer needed after verification
    ctx.raw_conversations_by_id = {}

    return ctx
