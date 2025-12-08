"""
Pipeline phase functions for the LLM Importer CLI.

Each phase function takes a PipelineContext, performs work, and returns
the updated context. This enables clean phase-based orchestration in main().
"""

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from src.parser import (
    parse_from_raw,
    get_stats_from_parsed,
    detect_export_type,
    load_conversations,
    load_claude_memories,
    format_memories_for_distiller,
    format_user_profile_for_distiller,
)
from src.chunker import chunk_conversations, Chunk, DEFAULT_CHUNK_SIZE, prepare_conversations
from src.providers import LLMProvider, APIProvider
from src.extractor import ExtractedFact
from src.processor import extract_and_verify_chunk, process_all_chunks, ParallelResult, ChunkResult
from src.aggregator import aggregate, AggregatedFact, group_by_category
from src.deduplicator import deduplicate, DeduplicatedFact
from src.distiller import distill, DistilledProfile, write_markdown, write_json
from src.checkpoint import hash_file, save_checkpoint, get_remaining_chunks, load_checkpoint

from src.cli.types import PipelineContext
from src.cli.display import (
    format_elapsed_time,
    show_stats_table,
    show_user_profile_preview,
    show_claude_memories_preview,
    show_chunk_table,
    show_phase_header,
    show_phase_complete,
)
from src.cli.validation import find_memories_json
from src.cli.providers import select_provider as _select_provider
from src.cli.checkpoints import get_checkpoint_path, check_existing_checkpoint
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


def phase_deduplicate(ctx: PipelineContext) -> PipelineContext:
    """Semantically deduplicate aggregated facts using LLM.

    Two-phase deduplication:
    - Phase 1: Within-category dedup (timestamp-batched, merge-sort)
    - Phase 2: Cross-category merge-sort (catches miscategorized duplicates)

    Includes engineering data output for monitoring and improvement.
    Saves checkpoint after completion for recovery.

    Args:
        ctx: Pipeline context with aggregated_facts and provider.

    Returns:
        Updated context with deduplicated_facts.

    Example:
        >>> ctx = phase_deduplicate(ctx)
        >>> print(f"Deduplicated to {len(ctx.deduplicated_facts)} unique facts")
    """
    console = ctx.console
    checkpoint_path = get_checkpoint_path(ctx.input_file)
    start_time = time.time()

    show_phase_header(console, 5, "DEDUPLICATE")

    # Check if dedup already completed (resume support)
    existing_checkpoint = load_checkpoint(checkpoint_path)
    if existing_checkpoint and existing_checkpoint.get("dedup_completed"):
        dedup_facts_data = existing_checkpoint.get("deduplicated_facts", [])
        ctx.deduplicated_facts = [
            DeduplicatedFact(**f) if isinstance(f, dict) else f
            for f in dedup_facts_data
        ]
        console.print(
            f"[green]Checkpoint found: {len(ctx.deduplicated_facts)} deduplicated facts loaded[/green]"
        )
        show_phase_complete(console, time.time() - start_time)
        return ctx

    if not ctx.aggregated_facts:
        console.print("[yellow]No facts to deduplicate.[/yellow]")
        ctx.deduplicated_facts = []
        show_phase_complete(console, time.time() - start_time)
        return ctx

    input_count = len(ctx.aggregated_facts)

    # Show category breakdown before dedup (engineering data)
    by_category = group_by_category(ctx.aggregated_facts)
    console.print(f"Deduplicating {input_count} facts...")
    console.print("[dim]Category breakdown:[/dim]")
    for cat, facts in sorted(by_category.items(), key=lambda x: -len(x[1])):
        console.print(f"  [dim]{cat}: {len(facts)}[/dim]")

    # Run deduplication
    try:
        with console.status("[bold blue]Waiting for LLM response..."):
            ctx.deduplicated_facts = deduplicate(ctx.aggregated_facts, ctx.provider)
    except Exception as e:
        console.print(f"\n[red]Dedup error:[/red] {e}")
        console.print("[yellow]Aggregated facts preserved in checkpoint for retry.[/yellow]")
        raise

    elapsed = time.time() - start_time
    output_count = len(ctx.deduplicated_facts)
    reduction_pct = ((input_count - output_count) / input_count * 100) if input_count > 0 else 0

    # Summary
    console.print(
        f"[green]✓ {input_count} → {output_count} facts "
        f"({reduction_pct:.0f}% reduction)[/green]"
    )

    # Save checkpoint with dedup results
    save_checkpoint(checkpoint_path, build_checkpoint_dict(
        source_file_hash=hash_file(ctx.input_file),
        completed_chunks=range(len(ctx.chunks)),
        verified_facts=ctx.verified_facts,
        dedup_completed=True,
        deduplicated_facts=ctx.deduplicated_facts,
    ))

    ctx.phase_timings["deduplicate"] = elapsed
    show_phase_complete(console, elapsed)

    return ctx


def phase_distill(ctx: PipelineContext) -> PipelineContext:
    """Compress deduplicated facts into final memory profile.

    Uses LLM to synthesize facts into a coherent, categorized profile.
    If trusted_context is available (Claude memories or ChatGPT custom
    instructions), merges with existing profile rather than creating
    from scratch.

    Includes checkpoint support: if distill_completed is True in checkpoint,
    loads the saved profile instead of re-running the LLM.

    Args:
        ctx: Pipeline context with deduplicated_facts and provider.

    Returns:
        Updated context with distilled_profile.

    Raises:
        Exception: If LLM call fails.

    Example:
        >>> ctx = phase_distill(ctx)
        >>> print(f"Profile for '{ctx.distilled_profile.name}'")
    """
    console = ctx.console
    checkpoint_path = get_checkpoint_path(ctx.input_file)
    start_time = time.time()

    show_phase_header(console, 6, "DISTILL")

    # Check if distill already completed (resume support)
    existing_checkpoint = load_checkpoint(checkpoint_path)
    if existing_checkpoint and existing_checkpoint.get("distill_completed"):
        profile_data = existing_checkpoint.get("distilled_profile", {})
        ctx.distilled_profile = DistilledProfile(**profile_data)
        console.print(
            f"[green]Checkpoint found: Profile for '{ctx.distilled_profile.name}' loaded[/green]"
        )
        show_phase_complete(console, time.time() - start_time)
        return ctx

    if not ctx.deduplicated_facts:
        console.print("[yellow]No facts to distill.[/yellow]")
        ctx.distilled_profile = None
        show_phase_complete(console, time.time() - start_time)
        return ctx

    # Engineering data: input count
    input_count = len(ctx.deduplicated_facts)
    source_info = f"{ctx.export_type or 'Unknown'} export ({ctx.stats.total_conversations} conversations)"

    console.print(f"Distilling {input_count} facts into memory profile...")

    # Run distillation
    try:
        with console.status("[bold blue]Waiting for LLM response..."):
            ctx.distilled_profile = distill(
                facts=ctx.deduplicated_facts,
                provider=ctx.provider,
                trusted_context=ctx.trusted_context,
                source_info=source_info,
            )
    except Exception as e:
        console.print(f"\n[red]Distill error:[/red] {e}")
        console.print("[yellow]Deduplicated facts preserved in checkpoint for retry.[/yellow]")
        raise

    elapsed = time.time() - start_time

    # Summary
    output_count = sum(len(facts) for facts in ctx.distilled_profile.categories.values())
    console.print(
        f"[green]✓ Profile '{ctx.distilled_profile.name}' generated "
        f"({output_count} organized facts)[/green]"
    )

    # Save checkpoint with distill results
    save_checkpoint(checkpoint_path, build_checkpoint_dict(
        source_file_hash=hash_file(ctx.input_file),
        completed_chunks=range(len(ctx.chunks)),
        verified_facts=ctx.verified_facts,
        dedup_completed=True,
        deduplicated_facts=ctx.deduplicated_facts,
        distill_completed=True,
        distilled_profile=ctx.distilled_profile,
    ))

    ctx.phase_timings["distill"] = elapsed
    show_phase_complete(console, elapsed)

    return ctx
