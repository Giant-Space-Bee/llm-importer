"""
Pipeline phase functions for the LLM Importer CLI.

Each phase function takes a PipelineContext, performs work, and returns
the updated context. This enables clean phase-based orchestration in main().
"""

import time
from dataclasses import asdict
from typing import Any, Dict, List

from src.parser import (
    parse_all,
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
from src.processor import extract_and_verify_chunk, process_all_chunks, ParallelResult
from src.aggregator import aggregate, AggregatedFact, group_by_category
from src.deduplicator import deduplicate, DeduplicatedFact
from src.checkpoint import hash_file, save_checkpoint, get_remaining_chunks, load_checkpoint

from src.cli.types import PipelineContext
from src.cli.display import (
    format_elapsed_time,
    show_stats_table,
    show_user_profile_preview,
    show_claude_memories_preview,
    show_chunk_table,
)
from src.cli.validation import find_memories_json
from src.cli.providers import select_provider as _select_provider
from src.cli.checkpoints import get_checkpoint_path, check_existing_checkpoint


# Demo mode uses smaller chunks for faster testing
DEMO_CHUNK_SIZE = 4096


def phase_parse(ctx: PipelineContext) -> PipelineContext:
    """Parse conversations and load trusted context.

    Loads and parses conversations from the input file, detects export type,
    and loads any available trusted context (Claude memories or ChatGPT profile).

    Args:
        ctx: Pipeline context with input_file set.

    Returns:
        Updated context with:
        - conversations: Parsed conversation objects
        - raw_conversations_by_id: Raw dicts for verification
        - user_profile: Extracted custom instructions (if any)
        - claude_memories: Claude memories (if available)
        - trusted_context: Formatted baseline for distiller
        - stats: Conversation statistics
        - export_type: Detected export format

    Example:
        >>> ctx = PipelineContext(input_file="conversations.json", console=console)
        >>> ctx = phase_parse(ctx)
        >>> print(f"Found {len(ctx.conversations)} conversations")
    """
    console = ctx.console

    # Parse conversations
    console.print("\n[bold]Loading and parsing conversations...[/bold]")
    conversations, user_profile = parse_all(ctx.input_file)
    stats = get_stats_from_parsed(conversations, user_profile)

    # Load raw conversations for verification (need full message tree)
    raw_convos = load_conversations(ctx.input_file)

    # Detect export type first (determines which ID field to use)
    export_type = detect_export_type(raw_convos)

    # Build lookup using correct ID field (ChatGPT uses 'id', Claude uses 'uuid')
    id_field = "uuid" if export_type == "claude" else "id"
    conversations_by_id = {c[id_field]: c for c in raw_convos}

    # Show stats
    show_stats_table(console, stats)

    # Show user profile (free wins!)
    if user_profile:
        show_user_profile_preview(console, user_profile)

    # Load Claude memories if available (trusted baseline)
    claude_memories = None
    trusted_context = None

    if export_type == "claude":
        memories_path = find_memories_json(ctx.input_file)
        if memories_path:
            claude_memories = load_claude_memories(str(memories_path.parent))
            if claude_memories:
                trusted_context = format_memories_for_distiller(claude_memories)
                show_claude_memories_preview(console, claude_memories)

    # Use ChatGPT user profile as trusted baseline if no Claude memories
    if trusted_context is None and user_profile:
        trusted_context = format_user_profile_for_distiller(user_profile)
        console.print("\n[bold green]Using custom instructions as trusted baseline[/bold green]")

    # Update context
    ctx.conversations = conversations
    ctx.raw_conversations_by_id = conversations_by_id
    ctx.user_profile = user_profile
    ctx.claude_memories = claude_memories
    ctx.trusted_context = trusted_context
    ctx.stats = stats
    ctx.export_type = export_type

    return ctx


def phase_select_provider(ctx: PipelineContext, choice: str) -> PipelineContext:
    """Select and configure the LLM provider.

    Args:
        ctx: Pipeline context.
        choice: Provider choice ("local" or "api").

    Returns:
        Updated context with provider set.

    Example:
        >>> ctx = phase_select_provider(ctx, "local")
        >>> print(ctx.provider.is_local)
        True
    """
    ctx.provider = _select_provider(ctx.console, choice)
    return ctx


def phase_chunk(ctx: PipelineContext) -> PipelineContext:
    """Chunk conversations into processable batches.

    Uses smaller chunk size in demo mode. For API provider, uses quality-first
    8k chunks and prepares oversized conversations by splitting them.

    Args:
        ctx: Pipeline context with conversations populated.

    Returns:
        Updated context with chunks list.

    Example:
        >>> ctx = phase_chunk(ctx)
        >>> print(f"Created {len(ctx.chunks)} chunks")
    """
    console = ctx.console
    conversations = ctx.conversations

    # Determine chunk size based on mode and provider
    if ctx.demo_mode:
        chunk_size = DEMO_CHUNK_SIZE
    elif isinstance(ctx.provider, APIProvider):
        # API mode: use quality-first chunking
        if ctx.tpm_override:
            ctx.provider.tpm = ctx.tpm_override
        chunk_size = ctx.provider.get_safe_chunk_size()
        max_concurrent = ctx.provider.get_max_concurrent()
        ctx.max_concurrent = max_concurrent
        console.print(
            f"[dim]TPM: {ctx.provider.tpm:,} → chunk size: {chunk_size:,}, "
            f"concurrent: {max_concurrent}[/dim]"
        )
        # Split oversized conversations
        original_count = len(conversations)
        conversations = prepare_conversations(conversations, chunk_size)
        if len(conversations) > original_count:
            console.print(
                f"[yellow]Prepared:[/yellow] {original_count} conversations → "
                f"{len(conversations)} (split oversized)"
            )
    else:
        # Local LLM: use full chunks
        chunk_size = DEFAULT_CHUNK_SIZE

    ctx.chunk_size = chunk_size

    console.print(f"\n[bold]Chunking conversations...[/bold] (max {chunk_size:,} tokens)")
    chunks = chunk_conversations(conversations, max_tokens=chunk_size)

    # Demo mode: first chunk only
    if ctx.demo_mode:
        chunks = chunks[:1]
        console.print("[yellow]Demo mode:[/yellow] Processing first chunk only")

    ctx.chunks = chunks
    show_chunk_table(console, chunks)

    return ctx


def phase_check_resume(ctx: PipelineContext) -> PipelineContext:
    """Check for checkpoint and set up resume state.

    Args:
        ctx: Pipeline context with chunks populated.

    Returns:
        Updated context with remaining_indices and existing_facts.

    Example:
        >>> ctx = phase_check_resume(ctx)
        >>> print(f"{len(ctx.remaining_indices)} chunks remaining")
    """
    completed_chunks, existing_facts, signal = check_existing_checkpoint(
        ctx.console, ctx.input_file, ctx.resume_mode
    )

    if ctx.resume_mode and signal == -1:
        # Resuming from checkpoint
        ctx.remaining_indices = get_remaining_chunks(
            {"completed_chunks": completed_chunks},
            len(ctx.chunks)
        )
        ctx.existing_facts = existing_facts
    else:
        # Starting fresh
        ctx.remaining_indices = list(range(len(ctx.chunks)))
        ctx.existing_facts = []

    if not ctx.remaining_indices:
        ctx.console.print("\n[green]All chunks already processed![/green]")
    else:
        ctx.console.print(f"\n[bold]Processing {len(ctx.remaining_indices)} chunks...[/bold]")

    return ctx


def process_sequential_with_checkpoints(
    ctx: PipelineContext,
) -> List[ExtractedFact]:
    """Process chunks sequentially with per-chunk checkpointing.

    Used for local LLM where each chunk takes minutes.
    Saves progress after each chunk so crashes don't lose work.

    Args:
        ctx: Pipeline context with chunks, provider, and checkpoint state.

    Returns:
        All verified facts (existing + new).

    Example:
        >>> facts = process_sequential_with_checkpoints(ctx)
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
    completed = set(range(len(chunks))) - set(remaining_indices)

    for chunk_idx in remaining_indices:
        chunk = chunks[chunk_idx]
        start_time = time.time()

        console.print(
            f"\n[cyan]Processing chunk {chunk_idx + 1}/{len(chunks)}[/cyan] "
            f"({chunk.token_count:,} tokens, {len(chunk.conversations)} convos)"
        )

        # Extract and verify
        new_facts = extract_and_verify_chunk(chunk, provider, conversations_by_id)
        all_facts.extend(new_facts)

        elapsed = time.time() - start_time
        console.print(
            f"  [green]+{len(new_facts)} verified facts[/green] "
            f"({len(all_facts)} total) [{format_elapsed_time(elapsed)}]"
        )

        # Save checkpoint
        completed.add(chunk_idx)
        save_checkpoint(checkpoint_path, {
            "source_file_hash": hash_file(input_file),
            "completed_chunks": sorted(completed),
            "verified_facts": [
                asdict(f) if isinstance(f, ExtractedFact) else f
                for f in all_facts
            ]
        })

    return all_facts  # type: ignore


def phase_extract(ctx: PipelineContext) -> PipelineContext:
    """Run extraction on all remaining chunks.

    Uses sequential processing for local LLM or parallel for API.

    Args:
        ctx: Pipeline context with chunks and provider.

    Returns:
        Updated context with verified_facts.

    Raises:
        KeyboardInterrupt: If user interrupts processing.

    Example:
        >>> ctx = phase_extract(ctx)
        >>> print(f"Extracted {len(ctx.verified_facts)} facts")
    """
    console = ctx.console
    provider = ctx.provider

    if not ctx.remaining_indices:
        # All chunks already processed
        ctx.verified_facts = [
            ExtractedFact(**f) if isinstance(f, dict) else f
            for f in ctx.existing_facts
        ]
        return ctx

    if provider.is_local:
        # Local LLM: sequential with per-chunk checkpointing
        verified_facts = process_sequential_with_checkpoints(ctx)
    else:
        # API: parallel processing with per-chunk checkpointing
        start_time = time.time()
        checkpoint_path = get_checkpoint_path(ctx.input_file)
        file_hash = hash_file(ctx.input_file)

        # Get only the chunks we need to process
        chunks_to_process = [ctx.chunks[i] for i in ctx.remaining_indices]

        console.print(
            f"[dim]Processing {len(chunks_to_process)} chunks in parallel...[/dim]"
        )

        # Checkpoint callback - saves after each chunk completes
        def on_chunk_complete(completed_indices, all_facts):
            save_checkpoint(checkpoint_path, {
                "source_file_hash": file_hash,
                "completed_chunks": sorted(completed_indices),
                "verified_facts": [
                    asdict(f) if isinstance(f, ExtractedFact) else f
                    for f in all_facts
                ]
            })

        # Process with checkpointing
        result: ParallelResult = process_all_chunks(
            chunks_to_process,
            provider,
            ctx.raw_conversations_by_id,
            chunk_indices=ctx.remaining_indices,
            on_chunk_complete=on_chunk_complete,
            existing_facts=list(ctx.existing_facts)
        )

        elapsed = time.time() - start_time

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
        else:
            console.print(
                f"  [green]Extracted {len(result.facts) - len(ctx.existing_facts)} "
                f"new verified facts[/green] [{format_elapsed_time(elapsed)}]"
            )

        verified_facts = result.facts

    ctx.verified_facts = verified_facts
    return ctx


def phase_aggregate(ctx: PipelineContext) -> PipelineContext:
    """Aggregate and deduplicate extracted facts.

    Args:
        ctx: Pipeline context with verified_facts.

    Returns:
        Updated context with aggregated_facts.

    Example:
        >>> ctx = phase_aggregate(ctx)
        >>> print(f"Aggregated to {len(ctx.aggregated_facts)} unique facts")
    """
    console = ctx.console

    if ctx.verified_facts:
        console.print("\n[bold]Aggregating facts...[/bold]")
        ctx.aggregated_facts = aggregate(ctx.verified_facts)
    else:
        console.print("\n[yellow]No facts extracted.[/yellow]")
        ctx.aggregated_facts = []

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

    # Check if dedup already completed (resume support)
    existing_checkpoint = load_checkpoint(checkpoint_path)
    if existing_checkpoint and existing_checkpoint.get("dedup_completed"):
        dedup_facts_data = existing_checkpoint.get("deduplicated_facts", [])
        ctx.deduplicated_facts = [
            DeduplicatedFact(**f) if isinstance(f, dict) else f
            for f in dedup_facts_data
        ]
        console.print(
            f"\n[green]Dedup checkpoint found:[/green] "
            f"{len(ctx.deduplicated_facts)} deduplicated facts loaded"
        )
        return ctx

    if not ctx.aggregated_facts:
        console.print("\n[yellow]No facts to deduplicate.[/yellow]")
        ctx.deduplicated_facts = []
        return ctx

    input_count = len(ctx.aggregated_facts)

    # Show category breakdown before dedup (engineering data)
    by_category = group_by_category(ctx.aggregated_facts)
    console.print(f"\n[bold]Deduplicating {input_count} aggregated facts...[/bold]")
    console.print("[dim]Category breakdown before dedup:[/dim]")
    for cat, facts in sorted(by_category.items(), key=lambda x: -len(x[1])):
        console.print(f"  [dim]{cat}: {len(facts)}[/dim]")

    # Run deduplication with timing
    start_time = time.time()
    try:
        ctx.deduplicated_facts = deduplicate(ctx.aggregated_facts, ctx.provider)
    except Exception as e:
        console.print(f"\n[red]Dedup error:[/red] {e}")
        console.print("[yellow]Aggregated facts preserved in checkpoint for retry.[/yellow]")
        raise

    elapsed = time.time() - start_time
    output_count = len(ctx.deduplicated_facts)
    reduction_pct = ((input_count - output_count) / input_count * 100) if input_count > 0 else 0

    # Engineering stats
    console.print(
        f"  [green]Reduced {input_count} → {output_count} facts "
        f"({reduction_pct:.1f}% reduction)[/green] [{format_elapsed_time(elapsed)}]"
    )

    # Save checkpoint with dedup results
    save_checkpoint(checkpoint_path, {
        "source_file_hash": hash_file(ctx.input_file),
        "completed_chunks": list(range(len(ctx.chunks))),
        "verified_facts": [
            asdict(f) if isinstance(f, ExtractedFact) else f
            for f in ctx.verified_facts
        ],
        "dedup_completed": True,
        "deduplicated_facts": [
            asdict(f) if isinstance(f, DeduplicatedFact) else f
            for f in ctx.deduplicated_facts
        ],
    })
    console.print("[dim]Dedup checkpoint saved.[/dim]")

    return ctx
