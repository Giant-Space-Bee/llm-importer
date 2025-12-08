"""
Deduplicate phase: Semantically deduplicate aggregated facts using LLM.

This phase performs two-phase deduplication:
- Phase 1: Within-category dedup (timestamp-batched, merge-sort)
- Phase 2: Cross-category merge-sort (catches miscategorized duplicates)
"""

import time

from src.aggregator import group_by_category
from src.deduplicator import deduplicate, DeduplicatedFact
from src.checkpoint import hash_file, save_checkpoint, load_checkpoint
from src.cli.types import PipelineContext
from src.cli.display import show_phase_header, show_phase_complete
from src.cli.checkpoints import get_checkpoint_path
from src.cli.phases._checkpoint import build_checkpoint_dict


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
