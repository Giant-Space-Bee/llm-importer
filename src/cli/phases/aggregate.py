"""
Aggregate phase: Group and count extracted facts.

This phase aggregates verified facts by normalizing and grouping them,
counting occurrences and tracking the newest source for each unique fact.
"""

import time

from src.aggregator import aggregate
from src.cli.types import PipelineContext
from src.cli.display import show_phase_header, show_phase_complete


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
    start_time = time.time()

    show_phase_header(console, 4, "AGGREGATE")

    if ctx.verified_facts:
        ctx.aggregated_facts = aggregate(ctx.verified_facts)
        console.print(
            f"[green]✓ {len(ctx.verified_facts)} facts → "
            f"{len(ctx.aggregated_facts)} unique[/green]"
        )
    else:
        console.print("[yellow]No facts extracted.[/yellow]")
        ctx.aggregated_facts = []

    elapsed = time.time() - start_time
    ctx.phase_timings["aggregate"] = elapsed
    show_phase_complete(console, elapsed)

    return ctx
