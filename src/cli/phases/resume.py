"""
Resume phase: Check for existing checkpoint and set up resume state.

This phase checks for an existing checkpoint file and determines which
chunks still need to be processed vs which have already been completed.
"""

from src.checkpoint import get_remaining_chunks
from src.cli.types import PipelineContext
from src.cli.checkpoints import check_existing_checkpoint


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
