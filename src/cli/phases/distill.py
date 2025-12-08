"""
Distill phase: Compress deduplicated facts into final memory profile.

This phase uses LLM to synthesize facts into a coherent, categorized profile.
If trusted_context is available (Claude memories or ChatGPT custom instructions),
it merges with existing profile rather than creating from scratch.
"""

import time

from src.distiller import distill, DistilledProfile
from src.checkpoint import hash_file, save_checkpoint, load_checkpoint
from src.cli.types import PipelineContext
from src.cli.display import show_phase_header, show_phase_complete
from src.cli.checkpoints import get_checkpoint_path
from src.cli.phases._checkpoint import build_checkpoint_dict


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
