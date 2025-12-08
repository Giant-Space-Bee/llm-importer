"""
Output phase: Write distilled profile to files.

This phase writes the final memory profile to both markdown (human-readable)
and JSON (machine-importable) formats in the output directory.
"""

import time
from pathlib import Path

from src.distiller import write_markdown, write_json
from src.cli.types import PipelineContext
from src.cli.display import show_phase_header, show_phase_complete


def phase_output(ctx: PipelineContext) -> PipelineContext:
    """Write distilled profile to output files.

    Creates both markdown (human-readable) and JSON (machine-importable)
    versions of the memory profile in the output/ directory.

    Args:
        ctx: Pipeline context with distilled_profile.

    Returns:
        Updated context with output_md_path and output_json_path.

    Example:
        >>> ctx = phase_output(ctx)
        >>> print(f"Output: {ctx.output_md_path}")
    """
    console = ctx.console
    start_time = time.time()

    show_phase_header(console, 7, "OUTPUT")

    if not ctx.distilled_profile:
        console.print("[yellow]No profile to output.[/yellow]")
        show_phase_complete(console, time.time() - start_time)
        return ctx

    output_dir = Path.cwd() / "output"
    output_dir.mkdir(exist_ok=True)

    # Write both formats
    ctx.output_md_path = write_markdown(ctx.distilled_profile, output_dir)
    ctx.output_json_path = write_json(ctx.distilled_profile, output_dir)

    console.print(f"[green]✓ Markdown:[/green] {ctx.output_md_path}")
    console.print(f"[green]✓ JSON:[/green] {ctx.output_json_path}")

    elapsed = time.time() - start_time
    ctx.phase_timings["output"] = elapsed
    show_phase_complete(console, elapsed)

    return ctx
