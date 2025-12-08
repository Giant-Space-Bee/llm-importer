"""
Chunk phase: Split conversations into processable batches.

This phase chunks the parsed conversations into token-sized batches
that can be processed by the LLM within its context window.
"""

import time

from src.chunker import chunk_conversations, DEFAULT_CHUNK_SIZE, prepare_conversations
from src.providers import APIProvider
from src.cli.types import PipelineContext
from src.cli.display import show_phase_header, show_phase_complete, show_chunk_table


# Demo mode uses smaller chunks for faster testing
DEMO_CHUNK_SIZE = 4096


def phase_chunk(ctx: PipelineContext) -> PipelineContext:
    """Chunk conversations into processable batches.

    Uses smaller chunk size in demo mode. Both API and local providers
    split oversized conversations and respect --chunk-size override.

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
    start_time = time.time()

    show_phase_header(console, 2, "CHUNK")

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
        # Local LLM: respect override, split oversized convos
        chunk_size = ctx.chunk_size_override or DEFAULT_CHUNK_SIZE
        original_count = len(conversations)
        conversations = prepare_conversations(conversations, chunk_size)
        if len(conversations) > original_count:
            console.print(
                f"[yellow]Prepared:[/yellow] {original_count} conversations → "
                f"{len(conversations)} (split oversized)"
            )

    ctx.chunk_size = chunk_size

    console.print(f"\n[bold]Chunking conversations...[/bold] (max {chunk_size:,} tokens)")
    chunks = chunk_conversations(conversations, max_tokens=chunk_size)

    # Demo mode: first chunk only
    if ctx.demo_mode:
        chunks = chunks[:1]
        console.print("[yellow]Demo mode:[/yellow] Processing first chunk only")

    ctx.chunks = chunks
    show_chunk_table(console, chunks)

    elapsed = time.time() - start_time
    ctx.phase_timings["chunk"] = elapsed
    show_phase_complete(console, elapsed)

    # Free memory - parsed conversations no longer needed after chunking
    ctx.conversations = []

    return ctx
