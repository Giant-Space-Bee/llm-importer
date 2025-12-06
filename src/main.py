"""
main.py - CLI entry point for LLM Importer.

Full pipeline: Parse -> Chunk -> Extract -> Verify -> Aggregate

Usage:
    python -m src.main                    # Interactive mode
    python -m src.main conversations.json # Specify input
    python -m src.main --demo             # Demo: 4k chunks, first chunk only
    python -m src.main --provider local   # Use local LLM (sequential)
    python -m src.main --provider api     # Use Anthropic API (parallel)
    python -m src.main --resume           # Resume from checkpoint
"""

from dotenv import load_dotenv

load_dotenv()  # Load .env file (for ANTHROPIC_API_KEY, etc.)

import argparse

from rich.console import Console

# Backward-compatible re-exports for tests
from src.cli import (
    # Types
    ValidationResult,
    PipelineContext,
    # Display
    get_banner,
    show_banner,
    format_file_size,
    format_elapsed_time,
    get_coming_soon_message,
    show_results,
    # Validation
    DEFAULT_INPUT_PATH,
    SUPPORTED_EXPORTS,
    validate_input_file,
    find_memories_json,
    is_supported_export,
    # Providers
    PROVIDER_LOCAL,
    PROVIDER_API,
    get_provider,
    select_provider,
    # Checkpoints
    CHECKPOINT_DIR,
    get_checkpoint_path,
    check_existing_checkpoint,
    # Phases
    DEMO_CHUNK_SIZE,
    phase_parse,
    phase_select_provider,
    phase_chunk,
    phase_check_resume,
    phase_extract,
    phase_aggregate,
    process_sequential_with_checkpoints,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace with input, demo, provider, and resume.

    Example:
        >>> args = parse_args()
        >>> print(args.input)
        conversations.json
    """
    parser = argparse.ArgumentParser(
        description="LLM Importer - Extract AI memories from ChatGPT/Claude exports"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_INPUT_PATH,
        help=f"Input file (default: {DEFAULT_INPUT_PATH})"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Demo mode: 4k chunks, first chunk only"
    )
    parser.add_argument(
        "--provider",
        choices=[PROVIDER_LOCAL, PROVIDER_API],
        help="LLM provider (default: prompt)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--tpm",
        type=int,
        default=None,
        help="Tokens per minute limit for API mode (default: 20000 for Tier 1)"
    )
    return parser.parse_args()


def main() -> int:
    """Main CLI entry point.

    Orchestrates the full pipeline:
    1. Parse arguments and validate input
    2. Parse conversations and load trusted context
    3. Select LLM provider
    4. Chunk conversations
    5. Check for resume state
    6. Extract facts from chunks
    7. Aggregate results

    Returns:
        Exit code (0 for success, 1 for error).

    Example:
        >>> exit_code = main()
        >>> sys.exit(exit_code)
    """
    args = parse_args()
    console = Console()
    show_banner(console)

    # Validate input file
    result = validate_input_file(args.input)
    if not result.valid:
        console.print(f"[red]Error:[/red] {result.error}")
        return 1
    console.print(f"[green]Found:[/green] {args.input} ({format_file_size(result.size_bytes)})")

    # Initialize context
    ctx = PipelineContext(
        input_file=args.input,
        console=console,
        demo_mode=args.demo,
        resume_mode=args.resume,
        tpm_override=args.tpm,
    )

    # Run phases
    try:
        ctx = phase_parse(ctx)
        ctx = phase_select_provider(ctx, args.provider)
        ctx = phase_chunk(ctx)
        ctx = phase_check_resume(ctx)
        ctx = phase_extract(ctx)
        ctx = phase_aggregate(ctx)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Progress saved to checkpoint.[/yellow]")
        return 1
    except Exception as e:
        console.print(f"\n[red]Error during extraction:[/red] {e}")
        if ctx.provider and ctx.provider.is_local:
            console.print("[dim]Make sure LM Studio is running at http://127.0.0.1:1234[/dim]")
        return 1

    # Show results
    if ctx.aggregated_facts:
        show_results(console, ctx.aggregated_facts)

    console.print("\n[green]Done![/green]")
    return 0


if __name__ == "__main__":
    exit(main())
