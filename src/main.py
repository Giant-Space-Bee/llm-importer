"""
main.py - CLI entry point for LLM Importer

Full pipeline: Parse → Chunk → Extract → Verify → Aggregate

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
import hashlib
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from src.parser import (
    parse_all,
    get_stats_from_parsed,
    detect_export_type,
    load_conversations,
    load_claude_memories,
    format_memories_for_distiller,
    format_user_profile_for_distiller,
    ExportType,
    ClaudeMemories,
)
from src.chunker import chunk_conversations, Chunk, DEFAULT_CHUNK_SIZE
from src.providers import LocalProvider, APIProvider, LLMProvider
from src.extractor import ExtractedFact
from src.processor import extract_and_verify_chunk, process_all_chunks
from src.aggregator import aggregate, group_by_category, AggregatedFact
from src.checkpoint import (
    hash_file,
    save_checkpoint,
    load_checkpoint,
    should_resume,
    get_remaining_chunks,
)


# Supported export types
SUPPORTED_EXPORTS = {"chatgpt", "claude"}


def find_memories_json(input_path: str) -> Optional[Path]:
    """
    Find memories.json for a Claude export.

    Claude exports are folders containing multiple JSON files.
    This function handles both:
    - Direct folder path: /path/to/data-timestamp-batch-N/
    - File path: /path/to/data-timestamp-batch-N/conversations.json

    Args:
        input_path: Path to Claude export (folder or conversations.json)

    Returns:
        Path to memories.json if found, None otherwise
    """
    path = Path(input_path)

    # If it's a file, check parent folder
    if path.is_file():
        folder = path.parent
    else:
        folder = path

    memories_path = folder / "memories.json"
    if memories_path.exists():
        return memories_path

    return None


def is_supported_export(export_type: ExportType) -> bool:
    """Check if an export type is currently supported."""
    return export_type in SUPPORTED_EXPORTS


def get_coming_soon_message(export_type: ExportType) -> str:
    """
    Get a friendly message for unsupported export types.

    Args:
        export_type: The detected export type

    Returns:
        User-friendly message explaining support status
    """
    # Both ChatGPT and Claude are now supported
    return (
        "Export format not recognized.\n"
        "Currently supported: ChatGPT, Claude\n"
        "More formats coming soon!"
    )

# Constants
DEFAULT_INPUT_PATH = "conversations.json"
PROVIDER_LOCAL = "local"
PROVIDER_API = "api"
DEMO_CHUNK_SIZE = 4096  # Smaller chunks for demo mode
CHECKPOINT_DIR = Path("checkpoints")


def get_checkpoint_path(input_file: str) -> Path:
    """
    Get checkpoint path for a given input file.

    Uses hash of absolute path to ensure unique checkpoint per input file,
    avoiding collisions when files have the same name in different directories.
    """
    abs_path = Path(input_file).resolve()
    path_hash = hashlib.sha256(str(abs_path).encode()).hexdigest()[:12]
    return CHECKPOINT_DIR / f"checkpoint_{path_hash}.json"


def get_provider(choice: str) -> LLMProvider:
    """
    Create and return the appropriate LLM provider.

    Args:
        choice: Either PROVIDER_LOCAL or PROVIDER_API

    Returns:
        LLMProvider instance

    Raises:
        ValueError: If choice is invalid or API key missing for API provider
    """
    if choice == PROVIDER_LOCAL:
        return LocalProvider()
    elif choice == PROVIDER_API:
        return APIProvider()  # Will raise ValueError if no API key
    else:
        raise ValueError(f"Invalid provider choice: {choice}. Use '{PROVIDER_LOCAL}' or '{PROVIDER_API}'")


def select_provider(console: Console, choice: Optional[str] = None) -> LLMProvider:
    """
    Select LLM provider interactively or from CLI arg.

    Args:
        console: Rich console for output
        choice: Provider choice from CLI, or None to prompt

    Returns:
        LLMProvider instance
    """
    if choice is None:
        choice = Prompt.ask(
            "Select provider",
            choices=[PROVIDER_LOCAL, PROVIDER_API],
            default=PROVIDER_LOCAL
        )

    try:
        provider = get_provider(choice)
        if provider.is_local:
            console.print(f"[cyan]Using:[/cyan] Local LLM (sequential processing)")
        else:
            console.print(f"[cyan]Using:[/cyan] Anthropic API (parallel processing)")
        return provider
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)


@dataclass
class ValidationResult:
    """Result of validating an input file."""
    valid: bool
    size_bytes: int
    error: Optional[str]


def get_banner() -> str:
    """Return the CLI banner text."""
    return """
    LLM IMPORTER
    Extract your AI memories

    Import your ChatGPT or Claude history into any AI assistant.
    """


def validate_input_file(path: str) -> ValidationResult:
    """
    Validate input file exists and get stats.

    Returns ValidationResult with:
    - valid: True if file exists and is readable
    - size_bytes: File size (0 if invalid)
    - error: Error message (None if valid)
    """
    if not path:
        return ValidationResult(valid=False, size_bytes=0, error="No path provided")

    file_path = Path(path)

    if not file_path.exists():
        return ValidationResult(
            valid=False,
            size_bytes=0,
            error=f"File not found: {path}"
        )

    if not file_path.is_file():
        return ValidationResult(
            valid=False,
            size_bytes=0,
            error=f"Path is not a file: {path}"
        )

    try:
        size = file_path.stat().st_size
        return ValidationResult(valid=True, size_bytes=size, error=None)
    except OSError as e:
        return ValidationResult(valid=False, size_bytes=0, error=str(e))


def format_file_size(size_bytes: int) -> str:
    """Convert bytes to human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def format_elapsed_time(seconds: float) -> str:
    """Format elapsed seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def check_existing_checkpoint(
    console: Console,
    input_file: str,
    use_resume: bool
) -> Tuple[List[int], List[Dict[str, Any]], int]:
    """
    Check for existing checkpoint and handle resume logic.

    Args:
        console: Rich console for output
        input_file: Path to input file
        use_resume: Whether --resume flag was passed

    Returns:
        Tuple of (remaining_chunk_indices, existing_facts, total_chunks)
        Note: total_chunks is 0 if not resuming (caller needs to compute it)
    """
    checkpoint_path = get_checkpoint_path(input_file)

    if should_resume(checkpoint_path, input_file):
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            completed = checkpoint.get("completed_chunks", [])
            existing_facts = checkpoint.get("verified_facts", [])

            if use_resume:
                console.print(
                    f"[yellow]Resuming from checkpoint:[/yellow] "
                    f"{len(completed)} chunks done, {len(existing_facts)} facts"
                )
                # Return with -1 for total_chunks to signal "use checkpoint info"
                return completed, existing_facts, -1
            else:
                console.print(
                    f"[yellow]Note:[/yellow] Valid checkpoint found "
                    f"({len(completed)} chunks, {len(existing_facts)} facts). "
                    f"Use --resume to continue, or this will start fresh."
                )

    return [], [], 0


def process_sequential_with_checkpoints(
    console: Console,
    chunks: List[Chunk],
    remaining_indices: List[int],
    provider: LLMProvider,
    conversations_by_id: Dict[str, Any],
    input_file: str,
    existing_facts: List[Dict[str, Any]]
) -> List[ExtractedFact]:
    """
    Process chunks sequentially with per-chunk checkpointing.

    Used for local LLM where each chunk takes minutes.
    Saves progress after each chunk so crashes don't lose work.

    Args:
        console: Rich console for output
        chunks: All chunks
        remaining_indices: Which chunk indices still need processing
        provider: LLM provider
        conversations_by_id: Raw conversation dicts for verification
        input_file: Path for checkpoint
        existing_facts: Previously extracted facts (from checkpoint)

    Returns:
        All verified facts (existing + new)
    """
    checkpoint_path = get_checkpoint_path(input_file)
    all_facts: List[Any] = list(existing_facts)  # May be dicts or ExtractedFact
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


def show_results(console: Console, aggregated: List[AggregatedFact]) -> None:
    """Display final aggregated results."""
    by_category = group_by_category(aggregated)

    # Summary table
    summary = Table(title="Extraction Summary", show_header=True)
    summary.add_column("Category", style="cyan")
    summary.add_column("Unique Facts", style="green", justify="right")
    summary.add_column("Total Mentions", style="yellow", justify="right")

    total_unique = 0
    total_mentions = 0
    categories = ["personal", "professional", "family", "preferences", "interests", "personality"]

    for cat in categories:
        facts = by_category.get(cat, [])
        unique = len(facts)
        mentions = sum(f.frequency for f in facts)
        total_unique += unique
        total_mentions += mentions
        if unique > 0:  # Only show categories with facts
            summary.add_row(cat, str(unique), str(mentions))

    summary.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total_unique}[/bold]",
        f"[bold]{total_mentions}[/bold]"
    )
    console.print(summary)

    # Top facts preview
    if aggregated:
        console.print("\n[bold]Top Facts (by frequency):[/bold]")
        sorted_facts = sorted(aggregated, key=lambda f: f.frequency, reverse=True)
        for f in sorted_facts[:10]:
            console.print(
                f"  [{f.fact.category}] {f.fact.fact[:60]}{'...' if len(f.fact.fact) > 60 else ''} "
                f"(x{f.frequency})"
            )


def show_banner(console: Console) -> None:
    """Display the welcome banner."""
    console.print(Panel(
        get_banner().strip(),
        title="[bold cyan]LLM IMPORTER[/bold cyan]",
        border_style="cyan"
    ))


def get_input_file(console: Console) -> Optional[str]:
    """Prompt user for input file path."""
    console.print()
    path = Prompt.ask(
        "[bold]Input file[/bold]",
        default=DEFAULT_INPUT_PATH
    )

    result = validate_input_file(path)

    if result.valid:
        console.print(f"[green]Found:[/green] {path} ({format_file_size(result.size_bytes)})")
        return path
    else:
        console.print(f"[red]Error:[/red] {result.error}")
        return None


def main():
    """Main CLI entry point."""
    # Parse arguments
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
    args = parser.parse_args()

    console = Console()
    show_banner(console)

    # Validate input file
    input_file = args.input
    result = validate_input_file(input_file)
    if not result.valid:
        console.print(f"[red]Error:[/red] {result.error}")
        return 1
    console.print(f"[green]Found:[/green] {input_file} ({format_file_size(result.size_bytes)})")

    # Check for existing checkpoint
    completed_chunks, existing_facts, _ = check_existing_checkpoint(
        console, input_file, args.resume
    )

    # Parse conversations
    console.print("\n[bold]Loading and parsing conversations...[/bold]")
    conversations, user_profile = parse_all(input_file)
    stats = get_stats_from_parsed(conversations, user_profile)

    # Load raw conversations for verification (need full message tree)
    raw_convos = load_conversations(input_file)
    conversations_by_id = {c["id"]: c for c in raw_convos}

    # Detect export type for Claude memories
    export_type = detect_export_type(raw_convos)

    # Show stats table
    table = Table(title="Conversation Stats", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Conversations", str(stats.total_conversations))
    table.add_row("Total Messages", str(stats.total_messages))
    table.add_row("User Messages", str(stats.user_messages))
    table.add_row("Total Characters", f"{stats.total_chars:,}")
    console.print(table)

    # Show user profile (free wins!)
    if user_profile:
        console.print("\n[bold green]Found user profile (custom instructions)![/bold green]")
        console.print(Panel(
            user_profile.user_profile[:500] + ("..." if len(user_profile.user_profile) > 500 else ""),
            title="User Profile Preview",
            border_style="green"
        ))

    # Load Claude memories if available (trusted baseline)
    claude_memories: Optional[ClaudeMemories] = None
    trusted_context: Optional[str] = None
    if export_type == "claude":
        memories_path = find_memories_json(input_file)
        if memories_path:
            claude_memories = load_claude_memories(str(memories_path.parent))
            if claude_memories:
                trusted_context = format_memories_for_distiller(claude_memories)
                mem_preview = claude_memories.conversations_memory[:300]
                if len(claude_memories.conversations_memory) > 300:
                    mem_preview += "..."
                console.print("\n[bold green]Found Claude memories (trusted baseline)![/bold green]")
                console.print(Panel(
                    mem_preview,
                    title=f"Claude Memories ({len(claude_memories.project_memories)} projects)",
                    border_style="green"
                ))

    # Use ChatGPT user profile as trusted baseline if no Claude memories
    # Priority: Claude memories > ChatGPT user profile (Claude is richer)
    if trusted_context is None and user_profile:
        trusted_context = format_user_profile_for_distiller(user_profile)
        console.print("\n[bold green]Using custom instructions as trusted baseline[/bold green]")

    # Chunk conversations
    chunk_size = DEMO_CHUNK_SIZE if args.demo else DEFAULT_CHUNK_SIZE
    console.print(f"\n[bold]Chunking conversations...[/bold] (max {chunk_size:,} tokens)")
    chunks = chunk_conversations(conversations, max_tokens=chunk_size)

    # Demo mode: first chunk only
    if args.demo:
        chunks = chunks[:1]
        console.print("[yellow]Demo mode:[/yellow] Processing first chunk only")

    # Show chunk stats
    chunk_table = Table(title="Chunk Breakdown", show_header=True)
    chunk_table.add_column("Chunk", style="cyan", justify="right")
    chunk_table.add_column("Conversations", style="green", justify="right")
    chunk_table.add_column("Tokens", style="yellow", justify="right")

    for chunk in chunks[:10]:
        chunk_table.add_row(
            str(chunk.id),
            str(len(chunk.conversations)),
            f"{chunk.token_count:,}"
        )

    if len(chunks) > 10:
        chunk_table.add_row("...", "...", "...")
        chunk_table.add_row(
            f"Total: {len(chunks)}",
            str(sum(len(c.conversations) for c in chunks)),
            f"{sum(c.token_count for c in chunks):,}"
        )
    console.print(chunk_table)

    # Select provider
    provider = select_provider(console, args.provider)

    # Compute remaining chunks
    if args.resume and completed_chunks:
        remaining_indices = get_remaining_chunks(
            {"completed_chunks": completed_chunks},
            len(chunks)
        )
    else:
        remaining_indices = list(range(len(chunks)))
        existing_facts = []

    if not remaining_indices:
        console.print("\n[green]All chunks already processed![/green]")
    else:
        console.print(f"\n[bold]Processing {len(remaining_indices)} chunks...[/bold]")

    # Process chunks
    try:
        if provider.is_local:
            # Local LLM: sequential with per-chunk checkpointing
            verified_facts = process_sequential_with_checkpoints(
                console,
                chunks,
                remaining_indices,
                provider,
                conversations_by_id,
                input_file,
                existing_facts
            )
        else:
            # API: parallel processing, checkpoint at end
            start_time = time.time()
            console.print("[dim]Processing in parallel...[/dim]")
            new_facts = process_all_chunks(chunks, provider, conversations_by_id)
            elapsed = time.time() - start_time
            console.print(
                f"  [green]Extracted {len(new_facts)} verified facts[/green] "
                f"[{format_elapsed_time(elapsed)}]"
            )
            verified_facts = list(existing_facts) + new_facts

            # Save final checkpoint
            checkpoint_path = get_checkpoint_path(input_file)
            save_checkpoint(checkpoint_path, {
                "source_file_hash": hash_file(input_file),
                "completed_chunks": list(range(len(chunks))),
                "verified_facts": [
                    asdict(f) if isinstance(f, ExtractedFact) else f
                    for f in verified_facts
                ]
            })

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Progress saved to checkpoint.[/yellow]")
        return 1
    except Exception as e:
        console.print(f"\n[red]Error during extraction:[/red] {e}")
        if provider.is_local:
            console.print("[dim]Make sure LM Studio is running at http://127.0.0.1:1234[/dim]")
        return 1

    # Aggregate
    if verified_facts:
        console.print("\n[bold]Aggregating facts...[/bold]")
        aggregated = aggregate(verified_facts)
        show_results(console, aggregated)
    else:
        console.print("\n[yellow]No facts extracted.[/yellow]")

    console.print("\n[green]Done![/green]")
    return 0


if __name__ == "__main__":
    exit(main())
