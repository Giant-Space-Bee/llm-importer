"""
main.py - CLI entry point

Stage 1: CLI Shell
- Show banner
- Accept input file (default: conversations.json)
- Validate file exists
- Show file stats
- Exit cleanly

Stage 2: Parser
- Load and parse conversations.json
- Show stats (conversations, messages, chars)
- Extract user_editable_context (free wins!)

Stage 3: Chunker
- Batch conversations into token-limited chunks
- Show chunk breakdown

Stage 4: Extractor
- Connect to local LLM
- Extract facts from first chunk
- Show extracted facts with source_quote
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from src.parser import (
    parse_all,
    get_stats_from_parsed,
    detect_export_type,
    load_conversations,
    ExportType,
)
from src.chunker import chunk_conversations, DEFAULT_CHUNK_SIZE
from src.providers import LocalProvider, APIProvider, LLMProvider
from src.extractor import extract_chunk


# Supported export types
SUPPORTED_EXPORTS = {"chatgpt"}


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
    if export_type == "claude":
        return (
            "Claude exports are coming soon!\n"
            "Currently supported: ChatGPT\n"
            "Claude support is on the roadmap."
        )
    else:
        return (
            "Export format not recognized.\n"
            "Currently supported: ChatGPT\n"
            "More formats coming soon!"
        )

# Constants
DEFAULT_INPUT_PATH = "conversations.json"
PROVIDER_LOCAL = "local"
PROVIDER_API = "api"


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

    Import your ChatGPT history into any AI assistant.
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
    console = Console()

    show_banner(console)

    input_file = get_input_file(console)

    if input_file is None:
        console.print("\n[yellow]Exiting.[/yellow]")
        return 1

    # Stage 2: Parse and show stats (single load)
    console.print("\n[bold]Loading and parsing conversations...[/bold]")
    conversations, user_profile = parse_all(input_file)
    stats = get_stats_from_parsed(conversations, user_profile)

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

    # Stage 3: Chunk conversations
    console.print("\n[bold]Chunking conversations...[/bold]")
    chunks = chunk_conversations(conversations, max_tokens=DEFAULT_CHUNK_SIZE)

    # Show chunk stats
    chunk_table = Table(title="Chunk Breakdown", show_header=True)
    chunk_table.add_column("Chunk", style="cyan", justify="right")
    chunk_table.add_column("Conversations", style="green", justify="right")
    chunk_table.add_column("Tokens", style="yellow", justify="right")

    for chunk in chunks[:10]:  # Show first 10
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

    # Stage 4: Extract from first chunk
    console.print("\n[bold]Stage 4: Extraction[/bold]")
    console.print(f"Ready to extract facts from chunk 0 ({chunks[0].token_count:,} tokens)")
    console.print("[dim]This will call your local LLM at http://127.0.0.1:1234[/dim]")

    proceed = Prompt.ask("\nProceed with extraction?", choices=["y", "n"], default="y")
    if proceed != "y":
        console.print("\n[yellow]Skipping extraction.[/yellow]")
        return 0

    console.print("\n[bold cyan]Calling LLM...[/bold cyan] (this may take a minute)")

    try:
        provider = LocalProvider()
        facts = extract_chunk(chunks[0], provider)

        console.print(f"\n[bold green]Extracted {len(facts)} facts![/bold green]")

        # Show the facts
        if facts:
            fact_table = Table(title="Extracted Facts (First 10)", show_header=True)
            fact_table.add_column("Category", style="cyan")
            fact_table.add_column("Fact", style="green")
            fact_table.add_column("Source Quote", style="yellow", max_width=40)

            for fact in facts[:10]:
                quote_preview = fact.source_quote[:37] + "..." if len(fact.source_quote) > 40 else fact.source_quote
                fact_table.add_row(fact.category, fact.fact, quote_preview)

            if len(facts) > 10:
                fact_table.add_row("...", f"({len(facts) - 10} more)", "...")

            console.print(fact_table)

    except Exception as e:
        console.print(f"\n[red]Error during extraction:[/red] {e}")
        console.print("[dim]Make sure LM Studio is running at http://127.0.0.1:1234[/dim]")
        return 1

    # Stage 4 complete
    console.print("\n[dim]Stage 4 complete. More stages coming soon...[/dim]")
    return 0


if __name__ == "__main__":
    exit(main())
