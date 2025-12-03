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
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from src.parser import (
    load_conversations,
    get_conversation_stats,
    extract_user_profile,
)

# Constants
DEFAULT_INPUT_PATH = "conversations.json"


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

    # Stage 2: Parse and show stats
    console.print("\n[bold]Loading conversations...[/bold]")
    conversations = load_conversations(input_file)
    stats = get_conversation_stats(conversations)

    # Show stats table
    table = Table(title="Conversation Stats", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Conversations", str(stats.total_conversations))
    table.add_row("Total Messages", str(stats.total_messages))
    table.add_row("User Messages", str(stats.user_messages))
    table.add_row("Total Characters", f"{stats.total_chars:,}")

    console.print(table)

    # Extract user profile (free wins!)
    user_profile = extract_user_profile(conversations)
    if user_profile:
        console.print("\n[bold green]Found user profile (custom instructions)![/bold green]")
        console.print(Panel(
            user_profile.user_profile[:500] + ("..." if len(user_profile.user_profile) > 500 else ""),
            title="User Profile Preview",
            border_style="green"
        ))

    # Stage 2 complete
    console.print("\n[dim]Stage 2 complete. More stages coming soon...[/dim]")
    return 0


if __name__ == "__main__":
    exit(main())
