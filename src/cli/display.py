"""
CLI display functions for Rich console output.

This module handles all visual output: banners, tables, progress indicators,
and result summaries using the Rich library.
"""

from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.parser import ConversationStats, UserProfile, ClaudeMemories
from src.chunker import Chunk
from src.aggregator import AggregatedFact, group_by_category


def get_banner() -> str:
    """Return the CLI banner text.

    Returns:
        Multi-line string containing the welcome banner.

    Example:
        >>> banner = get_banner()
        >>> print(banner)
    """
    return """
    LLM IMPORTER
    Extract your AI memories

    Import your ChatGPT or Claude history into any AI assistant.
    """


def show_banner(console: Console) -> None:
    """Display the welcome banner in a styled panel.

    Args:
        console: Rich console instance for output.

    Example:
        >>> console = Console()
        >>> show_banner(console)
    """
    console.print(Panel(
        get_banner().strip(),
        title="[bold cyan]LLM IMPORTER[/bold cyan]",
        border_style="cyan"
    ))


def format_file_size(size_bytes: int) -> str:
    """Convert bytes to human-readable string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Human-readable size string (e.g., "1.5 MB").

    Example:
        >>> format_file_size(1024)
        '1.0 KB'
        >>> format_file_size(1048576)
        '1.00 MB'
    """
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def format_elapsed_time(seconds: float) -> str:
    """Format elapsed seconds as human-readable string.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        Human-readable duration (e.g., "2m 30s").

    Example:
        >>> format_elapsed_time(90)
        '1m 30s'
        >>> format_elapsed_time(3661)
        '1h 1m'
    """
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


def get_coming_soon_message(export_type: str) -> str:
    """Get a friendly message for unsupported export types.

    Args:
        export_type: The detected export type.

    Returns:
        User-friendly message explaining support status.

    Example:
        >>> msg = get_coming_soon_message("unknown")
        >>> print(msg)
    """
    return (
        "Export format not recognized.\n"
        "Currently supported: ChatGPT, Claude\n"
        "More formats coming soon!"
    )


def show_stats_table(console: Console, stats: ConversationStats) -> None:
    """Display conversation statistics in a formatted table.

    Args:
        console: Rich console instance for output.
        stats: Conversation statistics to display.

    Example:
        >>> console = Console()
        >>> show_stats_table(console, stats)
    """
    table = Table(title="Conversation Stats", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Conversations", str(stats.total_conversations))
    table.add_row("Total Messages", str(stats.total_messages))
    table.add_row("User Messages", str(stats.user_messages))
    table.add_row("Total Characters", f"{stats.total_chars:,}")
    console.print(table)


def show_user_profile_preview(
    console: Console,
    user_profile: UserProfile,
    max_chars: int = 500
) -> None:
    """Display a preview of the user profile (custom instructions).

    Args:
        console: Rich console instance for output.
        user_profile: User profile to preview.
        max_chars: Maximum characters to show before truncating.

    Example:
        >>> console = Console()
        >>> show_user_profile_preview(console, profile)
    """
    console.print("\n[bold green]Found user profile (custom instructions)![/bold green]")
    preview = user_profile.user_profile[:max_chars]
    if len(user_profile.user_profile) > max_chars:
        preview += "..."
    console.print(Panel(
        preview,
        title="User Profile Preview",
        border_style="green"
    ))


def show_claude_memories_preview(
    console: Console,
    memories: ClaudeMemories,
    max_chars: int = 300
) -> None:
    """Display a preview of Claude memories (trusted baseline).

    Args:
        console: Rich console instance for output.
        memories: Claude memories to preview.
        max_chars: Maximum characters to show before truncating.

    Example:
        >>> console = Console()
        >>> show_claude_memories_preview(console, memories)
    """
    preview = memories.conversations_memory[:max_chars]
    if len(memories.conversations_memory) > max_chars:
        preview += "..."
    console.print("\n[bold green]Found Claude memories (trusted baseline)![/bold green]")
    console.print(Panel(
        preview,
        title=f"Claude Memories ({len(memories.project_memories)} projects)",
        border_style="green"
    ))


def show_chunk_table(
    console: Console,
    chunks: List[Chunk],
    max_rows: int = 10
) -> None:
    """Display chunk breakdown in a formatted table.

    Args:
        console: Rich console instance for output.
        chunks: List of chunks to display.
        max_rows: Maximum rows to show before summarizing.

    Example:
        >>> console = Console()
        >>> show_chunk_table(console, chunks)
    """
    table = Table(title="Chunk Breakdown", show_header=True)
    table.add_column("Chunk", style="cyan", justify="right")
    table.add_column("Conversations", style="green", justify="right")
    table.add_column("Tokens", style="yellow", justify="right")

    for chunk in chunks[:max_rows]:
        table.add_row(
            str(chunk.id),
            str(len(chunk.conversations)),
            f"{chunk.token_count:,}"
        )

    if len(chunks) > max_rows:
        table.add_row("...", "...", "...")
        table.add_row(
            f"Total: {len(chunks)}",
            str(sum(len(c.conversations) for c in chunks)),
            f"{sum(c.token_count for c in chunks):,}"
        )

    console.print(table)


def show_results(console: Console, aggregated: List[AggregatedFact]) -> None:
    """Display final aggregated results with summary and top facts.

    Args:
        console: Rich console instance for output.
        aggregated: List of aggregated facts to display.

    Example:
        >>> console = Console()
        >>> show_results(console, aggregated_facts)
    """
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
