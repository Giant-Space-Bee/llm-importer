"""
CLI display functions for Rich console output.

This module handles all visual output: banners, tables, progress indicators,
and result summaries using the Rich library.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.parser import ConversationStats, UserProfile, ClaudeMemories
from src.chunker import Chunk
from src.aggregator import AggregatedFact, group_by_category
from src.deduplicator import DeduplicatedFact
from src.distiller import DistilledProfile
from src.config import (
    USER_PROFILE_PREVIEW_CHARS,
    CLAUDE_MEMORIES_PREVIEW_CHARS,
    CHUNK_TABLE_MAX_ROWS,
    TOP_FACTS_DISPLAY_COUNT,
    FACT_TRUNCATE_SHORT,
    FACT_TRUNCATE_MEDIUM,
    HORIZONTAL_RULE_WIDTH,
)


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
    max_chars: int = USER_PROFILE_PREVIEW_CHARS
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
    max_chars: int = CLAUDE_MEMORIES_PREVIEW_CHARS
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
    max_rows: int = CHUNK_TABLE_MAX_ROWS
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
        for f in sorted_facts[:TOP_FACTS_DISPLAY_COUNT]:
            truncated = f.fact.fact[:FACT_TRUNCATE_SHORT]
            ellipsis = '...' if len(f.fact.fact) > FACT_TRUNCATE_SHORT else ''
            console.print(
                f"  [{f.fact.category}] {truncated}{ellipsis} "
                f"(x{f.frequency})"
            )


def show_dedup_results(
    console: Console,
    facts: List[DeduplicatedFact],
    input_count: int,
) -> None:
    """Display deduplicated results with reduction stats.

    Shows summary table by category and lists all unique facts.
    Includes engineering data: input count, output count, reduction ratio.

    Args:
        console: Rich console instance for output.
        facts: List of deduplicated facts.
        input_count: Number of facts before deduplication (for stats).

    Example:
        >>> console = Console()
        >>> show_dedup_results(console, dedup_facts, input_count=100)
    """
    # Group by category
    by_category: Dict[str, List[DeduplicatedFact]] = defaultdict(list)
    for f in facts:
        by_category[f.category].append(f)

    # Calculate reduction
    output_count = len(facts)
    reduction_pct = ((input_count - output_count) / input_count * 100) if input_count > 0 else 0

    # Summary table
    summary = Table(title="Deduplication Results", show_header=True)
    summary.add_column("Category", style="cyan")
    summary.add_column("Facts", style="green", justify="right")

    categories = ["personal", "professional", "family", "preferences", "interests", "personality"]
    total = 0
    for cat in categories:
        count = len(by_category.get(cat, []))
        total += count
        if count > 0:
            summary.add_row(cat, str(count))

    summary.add_row("[bold]TOTAL[/bold]", f"[bold]{total}[/bold]")
    console.print(summary)

    # Engineering stats
    console.print(
        f"\n[dim]Dedup stats: {input_count} input → {output_count} output "
        f"({reduction_pct:.1f}% reduction)[/dim]"
    )

    # Show all facts by category
    if facts:
        console.print("\n[bold]Deduplicated Facts:[/bold]")
        for cat in categories:
            cat_facts = by_category.get(cat, [])
            if cat_facts:
                console.print(f"\n  [cyan]{cat.upper()}[/cyan]")
                for f in cat_facts:
                    # Truncate long facts for display
                    truncated = f.fact[:FACT_TRUNCATE_MEDIUM]
                    ellipsis = "..." if len(f.fact) > FACT_TRUNCATE_MEDIUM else ""
                    console.print(f"    - {truncated}{ellipsis}")


def show_final_results(
    console: Console,
    profile: DistilledProfile,
    md_path: Path,
    json_path: Path,
) -> None:
    """Display final results summary with profile preview.

    Shows a summary table of facts by category, lists output file paths,
    and previews the first few facts from select categories.

    Args:
        console: Rich console instance for output.
        profile: The distilled profile to display.
        md_path: Path to the written markdown file.
        json_path: Path to the written JSON file.

    Example:
        >>> console = Console()
        >>> show_final_results(console, profile, md_path, json_path)
    """
    categories = ["personal", "professional", "family", "preferences", "interests", "personality"]

    # Category summary table
    table = Table(title=f"Memory Profile: {profile.name}", show_header=True)
    table.add_column("Category", style="cyan")
    table.add_column("Facts", style="green", justify="right")

    total = 0
    for cat in categories:
        count = len(profile.categories.get(cat, []))
        total += count
        if count > 0:
            table.add_row(cat.title(), str(count))

    table.add_row("[bold]TOTAL[/bold]", f"[bold]{total}[/bold]")
    console.print(table)

    # Output files
    console.print(f"\n[bold]Output Files:[/bold]")
    console.print(f"  {md_path}")
    console.print(f"  {json_path}")

    # Preview first few facts from each category
    console.print(f"\n[bold]Preview:[/bold]")
    for cat in ["personal", "professional"]:
        facts = profile.categories.get(cat, [])[:2]
        if facts:
            console.print(f"  [cyan]{cat.title()}:[/cyan]")
            for f in facts:
                truncated = f[:FACT_TRUNCATE_SHORT]
                ellipsis = '...' if len(f) > FACT_TRUNCATE_SHORT else ''
                console.print(f"    - {truncated}{ellipsis}")


def show_phase_header(console: Console, phase_num: int, name: str) -> None:
    """Display phase header with horizontal rule.

    Creates a clear visual separator between pipeline phases with
    consistent formatting.

    Args:
        console: Rich console instance for output.
        phase_num: Phase number (1-based).
        name: Name of the phase (e.g., "PARSE", "EXTRACTION").

    Example:
        >>> console = Console()
        >>> show_phase_header(console, 1, "PARSE")
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
         PHASE 1: PARSE
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    console.print(f"\n{'━' * HORIZONTAL_RULE_WIDTH}")
    console.print(f" [bold]PHASE {phase_num}: {name}[/bold]")
    console.print('━' * HORIZONTAL_RULE_WIDTH)


def show_phase_complete(console: Console, elapsed: float) -> None:
    """Display phase completion with elapsed time.

    Args:
        console: Rich console instance for output.
        elapsed: Elapsed time in seconds.

    Example:
        >>> show_phase_complete(console, 125.3)
        Phase complete [2m 5s]
    """
    console.print(f"[dim]Phase complete [{format_elapsed_time(elapsed)}][/dim]")


def show_pipeline_summary(
    console: Console,
    stats,
    verified_count: int,
    hallucination_count: int,
    dedup_count: int,
    input_count: int,
    total_elapsed: float,
    output_path: Optional[Path] = None,
) -> None:
    """Display final pipeline summary with key statistics.

    Shows a comprehensive summary of the pipeline run including input stats,
    extraction results, deduplication ratio, and total time.

    Args:
        console: Rich console instance for output.
        stats: ConversationStats object with input statistics.
        verified_count: Number of facts extracted and verified.
        hallucination_count: Number of potential hallucinations detected.
        dedup_count: Number of facts after deduplication.
        input_count: Number of facts before deduplication.
        total_elapsed: Total pipeline elapsed time in seconds.
        output_path: Path to the output file (if any).

    Example:
        >>> show_pipeline_summary(console, stats, 321, 14, 180, 312, 503.2, Path("output/memory-profile.md"))
    """
    console.print(f"\n{'━' * HORIZONTAL_RULE_WIDTH}")
    console.print(" [bold]SUMMARY[/bold]")
    console.print('━' * HORIZONTAL_RULE_WIDTH)

    # Input stats
    if stats:
        console.print(
            f"Input:        {stats.total_conversations} conversations "
            f"({stats.total_chars:,} chars)"
        )

    # Extraction results
    if verified_count > 0:
        hallu_note = ""
        if hallucination_count > 0:
            hallu_note = f" ({hallucination_count} potential hallucinations)"
        console.print(f"Extracted:    {verified_count} facts{hallu_note}")

    # Dedup results
    if dedup_count > 0 and input_count > 0:
        reduction = ((input_count - dedup_count) / input_count) * 100
        console.print(
            f"Deduplicated: {dedup_count} final facts ({reduction:.0f}% reduction)"
        )

    # Total time
    console.print(f"Total time:   {format_elapsed_time(total_elapsed)}")

    # Output path
    if output_path:
        console.print(f"\nOutput: {output_path}")
