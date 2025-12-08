"""
Parse phase: Load and parse conversations from export file.

This phase loads the export file, detects the format (ChatGPT or Claude),
parses conversations, and loads any trusted context (memories or custom instructions).
"""

import time

from src.parser import (
    parse_from_raw,
    get_stats_from_parsed,
    detect_export_type,
    load_conversations,
    load_claude_memories,
    format_memories_for_distiller,
    format_user_profile_for_distiller,
)
from src.cli.types import PipelineContext
from src.cli.display import (
    show_stats_table,
    show_user_profile_preview,
    show_claude_memories_preview,
    show_phase_header,
    show_phase_complete,
)
from src.cli.validation import find_memories_json


def phase_parse(ctx: PipelineContext) -> PipelineContext:
    """Parse conversations and load trusted context.

    Loads and parses conversations from the input file, detects export type,
    and loads any available trusted context (Claude memories or ChatGPT profile).

    Args:
        ctx: Pipeline context with input_file set.

    Returns:
        Updated context with:
        - conversations: Parsed conversation objects
        - raw_conversations_by_id: Raw dicts for verification
        - user_profile: Extracted custom instructions (if any)
        - claude_memories: Claude memories (if available)
        - trusted_context: Formatted baseline for distiller
        - stats: Conversation statistics
        - export_type: Detected export format

    Example:
        >>> ctx = PipelineContext(input_file="conversations.json", console=console)
        >>> ctx = phase_parse(ctx)
        >>> print(f"Found {len(ctx.conversations)} conversations")
    """
    console = ctx.console
    start_time = time.time()

    show_phase_header(console, 1, "PARSE")

    # Single load - parse and keep raw for verification
    raw_convos = load_conversations(ctx.input_file)
    export_type = detect_export_type(raw_convos)
    conversations, user_profile = parse_from_raw(raw_convos, export_type)
    stats = get_stats_from_parsed(conversations, user_profile)

    # Build lookup using correct ID field (ChatGPT uses 'id', Claude uses 'uuid')
    id_field = "uuid" if export_type == "claude" else "id"

    # Check for duplicate conversation IDs (would cause silent data loss)
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    for c in raw_convos:
        cid = c.get(id_field) or ""
        if cid in seen_ids:
            duplicate_ids.add(cid)
        seen_ids.add(cid)

    if duplicate_ids:
        dup_list = sorted(duplicate_ids)  # Deterministic order for testing
        shown = dup_list[:5]
        remaining = len(dup_list) - 5
        raise ValueError(
            f"Duplicate conversation IDs found: {shown}"
            + (f" (and {remaining} more)" if remaining > 0 else "")
            + "\nThis may indicate a corrupted export file."
        )

    conversations_by_id = {c[id_field]: c for c in raw_convos}

    # Show stats
    show_stats_table(console, stats)

    # Show user profile (free wins!)
    if user_profile:
        show_user_profile_preview(console, user_profile)

    # Load Claude memories if available (trusted baseline)
    claude_memories = None
    trusted_context = None

    if export_type == "claude":
        memories_path = find_memories_json(ctx.input_file)
        if memories_path:
            claude_memories = load_claude_memories(str(memories_path.parent))
            if claude_memories:
                trusted_context = format_memories_for_distiller(claude_memories)
                show_claude_memories_preview(console, claude_memories)

    # Use ChatGPT user profile as trusted baseline if no Claude memories
    if trusted_context is None and user_profile:
        trusted_context = format_user_profile_for_distiller(user_profile)
        console.print("\n[bold green]Using custom instructions as trusted baseline[/bold green]")

    # Update context
    ctx.conversations = conversations
    ctx.raw_conversations_by_id = conversations_by_id
    ctx.user_profile = user_profile
    ctx.claude_memories = claude_memories
    ctx.trusted_context = trusted_context
    ctx.stats = stats
    ctx.export_type = export_type

    elapsed = time.time() - start_time
    ctx.phase_timings["parse"] = elapsed
    show_phase_complete(console, elapsed)

    return ctx
