"""
CLI package for the LLM Importer.

This package provides all CLI-specific functionality including:
- Type definitions (ValidationResult, PipelineContext)
- Display functions (banners, tables, formatting)
- Input validation and path helpers
- Provider selection UI
- Checkpoint orchestration
- Pipeline phase functions

Example:
    >>> from src.cli import (
    ...     PipelineContext, show_banner, validate_input_file,
    ...     phase_parse, phase_extract, phase_aggregate,
    ... )
    >>> ctx = PipelineContext(input_file="conversations.json", console=console)
    >>> ctx = phase_parse(ctx)
"""

# Types
from src.cli.types import ValidationResult, PipelineContext

# Display functions
from src.cli.display import (
    get_banner,
    show_banner,
    format_file_size,
    format_elapsed_time,
    get_coming_soon_message,
    show_stats_table,
    show_user_profile_preview,
    show_claude_memories_preview,
    show_chunk_table,
    show_results,
)

# Validation
from src.cli.validation import (
    DEFAULT_INPUT_PATH,
    SUPPORTED_EXPORTS,
    validate_input_file,
    find_memories_json,
    is_supported_export,
    get_input_file,
)

# Providers
from src.cli.providers import (
    PROVIDER_LOCAL,
    PROVIDER_API,
    get_provider,
    select_provider,
)

# Checkpoints
from src.cli.checkpoints import (
    CHECKPOINT_DIR,
    get_checkpoint_path,
    check_existing_checkpoint,
)

# Phases
from src.cli.phases import (
    DEMO_CHUNK_SIZE,
    phase_parse,
    phase_select_provider,
    phase_chunk,
    phase_check_resume,
    phase_extract,
    phase_aggregate,
    process_sequential_with_checkpoints,
)

__all__ = [
    # Types
    "ValidationResult",
    "PipelineContext",
    # Display
    "get_banner",
    "show_banner",
    "format_file_size",
    "format_elapsed_time",
    "get_coming_soon_message",
    "show_stats_table",
    "show_user_profile_preview",
    "show_claude_memories_preview",
    "show_chunk_table",
    "show_results",
    # Validation
    "DEFAULT_INPUT_PATH",
    "SUPPORTED_EXPORTS",
    "validate_input_file",
    "find_memories_json",
    "is_supported_export",
    "get_input_file",
    # Providers
    "PROVIDER_LOCAL",
    "PROVIDER_API",
    "get_provider",
    "select_provider",
    # Checkpoints
    "CHECKPOINT_DIR",
    "get_checkpoint_path",
    "check_existing_checkpoint",
    # Phases
    "DEMO_CHUNK_SIZE",
    "phase_parse",
    "phase_select_provider",
    "phase_chunk",
    "phase_check_resume",
    "phase_extract",
    "phase_aggregate",
    "process_sequential_with_checkpoints",
]
