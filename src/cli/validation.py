"""
Input file validation and path helpers.

This module handles validating input files exist and are readable,
finding related files (like memories.json), and checking export type support.
"""

from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.prompt import Prompt

from src.parser import ExportType
from src.cli.types import ValidationResult
from src.cli.display import format_file_size


# Constants
DEFAULT_INPUT_PATH = "conversations.json"
SUPPORTED_EXPORTS = {"chatgpt", "claude"}


def validate_input_file(path: str) -> ValidationResult:
    """Validate that an input file exists and is readable.

    Checks that the path:
    1. Is not empty
    2. Points to an existing file (not directory)
    3. Is readable (can get file stats)

    Args:
        path: Path to the input file.

    Returns:
        ValidationResult with valid=True if file is accessible,
        or valid=False with an error message explaining the problem.

    Example:
        >>> result = validate_input_file("conversations.json")
        >>> if result.valid:
        ...     print(f"File size: {result.size_bytes}")
        ... else:
        ...     print(f"Error: {result.error}")
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


def find_memories_json(input_path: str) -> Optional[Path]:
    """Find memories.json for a Claude export.

    Claude exports are folders containing multiple JSON files.
    This function handles both:
    - Direct folder path: /path/to/data-timestamp-batch-N/
    - File path: /path/to/data-timestamp-batch-N/conversations.json

    Args:
        input_path: Path to Claude export (folder or conversations.json).

    Returns:
        Path to memories.json if found, None otherwise.

    Example:
        >>> path = find_memories_json("claude_export/conversations.json")
        >>> if path:
        ...     print(f"Found memories at: {path}")
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
    """Check if an export type is currently supported.

    Args:
        export_type: The detected export type.

    Returns:
        True if the export type is supported for processing.

    Example:
        >>> is_supported_export("chatgpt")
        True
        >>> is_supported_export("unknown")
        False
    """
    return export_type in SUPPORTED_EXPORTS


def get_input_file(console: Console) -> Optional[str]:
    """Prompt user for input file path interactively.

    Displays a prompt, validates the entered path, and shows
    success or error message.

    Args:
        console: Rich console instance for output.

    Returns:
        Validated file path if successful, None if invalid.

    Example:
        >>> console = Console()
        >>> path = get_input_file(console)
        >>> if path:
        ...     print(f"Using: {path}")
    """
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
