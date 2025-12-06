"""
Checkpoint orchestration for resumable processing.

This module handles checkpoint path generation and resume logic,
wrapping the lower-level checkpoint module for CLI use.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rich.console import Console

from src.checkpoint import load_checkpoint, should_resume


# Checkpoint storage directory
CHECKPOINT_DIR = Path("checkpoints")


def get_checkpoint_path(input_file: str) -> Path:
    """Get checkpoint path for a given input file.

    Uses hash of absolute path to ensure unique checkpoint per input file,
    avoiding collisions when files have the same name in different directories.

    Args:
        input_file: Path to the input file.

    Returns:
        Path to the checkpoint file for this input.

    Example:
        >>> path = get_checkpoint_path("data/conversations.json")
        >>> print(path)
        checkpoints/checkpoint_a1b2c3d4e5f6.json
    """
    abs_path = Path(input_file).resolve()
    path_hash = hashlib.sha256(str(abs_path).encode()).hexdigest()[:12]
    return CHECKPOINT_DIR / f"checkpoint_{path_hash}.json"


def check_existing_checkpoint(
    console: Console,
    input_file: str,
    use_resume: bool
) -> Tuple[List[int], List[Dict[str, Any]], int]:
    """Check for existing checkpoint and handle resume logic.

    If a valid checkpoint exists:
    - With --resume: Returns completed chunks and existing facts to resume
    - Without --resume: Warns user about checkpoint and starts fresh

    Args:
        console: Rich console for output.
        input_file: Path to input file.
        use_resume: Whether --resume flag was passed.

    Returns:
        Tuple of (completed_chunk_indices, existing_facts, signal):
        - If resuming: (completed, facts, -1) where -1 signals "use checkpoint"
        - If not resuming: ([], [], 0) to start fresh

    Example:
        >>> console = Console()
        >>> completed, facts, sig = check_existing_checkpoint(
        ...     console, "conversations.json", use_resume=True
        ... )
        >>> if sig == -1:
        ...     print(f"Resuming with {len(facts)} existing facts")
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
                # Return with -1 for signal to indicate "use checkpoint info"
                return completed, existing_facts, -1
            else:
                console.print(
                    f"[yellow]Note:[/yellow] Valid checkpoint found "
                    f"({len(completed)} chunks, {len(existing_facts)} facts). "
                    f"Use --resume to continue, or this will start fresh."
                )

    return [], [], 0
