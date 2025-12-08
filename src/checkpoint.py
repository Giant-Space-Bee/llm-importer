"""
checkpoint.py - Crash-resilient checkpointing for extraction

Save state after each chunk, resume on crash.
Atomic writes prevent corruption.
Source hash detects if input file changed (invalidates checkpoint).
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Optional, List, Union

from src.config import FILE_READ_BUFFER_SIZE


def hash_file(path: Union[str, Path]) -> str:
    """
    Compute SHA256 hash of file contents.

    Args:
        path: Path to file

    Returns:
        Hash string in format "sha256:<hex_digest>"
    """
    path = Path(path)
    sha256 = hashlib.sha256()

    with open(path, 'rb') as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(FILE_READ_BUFFER_SIZE), b''):
            sha256.update(chunk)

    return f"sha256:{sha256.hexdigest()}"


def save_checkpoint(path: Union[str, Path], state: dict) -> None:
    """
    Atomically save checkpoint state.

    Writes to temp file in same directory, then renames.
    Rename is atomic on POSIX, so crash during write won't corrupt.

    Args:
        path: Where to save checkpoint
        state: Checkpoint state dict
    """
    path = Path(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory (same filesystem for atomic rename)
    fd, temp_path = tempfile.mkstemp(dir=parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(state, f, indent=2)

        # Atomic rename (POSIX guarantees this)
        os.replace(temp_path, path)
    except Exception:
        # Clean up temp file on failure
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def load_checkpoint(path: Union[str, Path]) -> Optional[dict]:
    """
    Load checkpoint from file.

    Args:
        path: Path to checkpoint file

    Returns:
        Checkpoint dict, or None if file doesn't exist or is corrupted
    """
    path = Path(path)

    if not path.exists():
        return None

    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        # Corrupted or unreadable
        return None


def should_resume(
    checkpoint_path: Union[str, Path],
    source_path: Union[str, Path]
) -> bool:
    """
    Check if valid checkpoint exists for this source file.

    A checkpoint is valid if:
    - It exists and is readable
    - Its source_file_hash matches the current source file

    Args:
        checkpoint_path: Path to checkpoint file
        source_path: Path to source file being processed

    Returns:
        True if valid checkpoint exists, False otherwise
    """
    checkpoint = load_checkpoint(checkpoint_path)

    if checkpoint is None:
        return False

    # Check source file hash matches
    stored_hash = checkpoint.get("source_file_hash", "")
    current_hash = hash_file(source_path)

    return stored_hash == current_hash


def get_remaining_chunks(
    checkpoint: Optional[dict],
    total_chunks: int
) -> List[int]:
    """
    Get list of chunk indices that still need processing.

    Args:
        checkpoint: Loaded checkpoint dict, or None if no checkpoint
        total_chunks: Total number of chunks to process

    Returns:
        List of chunk indices not yet completed, in order
    """
    all_chunks = set(range(total_chunks))

    if checkpoint is None:
        return list(range(total_chunks))

    completed = set(checkpoint.get("completed_chunks", []))
    remaining = all_chunks - completed

    return sorted(remaining)
