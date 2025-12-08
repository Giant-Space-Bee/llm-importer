"""
Pipeline phase functions for the LLM Importer CLI.

This package contains all pipeline phase functions, each in its own module.
Imports are re-exported here for backward compatibility.

Example:
    >>> from src.cli.phases import phase_parse, phase_extract
    >>> ctx = phase_parse(ctx)
"""

# Migrated phases
from src.cli.phases.aggregate import phase_aggregate
from src.cli.phases.output import phase_output
from src.cli.phases.resume import phase_check_resume
from src.cli.phases.provider import phase_select_provider
from src.cli.phases.chunk import phase_chunk, DEMO_CHUNK_SIZE
from src.cli.phases.parse import phase_parse

# Re-export from legacy module during migration
from src.cli._phases_legacy import (
    process_sequential_with_checkpoints,
    phase_extract,
    phase_deduplicate,
    phase_distill,
)

__all__ = [
    "DEMO_CHUNK_SIZE",
    "phase_parse",
    "phase_select_provider",
    "phase_chunk",
    "phase_check_resume",
    "process_sequential_with_checkpoints",
    "phase_extract",
    "phase_aggregate",
    "phase_deduplicate",
    "phase_distill",
    "phase_output",
]
