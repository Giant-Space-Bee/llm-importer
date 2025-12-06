"""
CLI type definitions for the LLM Importer pipeline.

This module contains dataclasses used throughout the CLI:
- ValidationResult: Result of input file validation
- PipelineContext: State container passed between pipeline phases
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.parser import (
    Conversation,
    UserProfile,
    ConversationStats,
    ExportType,
    ClaudeMemories,
)
from src.chunker import Chunk
from src.providers import LLMProvider
from src.extractor import ExtractedFact
from src.aggregator import AggregatedFact


@dataclass
class ValidationResult:
    """Result of validating an input file.

    Attributes:
        valid: True if file exists and is readable.
        size_bytes: File size in bytes (0 if invalid).
        error: Error message describing the validation failure, or None if valid.

    Example:
        >>> result = ValidationResult(valid=True, size_bytes=1024, error=None)
        >>> if result.valid:
        ...     print(f"File size: {result.size_bytes}")
    """

    valid: bool
    size_bytes: int
    error: Optional[str]


@dataclass
class PipelineContext:
    """State container passed between CLI phases.

    Initialized with required inputs, then populated progressively as each
    phase completes:

    1. phase_parse: conversations, stats, export_type, trusted_context
    2. phase_select_provider: provider, chunk_size, max_concurrent
    3. phase_chunk: chunks
    4. phase_check_resume: remaining_indices, existing_facts
    5. phase_extract: verified_facts
    6. phase_aggregate: aggregated_facts

    Attributes:
        input_file: Path to the input JSON file.
        console: Rich Console instance for output.
        demo_mode: If True, process only first chunk with smaller size.
        resume_mode: If True, resume from existing checkpoint.
        tpm_override: Optional tokens-per-minute override for API rate limiting.
        conversations: Parsed conversation objects.
        raw_conversations_by_id: Raw conversation dicts keyed by ID for verification.
        user_profile: Extracted user profile (custom instructions).
        claude_memories: Claude memories if available.
        trusted_context: Formatted trusted baseline for distiller.
        stats: Conversation statistics.
        export_type: Detected export type (chatgpt, claude, etc).
        provider: Selected LLM provider instance.
        chunk_size: Token limit per chunk.
        max_concurrent: Maximum concurrent API requests.
        chunks: Chunked conversation batches.
        remaining_indices: Chunk indices still needing processing.
        existing_facts: Previously extracted facts from checkpoint.
        verified_facts: Facts that passed verification.
        aggregated_facts: Final deduplicated and aggregated facts.

    Example:
        >>> ctx = PipelineContext(
        ...     input_file="conversations.json",
        ...     console=Console(),
        ... )
        >>> ctx = phase_parse(ctx)
        >>> print(f"Found {len(ctx.conversations)} conversations")
    """

    # Required inputs
    input_file: str
    console: Any  # Rich Console - using Any to avoid import in type hint

    # Optional inputs with defaults
    demo_mode: bool = False
    resume_mode: bool = False
    tpm_override: Optional[int] = None

    # Phase outputs (populated progressively)
    conversations: List[Conversation] = field(default_factory=list)
    raw_conversations_by_id: Dict[str, Any] = field(default_factory=dict)
    user_profile: Optional[UserProfile] = None
    claude_memories: Optional[ClaudeMemories] = None
    trusted_context: Optional[str] = None
    stats: Optional[ConversationStats] = None
    export_type: Optional[ExportType] = None
    provider: Optional[LLMProvider] = None
    chunk_size: int = 65536
    max_concurrent: int = 1
    chunks: List[Chunk] = field(default_factory=list)
    remaining_indices: List[int] = field(default_factory=list)
    existing_facts: List[Dict[str, Any]] = field(default_factory=list)
    verified_facts: List[ExtractedFact] = field(default_factory=list)
    aggregated_facts: List[AggregatedFact] = field(default_factory=list)
