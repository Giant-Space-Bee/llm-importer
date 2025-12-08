"""
config.py - Centralized configuration constants

All magic numbers in one place. Each constant is:
- Named descriptively
- Documented with its purpose
- Env-var configurable where appropriate (prefix: LLM_IMPORTER_)

See docs/magic-numbers.md for detailed explanations.
"""

import os

# =============================================================================
# CHUNKING CONSTANTS
# =============================================================================

# Default chunk size for batching conversations (tokens)
# Smaller = better LLM recall, larger = fewer API calls
# Env: LLM_IMPORTER_CHUNK_SIZE
_FALLBACK_CHUNK_SIZE = 8192  # 2^13 - good balance of quality and throughput
DEFAULT_CHUNK_SIZE = int(os.getenv("LLM_IMPORTER_CHUNK_SIZE", _FALLBACK_CHUNK_SIZE))

# Chunk size bounds
MIN_CHUNK_SIZE = 4096    # 2^12 - minimum viable for extraction
MAX_CHUNK_SIZE = 65536   # 2^16 - maximum before context issues

# Demo mode uses smaller chunks for quick testing
DEMO_CHUNK_SIZE = 4096

# Quality-optimal chunk size for API extraction
# Smaller chunks = better "lost in the middle" avoidance
QUALITY_CHUNK_SIZE = 8192  # 2^13

# Estimated tokens for conversation header (title, ID, dates)
HEADER_OVERHEAD_TOKENS = 50

# Characters to trim per iteration when splitting oversized text
CHUNK_TRIM_STEP = 100


# =============================================================================
# RATE LIMITING CONSTANTS
# =============================================================================

# Anthropic API defaults (Tier 1 limits)
_FALLBACK_TPM = 30000
DEFAULT_TPM = int(os.getenv("LLM_IMPORTER_TPM", _FALLBACK_TPM))
DEFAULT_RPM = 5  # Requests per minute

# Rate limit timing
SECONDS_PER_MINUTE = 60
RATE_LIMIT_BUFFER = 0.1  # Extra seconds to wait after rate limit window

# Parallelism limits
MAX_CONCURRENT_REQUESTS = 5  # Cap parallel API calls

# Token budget allocation
OUTPUT_RESERVE = 10000   # Reserved for output (8k output + 2k safety margin)
OUTPUT_ESTIMATE = 3000   # Conservative estimate for parallelism calculation


# =============================================================================
# RETRY CONSTANTS
# =============================================================================

# Exponential backoff for rate limit errors
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 5  # Doubles each retry: 5s, 10s, 20s


# =============================================================================
# LLM PROVIDER CONSTANTS
# =============================================================================

# Generation parameters
DEFAULT_TEMPERATURE = 0.3  # Lower = more deterministic for extraction
DEFAULT_MAX_TOKENS = 8192  # Output token limit

# Local LLM (LM Studio) defaults
LOCAL_LLM_HOST = "127.0.0.1"
LOCAL_LLM_PORT = 1234
LOCAL_LLM_BASE_URL = f"http://{LOCAL_LLM_HOST}:{LOCAL_LLM_PORT}/v1"
DEFAULT_LOCAL_MODEL = "lmstudio-community/ministral-3-14b-instruct-2512"

# Anthropic model
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
STRUCTURED_OUTPUTS_BETA = "structured-outputs-2025-11-13"


# =============================================================================
# VERIFICATION CONSTANTS
# =============================================================================

# Fuzzy matching for quote verification
# 0.85 = 85% similar - catches typos but rejects paraphrasing
_FALLBACK_FUZZY_THRESHOLD = 0.85
FUZZY_MATCH_THRESHOLD = float(
    os.getenv("LLM_IMPORTER_FUZZY_THRESHOLD", _FALLBACK_FUZZY_THRESHOLD)
)

# Short quotes need stricter matching to avoid false positives
SHORT_QUOTE_LENGTH = 20  # Quotes shorter than this get stricter threshold
SHORT_QUOTE_THRESHOLD = 0.90  # 90% match required for short quotes

# Fuzzy search window parameters
FUZZY_WINDOW_MIN = 10      # Minimum window size
FUZZY_WINDOW_MARGIN = 10   # How much smaller/larger than quote to search
FUZZY_WINDOW_EXPAND = 20   # How much larger than quote to search
FUZZY_SEARCH_STEP = 5      # Step size for sliding window (efficiency vs accuracy)


# =============================================================================
# DEDUPLICATION CONSTANTS
# =============================================================================

# Maximum facts per LLM deduplication call
DEFAULT_MAX_BATCH = 50


# =============================================================================
# DISPLAY CONSTANTS
# =============================================================================

# Preview text truncation limits
USER_PROFILE_PREVIEW_CHARS = 500
CLAUDE_MEMORIES_PREVIEW_CHARS = 300

# Table display limits
CHUNK_TABLE_MAX_ROWS = 10
TOP_FACTS_DISPLAY_COUNT = 10

# Fact text truncation for display
FACT_TRUNCATE_SHORT = 60   # For summary lists
FACT_TRUNCATE_MEDIUM = 80  # For detailed views

# Visual separators
HORIZONTAL_RULE_WIDTH = 60

# Duplicate ID preview limit in error messages
DUPLICATE_ID_PREVIEW_COUNT = 5


# =============================================================================
# FILE I/O CONSTANTS
# =============================================================================

# Standard buffer size for file hashing
FILE_READ_BUFFER_SIZE = 8192  # 8KB - standard efficient buffer
