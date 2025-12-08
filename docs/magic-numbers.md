# Magic Numbers Reference

All configurable constants for LLM Importer, centralized in `src/config.py`.

## Environment Variables

These can be set to override defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_IMPORTER_CHUNK_SIZE` | 8192 | Tokens per chunk for extraction |
| `LLM_IMPORTER_TPM` | 30000 | Tokens per minute rate limit |
| `LLM_IMPORTER_FUZZY_THRESHOLD` | 0.85 | Fuzzy match similarity threshold |

Example:
```bash
export LLM_IMPORTER_CHUNK_SIZE=65536  # Use larger chunks for local LLM
export LLM_IMPORTER_TPM=80000         # Higher tier API limits
```

---

## Chunking Constants

### `DEFAULT_CHUNK_SIZE` = 8192
**What it does:** Sets how many tokens go into each batch sent to the LLM for extraction.

**Why this value:** 8192 (2^13) balances extraction quality against throughput. Research shows LLMs suffer "lost in the middle" syndrome with larger contexts—information in the middle of long prompts gets worse recall. 8K is small enough for good recall while large enough to provide conversation context.

**Trade-offs:**
- Smaller (4K): Better recall per chunk, but more API calls
- Larger (64K): Fewer calls, but extraction quality degrades

### `MIN_CHUNK_SIZE` = 4096
**What it does:** Minimum allowed chunk size.

**Why this value:** Below 4K tokens, there's not enough context for meaningful extraction. Single messages often exceed 1-2K tokens.

### `MAX_CHUNK_SIZE` = 65536
**What it does:** Maximum chunk size (2^16).

**Why this value:** Claude's context window is 200K, but practical extraction quality degrades significantly above 64K. This is the ceiling for `--chunk-size` CLI option.

### `DEMO_CHUNK_SIZE` = 4096
**What it does:** Chunk size used in `--demo` mode.

**Why this value:** Small enough for quick testing (processes fast), large enough to demonstrate real extraction.

### `QUALITY_CHUNK_SIZE` = 8192
**What it does:** Optimal chunk size for API mode extraction quality.

**Why this value:** Same rationale as `DEFAULT_CHUNK_SIZE`. API mode uses this fixed size and compensates with parallelism for throughput.

### `HEADER_OVERHEAD_TOKENS` = 50
**What it does:** Estimated tokens for conversation metadata (title, ID, timestamps).

**Why this value:** Measured from actual formatted output. Conservative estimate to ensure chunks don't exceed limits after adding headers.

### `CHUNK_TRIM_STEP` = 100
**What it does:** When splitting oversized text, remove this many characters per iteration.

**Why this value:** 100 chars ≈ 25 tokens. Small enough for precision, large enough to converge quickly.

---

## Rate Limiting Constants

### `DEFAULT_TPM` = 30000
**What it does:** Default tokens-per-minute budget for Anthropic API.

**Why this value:** Anthropic Tier 1 limit is 40K TPM. We use 30K (75%) to leave headroom and avoid 429 errors.

### `DEFAULT_RPM` = 5
**What it does:** Requests per minute limit.

**Why this value:** Anthropic Tier 1 RPM limit. Not configurable via env var because TPM is the practical bottleneck.

### `SECONDS_PER_MINUTE` = 60
**What it does:** Seconds in a minute (for rate limit window calculations).

**Why this value:** It's... 60 seconds. Named for clarity in rate limit code.

### `RATE_LIMIT_BUFFER` = 0.1
**What it does:** Extra seconds to wait after rate limit window resets.

**Why this value:** 100ms buffer prevents edge-case 429s from clock skew between client and server.

### `MAX_CONCURRENT_REQUESTS` = 5
**What it does:** Maximum parallel API requests.

**Why this value:** Higher parallelism increases throughput but risks overwhelming the API. 5 concurrent with ~11K tokens each fits comfortably in 80K TPM (higher tiers).

### `OUTPUT_RESERVE` = 10000
**What it does:** Tokens reserved for LLM output when calculating input budget.

**Why this value:** Extraction prompts can generate up to 8K output tokens. 10K provides 2K safety margin.

### `OUTPUT_ESTIMATE` = 3000
**What it does:** Conservative output estimate for parallelism calculations.

**Why this value:** Actual outputs average 2-3K tokens. Used to calculate how many concurrent requests fit in TPM budget. More accurate than OUTPUT_RESERVE for throughput math.

---

## Retry Constants

### `MAX_RETRIES` = 3
**What it does:** Number of retry attempts on rate limit (429) errors.

**Why this value:** 3 retries with exponential backoff (5s, 10s, 20s = 35s total) usually clears transient rate limits without excessive waiting.

### `INITIAL_BACKOFF_SECONDS` = 5
**What it does:** First retry waits 5 seconds, then 10s, then 20s.

**Why this value:** 5 seconds is long enough for rate limit windows to partially clear, short enough to not feel stuck.

---

## LLM Provider Constants

### `DEFAULT_TEMPERATURE` = 0.3
**What it does:** LLM sampling temperature.

**Why this value:** Lower temperature = more deterministic = more consistent extraction. 0.3 allows some creativity for natural language while avoiding hallucinations.

### `DEFAULT_MAX_TOKENS` = 8192
**What it does:** Maximum output tokens per LLM call.

**Why this value:** Extraction can produce many facts from large chunks. 8K is generous while staying within model limits.

### `LOCAL_LLM_HOST` = "127.0.0.1"
### `LOCAL_LLM_PORT` = 1234
### `LOCAL_LLM_BASE_URL` = "http://127.0.0.1:1234/v1"
**What it does:** Default LM Studio connection settings.

**Why these values:** LM Studio's default local server address. The `/v1` suffix is for OpenAI-compatible API endpoints.

### `DEFAULT_MODEL` = "claude-sonnet-4-5-20250929"
**What it does:** Anthropic model for API mode.

**Why this value:** Claude Sonnet 4.5 offers best price/performance for extraction tasks. Supports structured outputs.

### `STRUCTURED_OUTPUTS_BETA` = "structured-outputs-2025-11-13"
**What it does:** Anthropic beta feature flag for JSON schema enforcement.

**Why this value:** Required header for structured output API. Date indicates beta version.

---

## Verification Constants

### `FUZZY_MATCH_THRESHOLD` = 0.85
**What it does:** Minimum similarity ratio for fuzzy quote matching.

**Why this value:** 85% catches common issues:
- Typos ("Rpd" vs "Rod")
- Minor punctuation differences
- Whitespace normalization artifacts

But rejects:
- Paraphrasing (70-80% similar)
- Hallucinated quotes (typically <60% similar)

### `SHORT_QUOTE_LENGTH` = 20
**What it does:** Quotes shorter than this get stricter matching.

**Why this value:** Short quotes are more likely to match by accident. "I like coffee" could fuzzy-match many things.

### `SHORT_QUOTE_THRESHOLD` = 0.90
**What it does:** Required similarity for short quotes.

**Why this value:** 90% for short quotes reduces false positives while still catching typos.

### `FUZZY_WINDOW_MIN` = 10
**What it does:** Minimum sliding window size for fuzzy search.

**Why this value:** Windows smaller than 10 chars are too granular for meaningful similarity.

### `FUZZY_WINDOW_MARGIN` = 10
**What it does:** Search windows quote_length ± this margin.

**Why this value:** Allows matching quotes that are slightly shorter than expected (words omitted).

### `FUZZY_WINDOW_EXPAND` = 20
**What it does:** Search windows up to quote_length + this.

**Why this value:** Allows matching quotes that are slightly longer (extra words included).

### `FUZZY_SEARCH_STEP` = 5
**What it does:** Sliding window moves 5 characters per step.

**Why this value:** Trade-off between accuracy and speed. Step of 1 is thorough but slow. Step of 5 is 5x faster with minimal accuracy loss (overlapping windows still catch matches).

---

## Deduplication Constants

### `DEFAULT_MAX_BATCH` = 50
**What it does:** Maximum facts per deduplication LLM call.

**Why this value:** 50 facts fit comfortably in context with good recall. Larger batches risk the LLM missing subtle duplicates in the middle.

---

## Display Constants

### `USER_PROFILE_PREVIEW_CHARS` = 500
**What it does:** Max characters shown in user profile preview.

**Why this value:** Long enough to show meaningful content, short enough to not flood terminal.

### `CLAUDE_MEMORIES_PREVIEW_CHARS` = 300
**What it does:** Max characters shown in Claude memories preview.

**Why this value:** Memories are typically shorter, 300 chars shows most of the content.

### `CHUNK_TABLE_MAX_ROWS` = 10
**What it does:** Maximum rows in chunk breakdown table.

**Why this value:** Shows enough detail to verify chunking worked, summarizes the rest to avoid scroll fatigue.

### `TOP_FACTS_DISPLAY_COUNT` = 10
**What it does:** Number of top facts shown in summary.

**Why this value:** Top 10 gives good overview without overwhelming.

### `FACT_TRUNCATE_SHORT` = 60
**What it does:** Truncate facts to 60 chars in summary lists.

**Why this value:** Fits on one terminal line with category and frequency.

### `FACT_TRUNCATE_MEDIUM` = 80
**What it does:** Truncate facts to 80 chars in detailed views.

**Why this value:** Standard terminal width minus indentation.

### `HORIZONTAL_RULE_WIDTH` = 60
**What it does:** Width of `━` separator lines.

**Why this value:** Matches typical terminal width, creates clear visual breaks.

### `DUPLICATE_ID_PREVIEW_COUNT` = 5
**What it does:** How many duplicate IDs to show in error messages.

**Why this value:** Shows enough examples to debug, doesn't spam terminal with hundreds of IDs.

---

## File I/O Constants

### `FILE_READ_BUFFER_SIZE` = 8192
**What it does:** Buffer size for file hashing operations.

**Why this value:** 8KB is the standard efficient buffer size for disk I/O. Matches OS page size on most systems.
