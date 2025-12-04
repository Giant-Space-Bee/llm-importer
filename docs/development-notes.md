# Development Notes

Historical research, stage findings, and implementation decisions. Reference as needed.

## LLM Integration Workflow

When building LLM-powered features:

1. Probe with tiny input (1-2 messages)
2. Check: thinking tokens? structured output? actual format returned?
3. Understand before scaling
4. Small prototype first, then full module

## Stage 6 API Provider Decision

**Model**: Claude Sonnet 4 (`claude-sonnet-4-5-20250929`) - no override
- Haiku 4.5 lacks structured outputs (as of Dec 2025)
- Cost: $3/$15 per MTok (input/output)
- Structured outputs via beta header: `structured-outputs-2025-11-13`
- Rate limits (Tier 1): ~5 RPM, ~20k TPM

**API Request Format** (Anthropic SDK):
```python
from anthropic import Anthropic, transform_schema

client = Anthropic()
response = client.beta.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=8192,
    betas=["structured-outputs-2025-11-13"],
    messages=[{"role": "user", "content": prompt}],
    output_format={
        "type": "json_schema",
        "schema": transform_schema(PydanticModel)
    }
)
```

## Stage 4 LLM Findings (Hermes 4 70B via LM Studio)

**Tested 2025-12-03:**

| Finding | Implication |
|---------|-------------|
| No `<think>` tags by default | Don't need to strip them unless explicitly prompted to think |
| `json_schema` mode works | Use `response_format: {type: "json_schema", json_schema: {...}}` for guaranteed valid JSON |
| Fast on small input (~2 sec) | 70B on 61k tokens will be slow - test with smaller chunks first |
| Clean JSON output | Parser handles it fine |
| Categories not matching ours | Must enforce our category list in the schema |

**Structured output request format (Local LLM):**
```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "facts",
      "strict": true,
      "schema": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "fact": {"type": "string"},
            "category": {"type": "string", "enum": ["personal","professional","family","preferences","interests","personality"]},
            "source_convo_id": {"type": "string"},
            "source_timestamp": {"type": "number"},
            "source_quote": {"type": "string"}
          },
          "required": ["fact", "category", "source_convo_id", "source_timestamp", "source_quote"]
        }
      }
    }
  }
}
```

**Recommendations:**
1. Use structured outputs (`json_schema`) for guaranteed valid JSON
2. Test with smaller chunks first (4096 = 2^12) before going to 65536
3. Enforce our category enum in the schema
4. Chunk sizes show as ~61k because convos don't pack perfectly to exactly 65536

## Stage 5 Verifier Findings (2025-12-03)

**Implementation:**
- `normalize_text()`: Handles curly quotes, em/en dashes, ellipsis, whitespace, case
- `verify_fact()`: Normalized substring match against full conversation
- `verify_all()`: Batch verification, returns (verified, discarded) lists

**Unicode normalizations:**
| From | To |
|------|-----|
| `'` `'` (U+2018, U+2019) | `'` (straight) |
| `"` `"` (U+201C, U+201D) | `"` (straight) |
| `—` (U+2014 em dash) | `-` |
| `–` (U+2013 en dash) | `-` |
| `…` (U+2026 ellipsis) | `...` |

**Integration test results:**
- 3 facts extracted from 4k chunk
- 3/3 verified (100%)
- 0 hallucinations

**Key design decisions:**
1. Verify against ALL messages (`flatten_tree`), not just user messages
2. Normalize both quote and conversation before matching
3. Log hallucinations to stderr for debugging

---

## Completed Refactors

**Parser modularization (Dec 2025):**
- Split `parser.py` (485 lines) into `src/parsers/` package:
  - `types.py`: Shared dataclasses (Message, Conversation, etc.)
  - `chatgpt.py`: ChatGPT tree parsing, user_editable_context
  - `claude.py`: Claude flat array parsing, ISO timestamps
  - `memories.py`: Trusted baseline handling (Claude memories, ChatGPT profile)
- `parser.py` is now a facade with re-exports
- All imports remain backward compatible

**ChatGPT user_editable_context as trusted baseline (Dec 2025):**
- `format_user_profile_for_distiller()` formats custom instructions for distiller
- `main.py` uses ChatGPT profile when Claude memories unavailable
- Priority: Claude memories > ChatGPT profile (Claude is richer)
