# Bug 001: Invalid JSON from Claude API Structured Outputs

## Summary
API extraction fails with `Invalid JSON from Claude API: Expecting property name enclosed in double quotes` after successfully processing several chunks.

## Severity
**High** - Causes complete extraction failure and loss of work (no checkpoint saved).

## Location
`src/providers.py:364-370` - `APIProvider.complete_structured()`

## Reproduction
```bash
python -m src.main "export_data/data-2025-12-03-23-45-23-batch-0000/conversations.json" --provider api
```

## Error Message
```
Error during extraction: Invalid JSON from Claude API: Expecting property name
enclosed in double quotes: line 1 column 2 (char 1)
```

## Analysis

### What's happening
1. The Claude API is called with structured outputs beta (`structured-outputs-2025-11-13`)
2. Response is extracted from `response.content[0].text`
3. `json.loads(content)` fails because content is not valid JSON

### The error location
```python
# src/providers.py:364-365
content = response.content[0].text
return json.loads(content)  # <-- Fails here
```

### Likely causes
1. **Empty response**: If the API returns empty string, `json.loads("")` throws a different error
2. **Error message in response**: The API might return an error string instead of JSON
3. **Malformed structured output**: Claude API beta might occasionally return malformed JSON (e.g., `{x:` instead of `{"x":`)
4. **Content block type mismatch**: `response.content[0]` might not be a `TextBlock` in all cases

### Evidence from logs
The error occurs AFTER successful extractions (we see hallucination checks happening), suggesting:
- The structured output works most of the time
- It fails on specific chunks or under certain conditions (rate limit recovery?)

## Suggested Fix

```python
def _do_call() -> dict:
    response = self._client.beta.messages.create(
        model=self.model,
        max_tokens=DEFAULT_MAX_TOKENS,
        betas=[self.STRUCTURED_OUTPUTS_BETA],
        messages=[{"role": "user", "content": prompt}],
        output_format={
            "type": "json_schema",
            "schema": schema
        }
    )
    self._update_usage(
        response.usage.input_tokens,
        response.usage.output_tokens
    )

    # Debug: Log what we actually received
    if not response.content:
        raise RuntimeError("Claude API returned empty content")

    content_block = response.content[0]

    # Check content block type
    if not hasattr(content_block, 'text'):
        raise RuntimeError(f"Unexpected content block type: {type(content_block)}")

    content = content_block.text

    # Debug logging for diagnosis
    if not content or not content.strip().startswith('{'):
        print(f"[DEBUG] Unexpected response content: {content[:200]!r}", file=sys.stderr)

    return json.loads(content)
```

## Additional Investigation Needed
1. Add debug logging to capture the actual response content when JSON parsing fails
2. Check if this correlates with rate limit retries (maybe the retry isn't working correctly?)
3. Verify the structured outputs beta is being used correctly per latest Anthropic docs

## Related
- Bug 002: Checkpoint not saved in parallel mode (this JSON error cascades to total work loss)
