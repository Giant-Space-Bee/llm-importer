# Bug 003: Hallucination Detection False Positives

## Summary
The verifier marks many legitimate facts as "hallucinations" because it requires exact substring match of `source_quote` in original text, but the LLM often paraphrases or slightly modifies quotes.

## Severity
**Medium** - Valid facts are being discarded, reducing extraction quality.

## Location
`src/verifier.py` - `verify_quote()` function

## Log Data from Extraction Run (Dec 5, 2025)

```
[HALLUCINATION] NOT FOUND: 'Rod wants me to build a platform, a software as a ...' -> 'Rod wants the user to build a SaaS platform...'
[HALLUCINATION] NOT FOUND: 'lily's 9 3/4 day so we can do the Harry potter thi...' -> 'Has a daughter named Lily who is interested in Har...'
[HALLUCINATION] NOT FOUND: 'I'm the head of AI...' -> 'Works as Head of AI at EvolveWell...'
[HALLUCINATION] NOT FOUND: 'KNOWN series about AI girlfriend obsession...' -> 'Created a neo-noir TikTok series called KNOWN abou...'
[HALLUCINATION] NOT FOUND: 'I'm the head of AI...' -> 'Works as Head of AI at EvolveWell...'
[HALLUCINATION] NOT FOUND: 'integrated into my Linear...' -> 'Uses Linear for project management...'
[HALLUCINATION] NOT FOUND: 'lily's 9 3/4 day so we can do the Harry potter thi...' -> 'Lily is interested in Harry Potter...'
[HALLUCINATION] NOT FOUND: 'EV charging (Eevee)...' -> 'Has an electric vehicle called Eevee...'
[HALLUCINATION] NOT FOUND: 'Haley's $1,252/bi weekly until mid january...' -> 'Haley earns $1,252 biweekly until mid-January...'
[HALLUCINATION] NOT FOUND: 'You were Head of AI at a startup...' -> 'Was previously Head of AI at a startup...'
[HALLUCINATION] NOT FOUND: 'lily's 9 3/4 day so we can do the Harry potter thi...' -> 'Has a daughter named Lily, who is having 'lily's 9...'
[HALLUCINATION] NOT FOUND: 'I'm the head of AI...' -> 'Works as Head of AI at EvolveWell...'
[HALLUCINATION] NOT FOUND: 'Coding Fox Corp federally incorporated...' -> 'Has a federally incorporated company in Canada...'
[HALLUCINATION] NOT FOUND: 'i liked this one too "Let yourself be silently dra...' -> 'Appreciates Rumi quotes and poetry...'
```

## Analysis

### Pattern 1: Partial quote vs full context
- Quote: `'I'm the head of AI...'`
- Fact: `'Works as Head of AI at EvolveWell...'`
- The LLM is INFERRING context (adding "EvolveWell") that isn't in the quote snippet

### Pattern 2: Reformatting/normalization
- Quote: `'Haley's $1,252/bi weekly until mid january...'`
- Fact: `'Haley earns $1,252 biweekly until mid-January...'`
- Same info but the LLM normalized "bi weekly" → "biweekly", capitalized "January"

### Pattern 3: Extracting key info from longer quote
- Quote: `'lily's 9 3/4 day so we can do the Harry potter thi...'`
- Fact: `'Lily is interested in Harry Potter...'`
- The LLM is extracting the MEANING, not keeping the exact quote

### Pattern 4: Contextual inference
- Quote: `'EV charging (Eevee)...'`
- Fact: `'Has an electric vehicle called Eevee...'`
- LLM infers "EV charging" → "has electric vehicle"

## Root Cause
The verifier does exact substring matching:
```python
def verify_quote(source_quote: str, conversation_text: str) -> bool:
    normalized_quote = normalize_text(source_quote)
    normalized_text = normalize_text(conversation_text)
    return normalized_quote in normalized_text
```

This works when the LLM provides EXACT verbatim quotes, but:
1. The extraction prompt may not emphasize "verbatim" strongly enough
2. The LLM may be paraphrasing to make facts clearer
3. Some facts are synthesized from multiple parts of conversation

## Possible Fixes

### Option A: Improve extraction prompt
Add stronger language about verbatim quotes:
```
CRITICAL: source_quote MUST be an EXACT copy-paste from the user's message.
Do not paraphrase, do not clean up typos, do not add context.
```

### Option B: Fuzzy matching in verifier
Use fuzzy string matching (e.g., Levenshtein distance or token overlap):
```python
from difflib import SequenceMatcher

def verify_quote_fuzzy(source_quote: str, conversation_text: str, threshold: float = 0.8) -> bool:
    # Try exact match first
    if normalize_text(source_quote) in normalize_text(conversation_text):
        return True

    # Fuzzy match as fallback
    ratio = SequenceMatcher(None, source_quote.lower(), conversation_text.lower()).ratio()
    return ratio >= threshold
```

### Option C: Token-based matching
Check if X% of tokens from quote appear in conversation:
```python
def verify_quote_tokens(source_quote: str, conversation_text: str, threshold: float = 0.7) -> bool:
    quote_tokens = set(source_quote.lower().split())
    text_tokens = set(conversation_text.lower().split())
    overlap = len(quote_tokens & text_tokens) / len(quote_tokens)
    return overlap >= threshold
```

### Option D: Both - prompt improvement + fuzzy fallback
Best of both worlds: improve prompt AND add fuzzy matching as safety net.

## Recommendation
**Option D** is recommended:
1. First, improve extraction prompt to get better verbatim quotes
2. Add fuzzy matching as fallback (with logging to track how often it's used)
3. Monitor false positive rate over time

## Test Cases
These should all PASS after the fix:
- `'I'm the head of AI'` should match text containing `"I'm the head of AI at EvolveWell"`
- `'bi weekly'` should match `"biweekly"` (normalization)
- `'Coding Fox Corp federally incorporated'` should match nearby text about incorporation

## Related
- This might explain why different runs produce different fact counts
- Could investigate: what % of "hallucinations" are true vs false positives?
