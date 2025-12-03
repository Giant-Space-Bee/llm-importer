# ChatGPT Export Format: conversations.json

> **Source**: ChatGPT data export
> **As of**: December 2025

## Overview

Single JSON file containing all conversation history as a flat array of conversation objects.

---

## Top-Level Structure

```json
[
  { /* conversation 1 */ },
  { /* conversation 2 */ },
  ...
]
```

---

## Conversation Object

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Primary identifier |
| `title` | string | Conversation title |
| `create_time` | float | Unix timestamp (with decimals) |
| `update_time` | float | Last update timestamp |
| `mapping` | object | **Tree structure of all messages** |
| `current_node` | UUID | Active message node ID |
| `default_model_slug` | string | Model used (e.g., "auto", "gpt-5-1") |
| `gizmo_id` | string/null | Custom GPT ID if applicable |
| `is_archived` | boolean | Archive status |
| `memory_scope` | string | Memory setting |

<details>
<summary>Less important fields</summary>

- `conversation_id` - duplicate of `id`
- `gizmo_type`, `plugin_ids`, `voice` - feature flags
- `is_starred`, `is_read_only`, `is_do_not_remember` - UI state
- `moderation_results`, `safe_urls`, `blocked_urls` - security (usually empty)
- `async_status`, `disabled_tool_ids`, `context_scopes` - internal
- `sugar_item_id`, `sugar_item_visible`, `is_study_mode`, `owner` - unknown/rare

</details>

---

## The `mapping` Object (Message Tree)

Tree structure supporting **branching conversations** (edits, regenerations).

```json
"mapping": {
  "uuid-1": {
    "id": "uuid-1",
    "message": null,        // root node has no message
    "parent": null,
    "children": ["uuid-2"]
  },
  "uuid-2": {
    "id": "uuid-2",
    "message": { /* message object */ },
    "parent": "uuid-1",
    "children": ["uuid-3", "uuid-4"]  // branches!
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Node identifier (matches key) |
| `message` | object/null | The message content (null for root) |
| `parent` | UUID/null | Parent node ID |
| `children` | UUID[] | Child node IDs |

---

## Message Object

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Message identifier |
| `author` | object | Who wrote it |
| `content` | object | The actual content |
| `create_time` | float/null | When created |
| `status` | string | "finished_successfully", "error", etc. |
| `metadata` | object | Rich metadata |
| `recipient` | string | Usually "all" or tool name |
| `channel` | string/null | "final" or "commentary" |

### Author Object

```json
"author": {
  "role": "user",     // "user" | "assistant" | "system" | "tool"
  "name": null,       // tool name if role is "tool"
  "metadata": {}
}
```

**Tool names** (when `role: "tool"`): `python`, `web`, `myfiles_browser`, `file_search`, `bio`, `canmore.*`, `mtbrowser.*`, custom GPT tools

---

## Content Types

### 1. `text` (most common)
```json
{ "content_type": "text", "parts": ["message text here"] }
```

### 2. `user_editable_context` (user's custom instructions)
```json
{
  "content_type": "user_editable_context",
  "user_profile": "Preferred name: Landon\nRole: Head of AI...",
  "user_instructions": "Follow the instructions below..."
}
```
Found on ~80% of conversations, always from `author.role: "user"`, always hidden.

### 3. `multimodal_text` (images)
```json
{
  "content_type": "multimodal_text",
  "parts": [
    {
      "content_type": "image_asset_pointer",
      "asset_pointer": "file-service://file-xxx",
      "width": 1194,
      "height": 738
    },
    "Can you analyze this?"
  ]
}
```

### 4. `code`
```json
{ "content_type": "code", "language": "python", "text": "print('hello')" }
```

### 5. `execution_output`
```json
{ "content_type": "execution_output", "text": "output here" }
```
From `author.role: "tool"`, `author.name: "python"`

### 6. `tether_browsing_display` / `tether_quote` (web browsing)
```json
{ "content_type": "tether_quote", "url": "...", "domain": "...", "text": "...", "title": "..." }
```

### 7. `thoughts` / `reasoning_recap` (o1/thinking models)
```json
{ "content_type": "thoughts", "parts": ["reasoning..."] }
```

### 8. `system_error`
```json
{ "content_type": "system_error", "text": "error message" }
```

---

## Key Metadata Fields

**User messages:**
- `request_id`, `turn_exchange_id` - tracking IDs
- `dictation` - voice input flag

**Assistant messages:**
- `model_slug` - actual model used (may differ from `default_model_slug`)
- `finish_details` - how generation ended

**Hidden messages:**
- `is_visually_hidden_from_conversation: true`

**Search decisions:**
```json
"sonic_classification_result": {
  "search_decision": false,
  "search_complexity_decision": "no_search"
}
```

---

## Model Slugs Found

**GPT-5**: `gpt-5`, `gpt-5-1`, `gpt-5-1-thinking`, `gpt-5-pro`, `gpt-5-instant`, `gpt-5-t-mini`

**O-series**: `o1`, `o1-pro`, `o3`, `o3-mini`, `o3-mini-high`, `o3-pro`, `o4-mini`, `o4-mini-high`

**GPT-4**: `gpt-4`, `gpt-4-1`, `gpt-4-5`, `gpt-4o`, `gpt-4o-mini`, `gpt-4o-jawbone`

**Other**: `auto`, `research`

---

## Message Status Values

- `finished_successfully` - completed normally
- `in_progress` - still generating (shouldn't appear in exports)
- `error` - generation failed
- `cancelled` - generation stopped

---

## Typical Message Sequence

1. Empty root node (`message: null`)
2. System message (empty, hidden)
3. User context message (`user_editable_context`, hidden)
4. First user message (`text`)
5. First assistant response (`text`)
6. ... conversation continues ...

---

## Edge Cases

- **Null timestamps**: System/context messages often have `create_time: null`
- **Parts is always an array**: Even single text → `parts: ["text"]`
- **Hidden messages**: Check `metadata.is_visually_hidden_from_conversation`
- **Two model fields**: `conversation.default_model_slug` vs `message.metadata.model_slug`
- **Large files**: Can exceed 50MB for active users

---

**Sample analyzed**: 450 conversations, 21,530 messages, 17M characters, Dec 2024 - Dec 2025
