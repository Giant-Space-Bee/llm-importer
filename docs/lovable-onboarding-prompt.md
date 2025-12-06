# Onboarding Flow for Lovable

## Design Philosophy

This should feel like **Claude's onboarding** - premium, calm, intentional. Not startup-y or busy. Think:
- Generous whitespace
- Soft, muted colors (cream, warm grays, gentle accents)
- Typography that breathes (large, readable, unhurried)
- Transitions that feel smooth, not snappy
- Copy that sounds human, not corporate

**Core principle:** Every screen should feel like a conversation, not a form.

---

## Flow Overview

```
Welcome → Profile → "Have an AI?" fork
                         ↓ Yes
                    Choose AI (ChatGPT / Claude)
                         ↓
                    How to Export (modal)
                         ↓
                    Upload Files
                         ↓
                    "We're on it" → Main App
                    (processing happens in background, notify when done)
```

---

## Screen 1: Welcome

**Layout:** Centered, minimal

**Content:**
- Warm headline: "Let's get to know each other"
- Subtext: "This takes about 2 minutes"
- Single button: "Begin"

**Vibe:** Like meeting someone new. Calm, inviting.

---

## Screen 2: Basic Profile

**Layout:** Single column, centered, generous spacing

**Fields (keep minimal):**
1. **Your name** (text input)
2. **What should I call you?** (optional - nickname field, smaller/secondary)
3. **What brings you here?** (pill selector, multi-select allowed)
   - Work & productivity
   - Writing & creativity
   - Learning & research
   - Just exploring

**Button:** "Continue"

**Notes:**
- No required field asterisks - feels clinical
- Placeholder text should be helpful, not "Enter your name..."
- Pills should feel tappable, with satisfying selected state

---

## Screen 3: The Fork

**Layout:** Centered, decision-focused

**Headline:** "Already working with an AI?"

**Subtext:** "If you've been using ChatGPT or Claude, we can bring what they learned about you along for the ride."

**Two options (large, card-style buttons):**

1. **"Yes, let's import"**
   - Subtext: "Bring my history with me"

2. **"No, fresh start"**
   - Subtext: "I'll teach you as we go"

**Notes:**
- "Fresh start" should feel equally valid, not like a lesser option
- Cards should have subtle hover states
- No pressure, no FOMO copy

---

## Screen 4: Choose Your AI

*Only shown if user selected "Yes, let's import"*

**Layout:** Two cards side by side (stack on mobile)

**Headline:** "Where are you coming from?"

### Card 1: ChatGPT
- OpenAI logo (placeholder: green/black circle icon)
- "ChatGPT"
- Subtext: "by OpenAI"

### Card 2: Claude
- Anthropic logo (placeholder: coral/orange abstract icon)
- "Claude"
- Subtext: "by Anthropic"

**Interaction:**
- Click to select (single select)
- Selected state: subtle border or background shift
- "Continue" button appears/enables after selection

---

## Screen 5: How to Export (Modal)

*Appears as overlay before showing upload UI*

**Modal design:**
- Centered, max-width ~500px
- Soft shadow, rounded corners
- Dismissible only via "Got it" button (not backdrop click - they need to see this)

### ChatGPT Version

**Header:** "How to export from ChatGPT"

**Steps (with placeholder screenshot areas):**

```
1. Go to chatgpt.com and click your profile picture
   [Screenshot placeholder: Profile menu location]

2. Click "Settings"
   [Screenshot placeholder: Settings option]

3. Go to "Data controls" → "Export data"
   [Screenshot placeholder: Export button]

4. Check your email and download the ZIP file

5. Unzip it and find "conversations.json"
   [Screenshot placeholder: File in folder]
```

**Footer:** "Got it" button (primary, centered)

---

### Claude Version

**Header:** "How to export from Claude"

**Steps:**

```
1. Go to claude.ai and click your profile picture
   [Screenshot placeholder: Profile menu]

2. Click "Settings" → "Account"
   [Screenshot placeholder: Account settings]

3. Click "Export Data" and wait for the download
   [Screenshot placeholder: Export button]

4. Open the downloaded folder (named "data-..." something)

5. You'll need two files:
   • conversations.json
   • memories.json ← this is the important one!
   [Screenshot placeholder: Both files highlighted]
```

**Footer:** "Got it" button

---

## Screen 6: Upload Files

*After modal is dismissed*

**Layout:** Centered upload zone

### ChatGPT Version

**Headline:** "Drop your file here"

**Upload zone:**
- Large dashed-border rectangle
- Icon: Document/file icon
- Primary text: "conversations.json"
- Secondary text: "or click to browse"
- Accepted: `.json` files only

**States:**
- Default: Dashed border, muted
- Drag hover: Border becomes solid, subtle highlight
- Uploaded: Checkmark, filename shown, "Remove" option
- Error: Red border, error message below

---

### Claude Version

**Headline:** "Drop your files here"

**Two upload zones (stacked or side by side):**

**Zone 1:**
- Label: "conversations.json"
- Required indicator (subtle)

**Zone 2:**
- Label: "memories.json"
- Required indicator
- Helper text: "This contains your saved memories"

**Continue button:** Only enabled when both files uploaded

---

## Screen 7: "We're On It"

*Final onboarding screen - processing happens in background*

**Layout:** Centered, celebratory but calm

**Headline:** "You're all set"

**Subtext:** "We're reading through your conversations now. This can take a few minutes for larger exports."

**Secondary text:** "We'll let you know when your memories are ready."

**Visual:** Subtle animation - could be a gentle pulse, floating dots, or similar. Nothing aggressive.

**Button:** "Start exploring" → goes to main app

**Background behavior:**
- Processing continues in background
- When complete: toast notification or badge appears
- User can view extracted memories from settings/profile

---

## Error States to Design

1. **Wrong file type**
   - "This doesn't look like the right file. We need a .json file."

2. **Wrong JSON structure**
   - "This JSON file doesn't look like a ChatGPT/Claude export. Double-check you grabbed the right one?"

3. **File too large** (if you set a limit)
   - "This file is pretty large. Give us a moment..." (or actual limit message)

4. **Missing required file** (Claude - only one of two uploaded)
   - "We still need your memories.json file to continue"

---

## Assets Needed

| Asset | Description | Status |
|-------|-------------|--------|
| ChatGPT logo | OpenAI logo or stylized icon | Placeholder OK |
| Claude logo | Anthropic logo or stylized icon | Placeholder OK |
| ChatGPT export screenshot 1 | Profile menu | Need |
| ChatGPT export screenshot 2 | Settings location | Need |
| ChatGPT export screenshot 3 | Data controls / Export | Need |
| ChatGPT export screenshot 4 | File in unzipped folder | Need |
| Claude export screenshot 1 | Profile menu | Need |
| Claude export screenshot 2 | Account settings | Need |
| Claude export screenshot 3 | Export button | Need |
| Claude export screenshot 4 | Both files in folder | Need |

---

## Technical Notes

**File requirements:**
- ChatGPT: 1 file (`conversations.json`)
- Claude: 2 files (`conversations.json` + `memories.json`)

**Validation:**
- File extension: `.json`
- Basic structure check (has expected top-level keys)
- ChatGPT: should have array with objects containing `mapping` field
- Claude: conversations should have `uuid` and `chat_messages` fields

**Upload endpoint:** [Placeholder - TBD]

**Notification system:**
- When processing completes, show notification
- Store extraction results for viewing later

---

## Copy Tone Guide

**Do:**
- Sound like a helpful friend
- Use "we" and "you" naturally
- Keep sentences short
- Assume intelligence, explain process

**Don't:**
- Sound corporate or formal
- Use jargon (API, JSON structure, etc. - unless necessary)
- Rush or pressure
- Over-explain

**Examples:**
- Good: "We'll let you know when your memories are ready"
- Bad: "Processing will complete asynchronously and you will receive a notification upon completion"

- Good: "This doesn't look like the right file"
- Bad: "Error: Invalid JSON schema detected"
