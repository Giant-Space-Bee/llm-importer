"""
parsers/memories.py - Claude memories and ChatGPT user profile handling

Handles trusted baseline content:
- Claude: memories.json with conversations_memory and project_memories
- ChatGPT: user_editable_context (custom instructions)

Both are "free wins" - already curated, passed to distiller as trusted_context.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from src.parsers.types import UserProfile


@dataclass
class ClaudeMemories:
    """
    Parsed Claude memories.json content.

    Claude exports include a memories.json with pre-synthesized user profile.
    This is a "trusted baseline" - already curated by Claude, not raw data.
    """
    conversations_memory: str  # Main biography prose (markdown)
    project_memories: Dict[str, str]  # UUID -> prose per project
    account_uuid: str


def load_claude_memories(folder_path: str) -> Optional[ClaudeMemories]:
    """
    Load memories.json from a Claude export folder.

    Claude exports are folders with structure:
        data-{timestamp}-batch-{N}/
        ├── conversations.json
        ├── memories.json      ← this file
        ├── projects.json
        └── users.json

    Args:
        folder_path: Path to the Claude export folder

    Returns:
        ClaudeMemories if found and valid, None otherwise
    """
    memories_path = Path(folder_path) / "memories.json"

    if not memories_path.exists():
        return None

    try:
        with open(memories_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # memories.json is an array with a single object
        if not data or not isinstance(data, list) or len(data) == 0:
            return None

        mem_obj = data[0]

        # Extract fields with defaults for missing data
        conversations_memory = mem_obj.get("conversations_memory", "")
        project_memories_raw = mem_obj.get("project_memories", {})
        account_uuid = mem_obj.get("account_uuid", "")

        # Flatten project_memories: each value is a dict with prose fields
        # We'll concatenate all prose fields into a single string per project
        project_memories: Dict[str, str] = {}
        if isinstance(project_memories_raw, dict):
            for uuid, mem_data in project_memories_raw.items():
                if isinstance(mem_data, str):
                    # Already a string
                    project_memories[uuid] = mem_data
                elif isinstance(mem_data, dict):
                    # Concatenate all string values (the prose sections)
                    sections = []
                    for key, value in mem_data.items():
                        if isinstance(value, str) and key != "uuid":
                            sections.append(f"**{key}**\n\n{value}")
                    project_memories[uuid] = "\n\n".join(sections)

        # Check if we have any actual content
        if not conversations_memory and not project_memories:
            return None

        return ClaudeMemories(
            conversations_memory=conversations_memory,
            project_memories=project_memories,
            account_uuid=account_uuid
        )

    except (json.JSONDecodeError, KeyError, TypeError):
        # Malformed file - return None, caller can proceed without memories
        return None


def format_memories_for_distiller(memories: ClaudeMemories) -> str:
    """
    Convert ClaudeMemories to prose string for distiller context.

    Combines conversations_memory and project_memories into a single
    formatted string that the distiller can use as trusted context.

    Args:
        memories: Parsed ClaudeMemories object

    Returns:
        Formatted prose string
    """
    sections = []

    # Add conversations memory (the main user profile)
    if memories.conversations_memory:
        sections.append(memories.conversations_memory)

    # Add project memories if present
    if memories.project_memories:
        sections.append("\n---\n\n## Project-Specific Context\n")
        for uuid, prose in memories.project_memories.items():
            # Use a separator between projects
            sections.append(f"### Project {uuid[:8]}...\n\n{prose}")

    return "\n\n".join(sections)


def format_user_profile_for_distiller(profile: UserProfile) -> str:
    """
    Convert ChatGPT UserProfile to prose string for distiller context.

    ChatGPT's user_editable_context contains two fields:
    - user_profile: "About me" section the user filled out
    - user_instructions: "How would you like ChatGPT to respond?" section

    Both are trusted baseline content - already curated by the user.

    Args:
        profile: Parsed UserProfile from ChatGPT export

    Returns:
        Formatted prose string for distiller trusted_context
    """
    sections = []

    if profile.user_profile:
        sections.append("## About the User\n")
        sections.append(profile.user_profile)

    if profile.user_instructions:
        sections.append("\n## User Preferences\n")
        sections.append(profile.user_instructions)

    return "\n".join(sections)
