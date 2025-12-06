"""
LLM provider selection UI.

This module handles the CLI interface for selecting between
local LLM (LM Studio) and API (Anthropic) providers.
"""

from typing import Optional

from rich.console import Console
from rich.prompt import Prompt

from src.providers import LocalProvider, APIProvider, LLMProvider


# Provider choice constants
PROVIDER_LOCAL = "local"
PROVIDER_API = "api"


def get_provider(choice: str) -> LLMProvider:
    """Create and return the appropriate LLM provider.

    Args:
        choice: Either PROVIDER_LOCAL or PROVIDER_API.

    Returns:
        Configured LLMProvider instance.

    Raises:
        ValueError: If choice is invalid or API key missing for API provider.

    Example:
        >>> provider = get_provider("local")
        >>> print(provider.is_local)
        True
    """
    if choice == PROVIDER_LOCAL:
        return LocalProvider()
    elif choice == PROVIDER_API:
        return APIProvider()  # Will raise ValueError if no API key
    else:
        raise ValueError(
            f"Invalid provider choice: {choice}. "
            f"Use '{PROVIDER_LOCAL}' or '{PROVIDER_API}'"
        )


def select_provider(console: Console, choice: Optional[str] = None) -> LLMProvider:
    """Select LLM provider interactively or from CLI arg.

    If no choice is provided, prompts the user to select. Displays
    confirmation message and returns the configured provider.

    Args:
        console: Rich console for output.
        choice: Provider choice from CLI, or None to prompt user.

    Returns:
        Configured LLMProvider instance.

    Raises:
        SystemExit: If provider creation fails (e.g., missing API key).

    Example:
        >>> console = Console()
        >>> provider = select_provider(console, "local")
        Using: Local LLM (sequential processing)
    """
    if choice is None:
        choice = Prompt.ask(
            "Select provider",
            choices=[PROVIDER_LOCAL, PROVIDER_API],
            default=PROVIDER_LOCAL
        )

    try:
        provider = get_provider(choice)
        if provider.is_local:
            console.print("[cyan]Using:[/cyan] Local LLM (sequential processing)")
        else:
            console.print("[cyan]Using:[/cyan] Anthropic API (parallel processing)")
        return provider
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)
