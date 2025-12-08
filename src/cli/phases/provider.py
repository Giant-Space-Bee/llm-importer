"""
Provider phase: Select and configure the LLM provider.

This phase initializes the LLM provider (local or API) based on user choice.
"""

from src.cli.types import PipelineContext
from src.cli.providers import select_provider as _select_provider


def phase_select_provider(ctx: PipelineContext, choice: str) -> PipelineContext:
    """Select and configure the LLM provider.

    Args:
        ctx: Pipeline context.
        choice: Provider choice ("local" or "api").

    Returns:
        Updated context with provider set.

    Example:
        >>> ctx = phase_select_provider(ctx, "local")
        >>> print(ctx.provider.is_local)
        True
    """
    ctx.provider = _select_provider(ctx.console, choice)
    return ctx
