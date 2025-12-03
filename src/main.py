"""
main.py - CLI entry point

Interactive menu for:
- Input file selection
- Extraction depth (quick/medium/full)
- LLM provider (local/API)
- Chunk size (default 65536)
- Output format (md/json/both)
"""

from pathlib import Path

# from rich.console import Console
# from rich.prompt import Prompt, Confirm

# from .parser import parse_all
# from .chunker import chunk_conversations
# from .extractor import extract_all
# from .verifier import verify_all
# from .aggregator import aggregate
# from .deduplicator import deduplicate
# from .distiller import distill
# from .providers import LocalProvider, APIProvider


def main():
    """
    Main entry point.

    Flow:
    1. Show welcome banner
    2. Get input file (default: conversations.json)
    3. Select extraction depth
    4. Select LLM provider
    5. Configure chunk size
    6. Select output format
    7. Run pipeline
    8. Show results
    """
    # TODO:
    # console = Console()
    # show_banner(console)
    #
    # # 1. Input file
    # input_path = get_input_file(console)
    #
    # # 2. Parse
    # conversations, user_profile = parse_all(input_path)
    # if user_profile:
    #     save_user_profile(user_profile)  # Free win!
    #
    # # 3. Chunk
    # chunk_size = get_chunk_size(console)  # Default 65536
    # chunks = chunk_conversations(conversations, chunk_size)
    #
    # # 4. Provider
    # provider = get_provider(console)
    #
    # # 5. Extract
    # raw_facts = extract_all(chunks, provider)
    #
    # # 6. Verify
    # verified, discarded = verify_all(raw_facts, conversations)
    # log_stats(verified, discarded)
    #
    # # 7. Aggregate
    # aggregated = aggregate(verified)
    #
    # # 8. Deduplicate
    # unique = deduplicate(aggregated, provider)
    #
    # # 9. Distill
    # md_path, json_path = distill(unique, provider, Path("output"))
    #
    # # 10. Done!
    # show_results(console, md_path, json_path)
    pass


def show_banner(console):
    """Show welcome banner."""
    # TODO: Rich panel with fox logo
    pass


def get_input_file(console) -> Path:
    """Prompt for input file."""
    # TODO: Default to conversations.json, show file stats
    pass


def get_extraction_depth(console) -> str:
    """
    Select extraction depth.

    - quick: user messages only
    - medium: user + conversation context
    - full: complete conversations
    """
    # TODO: Rich select menu
    pass


def get_provider(console):
    """
    Select LLM provider.

    Local: http://localhost:1234/v1 (sequential)
    API: base URL + key + optional RPM/TPM (parallel)
    """
    # TODO: Rich select, then config prompts
    pass


def get_chunk_size(console) -> int:
    """Get chunk size (default 65536 = 2^16)."""
    # TODO: Prompt with default, validate is power of 2 (optional)
    pass


if __name__ == "__main__":
    main()
