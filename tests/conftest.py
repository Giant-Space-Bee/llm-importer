"""
Shared pytest fixtures for LLM Importer tests.
"""

import pytest
from pathlib import Path


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def fixtures_dir():
    """Return the test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def chatgpt_fixture(fixtures_dir):
    """Return path to ChatGPT format sample file (always available)."""
    return fixtures_dir / "chatgpt_sample.json"


@pytest.fixture
def claude_fixture(fixtures_dir):
    """Return path to Claude format sample file (always available)."""
    return fixtures_dir / "claude_sample.json"


@pytest.fixture
def conversations_file(project_root):
    """Return path to real conversations.json if it exists (large file, optional).

    This fixture is for optional "real data" tests with hardcoded assertions.
    Tests using this fixture will be skipped if the file is not present.
    Prefer using chatgpt_fixture or claude_fixture for regular unit tests.
    """
    path = project_root / "conversations.json"
    if not path.exists():
        pytest.skip("conversations.json not present (optional real data test)")
    return path


@pytest.fixture
def sample_json_file(tmp_path):
    """Create a small sample JSON file for testing."""
    sample = tmp_path / "sample.json"
    sample.write_text('[{"id": "test", "title": "Test Conversation"}]')
    return sample
