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
def conversations_file(project_root):
    """Return path to conversations.json if it exists."""
    path = project_root / "conversations.json"
    if not path.exists():
        pytest.skip("conversations.json not present")
    return path


@pytest.fixture
def sample_json_file(tmp_path):
    """Create a small sample JSON file for testing."""
    sample = tmp_path / "sample.json"
    sample.write_text('[{"id": "test", "title": "Test Conversation"}]')
    return sample
