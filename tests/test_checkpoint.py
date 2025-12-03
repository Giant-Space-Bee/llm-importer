"""
Stage 6g: Checkpoint tests

TDD - tests written first, then implementation.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path


class TestHashFile:
    """Test file hashing for change detection."""

    def test_hash_file_returns_sha256(self):
        """Should return sha256 hash of file contents."""
        from src.checkpoint import hash_file

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            f.write('{"test": "data"}')
            temp_path = f.name

        try:
            result = hash_file(temp_path)
            assert result.startswith("sha256:")
            assert len(result) == 7 + 64  # "sha256:" + 64 hex chars
        finally:
            os.unlink(temp_path)

    def test_hash_file_consistent(self):
        """Same file should produce same hash."""
        from src.checkpoint import hash_file

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            f.write('consistent content')
            temp_path = f.name

        try:
            hash1 = hash_file(temp_path)
            hash2 = hash_file(temp_path)
            assert hash1 == hash2
        finally:
            os.unlink(temp_path)

    def test_hash_file_different_for_different_content(self):
        """Different content should produce different hash."""
        from src.checkpoint import hash_file

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            f.write('content A')
            path_a = f.name

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            f.write('content B')
            path_b = f.name

        try:
            hash_a = hash_file(path_a)
            hash_b = hash_file(path_b)
            assert hash_a != hash_b
        finally:
            os.unlink(path_a)
            os.unlink(path_b)


class TestSaveAndLoadCheckpoint:
    """Test checkpoint persistence."""

    def test_save_and_load_checkpoint(self):
        """Should save checkpoint and load it back."""
        from src.checkpoint import save_checkpoint, load_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "checkpoint.json"

            state = {
                "version": 1,
                "started_at": 1701619200,
                "last_update": 1701619500,
                "total_chunks": 10,
                "completed_chunks": [0, 1, 2],
                "facts": [{"fact": "test", "category": "personal"}],
                "source_file_hash": "sha256:abc123"
            }

            save_checkpoint(ckpt_path, state)
            loaded = load_checkpoint(ckpt_path)

            assert loaded == state

    def test_no_checkpoint_returns_none(self):
        """Should return None if no checkpoint exists."""
        from src.checkpoint import load_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "nonexistent.json"
            result = load_checkpoint(ckpt_path)
            assert result is None

    def test_atomic_write(self):
        """Should not corrupt checkpoint on crash (atomic write)."""
        from src.checkpoint import save_checkpoint, load_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "checkpoint.json"

            # Save initial state
            state1 = {
                "version": 1,
                "completed_chunks": [0],
                "facts": [],
                "source_file_hash": "sha256:first"
            }
            save_checkpoint(ckpt_path, state1)

            # Save second state (simulates update)
            state2 = {
                "version": 1,
                "completed_chunks": [0, 1],
                "facts": [{"fact": "new"}],
                "source_file_hash": "sha256:first"
            }
            save_checkpoint(ckpt_path, state2)

            # Verify we have the latest state
            loaded = load_checkpoint(ckpt_path)
            assert loaded == state2

            # Verify no temp files left behind
            files = list(Path(tmpdir).glob("*"))
            assert len(files) == 1
            assert files[0].name == "checkpoint.json"

    def test_corrupted_checkpoint_returns_none(self):
        """Should return None for corrupted/invalid JSON checkpoint."""
        from src.checkpoint import load_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "checkpoint.json"
            ckpt_path.write_text("{ invalid json }")

            result = load_checkpoint(ckpt_path)
            assert result is None


class TestShouldResume:
    """Test checkpoint validity checking."""

    def test_should_resume_true_when_valid(self):
        """Should return True when valid checkpoint exists for source."""
        from src.checkpoint import save_checkpoint, should_resume, hash_file

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create source file
            source_path = Path(tmpdir) / "source.json"
            source_path.write_text('{"conversations": []}')

            # Create matching checkpoint
            ckpt_path = Path(tmpdir) / "checkpoint.json"
            state = {
                "version": 1,
                "completed_chunks": [0, 1],
                "source_file_hash": hash_file(source_path)
            }
            save_checkpoint(ckpt_path, state)

            assert should_resume(ckpt_path, source_path) is True

    def test_should_resume_false_when_no_checkpoint(self):
        """Should return False when checkpoint doesn't exist."""
        from src.checkpoint import should_resume

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source.json"
            source_path.write_text('{"conversations": []}')
            ckpt_path = Path(tmpdir) / "nonexistent.json"

            assert should_resume(ckpt_path, source_path) is False

    def test_detect_source_file_changed(self):
        """Should invalidate checkpoint if source file hash changed."""
        from src.checkpoint import save_checkpoint, should_resume

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source.json"
            source_path.write_text('{"conversations": []}')

            # Create checkpoint with OLD hash
            ckpt_path = Path(tmpdir) / "checkpoint.json"
            state = {
                "version": 1,
                "completed_chunks": [0, 1],
                "source_file_hash": "sha256:old_hash_that_doesnt_match"
            }
            save_checkpoint(ckpt_path, state)

            # Source changed, checkpoint should be invalid
            assert should_resume(ckpt_path, source_path) is False


class TestGetRemainingChunks:
    """Test remaining chunk calculation."""

    def test_resume_from_partial(self):
        """Should correctly identify remaining chunks."""
        from src.checkpoint import get_remaining_chunks

        checkpoint = {
            "completed_chunks": [0, 1, 2, 3],
            "total_chunks": 10
        }

        remaining = get_remaining_chunks(checkpoint, 10)

        assert remaining == [4, 5, 6, 7, 8, 9]

    def test_no_checkpoint_returns_all(self):
        """Should return all chunks when checkpoint is None."""
        from src.checkpoint import get_remaining_chunks

        remaining = get_remaining_chunks(None, 5)

        assert remaining == [0, 1, 2, 3, 4]

    def test_all_completed_returns_empty(self):
        """Should return empty list when all chunks done."""
        from src.checkpoint import get_remaining_chunks

        checkpoint = {
            "completed_chunks": [0, 1, 2],
            "total_chunks": 3
        }

        remaining = get_remaining_chunks(checkpoint, 3)

        assert remaining == []

    def test_handles_out_of_order_completed(self):
        """Should handle completed_chunks in any order."""
        from src.checkpoint import get_remaining_chunks

        checkpoint = {
            "completed_chunks": [2, 0, 3, 1],  # Out of order
            "total_chunks": 6
        }

        remaining = get_remaining_chunks(checkpoint, 6)

        assert remaining == [4, 5]
