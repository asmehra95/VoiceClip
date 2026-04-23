"""Tests for voiceclip.utils — shared utilities."""

import os
import tempfile
import pytest

from voiceclip.utils import safe_unlink


class TestSafeUnlink:
    def test_deletes_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        assert f.exists()
        safe_unlink(str(f))
        assert not f.exists()

    def test_nonexistent_file_no_error(self):
        safe_unlink("/tmp/voiceclip_nonexistent_file_12345.wav")

    def test_none_path_no_error(self):
        safe_unlink(None)

    def test_empty_string_no_error(self):
        safe_unlink("")

    def test_directory_no_error(self, tmp_path):
        """Trying to unlink a directory should not raise."""
        safe_unlink(str(tmp_path))
