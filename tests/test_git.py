"""Tests for git root discovery."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from agent.git import find_git_root, load_git_info

git_available = shutil.which("git") is not None
needs_git = pytest.mark.skipif(git_available is False, reason="git missing")


def test_not_a_repo(tmp_path: Path) -> None:
    assert find_git_root(tmp_path) is None


@needs_git
def test_repo_root(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    root = find_git_root(sub)
    assert root is not None
    assert root.name == tmp_path.name


def test_git_info_outside_repo(tmp_path: Path) -> None:
    info = load_git_info(tmp_path)
    assert info.name == tmp_path.name
    assert info.branch == ""
    assert info.dirty is False
