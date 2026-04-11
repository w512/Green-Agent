"""Tests for git root discovery and tracked-file lookup."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from agent.git import find_git_root, is_tracked

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


@needs_git
def test_is_tracked(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    tracked = tmp_path / "a.txt"
    tracked.write_text("a")
    subprocess.run(["git", "add", "a.txt"], cwd=tmp_path, check=True)
    untracked = tmp_path / "b.txt"
    untracked.write_text("b")
    assert is_tracked(tmp_path, tracked)
    assert not is_tracked(tmp_path, untracked)
    assert not is_tracked(tmp_path, tmp_path / "missing.txt")


def test_is_tracked_outside_repo(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a")
    assert not is_tracked(tmp_path, tmp_path / "a.txt")
