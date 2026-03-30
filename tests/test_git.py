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


@needs_git
def test_git_info_inside_repo(tmp_path: Path) -> None:
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text("a")
    subprocess.run([*git, "add", "."], cwd=tmp_path, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "c"], cwd=tmp_path, check=True)
    info = load_git_info(tmp_path)
    assert info.branch == "main"
    assert len(info.hash) >= 7
    assert info.dirty is False
    (tmp_path / "b.txt").write_text("b")
    assert load_git_info(tmp_path).dirty is True


def test_git_info_outside_repo(tmp_path: Path) -> None:
    info = load_git_info(tmp_path)
    assert info.name == tmp_path.name
    assert info.branch == ""
    assert info.dirty is False
