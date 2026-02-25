"""Locate the enclosing git repository via the git CLI."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

GIT_TIMEOUT_SECONDS = 5.0


def _git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_SECONDS,
        check=True,
    )
    return result.stdout.strip()


def find_git_root(cwd: Path) -> Path | None:
    try:
        root = _git(["rev-parse", "--show-toplevel"], cwd)
    except (OSError, subprocess.SubprocessError):
        # not a git repo, or git is unavailable
        return None
    if not root:
        return None
    return Path(os.path.abspath(root))


@dataclass(frozen=True)
class GitInfo:
    name: str
    branch: str
    hash: str
    dirty: bool


def load_git_info(cwd: Path) -> GitInfo:
    name = cwd.name
    try:
        branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
        commit = _git(["rev-parse", "--short", "HEAD"], cwd)
        status = _git(["status", "--porcelain"], cwd)
    except (OSError, subprocess.SubprocessError):
        # not a git repo, or git is unavailable
        return GitInfo(name=name, branch="", hash="", dirty=False)
    return GitInfo(name=name, branch=branch, hash=commit, dirty=bool(status))
