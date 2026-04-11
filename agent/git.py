"""Locate the enclosing git repository via the git CLI."""

from __future__ import annotations

import os
import subprocess
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


def is_tracked(repo: Path, file_path: Path) -> bool:
    """True when `file_path` is in the index of the repository at `repo`."""
    try:
        _git(["ls-files", "--error-unmatch", "--", str(file_path)], repo)
    except (OSError, subprocess.SubprocessError):
        return False
    return True
