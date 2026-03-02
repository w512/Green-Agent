"""Recursive file walk that prunes VCS, dependency, and cache directories."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

SKIP_NAMES = frozenset(
    {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "dist",
        "coverage",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".cursor",
    }
)


def is_skipped_name(name: str) -> bool:
    return name in SKIP_NAMES


def is_skipped_rel(rel: str) -> bool:
    parts = rel.replace("\\", "/").split("/")
    return any(is_skipped_name(part) for part in parts)


def walk_files(
    abs_dir: Path, relative_dir: str = ""
) -> Iterator[tuple[Path, str]]:
    """Yield (absolute_path, posix_relative_path) for regular files.

    Entries are visited in name order. Directories listed in SKIP_NAMES
    are pruned. Symlinks are neither followed nor reported.
    """
    with os.scandir(abs_dir) as handle:
        entries = sorted(handle, key=lambda entry: entry.name)
    for entry in entries:
        if is_skipped_name(entry.name):
            continue
        abs_path = Path(entry.path)
        rel_path = entry.name
        if relative_dir:
            rel_path = f"{relative_dir}/{entry.name}"
        if entry.is_dir(follow_symlinks=False):
            yield from walk_files(abs_path, rel_path)
        elif entry.is_file(follow_symlinks=False):
            yield abs_path, rel_path
