"""Workspace containment: keep file operations inside the project root.

Checks are two-layered:

- lexical: the joined path must stay under the workspace root;
- physical (symlink-aware): the real path (or, for not-yet-existing
  targets, the real path of the nearest existing ancestor) must stay
  under the real workspace root.
"""

from __future__ import annotations

import errno
import os
import stat as stat_module
from dataclasses import dataclass
from pathlib import Path

from agent.git import find_git_root


class WorkspaceError(Exception):
    """A path failed workspace containment checks."""


def is_inside(root: Path, candidate: Path) -> bool:
    return candidate == root or root in candidate.parents


def nearest_existing_ancestor(candidate: Path) -> Path:
    current = candidate
    while True:
        try:
            current.lstat()
            return current
        except OSError as error:
            if error.errno not in (errno.ENOENT, errno.ENOTDIR):
                raise
        parent = current.parent
        if parent == current:
            return current
        current = parent


@dataclass(frozen=True)
class ResolvedPath:
    path: Path
    is_file: bool
    is_directory: bool


class Workspace:
    """A project root with symlink-aware path containment."""

    def __init__(self, root: str | Path | None = None) -> None:
        lexical_root = Path(os.path.abspath(root or os.getcwd()))
        self.root = lexical_root
        self.real_root = Path(os.path.realpath(lexical_root))
        self.git_root = find_git_root(lexical_root)

    def _resolve(self, relative_path: object, *, must_exist: bool) -> Path:
        if not isinstance(relative_path, str) or not relative_path:
            raise WorkspaceError("Path must be a non-empty string.")

        if os.path.isabs(relative_path):
            raise WorkspaceError(
                "Only paths relative to the workspace are allowed."
            )

        lexical_target = Path(os.path.abspath(self.root / relative_path))
        if not is_inside(self.root, lexical_target):
            raise WorkspaceError(f"Path escapes workspace: {relative_path}")

        if must_exist:
            real_target = Path(os.path.realpath(lexical_target, strict=True))
            if not is_inside(self.real_root, real_target):
                raise WorkspaceError(
                    f"Path resolves outside workspace: {relative_path}"
                )
            return lexical_target

        ancestor = nearest_existing_ancestor(lexical_target)
        real_ancestor = Path(os.path.realpath(ancestor))
        if not is_inside(self.real_root, real_ancestor):
            raise WorkspaceError(
                f"Path resolves outside workspace: {relative_path}"
            )
        return lexical_target

    def resolve_existing_file(self, relative_path: object) -> Path:
        """Resolve a path that must already exist inside the workspace."""
        return self._resolve(relative_path, must_exist=True)

    def resolve_writable_file(self, relative_path: object) -> Path:
        """Resolve a path for writing; the file may not exist yet."""
        return self._resolve(relative_path, must_exist=False)

    def resolve_existing_path(
        self, relative_path: str | None = None
    ) -> ResolvedPath:
        """Resolve an existing file or directory; defaults to the root."""
        target = relative_path if relative_path else "."
        lexical_target = self._resolve(target, must_exist=True)
        mode = lexical_target.lstat().st_mode
        return ResolvedPath(
            path=lexical_target,
            is_file=stat_module.S_ISREG(mode),
            is_directory=stat_module.S_ISDIR(mode),
        )
