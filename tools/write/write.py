"""write: create or overwrite a whole text file."""

from __future__ import annotations

import stat
from typing import TYPE_CHECKING, Any

from agent.params import require_text
from agent.textfile import atomic_write_file

if TYPE_CHECKING:
    from agent.tools import Environment


def human_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _content(args: dict[str, Any]) -> str:
    content = args.get("content")
    if not isinstance(content, str):
        raise ValueError("content must be a string.")
    return content


class WriteTool:
    needs_approval = True
    trust = "path"

    def __init__(self, env: Environment) -> None:
        self.workspace = env.workspace

    def _target_state(self, relative: str) -> str:
        try:
            target = self.workspace.resolve_writable_file(relative)
            target.lstat()
        except (OSError, ValueError):
            return "new file"
        return "overwrite"

    def describe(self, args: dict[str, Any]) -> str:
        content = args.get("content")
        size = len(content.encode("utf-8")) if isinstance(content, str) else 0
        relative = str(args.get("path"))
        state = self._target_state(relative)
        return f"write {relative} ({human_size(size)}, {state})"

    def execute(self, args: dict[str, Any]) -> str:
        relative = require_text(args.get("path"), "path")
        content = _content(args)

        file_path = self.workspace.resolve_writable_file(relative)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path = self.workspace.resolve_writable_file(relative)

        existed = False
        try:
            mode = file_path.lstat().st_mode
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISDIR(mode):
                raise ValueError(f"Path is a directory: {relative}")
            existed = True
            if file_path.read_bytes() == content.encode("utf-8"):
                return f"Unchanged {relative} (identical content)."

        atomic_write_file(file_path, content)
        size = human_size(len(content.encode("utf-8")))
        verb = "Overwrote" if existed else "Created"
        return f"{verb} {relative} ({size})."


def create_tool(env: Environment) -> WriteTool:
    return WriteTool(env)
