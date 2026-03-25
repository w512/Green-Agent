"""delete: remove one regular file."""

from __future__ import annotations

import stat
from typing import TYPE_CHECKING, Any

from agent.params import require_text

if TYPE_CHECKING:
    from agent.tools import Environment


class DeleteTool:
    needs_approval = True
    trust = "path"

    def __init__(self, env: Environment) -> None:
        self.workspace = env.workspace

    def describe(self, args: dict[str, Any]) -> str:
        return f"delete {args.get('path')}"

    def execute(self, args: dict[str, Any]) -> str:
        relative = require_text(args.get("path"), "path")
        file_path = self.workspace.resolve_existing_file(relative)
        mode = file_path.lstat().st_mode
        if stat.S_ISDIR(mode):
            raise ValueError(f"Path is a directory (not deleted): {relative}")
        if not stat.S_ISREG(mode):
            raise ValueError(f"Not a regular file: {relative}")
        file_path.unlink()
        return f"Deleted {relative}."


def create_tool(env: Environment) -> DeleteTool:
    return DeleteTool(env)
