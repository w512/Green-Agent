"""read: return a numbered slice of a UTF-8 text file in the workspace."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent.params import positive_int
from agent.textfile import format_numbered_lines, read_text_file, split_lines

if TYPE_CHECKING:
    from agent.tools import Environment

MAX_LIMIT = 2000


class ReadTool:
    needs_approval = False
    trust = "path"

    def __init__(self, env: Environment) -> None:
        self.workspace = env.workspace

    def execute(self, args: dict[str, Any]) -> str:
        offset = positive_int(args.get("offset"), "offset", 1)
        file_path = self.workspace.resolve_existing_file(args.get("path"))
        lines = split_lines(read_text_file(file_path))
        total = len(lines)
        if total == 0:
            return "(empty file)"
        remaining = max(0, total - offset + 1)
        requested = remaining
        if args.get("limit") is not None:
            requested = positive_int(args.get("limit"), "limit", 1)
        limit = min(requested, MAX_LIMIT, remaining)
        start = offset - 1
        chunk = lines[start : start + limit]
        return format_numbered_lines(chunk, offset, total)


def create_tool(env: Environment) -> ReadTool:
    return ReadTool(env)
