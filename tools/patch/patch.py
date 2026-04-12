"""patch: apply several unique replacements to one file, all-or-nothing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent.params import require_text, text
from agent.textfile import atomic_write_file, read_text_file, replace_unique

if TYPE_CHECKING:
    from agent.tools import Environment


def _hunks(args: dict[str, Any]) -> list[tuple[str, str]]:
    hunks = args.get("hunks")
    if not isinstance(hunks, list) or not hunks:
        raise ValueError("hunks must be a non-empty array.")
    result: list[tuple[str, str]] = []
    for index, hunk in enumerate(hunks):
        where = f"hunks[{index}]"
        if not isinstance(hunk, dict):
            raise ValueError(f"{where} must be an object.")
        old_text = require_text(hunk.get("old_text"), f"{where}.old_text")
        new_text = text(hunk.get("new_text"), f"{where}.new_text")
        result.append((old_text, new_text))
    return result


class PatchTool:
    needs_approval = True
    trust = "path"

    def __init__(self, env: Environment) -> None:
        self.workspace = env.workspace

    def describe(self, args: dict[str, Any]) -> str:
        hunks = args.get("hunks")
        count = len(hunks) if isinstance(hunks, list) else 0
        return f"patch {args.get('path')} ({count} hunks)"

    def execute(self, args: dict[str, Any]) -> str:
        relative = require_text(args.get("path"), "path")
        hunks = _hunks(args)
        file_path = self.workspace.resolve_existing_file(relative)
        content = read_text_file(file_path)
        for index, (old_text, new_text) in enumerate(hunks):
            where = f"hunks[{index}].old_text"
            content = replace_unique(content, old_text, new_text, where)
        atomic_write_file(file_path, content)
        return f"Patched {relative} ({len(hunks)} hunks)."


def create_tool(env: Environment) -> PatchTool:
    return PatchTool(env)
