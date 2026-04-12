"""edit: replace one unique fragment in an existing file."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent.params import require_text, text
from agent.textfile import atomic_write_file, read_text_file, replace_unique

if TYPE_CHECKING:
    from agent.tools import Environment


def line_count(content: str) -> int:
    ends_open = bool(content) and not content.endswith("\n")
    return content.count("\n") + (1 if ends_open else 0)


def _texts(args: dict[str, Any]) -> tuple[str, str]:
    old_text = require_text(args.get("old_text"), "old_text")
    new_text = text(args.get("new_text"), "new_text")
    if old_text == new_text:
        raise ValueError("old_text and new_text are identical; nothing to do.")
    return old_text, new_text


class EditTool:
    needs_approval = True
    trust = "path"

    def __init__(self, env: Environment) -> None:
        self.workspace = env.workspace

    def describe(self, args: dict[str, Any]) -> str:
        old_text = args.get("old_text")
        new_text = args.get("new_text")
        removed = line_count(old_text) if isinstance(old_text, str) else 0
        added = line_count(new_text) if isinstance(new_text, str) else 0
        return f"edit {args.get('path')} (-{removed} +{added} lines)"

    def execute(self, args: dict[str, Any]) -> str:
        relative = require_text(args.get("path"), "path")
        old_text, new_text = _texts(args)
        file_path = self.workspace.resolve_existing_file(relative)
        content = read_text_file(file_path)
        updated = replace_unique(content, old_text, new_text)
        atomic_write_file(file_path, updated)
        delta = f"-{line_count(old_text)} +{line_count(new_text)} lines"
        return f"Edited {relative} ({delta})."


def create_tool(env: Environment) -> EditTool:
    return EditTool(env)
