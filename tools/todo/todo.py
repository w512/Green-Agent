"""todo: in-memory task list for the current session."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent.params import require_text, text

if TYPE_CHECKING:
    from agent.tools import Environment

STATUSES = ("pending", "in_progress", "completed", "cancelled")
MARKS = {"pending": " ", "in_progress": ">", "completed": "x", "cancelled": "-"}


def normalize_item(item: object, index: int) -> tuple[str, str, str]:
    where = f"todos[{index}]"
    if not isinstance(item, dict):
        raise ValueError(f"{where} must be an object.")
    item_id = require_text(item.get("id"), f"{where}.id")
    content = text(item.get("content"), f"{where}.content")
    status = item.get("status")
    if status not in STATUSES:
        allowed = ", ".join(STATUSES)
        raise ValueError(f"{where}.status must be one of: {allowed}.")
    return item_id, content, status


def format_list(items: dict[str, tuple[str, str]]) -> str:
    if not items:
        return "(no todos)"
    lines = []
    for number, (item_id, (content, status)) in enumerate(items.items(), 1):
        lines.append(f"{number}. [{MARKS[status]}] {item_id}: {content}")
    return "\n".join(lines)


class TodoTool:
    needs_approval = False

    def __init__(self, env: Environment) -> None:
        self.items: dict[str, tuple[str, str]] = {}

    def execute(self, args: dict[str, Any]) -> str:
        todos = args.get("todos")
        if not isinstance(todos, list):
            raise ValueError("todos must be an array.")
        incoming = [normalize_item(item, i) for i, item in enumerate(todos)]
        if args.get("merge") is False:
            self.items = {}
        for item_id, content, status in incoming:
            self.items[item_id] = (content, status)
        return format_list(self.items)


def create_tool(env: Environment) -> TodoTool:
    return TodoTool(env)
