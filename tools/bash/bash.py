"""bash: run a shell command in the workspace."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent.command import DEFAULT_TIMEOUT, MAX_TIMEOUT, run_command
from agent.params import positive_int, require_text

if TYPE_CHECKING:
    from agent.tools import Environment

DESCRIBE_CHARS = 200


def resolve_timeout(value: object) -> float:
    seconds = positive_int(value, "timeout_seconds", int(DEFAULT_TIMEOUT))
    return float(min(seconds, int(MAX_TIMEOUT)))


class BashTool:
    needs_approval = True
    trust = "command"

    def __init__(self, env: Environment) -> None:
        self.workspace = env.workspace

    def describe(self, args: dict[str, Any]) -> str:
        command = str(args.get("command", "")).strip().replace("\n", " ")
        if len(command) > DESCRIBE_CHARS:
            command = command[: DESCRIBE_CHARS - 3] + "..."
        return f"bash: {command}"

    def execute(self, args: dict[str, Any]) -> str:
        command = require_text(args.get("command"), "command")
        timeout = resolve_timeout(args.get("timeout_seconds"))
        return run_command(command, self.workspace.root, timeout)


def create_tool(env: Environment) -> BashTool:
    return BashTool(env)
