"""Approval policy for tools that modify the workspace or run commands.

Layers, evaluated in order for a tool with `needs_approval=True`:

1. `auto_approve=True` allows everything (non-interactive runs).
2. Shell commands matching a dangerous pattern (destructive git, rm -rf,
   sudo, network transfer, inline interpreters, ...) always ask, even if
   the user chose "always" for the tool earlier.
3. A tool the user approved with "always" in this session is allowed.
4. Inside the trust root - the workspace, when it lives in a git
   repository - the action is allowed if git can undo it:
   - path tools: the target is new, or is tracked by git;
   - command tools: every path-like token stays inside the trust root
     and the command has no variables or substitutions to hide behind.
5. Otherwise the user is asked.

The shell command heuristics live in `agent.shellcheck`. This is a safety
net, not a sandbox: the shell can still reach the network and any path
the heuristics do not recognise.
"""

from __future__ import annotations

import enum
import os
from collections.abc import Callable
from pathlib import Path

from agent.git import is_tracked
from agent.shellcheck import command_stays_inside, dangerous_command
from agent.tools import Args, Tool
from agent.workspace import Workspace, is_inside


class Decision(enum.Enum):
    ALLOW = "allow"
    ALWAYS = "always"
    DENY = "deny"


Ask = Callable[[str], Decision]

ALLOW_ANSWERS = frozenset({"y", "yes"})
ALWAYS_ANSWERS = frozenset({"a", "always"})


def parse_answer(answer: str) -> Decision:
    normalized = answer.strip().lower()
    if normalized in ALLOW_ANSWERS:
        return Decision.ALLOW
    if normalized in ALWAYS_ANSWERS:
        return Decision.ALWAYS
    return Decision.DENY


class Permissions:
    """`ask` is the frontend's prompt: description -> Decision.

    `trust_root` defaults to the workspace root when the workspace lives
    in a git repository (git can undo changes there); pass a path to
    trust a different directory. Outside git there is no trust root and
    every write or command asks.
    """

    def __init__(
        self,
        workspace: Workspace,
        ask: Ask,
        *,
        auto_approve: bool = False,
        trust_root: Path | None = None,
    ) -> None:
        self.workspace = workspace
        self.auto_approve = auto_approve
        self._ask = ask
        if trust_root is None and workspace.git_root is not None:
            trust_root = workspace.root
        self.trust_root = trust_root
        self._always: set[str] = set()

    def approve(self, tool: Tool, args: Args) -> bool:
        if not tool.needs_approval(args) or self.auto_approve:
            return True

        kind = tool.trust(args)
        reason = None
        if kind == "command":
            command = args.get("command")
            if isinstance(command, str):
                reason = dangerous_command(command)

        if reason is None:
            if tool.name in self._always:
                return True
            if self._recoverable(kind, args):
                return True

        description = tool.describe(args)
        if reason:
            description = f"{description}  [{reason}]"
        decision = self._ask(description)
        if decision is Decision.ALWAYS:
            self._always.add(tool.name)
            return True
        return decision is Decision.ALLOW

    def _recoverable(self, kind: str, args: Args) -> bool:
        root = self.trust_root
        if root is None:
            return False
        if kind == "path":
            return self._path_recoverable(root, args.get("path"))
        if kind == "command":
            command = args.get("command")
            if not isinstance(command, str) or not command.strip():
                return False
            return command_stays_inside(root, self.workspace.root, command)
        return False

    def _path_recoverable(self, root: Path, relative: object) -> bool:
        if not isinstance(relative, str) or not relative:
            return False
        target = Path(os.path.abspath(self.workspace.root / relative))
        if not is_inside(root, target):
            return False
        try:
            target.lstat()
        except FileNotFoundError:
            return True  # nothing to lose yet
        except OSError:
            return False
        git_root = self.workspace.git_root
        return git_root is not None and is_tracked(git_root, target)
