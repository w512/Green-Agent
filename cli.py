"""Console frontend: renders agent events, reads tasks, asks for approval.

The agent core knows nothing about this module; it only emits events
through `on_event` and asks through the permissions callback.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import select
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent.agent import AgentError, run_agent
from agent.permissions import Decision, parse_answer

PREVIEW_LINES = 8
PREVIEW_CHARS = 600
ARG_VALUE_CHARS = 60
ARGS_CHARS = 200
PASTE_WINDOW_SECONDS = 0.05
HISTORY_FILE = Path.home() / ".pyagent_history"
HISTORY_LENGTH = 1000
EXIT_WORDS = frozenset({"exit", "quit", "/exit", "/quit"})

HELP = """Commands:
  /new            start a new conversation (forget the history)
  /model [name]   show or switch the model
  /tools          list available tools
  /help           show this help
  /exit           quit (also: exit, quit, Ctrl-D)

Input: end a line with a backslash to continue on the next line.
Pasted multi-line text is kept together as one task.
Approval prompts: y = allow once, a = allow this tool for the session,
anything else = deny."""


# --- colors -----------------------------------------------------------------


class Palette:
    """ANSI colors when writing to a terminal; plain text otherwise."""

    def __init__(self, enabled: bool | None = None) -> None:
        if enabled is None:
            enabled = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
        self.enabled = enabled

    def paint(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, text: str) -> str:
        return self.paint("1", text)

    def dim(self, text: str) -> str:
        return self.paint("2", text)

    def warn(self, text: str) -> str:
        return self.paint("1;33", text)

    def error(self, text: str) -> str:
        return self.paint("1;31", text)

    def tool(self, text: str) -> str:
        return self.paint("36", text)

    def prompt(self, text: str) -> str:
        """Bold prompt with readline-safe markers around the color codes."""
        if not self.enabled:
            return text
        return f"\001\033[1m\002{text}\001\033[0m\002"


# --- rendering --------------------------------------------------------------


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def compact_args(args: object, args_text: object = None) -> str:
    """One-line view of tool arguments with long values trimmed."""
    if not isinstance(args, dict):
        return _shorten(str(args_text or ""), ARGS_CHARS)
    trimmed = {
        key: _shorten(value, ARG_VALUE_CHARS)
        if isinstance(value, str)
        else value
        for key, value in args.items()
    }
    text = json.dumps(trimmed, ensure_ascii=False)
    return _shorten(text, ARGS_CHARS)


def preview(text: str) -> str:
    lines = text.splitlines() or [""]
    shown = lines[:PREVIEW_LINES]
    body = "\n".join(shown)[:PREVIEW_CHARS]
    if len(lines) > len(shown) or len(body) < len(text):
        body += f"\n... ({len(lines)} lines total)"
    return body


def render_event(event: dict[str, Any], palette: Palette) -> str:
    kind = event["type"]
    if kind == "step":
        step = f"step {event['step']}/{event['max_steps']}"
        return palette.dim(f"[{step} · {event['model']}]")
    if kind == "assistant":
        return event["text"]
    if kind == "tool":
        args = compact_args(event["args"], event["args_text"])
        return palette.tool(f"-> {event['name']} {args}")
    if kind == "result":
        text = preview(event["preview"])
        status = event["status"]
        if status == "ok":
            return palette.dim(text)
        if status == "denied":
            return palette.warn(text)
        return palette.error(text)
    return ""


# --- input ------------------------------------------------------------------


def setup_readline(history_file: Path = HISTORY_FILE) -> bool:
    """Enable line editing and persistent history when readline exists."""
    try:
        import readline
    except ImportError:  # pragma: no cover - Windows
        return False
    with contextlib.suppress(OSError):
        readline.read_history_file(history_file)
    readline.set_history_length(HISTORY_LENGTH)

    def save() -> None:
        try:
            history_file.parent.mkdir(parents=True, exist_ok=True)
            readline.write_history_file(history_file)
        except OSError:
            pass

    atexit.register(save)
    return True


def pending_input(timeout: float = PASTE_WINDOW_SECONDS) -> bool:
    """True when more typed/pasted input is already waiting on a tty."""
    if not sys.stdin.isatty():
        return False
    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
    except (OSError, ValueError):
        return False
    return bool(ready)


def read_task(palette: Palette) -> str | None:
    """Read one task; None on EOF, "" when the user cancelled the line.

    A trailing backslash continues on the next line; lines that arrive
    together (a paste) are joined into one task.
    """
    lines: list[str] = []
    try:
        lines.append(input(palette.prompt("\n> ")))
        while lines[-1].endswith("\\") or pending_input():
            if lines[-1].endswith("\\"):
                lines[-1] = lines[-1][:-1]
            lines.append(input(palette.prompt(". ")))
    except EOFError:
        print()
        return None
    except KeyboardInterrupt:
        print(palette.dim("\n(cancelled; Ctrl-D or 'exit' to quit)"))
        return ""
    return "\n".join(lines).strip()


def console_ask(palette: Palette) -> Callable[[str], Decision]:
    def ask(description: str) -> Decision:
        if not sys.stdin.isatty():
            notice = "No terminal to confirm; denying. Use --yes to allow."
            print(palette.warn(notice))
            return Decision.DENY
        try:
            answer = input(palette.warn(f"\nApprove: {description}? [y/N/a] "))
        except EOFError:
            return Decision.DENY
        return parse_answer(answer)

    return ask


# --- chat -------------------------------------------------------------------


class Chat:
    def __init__(
        self,
        *,
        provider: Any,
        registry: dict[str, Any],
        permissions: Any,
        max_steps: int,
        palette: Palette | None = None,
        out: Callable[[str], None] = print,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.permissions = permissions
        self.max_steps = max_steps
        self.palette = palette or Palette()
        self.out = out
        self.history: list[dict[str, Any]] | None = None
        self._steps = 0
        self._tools = 0

    def _on_event(self, event: dict[str, Any]) -> None:
        if event["type"] == "step":
            self._steps += 1
        elif event["type"] == "tool":
            self._tools += 1
        self.out(render_event(event, self.palette))

    def _summary(self, started: float) -> str:
        elapsed = time.monotonic() - started
        steps = f"{self._steps} step{'s' if self._steps != 1 else ''}"
        tools = f"{self._tools} tool call{'s' if self._tools != 1 else ''}"
        return self.palette.dim(f"[{steps} · {tools} · {elapsed:.1f}s]")

    def run_task(self, task: str) -> bool:
        """Run one task; True on a normal completion."""
        self._steps = self._tools = 0
        started = time.monotonic()
        try:
            result = run_agent(
                task,
                provider=self.provider,
                registry=self.registry,
                permissions=self.permissions,
                max_steps=self.max_steps,
                on_event=self._on_event,
                prior_messages=self.history,
            )
        except AgentError as failure:
            self.history = failure.messages
            self.out(self.palette.error(f"\n{failure}"))
            return False
        except KeyboardInterrupt:
            notice = "\nInterrupted; the last completed turn is kept."
            self.out(self.palette.warn(notice))
            return False
        except Exception as failure:  # noqa: BLE001 - keep the chat alive
            self.out(
                self.palette.error(f"\n{type(failure).__name__}: {failure}")
            )
            return False
        self.history = result.messages
        self.out(self._summary(started))
        return True

    def handle_command(self, line: str) -> bool:
        """Handle a slash command; False when `line` is not a command."""
        if not line.startswith("/"):
            return False
        command, _, argument = line.partition(" ")
        argument = argument.strip()
        if command == "/help":
            self.out(HELP)
        elif command == "/new":
            self.history = None
            self.out(self.palette.dim("New conversation."))
        elif command == "/model":
            if argument:
                self.provider.model = argument
            self.out(f"Model: {self.provider.model}")
        elif command == "/tools":
            self.out(", ".join(self.registry) or "(none)")
        else:
            self.out(self.palette.error(f"Unknown command: {command}"))
            self.out(self.palette.dim("Type /help for the list of commands."))
        return True

    def loop(self) -> int:
        while True:
            task = read_task(self.palette)
            if task is None or task.lower() in EXIT_WORDS:
                return 0
            if not task or self.handle_command(task):
                continue
            self.run_task(task)
