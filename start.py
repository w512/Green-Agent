#!/usr/bin/env python3
"""Entry point: a terminal chat with the coding agent."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.py"
TEMPLATE_PATH = BASE_DIR / "config.template.py"

DEFAULT_MAX_STEPS = 30
PREVIEW_LINES = 8
PREVIEW_CHARS = 600
EXIT_WORDS = frozenset({"exit", "quit", "q"})

USAGE = """Set API_KEY, BASE_URL, and MODEL in config.py.
Any OpenAI-compatible Chat Completions provider works.
Type a task at the prompt; 'exit' or Ctrl-D quits."""

APPROVAL_HINT = (
    "Writes and shell commands may ask for approval: "
    "y = once, a = always this session, Enter = deny."
)


# --- output -----------------------------------------------------------------


def _tty() -> bool:
    return sys.stdout.isatty()


def _paint(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _tty() else text


def warn(text: str) -> str:
    return _paint("1;33", text)


def error(text: str) -> str:
    return _paint("1;31", text)


def dim(text: str) -> str:
    return _paint("2", text)


def tool_line(text: str) -> str:
    return _paint("36", text)


def compact_args(args: object, args_text: object) -> str:
    if isinstance(args, dict):
        text = json.dumps(args, ensure_ascii=False)
    else:
        text = str(args_text or "")
    return text if len(text) <= 160 else text[:157] + "..."


def preview(text: str) -> str:
    lines = text.splitlines() or [""]
    shown = lines[:PREVIEW_LINES]
    body = "\n".join(shown)[:PREVIEW_CHARS]
    hidden = len(lines) - len(shown)
    if hidden > 0 or len(body) < len(text):
        body += f"\n... ({len(lines)} lines total)"
    return body


def print_event(event: dict[str, Any]) -> None:
    kind = event["type"]
    if kind == "step":
        step = f"step {event['step']}/{event['max_steps']}"
        print(dim(f"[{step} · {event['model']}]"))
    elif kind == "assistant":
        print(event["text"])
    elif kind == "tool":
        args = compact_args(event["args"], event["args_text"])
        print(tool_line(f"-> {event['name']} {args}"))
    elif kind == "result":
        status = event["status"]
        text = preview(event["preview"])
        if status == "ok":
            print(dim(text))
        elif status == "denied":
            print(warn(text))
        else:
            print(error(text))


# --- setup ------------------------------------------------------------------


def ensure_config() -> bool:
    if CONFIG_PATH.exists():
        return False
    shutil.copyfile(TEMPLATE_PATH, CONFIG_PATH)
    return True


def resolve_root(arg: str | None) -> Path:
    root = Path(arg or Path.cwd()).resolve()
    if not root.exists():
        raise SystemExit(error(f"Not found: {root}"))
    if not root.is_dir():
        raise SystemExit(error(f"Not a directory: {root}"))
    return root


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="start.py",
        description="Terminal chat with a coding agent.",
        epilog=USAGE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "root", nargs="?", help="project directory (default: current)"
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="auto-approve every tool call (non-interactive runs)",
    )
    return parser.parse_args(argv)


def console_ask(description: str) -> Any:
    from agent.permissions import Decision, parse_answer

    if not sys.stdin.isatty():
        print(warn("No terminal to confirm; denying. Use --yes to allow."))
        return Decision.DENY
    try:
        answer = input(warn(f"\nApprove: {description}? [y/N/a] "))
    except EOFError:
        return Decision.DENY
    return parse_answer(answer)


# --- chat -------------------------------------------------------------------


def read_task() -> str | None:
    try:
        return input(_paint("1", "\n> ")).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def chat(provider: Any, registry: dict[str, Any], permissions: Any) -> int:
    from agent.agent import AgentError, run_agent

    history: list[dict[str, Any]] | None = None
    while True:
        task = read_task()
        if task is None or task.lower() in EXIT_WORDS:
            return 0
        if not task:
            continue
        try:
            result = run_agent(
                task,
                provider=provider,
                registry=registry,
                permissions=permissions,
                max_steps=DEFAULT_MAX_STEPS,
                on_event=print_event,
                prior_messages=history,
            )
            history = result.messages
        except AgentError as failure:
            print(error(f"\n{failure}"))
            history = failure.messages
        except KeyboardInterrupt:
            print(warn("\nInterrupted; the last completed turn is kept."))
        except Exception as failure:  # noqa: BLE001 - keep the chat alive
            print(error(f"\n{type(failure).__name__}: {failure}"))


def main() -> int:
    if ensure_config():
        print(warn("Created config.py"))

    from agent.llm import ConfigError, create_provider
    from agent.permissions import Permissions
    from agent.tools import Environment, load_tools
    from agent.workspace import Workspace

    args = parse_args(sys.argv[1:])

    try:
        provider = create_provider(log=lambda text: print(warn(text)))
    except ConfigError as failure:
        print(error(f"{failure}\n"))
        print(USAGE)
        return 1

    workspace = Workspace(resolve_root(args.root))
    registry = load_tools(Environment(workspace))
    permissions = Permissions(workspace, auto_approve=args.yes, ask=console_ask)

    print(f"Workspace: {workspace.root}")
    print(f"Git root:  {workspace.git_root or '(not a git repo)'}")
    print(f"Tools:     {', '.join(registry) or '(none)'}")
    print(f"Model:     {provider.model}")
    if args.yes:
        print(warn("Auto-approve is on: every tool call runs unasked."))
    elif permissions.trust_root is None:
        print(dim("Not a git repository: every write or command will ask."))
    else:
        print(dim(APPROVAL_HINT))
    print(dim("Type a task; 'exit' or Ctrl-D to quit."))

    return chat(provider, registry, permissions)


if __name__ == "__main__":
    sys.exit(main())
