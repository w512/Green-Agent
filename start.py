#!/usr/bin/env python3
"""Entry point: configure, wire the pieces together, start the console chat."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.py"
TEMPLATE_PATH = BASE_DIR / "config.template.py"

DEFAULT_MAX_STEPS = 30

EPILOG = """Set API_KEY, BASE_URL, and MODEL in config.py (created on first
run). Any OpenAI-compatible Chat Completions provider works.

Examples:
  python start.py                       chat about the current directory
  python start.py ../my-project         chat about another project
  python start.py -y -t "run the tests" one task, auto-approved, then exit"""

APPROVAL_HINT = (
    "Writes and shell commands may ask for approval: "
    "y = once, a = always this session, Enter = deny."
)


def ensure_config() -> bool:
    if CONFIG_PATH.exists():
        return False
    shutil.copyfile(TEMPLATE_PATH, CONFIG_PATH)
    return True


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="start.py",
        description="Terminal chat with a coding agent.",
        epilog=EPILOG,
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
    parser.add_argument(
        "-t",
        "--task",
        metavar="TEXT",
        help="run one task and exit (exit code 1 on failure)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        metavar="N",
        help=f"model/tool round trips per task (default {DEFAULT_MAX_STEPS})",
    )
    parser.add_argument(
        "--model", metavar="NAME", help="override MODEL from config.py"
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="full-screen interface: project tree, file viewer, chat "
        "(needs the tui extra: uv sync --extra tui)",
    )
    args = parser.parse_args(argv)
    if args.tui and args.task is not None:
        parser.error("--tui and --task cannot be combined")
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    return args


def resolve_root(arg: str | None) -> Path:
    root = Path(arg or Path.cwd()).resolve()
    if not root.exists():
        raise SystemExit(f"Not found: {root}")
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    created = ensure_config()

    from agent.llm import ConfigError, create_provider
    from agent.permissions import Permissions
    from agent.tools import Environment, load_tools
    from agent.workspace import Workspace
    from cli import Chat, Palette, console_ask, setup_readline

    palette = Palette()
    if created:
        print(palette.warn(f"Created {CONFIG_PATH.name}"))

    try:
        provider = create_provider(log=lambda text: print(palette.warn(text)))
    except ConfigError as failure:
        print(palette.error(str(failure)))
        print(EPILOG)
        return 1
    if args.model:
        provider.model = args.model

    workspace = Workspace(resolve_root(args.root))
    registry = load_tools(Environment(workspace))

    if args.tui:
        try:
            from tui import run_tui
        except ImportError:
            print(palette.error("The TUI needs Textual: uv sync --extra tui"))
            return 1
        return run_tui(
            provider=provider,
            registry=registry,
            workspace=workspace,
            auto_approve=args.yes,
            max_steps=args.max_steps,
        )

    permissions = Permissions(
        workspace, auto_approve=args.yes, ask=console_ask(palette)
    )
    chat = Chat(
        provider=provider,
        registry=registry,
        permissions=permissions,
        max_steps=args.max_steps,
        palette=palette,
    )

    if args.task is not None:
        return 0 if chat.run_task(args.task) else 1

    print(f"Workspace: {workspace.root}")
    print(f"Git root:  {workspace.git_root or '(not a git repo)'}")
    print(f"Tools:     {', '.join(registry) or '(none)'}")
    print(f"Model:     {provider.model}")
    if args.yes:
        print(palette.warn("Auto-approve is on: every tool call runs unasked."))
    elif permissions.trust_root is None:
        print(palette.dim("Not a git repository: every write or command asks."))
    else:
        print(palette.dim(APPROVAL_HINT))
    print(palette.dim("Type a task, or /help. 'exit' or Ctrl-D quits."))

    setup_readline()
    return chat.loop()


if __name__ == "__main__":
    sys.exit(main())
