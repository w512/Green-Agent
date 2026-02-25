#!/usr/bin/env python3
"""Entry point. Phase 1: ensure config exists and open the workspace."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.py"
TEMPLATE_PATH = BASE_DIR / "config.template.py"

USAGE = """Usage: python start.py [project-root]

Set API_KEY, BASE_URL, and MODEL in config.py.
Any OpenAI-compatible Chat Completions provider works."""


def _tty() -> bool:
    return sys.stdout.isatty()


def warn(text: str) -> str:
    return f"\033[1;33m{text}\033[0m" if _tty() else text


def error(text: str) -> str:
    return f"\033[1;31m{text}\033[0m" if _tty() else text


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


def main() -> int:
    created = ensure_config()
    if created:
        print(warn("Created config.py"))

    import config
    from agent.workspace import Workspace

    extra = sys.argv[1:]
    if len(extra) > 1:
        print(error("Usage: python start.py [project-root]"))
        return 1

    root = resolve_root(extra[0] if extra else None)
    workspace = Workspace(root)

    print(f"Workspace: {workspace.root}")
    print(f"Git root:  {workspace.git_root or '(not a git repo)'}")

    if not (config.API_KEY or "").strip():
        print(error("\nAPI_KEY is not set.\n"))
        print(USAGE)
        return 1

    print(warn("\nAgent loop is not implemented yet (phase 1)."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
