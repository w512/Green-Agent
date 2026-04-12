"""check: syntax-check one file, or run the project's own check command."""

from __future__ import annotations

import json
import re
import shutil
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent.command import run_command, run_file
from agent.textfile import read_text_file

if TYPE_CHECKING:
    from agent.tools import Environment

PROJECT_CHECK_TIMEOUT = 300.0
MAKE_TARGET_RE = re.compile(r"^check\s*:", re.MULTILINE)


def has_path(args: dict[str, Any]) -> bool:
    path = args.get("path")
    return isinstance(path, str) and bool(path)


# --- in-process syntax checks -------------------------------------------------


def check_python(path: Path, source: str) -> str:
    try:
        compile(source, str(path), "exec")
    except SyntaxError as error:
        where = f"line {error.lineno}" if error.lineno else "unknown line"
        return f"Syntax error in {path.name}, {where}: {error.msg}"
    except ValueError as error:  # e.g. null bytes
        return f"Syntax error in {path.name}: {error}"
    return "OK: python syntax"


def check_json(path: Path, source: str) -> str:
    try:
        json.loads(source)
    except ValueError as error:
        return f"Syntax error in {path.name}: {error}"
    return "OK: valid JSON"


def check_toml(path: Path, source: str) -> str:
    try:
        tomllib.loads(source)
    except tomllib.TOMLDecodeError as error:
        return f"Syntax error in {path.name}: {error}"
    return "OK: valid TOML"


IN_PROCESS: dict[str, Callable[[Path, str], str]] = {
    ".py": check_python,
    ".pyi": check_python,
    ".json": check_json,
    ".toml": check_toml,
}

EXTERNAL: dict[str, tuple[str, list[str]]] = {
    ".js": ("node", ["--check"]),
    ".mjs": ("node", ["--check"]),
    ".cjs": ("node", ["--check"]),
    ".sh": ("bash", ["-n"]),
    ".bash": ("bash", ["-n"]),
}


# --- project check discovery --------------------------------------------------


def _pyproject_check(root: Path) -> str | None:
    path = root / "pyproject.toml"
    if not path.is_file():
        return None
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    command = data.get("tool", {}).get("green-agent", {}).get("check")
    return command if isinstance(command, str) and command.strip() else None


def _package_json_check(root: Path) -> str | None:
    path = root / "package.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    scripts = data.get("scripts") if isinstance(data, dict) else None
    if isinstance(scripts, dict) and isinstance(scripts.get("check"), str):
        return "npm run check"
    return None


def _makefile_check(root: Path) -> str | None:
    for name in ("Makefile", "makefile", "GNUmakefile"):
        path = root / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if MAKE_TARGET_RE.search(text):
            return "make check"
    return None


def project_check_command(root: Path) -> str | None:
    for finder in (_pyproject_check, _package_json_check, _makefile_check):
        command = finder(root)
        if command:
            return command
    return None


NO_PROJECT_CHECK = (
    "No project check command found. Define one in pyproject.toml "
    '([tool.green-agent] check = "..."), package.json (scripts.check), or a '
    "Makefile 'check' target; or run tests and linters directly with bash."
)


class CheckTool:
    def __init__(self, env: Environment) -> None:
        self.workspace = env.workspace

    @staticmethod
    def needs_approval(args: dict[str, Any]) -> bool:
        return not has_path(args)

    @staticmethod
    def trust(args: dict[str, Any]) -> str:
        return "path" if has_path(args) else "always"

    def describe(self, args: dict[str, Any]) -> str:
        if has_path(args):
            return f"check {args['path']}"
        command = project_check_command(self.workspace.root) or "(none)"
        return f"check: {command}"

    def _check_file(self, relative: str) -> str:
        path = self.workspace.resolve_existing_file(relative)
        suffix = path.suffix.lower()
        if suffix in IN_PROCESS:
            return IN_PROCESS[suffix](path, read_text_file(path))
        if suffix in EXTERNAL:
            program, flags = EXTERNAL[suffix]
            if shutil.which(program) is None:
                raise ValueError(
                    f"{program} is not installed; cannot check {relative}."
                )
            return run_file([program, *flags, str(path)], self.workspace.root)
        supported = ", ".join(sorted([*IN_PROCESS, *EXTERNAL]))
        kind = suffix or "files without extension"
        raise ValueError(
            f"No syntax checker for {kind}; supported: {supported}."
        )

    def execute(self, args: dict[str, Any]) -> str:
        if has_path(args):
            return self._check_file(args["path"])
        command = project_check_command(self.workspace.root)
        if command is None:
            raise ValueError(NO_PROJECT_CHECK)
        return run_command(command, self.workspace.root, PROJECT_CHECK_TIMEOUT)


def create_tool(env: Environment) -> CheckTool:
    return CheckTool(env)
