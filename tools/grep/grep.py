"""grep: search file contents line by line with a Python regex."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent.globmatch import match_glob
from agent.params import optional_text, positive_int, require_text
from agent.textfile import (
    TextFileError,
    read_text_file,
    split_lines,
    truncate_output,
)
from agent.walk import walk_files

if TYPE_CHECKING:
    from agent.tools import Environment

DEFAULT_MAX = 50
HARD_MAX = 200


def compile_pattern(pattern: str, ignore_case: bool) -> re.Pattern[str]:
    try:
        return re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as error:
        raise ValueError(f"Invalid regex pattern: {error}") from error


def resolve_max_matches(value: object) -> int:
    return min(positive_int(value, "max_matches", DEFAULT_MAX), HARD_MAX)


def grep_file(
    abs_path: Path,
    rel_path: str,
    regex: re.Pattern[str],
    out: list[str],
    max_matches: int,
) -> bool:
    """Append matches to `out`; True when the cap has been reached."""
    try:
        content = read_text_file(abs_path)
    except (OSError, TextFileError):
        return False  # unreadable or binary
    for index, line in enumerate(split_lines(content), start=1):
        if not regex.search(line):
            continue
        out.append(f"{rel_path}:{index}:{line}")
        if len(out) >= max_matches:
            return True
    return False


class GrepTool:
    needs_approval = False
    trust = "path"

    def __init__(self, env: Environment) -> None:
        self.workspace = env.workspace

    def _candidates(self, path_arg: object) -> Iterable[tuple[Path, str]]:
        target = self.workspace.resolve_existing_path(path_arg)
        rel = self.workspace.relative(target.path)
        if target.is_file:
            return [(target.path, rel or target.path.name)]
        if target.is_directory:
            return walk_files(target.path, rel)
        raise ValueError(f"Not a file or directory: {path_arg or '.'}")

    def execute(self, args: dict[str, Any]) -> str:
        pattern = require_text(args.get("pattern"), "pattern")
        regex = compile_pattern(pattern, args.get("ignore_case") is True)
        max_matches = resolve_max_matches(args.get("max_matches"))
        glob_filter = optional_text(args.get("glob"), "glob")

        lines: list[str] = []
        truncated = False
        for abs_path, rel_path in self._candidates(args.get("path")):
            if glob_filter and not match_glob(rel_path, glob_filter):
                continue
            if grep_file(abs_path, rel_path, regex, lines, max_matches):
                truncated = True
                break

        if not lines:
            return "No matches."
        body = "\n".join(lines)
        note = f"\n... stopped after {max_matches} matches" if truncated else ""
        return truncate_output(body + note)


def create_tool(env: Environment) -> GrepTool:
    return GrepTool(env)
