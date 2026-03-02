"""glob: list workspace files whose relative path matches a pattern."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent.globmatch import glob_to_regex, match_glob
from agent.params import require_text
from agent.textfile import truncate_output
from agent.walk import walk_files

if TYPE_CHECKING:
    from agent.tools import Environment

MAX_MATCHES = 200


class GlobTool:
    needs_approval = False
    trust = "path"

    def __init__(self, env: Environment) -> None:
        self.workspace = env.workspace

    def execute(self, args: dict[str, Any]) -> str:
        pattern = require_text(args.get("pattern"), "pattern")
        glob_to_regex(pattern)  # validate before walking

        path_arg = args.get("path")
        target = self.workspace.resolve_existing_path(path_arg)
        if not target.is_directory:
            raise ValueError(f"glob path must be a directory: {path_arg}")

        base_rel = self.workspace.relative(target.path)
        matches: list[str] = []
        extra = 0
        for _abs_path, rel_path in walk_files(target.path, base_rel):
            if not match_glob(rel_path, pattern):
                continue
            if len(matches) < MAX_MATCHES:
                matches.append(rel_path)
            else:
                extra += 1

        if not matches:
            return "No files matched."
        if extra > 0:
            matches.append(f"... {extra} more")
        return truncate_output("\n".join(matches))


def create_tool(env: Environment) -> GlobTool:
    return GlobTool(env)
