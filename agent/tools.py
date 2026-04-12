"""Tool registry: discovers tools/{name}/{name}.json + tools/{name}/{name}.py.

A tool directory contributes:

- `{name}.json` — OpenAI function definition sent to the model;
- `{name}.py` — a module exposing `create_tool(env)` that returns an
  object with `execute(args)` and optional attributes `needs_approval`
  (bool or callable of args, default False), `trust` ("path" | "command"
  | "always", or a callable of args; default "always"), and
  `describe(args) -> str`.

Directories that do not fit this shape are skipped silently; malformed
JSON or duplicate tool names raise.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.workspace import Workspace

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
FACTORY_NAME = "create_tool"
MODULE_PREFIX = "green_agent_tools"

Args = dict[str, Any]
TrustKind = str  # "path" | "command" | "always"


class ToolLoadError(Exception):
    """A tool directory is present but its definition is invalid."""


@dataclass(frozen=True)
class Environment:
    """Runtime state handed to tool factories."""

    workspace: Workspace


@dataclass(frozen=True)
class Tool:
    name: str
    definition: dict[str, Any]
    needs_approval: Callable[[Args], bool]
    trust: Callable[[Args], TrustKind]
    describe: Callable[[Args], str]
    execute: Callable[[Args], object]


def _as_callable(value: object) -> Callable[[Args], Any]:
    if callable(value):
        return value
    return lambda _args: value


def normalize_tool(impl: object, definition: dict[str, Any]) -> Tool:
    name = definition["function"]["name"]
    describe = getattr(impl, "describe", None)
    if not callable(describe):
        describe = lambda _args: name  # noqa: E731
    needs_approval = getattr(impl, "needs_approval", False)
    if not callable(needs_approval):
        needs_approval = needs_approval is True
    return Tool(
        name=name,
        definition=definition,
        needs_approval=_as_callable(needs_approval),
        trust=_as_callable(getattr(impl, "trust", "always")),
        describe=describe,
        execute=impl.execute,  # type: ignore[attr-defined]
    )


def _load_definition(json_path: Path) -> dict[str, Any]:
    try:
        definition = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ToolLoadError(f"Cannot read {json_path}: {error}") from error
    if not isinstance(definition, dict) or definition.get("type") != "function":
        raise ToolLoadError(f"Invalid tool definition: {json_path}")
    function = definition.get("function")
    name = function.get("name") if isinstance(function, dict) else None
    if not isinstance(name, str) or not name:
        raise ToolLoadError(f"Invalid tool definition: {json_path}")
    return definition


def _import_module(name: str, py_path: Path) -> Any:
    module_name = f"{MODULE_PREFIX}.{name}"
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise ToolLoadError(f"Cannot load tool module: {py_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def load_tool(tool_dir: Path, env: Environment) -> Tool | None:
    """Load one tool directory; None when it is not a tool."""
    if not tool_dir.is_dir():
        return None
    name = tool_dir.name
    py_path = tool_dir / f"{name}.py"
    json_path = tool_dir / f"{name}.json"
    if not py_path.is_file() or not json_path.is_file():
        return None
    module = _import_module(name, py_path)
    factory = getattr(module, FACTORY_NAME, None)
    if not callable(factory):
        return None
    impl = factory(env)
    if not callable(getattr(impl, "execute", None)):
        return None
    definition = _load_definition(json_path)
    return normalize_tool(impl, definition)


def load_tools(
    env: Environment, tools_dir: Path = TOOLS_DIR
) -> dict[str, Tool]:
    """Build the registry: tool name -> Tool, in directory name order."""
    registry: dict[str, Tool] = {}
    for tool_dir in sorted(tools_dir.iterdir()):
        tool = load_tool(tool_dir, env)
        if tool is None:
            continue
        if tool.name in registry:
            raise ToolLoadError(f"Duplicate tool name: {tool.name}")
        registry[tool.name] = tool
    return registry


def definitions(registry: dict[str, Tool]) -> list[dict[str, Any]]:
    return [tool.definition for tool in registry.values()]
