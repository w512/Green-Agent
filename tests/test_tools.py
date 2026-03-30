"""Tool registry: discovery, validation, and normalization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.tools import (
    Environment,
    Tool,
    ToolLoadError,
    definitions,
    load_tools,
    normalize_tool,
)
from agent.workspace import Workspace

GOOD_JSON = {
    "type": "function",
    "function": {"name": "echo", "parameters": {"type": "object"}},
}

GOOD_PY = """
class EchoTool:
    needs_approval = True
    trust = "command"

    def __init__(self, env):
        self.root = env.workspace.root

    def describe(self, args):
        return f"echo {args.get('text')}"

    def execute(self, args):
        return args.get("text", "")


def create_tool(env):
    return EchoTool(env)
"""


def make_tool(tools_dir: Path, name: str, py: str, spec: object) -> Path:
    tool_dir = tools_dir / name
    tool_dir.mkdir(parents=True)
    (tool_dir / f"{name}.py").write_text(py)
    text = spec if isinstance(spec, str) else json.dumps(spec)
    (tool_dir / f"{name}.json").write_text(text)
    return tool_dir


@pytest.fixture()
def env(tmp_path: Path) -> Environment:
    root = tmp_path / "ws"
    root.mkdir()
    return Environment(Workspace(root))


ALL_TOOLS = {
    "read", "glob", "grep",
    "write", "edit", "patch", "delete",
    "bash", "check", "fetch", "todo",
}  # fmt: skip
READ_ONLY = {"read", "glob", "grep", "todo"}


class TestBuiltinRegistry:
    def test_loads_all_tools(self, registry: dict[str, Tool]) -> None:
        assert set(registry) == ALL_TOOLS

    def test_definitions_shape(self, registry: dict[str, Tool]) -> None:
        for name, tool in registry.items():
            assert tool.definition["type"] == "function"
            assert tool.definition["function"]["name"] == name
            params = tool.definition["function"]["parameters"]
            assert params["type"] == "object"
        assert len(definitions(registry)) == len(registry)

    def test_approval_flags(self, registry: dict[str, Tool]) -> None:
        for name, tool in registry.items():
            if name == "check":
                continue
            expected = name not in READ_ONLY
            assert tool.needs_approval({"path": "x"}) is expected, name
        # check: syntax-checking a file is free, project checks are not
        assert registry["check"].needs_approval({"path": "x.py"}) is False
        assert registry["check"].needs_approval({}) is True

    def test_trust_kinds(self, registry: dict[str, Tool]) -> None:
        kinds = {
            name: tool.trust({"path": "x"}) for name, tool in registry.items()
        }
        assert kinds["bash"] == "command"
        assert kinds["fetch"] == "always"
        assert registry["check"].trust({}) == "always"
        for name in (
            "read",
            "glob",
            "grep",
            "write",
            "edit",
            "patch",
            "delete",
        ):
            assert kinds[name] == "path", name


class TestDiscovery:
    def test_custom_tool(self, tmp_path: Path, env: Environment) -> None:
        tools_dir = tmp_path / "tools"
        make_tool(tools_dir, "echo", GOOD_PY, GOOD_JSON)
        registry = load_tools(env, tools_dir)
        tool = registry["echo"]
        assert tool.needs_approval({}) is True
        assert tool.trust({}) == "command"
        assert tool.describe({"text": "hi"}) == "echo hi"
        assert tool.execute({"text": "hi"}) == "hi"

    def test_skips_files_and_incomplete_dirs(
        self, tmp_path: Path, env: Environment
    ) -> None:
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "README.md").write_text("docs")
        (tools_dir / "nojson").mkdir()
        (tools_dir / "nojson" / "nojson.py").write_text("x = 1")
        (tools_dir / "nopy").mkdir()
        (tools_dir / "nopy" / "nopy.json").write_text("{}")
        assert load_tools(env, tools_dir) == {}

    def test_skips_module_without_factory(
        self, tmp_path: Path, env: Environment
    ) -> None:
        tools_dir = tmp_path / "tools"
        make_tool(tools_dir, "echo", "VALUE = 1\n", GOOD_JSON)
        assert load_tools(env, tools_dir) == {}

    def test_skips_factory_without_execute(
        self, tmp_path: Path, env: Environment
    ) -> None:
        tools_dir = tmp_path / "tools"
        py = "def create_tool(env):\n    return object()\n"
        make_tool(tools_dir, "echo", py, GOOD_JSON)
        assert load_tools(env, tools_dir) == {}

    def test_malformed_json_raises(
        self, tmp_path: Path, env: Environment
    ) -> None:
        tools_dir = tmp_path / "tools"
        make_tool(tools_dir, "echo", GOOD_PY, "{not json")
        with pytest.raises(ToolLoadError, match="Cannot read"):
            load_tools(env, tools_dir)

    @pytest.mark.parametrize(
        "spec",
        [
            {"type": "tool", "function": {"name": "echo"}},
            {"type": "function", "function": {}},
            {"type": "function"},
            [],
        ],
    )
    def test_invalid_shape_raises(
        self, tmp_path: Path, env: Environment, spec: object
    ) -> None:
        tools_dir = tmp_path / "tools"
        make_tool(tools_dir, "echo", GOOD_PY, spec)
        with pytest.raises(ToolLoadError, match="Invalid tool definition"):
            load_tools(env, tools_dir)

    def test_duplicate_name_raises(
        self, tmp_path: Path, env: Environment
    ) -> None:
        tools_dir = tmp_path / "tools"
        make_tool(tools_dir, "a", GOOD_PY, GOOD_JSON)
        make_tool(tools_dir, "b", GOOD_PY, GOOD_JSON)
        with pytest.raises(ToolLoadError, match="Duplicate tool name"):
            load_tools(env, tools_dir)

    def test_import_error_propagates_and_cleans_up(
        self, tmp_path: Path, env: Environment
    ) -> None:
        import sys

        tools_dir = tmp_path / "tools"
        make_tool(
            tools_dir, "broken", 'raise RuntimeError("boom")\n', GOOD_JSON
        )
        with pytest.raises(RuntimeError, match="boom"):
            load_tools(env, tools_dir)
        assert "pyagent_tools.broken" not in sys.modules

    def test_factory_receives_environment(
        self, tmp_path: Path, env: Environment
    ) -> None:
        tools_dir = tmp_path / "tools"
        py = (
            "def create_tool(env):\n"
            "    class T:\n"
            "        def execute(self, args):\n"
            "            return str(env.workspace.root)\n"
            "    return T()\n"
        )
        make_tool(tools_dir, "echo", py, GOOD_JSON)
        registry = load_tools(env, tools_dir)
        assert registry["echo"].execute({}) == str(env.workspace.root)


class TestNormalize:
    def test_defaults(self) -> None:
        class Impl:
            def execute(self, args: dict) -> str:
                return "ok"

        tool = normalize_tool(Impl(), GOOD_JSON)
        assert tool.name == "echo"
        assert tool.needs_approval({}) is False
        assert tool.trust({}) == "always"
        assert tool.describe({}) == "echo"
        assert tool.execute({}) == "ok"

    def test_callable_trust(self) -> None:
        class Impl:
            def trust(self, args: dict) -> str:
                return "command" if "command" in args else "path"

            def execute(self, args: dict) -> str:
                return ""

        tool = normalize_tool(Impl(), GOOD_JSON)
        assert tool.trust({"command": "ls"}) == "command"
        assert tool.trust({"path": "x"}) == "path"
