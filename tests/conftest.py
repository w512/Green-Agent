"""Shared fixtures: a sample project tree and a loaded tool registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.tools import Environment, Tool, load_tools
from agent.workspace import Workspace


def write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """A small project with text, binary, and skipped directories."""
    root = tmp_path / "project"
    write(root / "README.md", "# Title\nhello world\n")
    app = "def main():\n    print('Hello')\n\nmain()\n"
    write(root / "src" / "app.py", app)
    write(root / "src" / "util.py", "VALUE = 42\nHELLO = 'x'\n")
    write(root / "src" / "__pycache__" / "app.pyc", b"\x00\x01cache")
    write(root / "docs" / "guide.md", "Guide\n")
    write(root / "bin" / "data.bin", b"\x00\x01\x02hello")
    write(root / "empty.txt", "")
    write(root / "node_modules" / "pkg" / "index.js", "hello\n")
    write(root / ".git" / "HEAD", "ref: refs/heads/main\n")
    return root


@pytest.fixture()
def ws(project: Path) -> Workspace:
    return Workspace(project)


@pytest.fixture()
def registry(ws: Workspace) -> dict[str, Tool]:
    return load_tools(Environment(ws))
