from __future__ import annotations

from pathlib import Path

import pytest

from agent.tools import Tool
from agent.workspace import WorkspaceError


@pytest.fixture()
def glob(registry: dict[str, Tool]) -> Tool:
    return registry["glob"]


def test_recursive_pattern(glob: Tool) -> None:
    result = glob.execute({"pattern": "**/*.py"})
    assert result == "src/app.py\nsrc/util.py"  # __pycache__ skipped


def test_basename_pattern(glob: Tool) -> None:
    assert glob.execute({"pattern": "*.md"}) == "README.md\ndocs/guide.md"


def test_directory_pattern(glob: Tool) -> None:
    assert glob.execute({"pattern": "src/*.py"}) == "src/app.py\nsrc/util.py"


def test_subdirectory_path_keeps_root_relative(glob: Tool) -> None:
    result = glob.execute({"pattern": "*.py", "path": "src"})
    assert result == "src/app.py\nsrc/util.py"


def test_skipped_directories(glob: Tool) -> None:
    assert glob.execute({"pattern": "*.js"}) == "No files matched."
    assert glob.execute({"pattern": "HEAD"}) == "No files matched."


def test_no_match(glob: Tool) -> None:
    assert glob.execute({"pattern": "*.rs"}) == "No files matched."


def test_path_must_be_directory(glob: Tool) -> None:
    with pytest.raises(ValueError, match="must be a directory"):
        glob.execute({"pattern": "*", "path": "README.md"})


def test_path_escape(glob: Tool) -> None:
    with pytest.raises(WorkspaceError):
        glob.execute({"pattern": "*", "path": "../"})


@pytest.mark.parametrize("pattern", [None, "", 3])
def test_invalid_pattern(glob: Tool, pattern: object) -> None:
    with pytest.raises(ValueError, match="pattern must be"):
        glob.execute({"pattern": pattern})


def test_match_cap(glob: Tool, project: Path) -> None:
    many = project / "many"
    many.mkdir()
    for i in range(205):
        (many / f"f{i:03}.txt").write_text("x")
    result = glob.execute({"pattern": "many/*.txt"})
    lines = result.split("\n")
    assert len(lines) == 201
    assert lines[-1] == "... 5 more"
