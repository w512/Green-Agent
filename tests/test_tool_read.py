from __future__ import annotations

from pathlib import Path

import pytest

from agent.textfile import TextFileError
from agent.tools import Tool
from agent.workspace import WorkspaceError


@pytest.fixture()
def read(registry: dict[str, Tool]) -> Tool:
    return registry["read"]


def test_whole_file(read: Tool) -> None:
    assert read.execute({"path": "README.md"}) == "1|# Title\n2|hello world"


def test_offset_and_limit(read: Tool) -> None:
    result = read.execute({"path": "src/app.py", "offset": 2, "limit": 2})
    assert result == "2|    print('Hello')\n3|\n... 1 lines not shown"


def test_float_numbers_are_truncated(read: Tool) -> None:
    result = read.execute({"path": "src/app.py", "offset": 2.9, "limit": 1.5})
    assert result.startswith("2|")
    assert result.count("\n") == 1  # one line + note


def test_offset_past_end(read: Tool) -> None:
    result = read.execute({"path": "README.md", "offset": 10})
    assert result == "(no lines at offset 10; file has 2 lines)"


def test_empty_file(read: Tool) -> None:
    assert read.execute({"path": "empty.txt"}) == "(empty file)"


def test_hard_cap(read: Tool, project: Path) -> None:
    big = project / "big.txt"
    big.write_text("\n".join(f"line {i}" for i in range(1, 2501)) + "\n")
    result = read.execute({"path": "big.txt", "limit": 5000})
    lines = result.split("\n")
    assert len(lines) == 2001
    assert lines[0] == "   1|line 1"
    assert lines[-1] == "... 500 lines not shown"


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"path": "README.md", "offset": 0}, "offset must be a positive"),
        ({"path": "README.md", "limit": 0}, "limit must be a positive"),
        ({"path": "README.md", "offset": "3"}, "offset must be a number"),
        ({"path": "README.md", "limit": True}, "limit must be a number"),
        ({"path": "README.md", "limit": float("inf")}, "finite number"),
    ],
)
def test_bad_numbers(read: Tool, args: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        read.execute(args)


def test_binary_file(read: Tool) -> None:
    with pytest.raises(TextFileError, match="looks binary"):
        read.execute({"path": "bin/data.bin"})


def test_missing_file(read: Tool) -> None:
    with pytest.raises(WorkspaceError, match="Not found: nope.txt"):
        read.execute({"path": "nope.txt"})


@pytest.mark.parametrize("path", [None, "", "../x", "/etc/passwd"])
def test_bad_paths(read: Tool, path: object) -> None:
    with pytest.raises(WorkspaceError):
        read.execute({"path": path})
