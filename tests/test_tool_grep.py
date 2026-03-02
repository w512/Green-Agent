from __future__ import annotations

import pytest

from agent.tools import Tool
from agent.workspace import WorkspaceError


@pytest.fixture()
def grep(registry: dict[str, Tool]) -> Tool:
    return registry["grep"]


def test_basic_match(grep: Tool) -> None:
    assert grep.execute({"pattern": "hello"}) == "README.md:2:hello world"


def test_ignore_case(grep: Tool) -> None:
    result = grep.execute({"pattern": "HELLO", "ignore_case": True})
    assert result == (
        "README.md:2:hello world\n"
        "src/app.py:2:    print('Hello')\n"
        "src/util.py:2:HELLO = 'x'"
    )


def test_case_sensitive_by_default(grep: Tool) -> None:
    assert grep.execute({"pattern": "HELLO"}) == "src/util.py:2:HELLO = 'x'"


def test_regex_features(grep: Tool) -> None:
    result = grep.execute({"pattern": r"^\w+ = \d+$"})
    assert result == "src/util.py:1:VALUE = 42"


def test_glob_filter(grep: Tool) -> None:
    result = grep.execute(
        {"pattern": "hello", "ignore_case": True, "glob": "*.py"}
    )
    assert result == (
        "src/app.py:2:    print('Hello')\nsrc/util.py:2:HELLO = 'x'"
    )


def test_max_matches(grep: Tool) -> None:
    result = grep.execute({"pattern": ".", "max_matches": 2})
    lines = result.split("\n")
    assert len(lines) == 3
    assert lines[-1] == "... stopped after 2 matches"


def test_hard_cap(grep: Tool) -> None:
    result = grep.execute({"pattern": "zzz", "max_matches": 10_000})
    assert result == "No matches."  # accepted; capped silently


def test_file_target(grep: Tool) -> None:
    result = grep.execute({"pattern": "main", "path": "src/app.py"})
    assert result == "src/app.py:1:def main():\nsrc/app.py:4:main()"


def test_directory_target(grep: Tool) -> None:
    result = grep.execute({"pattern": "hello", "path": "docs"})
    assert result == "No matches."


def test_skips_binary_and_pruned_dirs(grep: Tool) -> None:
    # bin/data.bin and node_modules/pkg/index.js both contain "hello"
    result = grep.execute({"pattern": "hello", "glob": "*.bin"})
    assert result == "No matches."
    result = grep.execute({"pattern": "hello", "glob": "*.js"})
    assert result == "No matches."


def test_invalid_regex(grep: Tool) -> None:
    with pytest.raises(ValueError, match="Invalid regex pattern"):
        grep.execute({"pattern": "("})


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"pattern": ""}, "pattern must be a non-empty string"),
        ({"pattern": None}, "pattern must be a non-empty string"),
        ({"pattern": "x", "glob": 5}, "glob must be a string"),
        ({"pattern": "x", "max_matches": 0}, "max_matches must be a positive"),
    ],
)
def test_bad_args(grep: Tool, args: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        grep.execute(args)


def test_path_escape(grep: Tool) -> None:
    with pytest.raises(WorkspaceError):
        grep.execute({"pattern": "x", "path": "../"})
