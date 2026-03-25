from __future__ import annotations

import pytest

from agent.globmatch import glob_to_regex, match_glob


@pytest.mark.parametrize(
    ("path", "pattern"),
    [
        ("app.py", "*.py"),
        ("src/app.py", "*.py"),  # basename fallback
        ("src/app.py", "src/*.py"),
        ("src/app.py", "**/*.py"),
        ("app.py", "**/*.py"),
        ("a/b/c/d.py", "a/**/d.py"),
        ("a/d.py", "a/**/d.py"),
        ("src/app.py", "src/app.py"),
        ("x1.txt", "x?.txt"),
        ("a+b.py", "a+b.py"),
        ("dir/sub/file.txt", "dir/**"),
        (r"src\app.py", "src/*.py"),  # backslashes normalized
    ],
)
def test_matches(path: str, pattern: str) -> None:
    assert match_glob(path, pattern)


@pytest.mark.parametrize(
    ("path", "pattern"),
    [
        ("src/app.py", "*.js"),
        ("src/app.py", "lib/*.py"),
        ("a/b/c.py", "a/*.py"),  # * does not cross /
        ("a/b.txt", "a?b.txt"),  # ? does not cross /
        ("aab.py", "a+b.py"),  # regex chars are literal
        ("app.pyc", "*.py"),
        ("src/app.py", "app"),
        ("test_calc.py", "**/calc.py"),  # ** matches directories only
        ("src/test_calc.py", "**/calc.py"),
    ],
)
def test_non_matches(path: str, pattern: str) -> None:
    assert not match_glob(path, pattern)


@pytest.mark.parametrize("bad", ["", None, 5])
def test_invalid_pattern(bad: object) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        glob_to_regex(bad)
