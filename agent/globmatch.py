"""Minimal glob matcher supporting *, **, and ? over posix relative paths.

Unlike fnmatch, `*` and `?` never cross a `/`; only `**` does.
A pattern matches if it matches the whole relative path or the basename.
"""

from __future__ import annotations

import re
from functools import lru_cache

_GLOB_TOKEN = re.compile(r"\*\*/?|\*|\?|[^*?]+")
# "**/" is zero or more whole directory components, so "**/x.py" does not
# match "test_x.py"; a bare "**" matches anything including "/".
_GLOB_ATOM = {"**/": "(?:.*/)?", "**": ".*", "*": "[^/]*", "?": "[^/]"}


def to_posix(value: str) -> str:
    return value.replace("\\", "/")


@lru_cache(maxsize=256)
def _compile(pattern: str) -> re.Pattern[str]:
    tokens = _GLOB_TOKEN.findall(to_posix(pattern))
    atoms = [_GLOB_ATOM.get(token) or re.escape(token) for token in tokens]
    return re.compile("".join(atoms))


def glob_to_regex(pattern: object) -> re.Pattern[str]:
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("glob pattern must be a non-empty string.")
    return _compile(pattern)


def match_glob(relative_path: str, pattern: str) -> bool:
    posix = to_posix(relative_path)
    regex = glob_to_regex(pattern)
    if regex.fullmatch(posix):
        return True
    base = posix.rsplit("/", 1)[-1]
    return regex.fullmatch(base) is not None
