"""UTF-8 text file helpers shared by tools."""

from __future__ import annotations

import os
import time
from pathlib import Path

BINARY_PROBE = 8 * 1024
MAX_OUTPUT_CHARS = 50_000


class TextFileError(ValueError):
    """File content is binary or not valid UTF-8."""


def is_binary(data: bytes) -> bool:
    return b"\x00" in data[:BINARY_PROBE]


def decode_text(data: bytes, label: str = "content") -> str:
    if is_binary(data):
        raise TextFileError(f"{label} looks binary (NUL in the first 8 KiB).")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise TextFileError(f"{label} is not valid UTF-8.") from error


def read_text_file(file_path: Path) -> str:
    return decode_text(file_path.read_bytes(), str(file_path))


def atomic_write_file(file_path: Path, content: str) -> None:
    """Write via a temp file in the same directory, then rename over."""
    stamp = f"{os.getpid()}.{time.time_ns()}"
    tmp = file_path.parent / f".{file_path.name}.{stamp}.tmp"
    try:
        tmp.write_bytes(content.encode("utf-8"))
        os.replace(tmp, file_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def split_lines(content: str) -> list[str]:
    """Split on newline; a trailing newline does not add an empty line."""
    if not content:
        return []
    lines = content.split("\n")
    if lines[-1] == "":
        lines.pop()
    return lines


def truncate_output(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n...[output truncated]"


def format_numbered_lines(
    lines: list[str], start_line: int, total_lines: int
) -> str:
    if total_lines == 0:
        return "(empty file)"
    if not lines:
        note = f"file has {total_lines} lines"
        return f"(no lines at offset {start_line}; {note})"
    end_line = start_line + len(lines) - 1
    width = len(str(total_lines))
    body = [
        f"{start_line + index:>{width}}|{line}"
        for index, line in enumerate(lines)
    ]
    remaining = total_lines - end_line
    if remaining > 0:
        body.append(f"... {remaining} lines not shown")
    return "\n".join(body)
