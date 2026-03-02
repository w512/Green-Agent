from __future__ import annotations

from pathlib import Path

import pytest

from agent.textfile import (
    TextFileError,
    atomic_write_file,
    decode_text,
    format_numbered_lines,
    is_binary,
    read_text_file,
    split_lines,
    truncate_output,
)


class TestDecode:
    def test_plain_text(self) -> None:
        assert decode_text(b"hi\n") == "hi\n"

    def test_nul_is_binary(self) -> None:
        assert is_binary(b"abc\x00def")
        with pytest.raises(TextFileError, match="looks binary"):
            decode_text(b"abc\x00def", "x.bin")

    def test_nul_beyond_probe_is_not_detected(self) -> None:
        data = b"a" * (8 * 1024) + b"\x00"
        assert not is_binary(data)

    def test_invalid_utf8(self) -> None:
        with pytest.raises(TextFileError, match="not valid UTF-8"):
            decode_text(b"\xff\xfe")

    def test_read_text_file(self, tmp_path: Path) -> None:
        target = tmp_path / "a.txt"
        target.write_text("content", encoding="utf-8")
        assert read_text_file(target) == "content"


class TestAtomicWrite:
    def test_creates_and_overwrites(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        atomic_write_file(target, "one")
        assert target.read_text() == "one"
        atomic_write_file(target, "two")
        assert target.read_text() == "two"

    def test_no_temp_files_left(self, tmp_path: Path) -> None:
        atomic_write_file(tmp_path / "out.txt", "x")
        assert [p.name for p in tmp_path.iterdir()] == ["out.txt"]

    def test_cleanup_on_failure(self, tmp_path: Path) -> None:
        target = tmp_path / "dir"
        target.mkdir()
        with pytest.raises(OSError):
            atomic_write_file(target, "x")  # cannot rename a file over a dir
        assert [p.name for p in tmp_path.iterdir()] == ["dir"]


class TestSplitLines:
    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            ("", []),
            ("\n", [""]),
            ("a", ["a"]),
            ("a\n", ["a"]),
            ("a\nb", ["a", "b"]),
            ("a\nb\n", ["a", "b"]),
            ("a\nb\n\n", ["a", "b", ""]),
        ],
    )
    def test_cases(self, content: str, expected: list[str]) -> None:
        assert split_lines(content) == expected


class TestTruncate:
    def test_short_unchanged(self) -> None:
        assert truncate_output("abc", 10) == "abc"

    def test_long_truncated(self) -> None:
        result = truncate_output("x" * 20, 10)
        assert result == "x" * 10 + "\n...[output truncated]"


class TestFormatNumberedLines:
    def test_empty_file(self) -> None:
        assert format_numbered_lines([], 1, 0) == "(empty file)"

    def test_offset_past_end(self) -> None:
        result = format_numbered_lines([], 10, 3)
        assert result == "(no lines at offset 10; file has 3 lines)"

    def test_padding_and_remaining(self) -> None:
        result = format_numbered_lines(["a", "b"], 9, 12)
        assert result == " 9|a\n10|b\n... 2 lines not shown"

    def test_complete_has_no_note(self) -> None:
        assert format_numbered_lines(["a", "b"], 1, 2) == "1|a\n2|b"
