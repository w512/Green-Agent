"""write, edit, patch, delete tools and the replace_unique helper."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.textfile import replace_unique, truncate_middle
from agent.tools import Tool
from agent.workspace import WorkspaceError


@pytest.fixture()
def write(registry: dict[str, Tool]) -> Tool:
    return registry["write"]


@pytest.fixture()
def edit(registry: dict[str, Tool]) -> Tool:
    return registry["edit"]


@pytest.fixture()
def patch(registry: dict[str, Tool]) -> Tool:
    return registry["patch"]


@pytest.fixture()
def delete(registry: dict[str, Tool]) -> Tool:
    return registry["delete"]


# --- helpers ----------------------------------------------------------------


class TestReplaceUnique:
    def test_single_match(self) -> None:
        assert replace_unique("a b c", "b", "X") == "a X c"

    def test_literal_replacement(self) -> None:
        # no regex/backreference semantics in the replacement
        assert replace_unique("x", "x", r"\1 $& $1") == r"\1 $& $1"

    def test_empty_old_text(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            replace_unique("abc", "", "x")

    def test_not_found_plain(self) -> None:
        with pytest.raises(ValueError, match=r"was not found\.$"):
            replace_unique("abc", "zzz", "x")

    def test_not_found_whitespace_hint(self) -> None:
        with pytest.raises(ValueError, match="leading/trailing whitespace"):
            replace_unique("def f():\n    pass\n", "  pass\n\n", "x")
        with pytest.raises(ValueError, match="indentation"):
            replace_unique("if x:\n\treturn 1\n", "if x:\n    return 1", "x")

    def test_not_found_first_line_hint(self) -> None:
        with pytest.raises(ValueError, match="first line exists"):
            replace_unique("alpha\nbeta\n", "alpha\ngamma", "x")

    def test_multiple_matches(self) -> None:
        with pytest.raises(ValueError, match="occurs 3 times"):
            replace_unique("a a a", "a", "x")

    def test_crlf_file_accepts_lf_fragment(self) -> None:
        content = "one\r\ntwo\r\nthree\r\n"
        result = replace_unique(content, "two\nthree", "2\n3")
        assert result == "one\r\n2\r\n3\r\n"

    def test_where_label(self) -> None:
        with pytest.raises(ValueError, match=r"hunks\[2\]\.old_text was not"):
            replace_unique("abc", "z", "x", where="hunks[2].old_text")


class TestTruncateMiddle:
    def test_short(self) -> None:
        assert truncate_middle("abc", 10) == "abc"

    def test_keeps_head_and_tail(self) -> None:
        text = "H" * 50 + "M" * 100 + "T" * 50
        result = truncate_middle(text, 100)
        assert result.startswith("H" * 50)
        assert result.endswith("T" * 50)
        assert "[100 chars omitted]" in result


# --- write ------------------------------------------------------------------


class TestWrite:
    def test_create_with_parents(self, write: Tool, project: Path) -> None:
        result = write.execute({"path": "a/b/new.txt", "content": "hi\n"})
        assert result == "Created a/b/new.txt (3 B)."
        assert (project / "a" / "b" / "new.txt").read_text() == "hi\n"

    def test_overwrite(self, write: Tool, project: Path) -> None:
        result = write.execute({"path": "README.md", "content": "new"})
        assert result == "Overwrote README.md (3 B)."
        assert (project / "README.md").read_text() == "new"

    def test_identical_content_is_noop(self, write: Tool) -> None:
        content = "# Title\nhello world\n"
        result = write.execute({"path": "README.md", "content": content})
        assert result.startswith("Unchanged README.md")

    def test_directory_target(self, write: Tool) -> None:
        with pytest.raises(ValueError, match="is a directory"):
            write.execute({"path": "src", "content": "x"})

    def test_escape(self, write: Tool) -> None:
        with pytest.raises(WorkspaceError):
            write.execute({"path": "../x.txt", "content": "x"})

    def test_bad_args(self, write: Tool) -> None:
        with pytest.raises(ValueError, match="content must be a string"):
            write.execute({"path": "x.txt", "content": 5})
        with pytest.raises(ValueError, match="path must be a non-empty"):
            write.execute({"path": "", "content": "x"})

    def test_describe(self, write: Tool) -> None:
        text = write.describe({"path": "README.md", "content": "x" * 2048})
        assert text == "write README.md (2.0 KB, overwrite)"
        text = write.describe({"path": "nope.txt", "content": "abc"})
        assert text == "write nope.txt (3 B, new file)"

    def test_unicode_size(self, write: Tool) -> None:
        result = write.execute({"path": "u.txt", "content": "привет"})
        assert result == "Created u.txt (12 B)."


# --- edit -------------------------------------------------------------------


class TestEdit:
    def test_replace(self, edit: Tool, project: Path) -> None:
        result = edit.execute(
            {"path": "src/util.py", "old_text": "42", "new_text": "43"}
        )
        assert result == "Edited src/util.py (-1 +1 lines)."
        assert (project / "src" / "util.py").read_text() == (
            "VALUE = 43\nHELLO = 'x'\n"
        )

    def test_multiline_and_deletion(self, edit: Tool, project: Path) -> None:
        result = edit.execute(
            {
                "path": "src/app.py",
                "old_text": "\n\nmain()\n",
                "new_text": "\n",
            }
        )
        assert result == "Edited src/app.py (-3 +1 lines)."
        assert (project / "src" / "app.py").read_text() == (
            "def main():\n    print('Hello')\n"
        )

    def test_not_found_with_hint(self, edit: Tool) -> None:
        with pytest.raises(ValueError, match="whitespace"):
            edit.execute(
                {
                    "path": "src/app.py",
                    "old_text": "\tprint('Hello')",
                    "new_text": "x",
                }
            )

    def test_ambiguous(self, edit: Tool, project: Path) -> None:
        (project / "dup.txt").write_text("x\nx\n")
        with pytest.raises(ValueError, match="occurs 2 times"):
            edit.execute({"path": "dup.txt", "old_text": "x", "new_text": "y"})

    def test_identical_texts(self, edit: Tool) -> None:
        with pytest.raises(ValueError, match="identical"):
            edit.execute(
                {"path": "README.md", "old_text": "a", "new_text": "a"}
            )

    def test_missing_file(self, edit: Tool) -> None:
        with pytest.raises(OSError):
            edit.execute({"path": "nope.txt", "old_text": "a", "new_text": "b"})

    def test_binary_file(self, edit: Tool) -> None:
        with pytest.raises(ValueError, match="binary"):
            edit.execute(
                {"path": "bin/data.bin", "old_text": "a", "new_text": "b"}
            )

    def test_describe(self, edit: Tool) -> None:
        text = edit.describe(
            {"path": "a.py", "old_text": "1\n2\n3", "new_text": "x"}
        )
        assert text == "edit a.py (-3 +1 lines)"


# --- patch ------------------------------------------------------------------


class TestPatch:
    def test_multiple_hunks(self, patch: Tool, project: Path) -> None:
        result = patch.execute(
            {
                "path": "src/util.py",
                "hunks": [
                    {"old_text": "42", "new_text": "1"},
                    {"old_text": "'x'", "new_text": "'y'"},
                ],
            }
        )
        assert result == "Patched src/util.py (2 hunks)."
        assert (project / "src" / "util.py").read_text() == (
            "VALUE = 1\nHELLO = 'y'\n"
        )

    def test_all_or_nothing(self, patch: Tool, project: Path) -> None:
        before = (project / "src" / "util.py").read_text()
        with pytest.raises(ValueError, match=r"hunks\[1\]\.old_text was not"):
            patch.execute(
                {
                    "path": "src/util.py",
                    "hunks": [
                        {"old_text": "42", "new_text": "1"},
                        {"old_text": "missing", "new_text": "z"},
                    ],
                }
            )
        assert (project / "src" / "util.py").read_text() == before

    def test_later_hunk_sees_earlier_result(
        self, patch: Tool, project: Path
    ) -> None:
        patch.execute(
            {
                "path": "src/util.py",
                "hunks": [
                    {"old_text": "VALUE", "new_text": "COUNT"},
                    {"old_text": "COUNT = 42", "new_text": "COUNT = 0"},
                ],
            }
        )
        assert (project / "src" / "util.py").read_text().startswith("COUNT = 0")

    @pytest.mark.parametrize(
        ("hunks", "message"),
        [
            ([], "non-empty array"),
            ("x", "non-empty array"),
            ([5], r"hunks\[0\] must be an object"),
            (
                [{"old_text": "", "new_text": "x"}],
                r"old_text must be a non-empty",
            ),
            ([{"old_text": "a", "new_text": 1}], r"new_text must be a string"),
        ],
    )
    def test_validation(self, patch: Tool, hunks: object, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            patch.execute({"path": "README.md", "hunks": hunks})

    def test_describe(self, patch: Tool) -> None:
        assert patch.describe({"path": "a", "hunks": [{}, {}]}) == (
            "patch a (2 hunks)"
        )


# --- delete -----------------------------------------------------------------


class TestDelete:
    def test_delete_file(self, delete: Tool, project: Path) -> None:
        assert (
            delete.execute({"path": "docs/guide.md"})
            == "Deleted docs/guide.md."
        )
        assert not (project / "docs" / "guide.md").exists()

    def test_refuses_directory(self, delete: Tool, project: Path) -> None:
        with pytest.raises(ValueError, match="is a directory"):
            delete.execute({"path": "docs"})
        assert (project / "docs").is_dir()

    def test_refuses_symlink(self, delete: Tool, project: Path) -> None:
        os.symlink(project / "README.md", project / "link.md")
        with pytest.raises(ValueError, match="Not a regular file"):
            delete.execute({"path": "link.md"})
        assert (project / "README.md").exists()

    def test_missing(self, delete: Tool) -> None:
        with pytest.raises(OSError):
            delete.execute({"path": "nope.txt"})

    def test_escape(self, delete: Tool) -> None:
        with pytest.raises(WorkspaceError):
            delete.execute({"path": "../x"})
