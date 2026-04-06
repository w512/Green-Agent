"""Containment tests: lexical escapes, absolute paths, symlinks."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.workspace import (
    Workspace,
    WorkspaceError,
    is_inside,
    nearest_existing_ancestor,
)


@pytest.fixture()
def ws(tmp_path: Path) -> Workspace:
    (tmp_path / "root").mkdir()
    (tmp_path / "root" / "file.txt").write_text("hello\n")
    (tmp_path / "root" / "sub").mkdir()
    (tmp_path / "root" / "sub" / "nested.txt").write_text("nested\n")
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.txt").write_text("secret\n")
    return Workspace(tmp_path / "root")


class TestIsInside:
    def test_root_itself(self, tmp_path: Path) -> None:
        assert is_inside(tmp_path, tmp_path)

    def test_child(self, tmp_path: Path) -> None:
        assert is_inside(tmp_path, tmp_path / "a" / "b")

    def test_sibling_with_common_prefix(self, tmp_path: Path) -> None:
        root = tmp_path / "ws"
        sibling = tmp_path / "ws2"
        assert not is_inside(root, sibling)

    def test_parent(self, tmp_path: Path) -> None:
        assert not is_inside(tmp_path / "a", tmp_path)


class TestNearestExistingAncestor:
    def test_existing_path(self, tmp_path: Path) -> None:
        assert nearest_existing_ancestor(tmp_path) == tmp_path

    def test_missing_levels(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "c"
        assert nearest_existing_ancestor(target) == tmp_path

    def test_file_in_the_middle(self, tmp_path: Path) -> None:
        blocker = tmp_path / "file.txt"
        blocker.write_text("x")
        target = blocker / "sub" / "deep"
        assert nearest_existing_ancestor(target) == blocker


class TestResolveExistingFile:
    def test_plain_file(self, ws: Workspace) -> None:
        resolved = ws.resolve_existing_file("file.txt")
        assert resolved == ws.root / "file.txt"

    def test_nested_file(self, ws: Workspace) -> None:
        resolved = ws.resolve_existing_file("sub/nested.txt")
        assert resolved == ws.root / "sub" / "nested.txt"

    def test_missing_file(self, ws: Workspace) -> None:
        with pytest.raises(WorkspaceError, match="Not found: missing.txt"):
            ws.resolve_existing_file("missing.txt")

    def test_missing_file_hides_absolute_path(self, ws: Workspace) -> None:
        with pytest.raises(WorkspaceError) as info:
            ws.resolve_existing_file("sub/missing.txt")
        assert str(ws.root) not in str(info.value)

    def test_file_used_as_directory(self, ws: Workspace) -> None:
        with pytest.raises(WorkspaceError, match="Not found: file.txt/x"):
            ws.resolve_existing_file("file.txt/x")

    @pytest.mark.parametrize("bad", ["", None, 42, ["file.txt"]])
    def test_non_string_or_empty(self, ws: Workspace, bad: object) -> None:
        with pytest.raises(WorkspaceError, match="non-empty string"):
            ws.resolve_existing_file(bad)

    def test_absolute_path(self, ws: Workspace) -> None:
        absolute = str(ws.root / "file.txt")
        with pytest.raises(WorkspaceError, match="relative"):
            ws.resolve_existing_file(absolute)

    @pytest.mark.parametrize(
        "escape",
        ["../outside/secret.txt", "sub/../../outside/secret.txt", ".."],
    )
    def test_lexical_escape(self, ws: Workspace, escape: str) -> None:
        with pytest.raises(WorkspaceError, match="escapes workspace"):
            ws.resolve_existing_file(escape)

    def test_symlinked_file_outside(self, ws: Workspace) -> None:
        link = ws.root / "link.txt"
        os.symlink(ws.root.parent / "outside" / "secret.txt", link)
        with pytest.raises(WorkspaceError, match="resolves outside"):
            ws.resolve_existing_file("link.txt")

    def test_symlinked_dir_outside(self, ws: Workspace) -> None:
        link = ws.root / "linkdir"
        os.symlink(ws.root.parent / "outside", link)
        with pytest.raises(WorkspaceError, match="resolves outside"):
            ws.resolve_existing_file("linkdir/secret.txt")

    def test_symlink_inside_is_allowed(self, ws: Workspace) -> None:
        link = ws.root / "alias.txt"
        os.symlink(ws.root / "file.txt", link)
        assert ws.resolve_existing_file("alias.txt") == link


class TestResolveWritableFile:
    def test_new_file(self, ws: Workspace) -> None:
        resolved = ws.resolve_writable_file("new.txt")
        assert resolved == ws.root / "new.txt"

    def test_new_nested_dirs(self, ws: Workspace) -> None:
        resolved = ws.resolve_writable_file("a/b/c.txt")
        assert resolved == ws.root / "a" / "b" / "c.txt"

    def test_lexical_escape(self, ws: Workspace) -> None:
        with pytest.raises(WorkspaceError, match="escapes workspace"):
            ws.resolve_writable_file("../evil.txt")

    def test_new_file_under_outside_symlink(self, ws: Workspace) -> None:
        link = ws.root / "linkdir"
        os.symlink(ws.root.parent / "outside", link)
        with pytest.raises(WorkspaceError, match="resolves outside"):
            ws.resolve_writable_file("linkdir/new.txt")

    def test_missing_ancestors_inside(self, ws: Workspace) -> None:
        resolved = ws.resolve_writable_file("sub/deep/new.txt")
        assert resolved == ws.root / "sub" / "deep" / "new.txt"


class TestResolveExistingPath:
    def test_default_is_root_dir(self, ws: Workspace) -> None:
        info = ws.resolve_existing_path()
        assert info.path == ws.root
        assert info.is_directory and not info.is_file

    def test_empty_string_is_root(self, ws: Workspace) -> None:
        info = ws.resolve_existing_path("")
        assert info.path == ws.root

    def test_file(self, ws: Workspace) -> None:
        info = ws.resolve_existing_path("file.txt")
        assert info.is_file and not info.is_directory

    def test_directory(self, ws: Workspace) -> None:
        info = ws.resolve_existing_path("sub")
        assert info.is_directory and not info.is_file


class TestRelative:
    def test_root_is_empty(self, ws: Workspace) -> None:
        assert ws.relative(ws.root) == ""

    def test_nested_is_posix(self, ws: Workspace) -> None:
        assert ws.relative(ws.root / "sub" / "nested.txt") == "sub/nested.txt"


class TestSymlinkedRoot:
    def test_files_resolve_through_symlinked_root(self, tmp_path: Path) -> None:
        actual = tmp_path / "actual"
        actual.mkdir()
        (actual / "file.txt").write_text("x")
        link_root = tmp_path / "link-root"
        os.symlink(actual, link_root)
        ws = Workspace(link_root)
        assert ws.root == link_root
        assert ws.real_root == Path(os.path.realpath(actual))
        resolved = ws.resolve_existing_file("file.txt")
        assert resolved == link_root / "file.txt"
