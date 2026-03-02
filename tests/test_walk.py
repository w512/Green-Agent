from __future__ import annotations

import os
from pathlib import Path

from agent.walk import is_skipped_rel, walk_files


def rels(root: Path, base: str = "") -> list[str]:
    return [rel for _abs, rel in walk_files(root, base)]


def test_skips_vcs_deps_and_caches(project: Path) -> None:
    found = rels(project)
    assert found == [
        "README.md",
        "bin/data.bin",
        "docs/guide.md",
        "empty.txt",
        "src/app.py",
        "src/util.py",
    ]


def test_absolute_paths_match(project: Path) -> None:
    for abs_path, rel in walk_files(project):
        assert abs_path == project / rel
        assert abs_path.is_file()


def test_relative_dir_prefix(project: Path) -> None:
    assert rels(project / "src", "src") == ["src/app.py", "src/util.py"]


def test_symlinks_are_ignored(project: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "leak.txt").write_text("x")
    os.symlink(outside, project / "linkdir")
    os.symlink(outside / "leak.txt", project / "link.txt")
    found = rels(project)
    assert not any("link" in rel for rel in found)


def test_is_skipped_rel() -> None:
    assert is_skipped_rel("node_modules/pkg/index.js")
    assert is_skipped_rel("src/__pycache__/x.pyc")
    assert is_skipped_rel("a\\.git\\config")
    assert not is_skipped_rel("src/app.py")
