"""Shell command heuristics: tokenization, danger detection, containment."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.shellcheck import (
    command_stays_inside,
    dangerous_command,
    looks_like_path,
    simple_commands,
    tokenize,
)


class TestTokenize:
    def test_operators_and_quotes(self) -> None:
        tokens = tokenize('rm -rf "a b" && echo x > out; ls | wc')
        assert tokens[:4] == ["rm", "-rf", "a b", "&&"]
        assert tokens[4:9] == ["echo", "x", ">", "out", ";"]
        assert tokens[9:] == ["ls", "|", "wc"]

    def test_unbalanced_quotes_fall_back(self) -> None:
        assert tokenize('echo "oops') == ["echo", '"oops']

    def test_simple_commands_unwrap_and_skip_redirects(self) -> None:
        tokens = tokenize("FOO=1 env BAR=2 nice -n 5 git push; echo hi > out")
        assert list(simple_commands(tokens)) == [
            ("git", ["push"]),
            ("echo", ["hi"]),
        ]

    def test_basename_of_command_path(self) -> None:
        assert list(simple_commands(tokenize("/bin/rm -r x"))) == [
            ("rm", ["-r", "x"])
        ]


class TestDangerous:
    @pytest.mark.parametrize(
        "command",
        [
            "rm -rf build",
            "rm -r build",
            "rm -f x",
            "rm --recursive x",
            "/usr/bin/rm -Rf x",
            "ls; rm -rf x",
            "echo a | xargs rm -rf",
            "env FOO=1 rm -rf x",
            "sudo ls",
            "su root",
            "curl https://example.com",
            "wget x",
            "ssh host",
            "scp a host:b",
            "rsync -a . host:/x",
            "kill -9 1",
            "pkill python",
            "chown -R me .",
            "chmod -R 777 .",
            "dd if=/dev/zero of=x",
            "git push origin main",
            "git push --force",
            "git -C sub push",
            "git reset --hard",
            "git clean -fd",
            "git rebase -i HEAD~3",
            "git restore .",
            "git checkout -- file.txt",
            "git checkout .",
            "git checkout -f main",
            "git checkout feature/x",
            "git branch -D old",
            "git branch --delete old",
            "git tag -d v1",
            "git stash drop",
            "git stash clear",
            "find . -name '*.pyc' -delete",
            "find . -exec rm {} \\;",
            "sh -c 'rm -rf x'",
            "bash -c ls",
            "python -c 'print(1)'",
            "python3.11 -c x",
            "node -e 'x'",
            "perl -e x",
            "eval ls",
        ],
    )
    def test_flagged(self, command: str) -> None:
        assert dangerous_command(command) is not None

    @pytest.mark.parametrize(
        "command",
        [
            "ls -la",
            "rm file.txt",
            "rm -v file.txt",
            "git status",
            "git --no-pager diff",
            "git --version",
            "git -C sub status",
            "git add .",
            "git commit -m x",
            "git checkout -b feature",
            "git checkout main",
            "git branch",
            "git branch new",
            "git stash",
            "git stash pop",
            "git log --oneline",
            "chmod +x run.sh",
            "find . -name '*.py'",
            "python -m pytest",
            "python script.py",
            "node index.js",
            "pip install requests",
            "uv sync",
            "npm test",
            "echo hi > out.txt",
            "cat a | grep b",
            "make build",
        ],
    )
    def test_not_flagged(self, command: str) -> None:
        assert dangerous_command(command) is None

    def test_reason_text(self) -> None:
        assert dangerous_command("rm -rf x") == "rm -r/-f"
        assert dangerous_command("git push") == "git push"
        assert dangerous_command("sudo x") == "sudo"


class TestPathHeuristics:
    @pytest.mark.parametrize(
        "token", [".", "..", "~", "~/x", "/etc", "./a", "../a", "a/b", "a\\b"]
    )
    def test_looks_like_path(self, token: str) -> None:
        assert looks_like_path(token)

    @pytest.mark.parametrize(
        "token", ["", "ls", "-rf", "file.txt", "http://x/y"]
    )
    def test_not_a_path(self, token: str) -> None:
        assert not looks_like_path(token)

    def test_stays_inside(self, tmp_path: Path) -> None:
        root = tmp_path / "ws"
        root.mkdir()
        inside = [
            "ls",
            "cat src/app.py",
            "ls ./sub/../sub",
            "echo hi > out.txt",
            "curl http://example.com/x",
            "python -m pytest tests/",
            "OUT=dist/build make",
            "cat x 2>/dev/null",
            "ls > /dev/null",
        ]
        for command in inside:
            assert command_stays_inside(root, root, command), command

    def test_leaves(self, tmp_path: Path) -> None:
        root = tmp_path / "ws"
        root.mkdir()
        outside = [
            "cat ../secret",
            "cat /etc/passwd",
            "ls ~",
            "ls ~/x",
            "cd .. && ls",
            "echo x > /tmp/out",
            "make --out=/tmp/x",
            "echo $HOME",
            "cat $(find x)",
            "echo `pwd`",
            "ls sub/../..",
        ]
        for command in outside:
            assert not command_stays_inside(root, root, command), command

    def test_subdir_cwd(self, tmp_path: Path) -> None:
        root = tmp_path / "ws"
        (root / "pkg").mkdir(parents=True)
        assert command_stays_inside(root, root / "pkg", "cat ../top.txt")
        assert not command_stays_inside(root, root / "pkg", "cat ../../x")
