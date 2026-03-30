"""Approval policy: dangerous commands, trust root, session decisions."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from agent.permissions import (
    Decision,
    Permissions,
    command_stays_inside,
    console_ask,
    dangerous_command,
    looks_like_path,
    parse_answer,
    simple_commands,
    tokenize,
)
from agent.tools import Tool
from agent.workspace import Workspace

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="no git")


# --- fakes ------------------------------------------------------------------


def make_tool(name: str, trust: str, needs_approval: bool = True) -> Tool:
    return Tool(
        name=name,
        definition={"type": "function", "function": {"name": name}},
        needs_approval=lambda _args: needs_approval,
        trust=lambda _args: trust,
        describe=lambda args: f"{name} {args}",
        execute=lambda _args: "ok",
    )


WRITE = make_tool("write", "path")
BASH = make_tool("bash", "command")
FETCH = make_tool("fetch", "always")
READ = make_tool("read", "path", needs_approval=False)


class Asker:
    def __init__(self, *decisions: Decision) -> None:
        self.decisions = list(decisions)
        self.prompts: list[str] = []

    def __call__(self, description: str) -> Decision:
        self.prompts.append(description)
        if not self.decisions:
            raise AssertionError(f"unexpected prompt: {description}")
        return self.decisions.pop(0)


@pytest.fixture()
def plain_ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "plain"
    root.mkdir()
    (root / "file.txt").write_text("x")
    return Workspace(root)


@pytest.fixture()
def git_ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "repo"
    root.mkdir()
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q"], cwd=root, check=True)
    (root / "tracked.txt").write_text("tracked")
    (root / "sub").mkdir()
    (root / "sub" / "deep.txt").write_text("deep")
    subprocess.run([*git, "add", "."], cwd=root, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "init"], cwd=root, check=True)
    (root / "untracked.txt").write_text("secret")
    (root / ".gitignore").write_text("ignored.txt\n")
    (root / "ignored.txt").write_text("secret")
    return Workspace(root)


# --- command analysis -------------------------------------------------------


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


# --- answers ----------------------------------------------------------------


class TestAnswers:
    @pytest.mark.parametrize("text", ["y", "Y", " yes "])
    def test_allow(self, text: str) -> None:
        assert parse_answer(text) is Decision.ALLOW

    @pytest.mark.parametrize("text", ["a", "always", "A"])
    def test_always(self, text: str) -> None:
        assert parse_answer(text) is Decision.ALWAYS

    @pytest.mark.parametrize("text", ["", "n", "no", "maybe", "yep"])
    def test_deny(self, text: str) -> None:
        assert parse_answer(text) is Decision.DENY

    def test_console_ask_without_tty_denies(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        assert console_ask("write x") is Decision.DENY
        assert "denying" in capsys.readouterr().out

    def test_console_ask_reads_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _prompt: "y")
        assert console_ask("write x") is Decision.ALLOW

    def test_console_ask_eof_denies(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        def raise_eof(_prompt: str) -> str:
            raise EOFError

        monkeypatch.setattr("builtins.input", raise_eof)
        assert console_ask("write x") is Decision.DENY


# --- policy -----------------------------------------------------------------


class TestPolicyBasics:
    def test_no_approval_needed(self, plain_ws: Workspace) -> None:
        perms = Permissions(plain_ws, ask=Asker())
        assert perms.approve(READ, {"path": "file.txt"})

    def test_auto_approve(self, plain_ws: Workspace) -> None:
        perms = Permissions(plain_ws, auto_approve=True, ask=Asker())
        assert perms.approve(WRITE, {"path": "../x"})
        assert perms.approve(BASH, {"command": "rm -rf /"})

    def test_outside_git_always_asks(self, plain_ws: Workspace) -> None:
        asker = Asker(Decision.ALLOW, Decision.DENY)
        perms = Permissions(plain_ws, ask=asker)
        assert perms.trust_root is None
        assert perms.approve(WRITE, {"path": "new.txt"})
        assert not perms.approve(BASH, {"command": "ls"})
        assert len(asker.prompts) == 2

    def test_prompt_uses_describe(self, plain_ws: Workspace) -> None:
        asker = Asker(Decision.DENY)
        Permissions(plain_ws, ask=asker).approve(WRITE, {"path": "a"})
        assert asker.prompts == ["write {'path': 'a'}"]

    def test_always_remembers_per_tool(self, plain_ws: Workspace) -> None:
        asker = Asker(Decision.ALWAYS, Decision.DENY)
        perms = Permissions(plain_ws, ask=asker)
        assert perms.approve(WRITE, {"path": "a"})
        assert perms.approve(WRITE, {"path": "b"})  # no prompt
        assert perms.approve(WRITE, {"path": "../c"})  # no prompt
        assert not perms.approve(BASH, {"command": "ls"})  # other tool asks
        assert len(asker.prompts) == 2

    def test_always_does_not_cover_dangerous(self, plain_ws: Workspace) -> None:
        asker = Asker(Decision.ALWAYS, Decision.DENY)
        perms = Permissions(plain_ws, ask=asker)
        assert perms.approve(BASH, {"command": "ls"})
        assert perms.approve(BASH, {"command": "make"})
        assert not perms.approve(BASH, {"command": "git push"})
        assert asker.prompts[-1].endswith("[git push]")

    def test_explicit_trust_root(self, plain_ws: Workspace) -> None:
        perms = Permissions(plain_ws, ask=Asker(), trust_root=plain_ws.root)
        assert perms.approve(BASH, {"command": "ls sub"})
        assert perms.approve(WRITE, {"path": "brand-new.txt"})


@needs_git
class TestPolicyInGit:
    def test_trust_root_is_workspace(self, git_ws: Workspace) -> None:
        perms = Permissions(git_ws, ask=Asker())
        assert perms.trust_root == git_ws.root

    def test_new_file_allowed(self, git_ws: Workspace) -> None:
        perms = Permissions(git_ws, ask=Asker())
        assert perms.approve(WRITE, {"path": "new.txt"})
        assert perms.approve(WRITE, {"path": "sub/nested/new.txt"})

    def test_tracked_file_allowed(self, git_ws: Workspace) -> None:
        perms = Permissions(git_ws, ask=Asker())
        assert perms.approve(WRITE, {"path": "tracked.txt"})
        assert perms.approve(WRITE, {"path": "sub/deep.txt"})

    @pytest.mark.parametrize("path", ["untracked.txt", "ignored.txt"])
    def test_untracked_file_asks(self, git_ws: Workspace, path: str) -> None:
        asker = Asker(Decision.DENY)
        perms = Permissions(git_ws, ask=asker)
        assert not perms.approve(WRITE, {"path": path})
        assert len(asker.prompts) == 1

    def test_path_outside_asks(self, git_ws: Workspace) -> None:
        asker = Asker(Decision.DENY)
        perms = Permissions(git_ws, ask=asker)
        assert not perms.approve(WRITE, {"path": "../elsewhere.txt"})

    def test_missing_path_arg_asks(self, git_ws: Workspace) -> None:
        asker = Asker(Decision.DENY)
        assert not Permissions(git_ws, ask=asker).approve(WRITE, {})

    def test_subdirectory_workspace_trusts_only_itself(
        self, git_ws: Workspace
    ) -> None:
        sub = Workspace(git_ws.root / "sub")
        assert sub.git_root == git_ws.root
        asker = Asker(Decision.DENY)
        perms = Permissions(sub, ask=asker)
        assert perms.trust_root == sub.root
        assert perms.approve(BASH, {"command": "cat deep.txt"})
        assert not perms.approve(BASH, {"command": "cat ../tracked.txt"})

    def test_command_inside_allowed(self, git_ws: Workspace) -> None:
        perms = Permissions(git_ws, ask=Asker())
        assert perms.approve(BASH, {"command": "python -m pytest sub"})
        assert perms.approve(BASH, {"command": "git status"})

    def test_command_outside_asks(self, git_ws: Workspace) -> None:
        asker = Asker(Decision.ALLOW)
        perms = Permissions(git_ws, ask=asker)
        assert perms.approve(BASH, {"command": "cat /etc/hosts"})
        assert len(asker.prompts) == 1

    def test_dangerous_command_asks_with_reason(
        self, git_ws: Workspace
    ) -> None:
        asker = Asker(Decision.DENY)
        perms = Permissions(git_ws, ask=asker)
        assert not perms.approve(BASH, {"command": "rm -rf sub"})
        assert asker.prompts[0].endswith("[rm -r/-f]")

    def test_empty_or_bad_command_asks(self, git_ws: Workspace) -> None:
        asker = Asker(Decision.DENY, Decision.DENY)
        perms = Permissions(git_ws, ask=asker)
        assert not perms.approve(BASH, {"command": ""})
        assert not perms.approve(BASH, {"command": 5})

    def test_always_kind_asks_even_in_git(self, git_ws: Workspace) -> None:
        asker = Asker(Decision.DENY)
        perms = Permissions(git_ws, ask=asker)
        assert not perms.approve(FETCH, {"url": "https://x"})
        assert len(asker.prompts) == 1
