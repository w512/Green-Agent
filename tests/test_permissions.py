"""Approval policy: answers, trust root, session decisions."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from agent.permissions import Decision, Permissions, parse_answer
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
