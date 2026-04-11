"""Approval policy for tools that modify the workspace or run commands.

Layers, evaluated in order for a tool with `needs_approval=True`:

1. `auto_approve=True` allows everything (non-interactive runs).
2. Shell commands matching a dangerous pattern (destructive git, rm -rf,
   sudo, network transfer, inline interpreters, ...) always ask, even if
   the user chose "always" for the tool earlier.
3. A tool the user approved with "always" in this session is allowed.
4. Inside the trust root - the workspace, when it lives in a git
   repository - the action is allowed if git can undo it:
   - path tools: the target is new, or is tracked by git;
   - command tools: every path-like token stays inside the trust root
     and the command has no variables or substitutions to hide behind.
5. Otherwise the user is asked.

This is a safety net, not a sandbox: the shell can still reach the
network and any path the heuristics do not recognise.
"""

from __future__ import annotations

import enum
import os
import re
import shlex
from collections.abc import Callable, Iterator
from pathlib import Path

from agent.git import is_tracked
from agent.tools import Args, Tool
from agent.workspace import Workspace, is_inside


class Decision(enum.Enum):
    ALLOW = "allow"
    ALWAYS = "always"
    DENY = "deny"


Ask = Callable[[str], Decision]

ALLOW_ANSWERS = frozenset({"y", "yes"})
ALWAYS_ANSWERS = frozenset({"a", "always"})

# --- shell command analysis ---------------------------------------------------

SEPARATORS = frozenset({";", "|", "||", "&&", "&", "(", ")"})
REDIRECTS = frozenset({"<", ">", ">>", "<<", "<<<", "<>", ">|"})
WRAPPERS = frozenset(
    {"env", "time", "nice", "nohup", "command", "builtin", "exec", "xargs"}
)
# nice -n 5, xargs -I {} / -n 1 / -P 4, env -u VAR / -C dir ...
WRAPPER_FLAGS_WITH_VALUE = frozenset(
    {"-n", "-I", "-P", "-L", "-d", "-E", "-u", "-C", "-s", "-a"}
)
ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*", re.DOTALL)
OPAQUE_RE = re.compile(r"[$`]")

ALWAYS_ASK = frozenset(
    {
        # privilege
        "sudo",
        "su",
        "doas",
        # network transfer / remote execution
        "curl",
        "wget",
        "ssh",
        "scp",
        "sftp",
        "rsync",
        "nc",
        "ncat",
        "telnet",
        "ftp",
        # disks and system state
        "dd",
        "mkfs",
        "fdisk",
        "diskutil",
        "shred",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "kill",
        "pkill",
        "killall",
        "crontab",
        "launchctl",
        "systemctl",
        "chown",
        "eval",
    }
)
GIT_ALWAYS_ASK = frozenset(
    {"push", "reset", "clean", "rebase", "restore", "filter-branch", "gc"}
)
GIT_CHECKOUT_DISCARD = frozenset({"--", ".", "-f", "--force", "-p", "--patch"})
GIT_DELETE_FLAGS = frozenset({"-d", "-D", "--delete"})
SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh", "fish"})
INTERPRETERS = frozenset({"node", "perl", "ruby", "php"})
FIND_ACTIONS = frozenset({"-delete", "-exec", "-execdir", "-ok", "-okdir"})
RM_FLAG_RE = re.compile(r"-[a-zA-Z]*[rRf][a-zA-Z]*")


def tokenize(command: str) -> list[str]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        return list(lexer)
    except ValueError:  # unbalanced quotes: degrade to whitespace split
        return command.split()


def simple_commands(tokens: list[str]) -> Iterator[tuple[str, list[str]]]:
    """Yield (command_name, args) for each simple command in a pipeline."""
    words: list[str] = []
    skip_next = False
    for token in [*tokens, ";"]:
        if skip_next:
            skip_next = False
            continue
        if token in REDIRECTS:
            skip_next = True
            continue
        if token in SEPARATORS:
            if words:
                yield from _unwrap(words)
            words = []
            continue
        words.append(token)


def _unwrap(words: list[str]) -> Iterator[tuple[str, list[str]]]:
    index = 0
    while index < len(words):
        word = words[index]
        if ASSIGNMENT_RE.fullmatch(word):
            index += 1
            continue
        name = os.path.basename(word)
        if name in WRAPPERS:
            index += 1
            while index < len(words) and words[index].startswith("-"):
                takes_value = words[index] in WRAPPER_FLAGS_WITH_VALUE
                index += 2 if takes_value else 1
            continue
        yield name, words[index + 1 :]
        return


GIT_GLOBAL_FLAGS_WITH_VALUE = frozenset(
    {"-C", "-c", "--git-dir", "--work-tree"}
)


def _git_subcommand(args: list[str]) -> tuple[str, list[str]] | None:
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in GIT_GLOBAL_FLAGS_WITH_VALUE:
            index += 2
            continue
        if arg.startswith("-"):
            index += 1
            continue
        return arg, args[index + 1 :]
    return None


def _git_reason(args: list[str]) -> str | None:
    found = _git_subcommand(args)
    if found is None:
        return None
    sub, rest = found
    if sub in GIT_ALWAYS_ASK:
        return f"git {sub}"
    if sub == "checkout" and any(
        arg in GIT_CHECKOUT_DISCARD or "/" in arg for arg in rest
    ):
        return "git checkout may discard changes"
    if sub in {"branch", "tag"} and any(
        arg in GIT_DELETE_FLAGS for arg in rest
    ):
        return f"git {sub} delete"
    if sub == "stash" and any(arg in {"drop", "clear"} for arg in rest):
        return "git stash drop"
    return None


def dangerous_reason(name: str, args: list[str]) -> str | None:
    if name in ALWAYS_ASK:
        return name
    if name == "rm" and any(RM_FLAG_RE.fullmatch(a) for a in args):
        return "rm -r/-f"
    if name == "rm" and ("--recursive" in args or "--force" in args):
        return "rm -r/-f"
    if name == "git":
        return _git_reason(args)
    if name == "chmod" and any(a.startswith("-") and "R" in a for a in args):
        return "chmod -R"
    if name == "find" and any(a in FIND_ACTIONS for a in args):
        return "find -delete/-exec"
    if name in SHELLS and "-c" in args:
        return f"{name} -c"
    inline = name in INTERPRETERS or name.startswith("python")
    if inline and any(a in {"-c", "-e"} for a in args):
        return f"{name} -c"
    return None


def dangerous_command(command: str) -> str | None:
    """Reason a command must always be confirmed, or None."""
    for name, args in simple_commands(tokenize(command)):
        reason = dangerous_reason(name, args)
        if reason:
            return reason
    return None


# --- path containment heuristics ---------------------------------------------


def looks_like_path(token: str) -> bool:
    if not token or "://" in token:
        return False
    if token in {".", ".."} or token[0] in "~/":
        return True
    if token.startswith(("./", "../")):
        return True
    return "/" in token or "\\" in token


def unwrap_assignment(token: str) -> str:
    _key, sep, value = token.partition("=")
    if sep and value and looks_like_path(value):
        return value
    return token


def resolve_against(cwd: Path, token: str) -> Path:
    if token == "~":
        return Path.home()
    if token.startswith(("~/", "~\\")):
        return Path.home() / token[2:]
    return Path(os.path.abspath(cwd / token))


SAFE_PATHS = frozenset({"/dev/null", "/dev/stdout", "/dev/stderr"})


def command_stays_inside(root: Path, cwd: Path, command: str) -> bool:
    """False if any path-like token leaves `root` or paths are opaque."""
    if OPAQUE_RE.search(command):
        return False
    for token in tokenize(command):
        candidate = unwrap_assignment(token)
        if candidate in SAFE_PATHS:
            continue
        if not looks_like_path(token) and not looks_like_path(candidate):
            continue
        if not is_inside(root, resolve_against(cwd, candidate)):
            return False
    return True


# --- interactive answers -----------------------------------------------------


def parse_answer(answer: str) -> Decision:
    normalized = answer.strip().lower()
    if normalized in ALLOW_ANSWERS:
        return Decision.ALLOW
    if normalized in ALWAYS_ANSWERS:
        return Decision.ALWAYS
    return Decision.DENY


# --- policy ------------------------------------------------------------------


class Permissions:
    """`ask` is the frontend's prompt: description -> Decision."""

    def __init__(
        self,
        workspace: Workspace,
        ask: Ask,
        *,
        auto_approve: bool = False,
        trust_root: Path | None | str = "auto",
    ) -> None:
        self.workspace = workspace
        self.auto_approve = auto_approve
        self._ask = ask
        if trust_root == "auto":
            trust_root = workspace.root if workspace.git_root else None
        self.trust_root: Path | None = (
            Path(trust_root) if isinstance(trust_root, (str, Path)) else None
        )
        self._always: set[str] = set()

    def approve(self, tool: Tool, args: Args) -> bool:
        if not tool.needs_approval(args) or self.auto_approve:
            return True

        kind = tool.trust(args)
        reason = None
        if kind == "command":
            command = args.get("command")
            if isinstance(command, str):
                reason = dangerous_command(command)

        if reason is None:
            if tool.name in self._always:
                return True
            if self._recoverable(kind, args):
                return True

        description = tool.describe(args)
        if reason:
            description = f"{description}  [{reason}]"
        decision = self._ask(description)
        if decision is Decision.ALWAYS:
            self._always.add(tool.name)
            return True
        return decision is Decision.ALLOW

    def _recoverable(self, kind: str, args: Args) -> bool:
        root = self.trust_root
        if root is None:
            return False
        if kind == "path":
            return self._path_recoverable(root, args.get("path"))
        if kind == "command":
            command = args.get("command")
            if not isinstance(command, str) or not command.strip():
                return False
            return command_stays_inside(root, self.workspace.root, command)
        return False

    def _path_recoverable(self, root: Path, relative: object) -> bool:
        if not isinstance(relative, str) or not relative:
            return False
        target = Path(os.path.abspath(self.workspace.root / relative))
        if not is_inside(root, target):
            return False
        try:
            target.lstat()
        except FileNotFoundError:
            return True  # nothing to lose yet
        except OSError:
            return False
        git_root = self.workspace.git_root
        return git_root is not None and is_tracked(git_root, target)
