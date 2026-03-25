"""Run shell commands and executables for tools; format their output.

Output format is stable so the model can parse it:

    exit_code: <n | timeout>
    stdout:
    ...
    stderr:
    ...

Empty sections are omitted. Long output keeps its head and tail.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
from collections.abc import Sequence
from pathlib import Path

from agent.textfile import truncate_middle

DEFAULT_TIMEOUT = 30.0
MAX_TIMEOUT = 600.0
MAX_OUTPUT_BYTES = 4 * 1024 * 1024

# Keep child processes non-interactive and their output plain.
QUIET_ENV = {
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_PAGER": "cat",
    "PAGER": "cat",
    "TERM": "dumb",
    "NO_COLOR": "1",
    "PYTHONUNBUFFERED": "1",
}


def shell_executable() -> str | None:
    return shutil.which("bash") or shutil.which("sh")


def _decode(data: bytes) -> str:
    return data[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")


def format_result(exit_code: int | str, stdout: str, stderr: str) -> str:
    blocks = [f"exit_code: {exit_code}"]
    if stdout.strip():
        blocks.append(f"stdout:\n{stdout.rstrip()}")
    if stderr.strip():
        blocks.append(f"stderr:\n{stderr.rstrip()}")
    return truncate_middle("\n".join(blocks))


def _kill_tree(process: subprocess.Popen[bytes]) -> None:
    try:
        if hasattr(os, "killpg"):
            os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover - non-POSIX
            process.kill()
    except ProcessLookupError:
        pass


def _run(
    argv: Sequence[str] | str,
    *,
    cwd: Path,
    timeout: float,
    shell: bool,
) -> str:
    env = {**os.environ, **QUIET_ENV}
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            shell=shell,
            executable=shell_executable() if shell else None,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as error:
        return format_result("error", "", str(error))

    try:
        out, err = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(process)
        out, err = process.communicate()
        notice = f"Command exceeded the {timeout:g} second timeout."
        stderr = f"{_decode(err)}\n{notice}".strip()
        return format_result("timeout", _decode(out), stderr)

    return format_result(process.returncode, _decode(out), _decode(err))


def run_command(
    command: str, cwd: Path, timeout: float = DEFAULT_TIMEOUT
) -> str:
    """Run `command` through the shell with `cwd` as working directory."""
    return _run(command, cwd=cwd, timeout=timeout, shell=True)


def run_file(
    argv: Sequence[str], cwd: Path, timeout: float = DEFAULT_TIMEOUT
) -> str:
    """Run an executable with arguments, no shell involved."""
    return _run(list(argv), cwd=cwd, timeout=timeout, shell=False)
