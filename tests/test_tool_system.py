"""bash, check, fetch, todo tools and the command runner."""

from __future__ import annotations

import io
import json
import shutil
import sys
import urllib.error
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

from agent.command import format_result, run_command, run_file
from agent.tools import Tool

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="no node")


@pytest.fixture()
def bash(registry: dict[str, Tool]) -> Tool:
    return registry["bash"]


@pytest.fixture()
def check(registry: dict[str, Tool]) -> Tool:
    return registry["check"]


@pytest.fixture()
def fetch(registry: dict[str, Tool]) -> Tool:
    return registry["fetch"]


@pytest.fixture()
def todo(registry: dict[str, Tool]) -> Tool:
    return registry["todo"]


# --- command runner ---------------------------------------------------------


class TestCommandRunner:
    def test_format_omits_empty_sections(self) -> None:
        assert format_result(0, "", "") == "exit_code: 0"
        assert format_result(1, "out\n", "") == "exit_code: 1\nstdout:\nout"
        assert format_result(2, "", "err") == "exit_code: 2\nstderr:\nerr"

    def test_run_command_success(self, tmp_path: Path) -> None:
        result = run_command("echo hi; echo err >&2", tmp_path)
        assert result == "exit_code: 0\nstdout:\nhi\nstderr:\nerr"

    def test_run_command_failure(self, tmp_path: Path) -> None:
        assert run_command("exit 3", tmp_path) == "exit_code: 3"

    def test_cwd(self, tmp_path: Path) -> None:
        result = run_command("pwd", tmp_path)
        assert result.endswith(str(tmp_path.resolve()))

    def test_stdin_is_closed(self, tmp_path: Path) -> None:
        result = run_command("cat", tmp_path)  # would hang on a tty
        assert result == "exit_code: 0"

    def test_quiet_env(self, tmp_path: Path) -> None:
        result = run_command("echo $GIT_TERMINAL_PROMPT $PAGER", tmp_path)
        assert result.endswith("0 cat")

    def test_timeout_kills_children(self, tmp_path: Path) -> None:
        result = run_command("sleep 5; echo late", tmp_path, timeout=0.3)
        assert result.startswith("exit_code: timeout")
        assert "0.3 second timeout" in result
        assert "late" not in result

    def test_bash_syntax_available(self, tmp_path: Path) -> None:
        result = run_command("[[ 1 == 1 ]] && echo bashism", tmp_path)
        assert result.endswith("bashism")

    def test_run_file(self, tmp_path: Path) -> None:
        result = run_file([sys.executable, "-c", "print(2+2)"], tmp_path)
        assert result == "exit_code: 0\nstdout:\n4"

    def test_run_file_missing_program(self, tmp_path: Path) -> None:
        result = run_file(["definitely-not-a-program-xyz"], tmp_path)
        assert result.startswith("exit_code: error")

    def test_invalid_utf8_output(self, tmp_path: Path) -> None:
        result = run_command("printf '\\xff\\xfe ok'", tmp_path)
        assert result.endswith("ok")


# --- bash -------------------------------------------------------------------


class TestBash:
    def test_runs_in_workspace(self, bash: Tool, project: Path) -> None:
        result = bash.execute({"command": "ls docs"})
        assert result == "exit_code: 0\nstdout:\nguide.md"

    def test_timeout_argument(self, bash: Tool) -> None:
        result = bash.execute({"command": "sleep 3", "timeout_seconds": 1})
        assert result.startswith("exit_code: timeout")

    def test_timeout_is_capped(self, bash: Tool) -> None:
        # only checks that an oversized value is accepted, not waited for
        result = bash.execute({"command": "true", "timeout_seconds": 10_000})
        assert result == "exit_code: 0"

    def test_bad_args(self, bash: Tool) -> None:
        with pytest.raises(ValueError, match="command must be"):
            bash.execute({"command": ""})
        with pytest.raises(ValueError, match="timeout_seconds must be"):
            bash.execute({"command": "true", "timeout_seconds": 0})

    def test_describe_truncates(self, bash: Tool) -> None:
        text = bash.describe({"command": "echo " + "x" * 300})
        assert text.startswith("bash: echo xxx")
        assert text.endswith("...")
        assert len(text) <= len("bash: ") + 200
        assert bash.describe({"command": "a\nb"}) == "bash: a b"


# --- check ------------------------------------------------------------------


class TestCheckFile:
    def test_python_ok(self, check: Tool) -> None:
        assert check.execute({"path": "src/app.py"}) == "OK: python syntax"

    def test_python_error(self, check: Tool, project: Path) -> None:
        (project / "bad.py").write_text("def f(:\n    pass\n")
        result = check.execute({"path": "bad.py"})
        assert result.startswith("Syntax error in bad.py, line 1:")

    def test_json(self, check: Tool, project: Path) -> None:
        (project / "ok.json").write_text('{"a": 1}')
        (project / "bad.json").write_text("{a}")
        assert check.execute({"path": "ok.json"}) == "OK: valid JSON"
        assert check.execute({"path": "bad.json"}).startswith("Syntax error")

    def test_toml(self, check: Tool, project: Path) -> None:
        (project / "ok.toml").write_text("a = 1\n")
        (project / "bad.toml").write_text("a = \n")
        assert check.execute({"path": "ok.toml"}) == "OK: valid TOML"
        assert check.execute({"path": "bad.toml"}).startswith("Syntax error")

    def test_shell(self, check: Tool, project: Path) -> None:
        (project / "ok.sh").write_text("echo hi\n")
        (project / "bad.sh").write_text("if true; then\n")
        assert check.execute({"path": "ok.sh"}) == "exit_code: 0"
        assert check.execute({"path": "bad.sh"}).startswith("exit_code: 2")

    @needs_node
    def test_javascript(self, check: Tool, project: Path) -> None:
        (project / "ok.js").write_text("const a = 1;\n")
        (project / "bad.js").write_text("const = ;\n")
        assert check.execute({"path": "ok.js"}) == "exit_code: 0"
        assert check.execute({"path": "bad.js"}).startswith("exit_code: 1")

    def test_unsupported_extension(self, check: Tool) -> None:
        with pytest.raises(ValueError, match="No syntax checker for .md"):
            check.execute({"path": "README.md"})

    def test_missing_program(
        self, check: Tool, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (project / "x.js").write_text("1")
        monkeypatch.setattr("shutil.which", lambda _name: None)
        with pytest.raises(ValueError, match="node is not installed"):
            check.execute({"path": "x.js"})

    def test_describe(self, check: Tool) -> None:
        assert check.describe({"path": "a.py"}) == "check a.py"


class TestCheckProject:
    def test_no_command(self, check: Tool) -> None:
        with pytest.raises(ValueError, match="No project check command"):
            check.execute({})
        assert check.describe({}) == "check: (none)"

    def test_pyproject(self, check: Tool, project: Path) -> None:
        (project / "pyproject.toml").write_text(
            '[tool.pyagent]\ncheck = "echo pyproject-check"\n'
        )
        assert check.describe({}) == "check: echo pyproject-check"
        assert check.execute({}) == "exit_code: 0\nstdout:\npyproject-check"

    def test_package_json(self, check: Tool, project: Path) -> None:
        (project / "package.json").write_text(
            json.dumps({"scripts": {"check": "echo x"}})
        )
        assert check.describe({}) == "check: npm run check"

    def test_makefile(self, check: Tool, project: Path) -> None:
        (project / "Makefile").write_text("check:\n\t@echo make-check\n")
        assert check.describe({}) == "check: make check"
        if shutil.which("make"):
            assert check.execute({}) == "exit_code: 0\nstdout:\nmake-check"

    def test_pyproject_wins(self, check: Tool, project: Path) -> None:
        (project / "pyproject.toml").write_text('[tool.pyagent]\ncheck = "a"\n')
        (project / "Makefile").write_text("check:\n\ttrue\n")
        assert check.describe({}) == "check: a"

    def test_makefile_without_target(self, check: Tool, project: Path) -> None:
        (project / "Makefile").write_text("build:\n\ttrue\n")
        assert check.describe({}) == "check: (none)"


# --- fetch ------------------------------------------------------------------


class FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        status: int = 200,
        url: str = "https://example.test/page",
        content_type: str = "text/html; charset=utf-8",
    ) -> None:
        self._body = io.BytesIO(body)
        self.status = status
        self._url = url
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def geturl(self) -> str:
        return self._url

    def getcode(self) -> int:
        return self.status

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        pass


HTML = b"""<html><head><title>Docs</title><style>p{}</style></head>
<body><script>var x = 1;</script><h1>Title</h1>
<p>Some   <b>bold</b> text.</p><ul><li>one</li><li>two</li></ul>
<pre>  keep   spacing</pre></body></html>"""


class TestFetch:
    @pytest.fixture(autouse=True)
    def no_network(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def blocked(*_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("network access in tests")

        monkeypatch.setattr("urllib.request.urlopen", blocked)

    def serve(self, monkeypatch: pytest.MonkeyPatch, response: Any) -> None:
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda *_a, **_k: response
        )

    def test_html_to_text(
        self, fetch: Tool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.serve(monkeypatch, FakeResponse(HTML))
        result = fetch.execute({"url": "https://example.test/page"})
        header, _, body = result.partition("\n\n")
        assert header == (
            "status: 200\nurl: https://example.test/page\n"
            "content-type: text/html"
        )
        assert "var x" not in body and "p{}" not in body
        assert "Docs" in body and "Title" in body
        assert "Some bold text." in body
        assert "- one\n- two" in body
        assert "  keep   spacing" in body

    def test_json_passthrough(
        self, fetch: Tool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.serve(
            monkeypatch,
            FakeResponse(b'{"a": 1}', content_type="application/json"),
        )
        result = fetch.execute({"url": "https://example.test/api"})
        assert result.endswith('\n\n{"a": 1}')

    def test_truncation(
        self, fetch: Tool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.serve(
            monkeypatch, FakeResponse(b"x" * 100, content_type="text/plain")
        )
        result = fetch.execute({"url": "https://e.test/", "max_chars": 10})
        assert result.endswith("x" * 10 + "\n...[body truncated at 10 chars]")

    def test_http_error_body_is_returned(
        self, fetch: Tool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        headers = Message()
        headers["Content-Type"] = "text/plain"
        error = urllib.error.HTTPError(
            "https://e.test/x", 404, "Not Found", headers, io.BytesIO(b"gone")
        )
        self.serve(monkeypatch, error)
        result = fetch.execute({"url": "https://e.test/x"})
        assert result.startswith("status: 404\n")
        assert result.endswith("\n\ngone")

    def test_binary_content_type(
        self, fetch: Tool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.serve(
            monkeypatch, FakeResponse(b"\x89PNG", content_type="image/png")
        )
        with pytest.raises(ValueError, match="Unsupported content type"):
            fetch.execute({"url": "https://e.test/i.png"})

    def test_redirect_to_non_http_rejected(
        self, fetch: Tool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.serve(monkeypatch, FakeResponse(b"x", url="file:///etc/passwd"))
        with pytest.raises(ValueError, match="Only http"):
            fetch.execute({"url": "https://e.test/"})

    def test_connection_failure(
        self, fetch: Tool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail(*_a: Any, **_k: Any) -> None:
            raise urllib.error.URLError("name resolution failed")

        monkeypatch.setattr("urllib.request.urlopen", fail)
        with pytest.raises(ValueError, match="fetch failed: name resolution"):
            fetch.execute({"url": "https://nope.invalid/"})

    def test_timeout(
        self, fetch: Tool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def slow(*_a: Any, **_k: Any) -> None:
            raise TimeoutError

        monkeypatch.setattr("urllib.request.urlopen", slow)
        with pytest.raises(ValueError, match="15 second timeout"):
            fetch.execute({"url": "https://slow.test/"})

    @pytest.mark.parametrize(
        "url", ["ftp://x/y", "file:///etc/passwd", "example.com", "", None]
    )
    def test_bad_urls(self, fetch: Tool, url: object) -> None:
        with pytest.raises(ValueError):
            fetch.execute({"url": url})

    def test_describe(self, fetch: Tool) -> None:
        assert fetch.describe({"url": "https://x"}) == "fetch https://x"


# --- todo -------------------------------------------------------------------


class TestTodo:
    def test_add_and_merge(self, todo: Tool) -> None:
        first = todo.execute(
            {
                "todos": [
                    {"id": "1", "content": "read", "status": "completed"},
                    {"id": "2", "content": "edit", "status": "in_progress"},
                ]
            }
        )
        assert first == "1. [x] 1: read\n2. [>] 2: edit"
        second = todo.execute(
            {
                "todos": [
                    {"id": "2", "content": "edit", "status": "completed"},
                    {"id": "3", "content": "test", "status": "pending"},
                ]
            }
        )
        assert second == "1. [x] 1: read\n2. [x] 2: edit\n3. [ ] 3: test"

    def test_replace(self, todo: Tool) -> None:
        todo.execute(
            {"todos": [{"id": "1", "content": "a", "status": "pending"}]}
        )
        result = todo.execute(
            {
                "todos": [{"id": "9", "content": "b", "status": "cancelled"}],
                "merge": False,
            }
        )
        assert result == "1. [-] 9: b"

    def test_empty_list(self, todo: Tool) -> None:
        assert todo.execute({"todos": [], "merge": False}) == "(no todos)"

    @pytest.mark.parametrize(
        ("todos", "message"),
        [
            ("x", "todos must be an array"),
            ([1], r"todos\[0\] must be an object"),
            ([{"id": "", "content": "a", "status": "pending"}], r"\.id must"),
            ([{"id": "1", "content": 2, "status": "pending"}], r"\.content"),
            ([{"id": "1", "content": "a", "status": "done"}], r"\.status must"),
        ],
    )
    def test_validation(self, todo: Tool, todos: object, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            todo.execute({"todos": todos})

    def test_state_is_per_registry(self, todo: Tool, ws: Any) -> None:
        from agent.tools import Environment, load_tools

        todo.execute(
            {"todos": [{"id": "1", "content": "a", "status": "pending"}]}
        )
        fresh = load_tools(Environment(ws))["todo"]
        assert fresh.execute({"todos": []}) == "(no todos)"
