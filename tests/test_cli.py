"""Console frontend: rendering, input handling, slash commands, one-shot."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import start
from agent.agent import AllowAll
from agent.permissions import Decision
from agent.tools import Tool
from cli import (
    Chat,
    Palette,
    compact_args,
    console_ask,
    pending_input,
    preview,
    read_task,
    render_event,
    setup_readline,
)

PLAIN = Palette(enabled=False)
COLOR = Palette(enabled=True)


# --- fakes ------------------------------------------------------------------


class FakeMessage:
    def __init__(self, content: object = None, tool_calls: Any = None) -> None:
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {
                        "name": c.function.name,
                        "arguments": c.function.arguments,
                    },
                }
                for c in self.tool_calls
            ]
        return {k: v for k, v in data.items() if v is not None}


def reply(content: object = None, calls: Any = None) -> SimpleNamespace:
    message = FakeMessage(content, calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def call(name: str, arguments: str) -> SimpleNamespace:
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(id="c1", function=function)


class FakeProvider:
    def __init__(self, responses: list[Any]) -> None:
        self.model = "fake"
        self.responses = list(responses)

    def respond(self, messages: list, tools: list) -> Any:
        return self.responses.pop(0)


def make_chat(responses: list[Any], registry: dict[str, Tool] | None = None):
    lines: list[str] = []
    chat = Chat(
        provider=FakeProvider(responses),
        registry=registry or {},
        permissions=AllowAll(),
        max_steps=5,
        palette=PLAIN,
        out=lines.append,
    )
    return chat, lines


# --- rendering --------------------------------------------------------------


class TestPalette:
    def test_disabled_is_plain(self) -> None:
        assert PLAIN.warn("x") == "x"
        assert PLAIN.prompt("> ") == "> "

    def test_enabled_wraps(self) -> None:
        assert COLOR.warn("x") == "\033[1;33mx\033[0m"
        assert COLOR.prompt("> ") == "\001\033[1m\002> \001\033[0m\002"

    def test_no_color_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        assert Palette().enabled is False


class TestCompactArgs:
    def test_short_dict(self) -> None:
        assert compact_args({"path": "a.py"}) == '{"path": "a.py"}'

    def test_long_values_trimmed_individually(self) -> None:
        text = compact_args({"path": "a.py", "content": "x" * 500})
        assert '"path": "a.py"' in text
        assert text.count("x") == 57
        assert text.endswith('..."}')

    def test_non_dict_uses_raw_text(self) -> None:
        assert compact_args(None, "{broken") == "{broken"
        assert compact_args(None, None) == ""

    def test_overall_cap(self) -> None:
        args = {f"k{i}": "v" * 50 for i in range(10)}
        assert len(compact_args(args)) == 200


class TestPreview:
    def test_short(self) -> None:
        assert preview("a\nb") == "a\nb"

    def test_many_lines(self) -> None:
        text = "\n".join(str(i) for i in range(20))
        result = preview(text)
        assert result.startswith("0\n1\n")
        assert result.endswith("\n... (20 lines total)")
        assert result.count("\n") == 8

    def test_long_single_line(self) -> None:
        result = preview("x" * 1000)
        assert result.startswith("x" * 600)
        assert result.endswith("(1 lines total)")

    def test_empty(self) -> None:
        assert preview("") == ""


class TestRenderEvent:
    def test_step(self) -> None:
        event = {"type": "step", "step": 2, "max_steps": 30, "model": "m"}
        assert render_event(event, PLAIN) == "[step 2/30 · m]"

    def test_assistant(self) -> None:
        assert render_event({"type": "assistant", "text": "hi"}, PLAIN) == "hi"

    def test_tool(self) -> None:
        event = {
            "type": "tool",
            "name": "read",
            "args": {"path": "x"},
            "args_text": "",
        }
        assert render_event(event, PLAIN) == '-> read {"path": "x"}'

    @pytest.mark.parametrize(
        ("status", "code"), [("ok", "2"), ("denied", "1;33"), ("error", "1;31")]
    )
    def test_result_colors(self, status: str, code: str) -> None:
        event = {
            "type": "result",
            "name": "x",
            "args": {},
            "status": status,
            "preview": "out",
        }
        assert render_event(event, COLOR) == f"\033[{code}mout\033[0m"

    def test_unknown_type(self) -> None:
        assert render_event({"type": "weird"}, PLAIN) == ""


# --- input ------------------------------------------------------------------


class TestReadTask:
    def feed(self, monkeypatch: pytest.MonkeyPatch, lines: list[Any]) -> None:
        queue = list(lines)

        def fake_input(_prompt: str) -> str:
            item = queue.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item

        monkeypatch.setattr("builtins.input", fake_input)
        monkeypatch.setattr("cli.pending_input", lambda: False)

    def test_single_line(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.feed(monkeypatch, ["  do it  "])
        assert read_task(PLAIN) == "do it"

    def test_backslash_continuation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.feed(monkeypatch, ["first \\", "second \\", "third"])
        assert read_task(PLAIN) == "first \nsecond \nthird"

    def test_paste_joins_pending_lines(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.feed(monkeypatch, ["a", "b", "c"])
        pending = iter([True, True, False])
        monkeypatch.setattr("cli.pending_input", lambda: next(pending))
        assert read_task(PLAIN) == "a\nb\nc"

    def test_eof_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        self.feed(monkeypatch, [EOFError()])
        assert read_task(PLAIN) is None

    def test_ctrl_c_cancels_line(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        self.feed(monkeypatch, [KeyboardInterrupt()])
        assert read_task(PLAIN) == ""
        assert "cancelled" in capsys.readouterr().out


class TestPendingInput:
    def test_non_tty_is_never_pending(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        assert pending_input() is False

    def test_select_failure_is_not_pending(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        def broken(*_a: Any, **_k: Any) -> None:
            raise ValueError("bad fd")

        monkeypatch.setattr("select.select", broken)
        assert pending_input() is False

    def test_ready_is_pending(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("select.select", lambda *_a: ([1], [], []))
        assert pending_input() is True


class TestReadline:
    def test_setup_registers_history(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("readline")
        saved: list[Any] = []
        monkeypatch.setattr("atexit.register", saved.append)
        history = tmp_path / "nested" / "history"
        assert setup_readline(history) is True
        assert len(saved) == 1
        saved[0]()  # the save hook creates the directory and the file
        assert history.exists()


class TestConsoleAsk:
    def test_without_tty_denies(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        assert console_ask(PLAIN)("write x") is Decision.DENY
        assert "--yes" in capsys.readouterr().out

    def test_eof_denies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        def eof(_prompt: str) -> str:
            raise EOFError

        monkeypatch.setattr("builtins.input", eof)
        assert console_ask(PLAIN)("write x") is Decision.DENY

    def test_answers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        prompts: list[str] = []

        def fake_input(prompt: str) -> str:
            prompts.append(prompt)
            return "a"

        monkeypatch.setattr("builtins.input", fake_input)
        assert console_ask(PLAIN)("write x") is Decision.ALWAYS
        assert prompts == ["\nApprove: write x? [y/N/a] "]


# --- chat -------------------------------------------------------------------


class TestChat:
    def test_run_task_success_and_summary(self) -> None:
        chat, lines = make_chat([reply("done")])
        assert chat.run_task("hi") is True
        assert lines[0] == "[step 1/5 · fake]"
        assert lines[1] == "done"
        assert lines[2].startswith("[1 step · 0 tool calls · ")
        assert chat.history is not None
        assert [m["role"] for m in chat.history] == [
            "system",
            "user",
            "assistant",
        ]

    def test_history_continues_and_new_resets(self) -> None:
        chat, _lines = make_chat([reply("one"), reply("two")])
        chat.run_task("a")
        chat.run_task("b")
        assert chat.history is not None and len(chat.history) == 5
        assert chat.handle_command("/new") is True
        assert chat.history is None

    def test_tool_call_counts(self, registry: dict[str, Tool]) -> None:
        first = reply(None, [call("read", '{"path": "README.md"}')])
        chat, lines = make_chat([first, reply("ok")], registry)
        assert chat.run_task("read it") is True
        assert lines[1] == '-> read {"path": "README.md"}'
        assert lines[2] == "1|# Title\n2|hello world"
        assert lines[-1].startswith("[2 steps · 1 tool call · ")

    def test_step_limit_keeps_history(self) -> None:
        looping = reply(None, [call("nope", "{}")])
        chat, lines = make_chat([looping] * 5)
        chat.max_steps = 2
        assert chat.run_task("go") is False
        assert lines[-1] == "\nAgent exceeded the maximum of 2 steps."
        assert chat.history is not None and len(chat.history) > 2

    @pytest.mark.parametrize(
        ("exc", "expected"),
        [
            (RuntimeError("api down"), "\nRuntimeError: api down"),
            (
                KeyboardInterrupt(),
                "\nInterrupted; the last completed turn is kept.",
            ),
        ],
    )
    def test_failures_are_reported(
        self, exc: BaseException, expected: str
    ) -> None:
        class Boom:
            model = "m"

            def respond(self, *_args: Any) -> Any:
                raise exc

        lines: list[str] = []
        chat = Chat(
            provider=Boom(),
            registry={},
            permissions=AllowAll(),
            max_steps=3,
            palette=PLAIN,
            out=lines.append,
        )
        chat.history = [{"role": "system", "content": "kept"}]
        assert chat.run_task("x") is False
        assert lines[-1] == expected
        assert chat.history == [{"role": "system", "content": "kept"}]

    def test_commands(self) -> None:
        chat, lines = make_chat([])
        chat.registry = {"read": None, "grep": None}
        assert chat.handle_command("plain text") is False
        assert chat.handle_command("/help") is True
        assert lines[-1].startswith("Commands:")
        chat.handle_command("/tools")
        assert lines[-1] == "read, grep"
        chat.handle_command("/model")
        assert lines[-1] == "Model: fake"
        chat.handle_command("/model other")
        assert lines[-1] == "Model: other"
        assert chat.provider.model == "other"
        chat.handle_command("/bogus")
        assert lines[-2] == "Unknown command: /bogus"

    def test_loop_exits_on_eof_and_exit_words(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chat, lines = make_chat([reply("answer")])
        inputs = iter(["", "/help", "hello", "exit"])
        monkeypatch.setattr("cli.read_task", lambda _p: next(inputs))
        assert chat.loop() == 0
        assert "answer" in lines
        monkeypatch.setattr("cli.read_task", lambda _p: None)
        assert chat.loop() == 0


# --- start.py ---------------------------------------------------------------


class TestStart:
    def test_parse_defaults(self) -> None:
        args = start.parse_args([])
        assert args.root is None and args.yes is False
        assert args.task is None and args.max_steps == 30

    def test_parse_all(self) -> None:
        args = start.parse_args(
            ["proj", "-y", "-t", "do", "--max-steps", "5", "--model", "m"]
        )
        assert (args.root, args.yes, args.task) == ("proj", True, "do")
        assert (args.max_steps, args.model) == (5, "m")

    def test_bad_max_steps(self) -> None:
        with pytest.raises(SystemExit):
            start.parse_args(["--max-steps", "0"])

    def test_resolve_root_errors(self, tmp_path: Any) -> None:
        with pytest.raises(SystemExit, match="Not found"):
            start.resolve_root(str(tmp_path / "nope"))
        file = tmp_path / "f"
        file.write_text("x")
        with pytest.raises(SystemExit, match="Not a directory"):
            start.resolve_root(str(file))

    def test_ensure_config_copies_template_once(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        template = tmp_path / "config.template.py"
        template.write_text("API_KEY = ''\n")
        config = tmp_path / "config.py"
        monkeypatch.setattr(start, "TEMPLATE_PATH", template)
        monkeypatch.setattr(start, "CONFIG_PATH", config)
        assert start.ensure_config() is True
        assert config.read_text() == "API_KEY = ''\n"
        config.write_text("edited")
        assert start.ensure_config() is False
        assert config.read_text() == "edited"

    @pytest.fixture()
    def wired(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
        """main() with the provider factory replaced by a scripted fake."""
        monkeypatch.setattr(start, "CONFIG_PATH", tmp_path / "config.py")
        monkeypatch.setattr(start, "TEMPLATE_PATH", tmp_path / "template.py")
        (tmp_path / "template.py").write_text("")
        holder: dict[str, Any] = {}

        def fake_create_provider(**_kwargs: Any) -> FakeProvider:
            return holder["provider"]

        import agent.llm

        monkeypatch.setattr(agent.llm, "create_provider", fake_create_provider)
        (tmp_path / "ws").mkdir()
        (tmp_path / "ws" / "note.txt").write_text("hello\n")
        holder["root"] = str(tmp_path / "ws")
        return holder

    def test_one_shot_success(
        self, wired: dict[str, Any], capsys: pytest.CaptureFixture
    ) -> None:
        first = reply(None, [call("read", '{"path": "note.txt"}')])
        wired["provider"] = FakeProvider([first, reply("it says hello")])
        code = start.main(["-y", "-t", "what is in note.txt?", wired["root"]])
        out = capsys.readouterr().out
        assert code == 0
        assert "-> read" in out and "it says hello" in out
        assert "Workspace:" not in out  # no banner in one-shot mode

    def test_one_shot_failure_exit_code(self, wired: dict[str, Any]) -> None:
        looping = reply(None, [call("read", '{"path": "note.txt"}')])
        wired["provider"] = FakeProvider([looping] * 3)
        code = start.main(["-y", "--max-steps", "2", "-t", "x", wired["root"]])
        assert code == 1

    def test_model_override(
        self, wired: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wired["provider"] = FakeProvider([reply("ok")])
        assert start.main(["--model", "other", "-t", "hi", wired["root"]]) == 0
        assert wired["provider"].model == "other"

    def test_interactive_banner_and_exit(
        self,
        wired: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        wired["provider"] = FakeProvider([])
        monkeypatch.setattr("cli.setup_readline", lambda: False)
        monkeypatch.setattr("cli.read_task", lambda _p: None)
        assert start.main([wired["root"]]) == 0
        out = capsys.readouterr().out
        assert "Created config.py" in out  # first run copies the template
        assert "Workspace:" in out and "Tools:" in out
        assert "Not a git repository" in out

    def test_missing_config_is_reported(
        self, wired: dict[str, Any], capsys: pytest.CaptureFixture
    ) -> None:
        import agent.llm

        def fail(**_kwargs: Any) -> None:
            raise agent.llm.ConfigError("API_KEY is not set.")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(agent.llm, "create_provider", fail)
            assert start.main(["-t", "hi", wired["root"]]) == 1
        out = capsys.readouterr().out
        assert "API_KEY is not set." in out
