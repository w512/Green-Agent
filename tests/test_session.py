"""Session: history handling, counters, and failure reporting."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent.session import Outcome, Session, plural
from agent.tools import Tool
from tests.conftest import AllowAll


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


def make_session(
    responses: list[Any], registry: dict[str, Tool] | None = None
) -> Session:
    return Session(
        provider=FakeProvider(responses),
        registry=registry or {},
        permissions=AllowAll(),
        max_steps=5,
    )


class TestOutcome:
    def test_plural(self) -> None:
        assert plural(1, "step") == "1 step"
        assert plural(0, "step") == "0 steps"
        assert plural(2, "tool call") == "2 tool calls"

    def test_summary_and_ok(self) -> None:
        outcome = Outcome("done", None, 2, 1, 3.14)
        assert outcome.ok is True
        assert outcome.summary() == "[2 steps · 1 tool call · 3.1s]"
        assert Outcome(None, "boom", 0, 0, 0.0).ok is False


class TestRun:
    def test_success_keeps_history_and_counts(
        self, registry: dict[str, Tool]
    ) -> None:
        first = reply(None, [call("read", '{"path": "README.md"}')])
        session = make_session([first, reply("ok")], registry)
        events: list[str] = []
        outcome = session.run(
            "read it", on_event=lambda e: events.append(e["type"])
        )
        assert outcome.ok and outcome.text == "ok"
        assert (outcome.steps, outcome.tool_calls) == (2, 1)
        assert outcome.seconds >= 0
        assert events == ["step", "tool", "result", "step", "assistant"]
        assert session.history is not None
        assert [m["role"] for m in session.history] == [
            "system",
            "user",
            "assistant",
            "tool",
            "assistant",
        ]

    def test_history_continues_and_reset(self) -> None:
        session = make_session([reply("one"), reply("two")])
        session.run("a")
        session.run("b")
        assert session.history is not None and len(session.history) == 5
        session.reset()
        assert session.history is None

    def test_agent_error_keeps_partial_history(self) -> None:
        looping = reply(None, [call("nope", "{}")])
        session = make_session([looping] * 5)
        session.max_steps = 2
        outcome = session.run("go")
        assert outcome.error == "Agent exceeded the maximum of 2 steps."
        assert outcome.text is None
        assert outcome.steps == 2
        assert session.history is not None and len(session.history) > 2

    def test_other_exception_leaves_history_untouched(self) -> None:
        class Boom:
            model = "m"

            def respond(self, *_args: Any) -> Any:
                raise RuntimeError("api down")

        session = Session(provider=Boom(), registry={}, permissions=AllowAll())
        session.history = [{"role": "system", "content": "kept"}]
        outcome = session.run("x")
        assert outcome.error == "RuntimeError: api down"
        assert session.history == [{"role": "system", "content": "kept"}]

    def test_keyboard_interrupt_propagates(self) -> None:
        class Interrupt:
            model = "m"

            def respond(self, *_args: Any) -> Any:
                raise KeyboardInterrupt

        session = Session(
            provider=Interrupt(), registry={}, permissions=AllowAll()
        )
        with pytest.raises(KeyboardInterrupt):
            session.run("x")

    def test_stop_is_forwarded(self) -> None:
        session = make_session([reply("never")])
        outcome = session.run("x", stop=lambda: True)
        assert outcome.error == "Stopped by user."
        assert outcome.steps == 0
