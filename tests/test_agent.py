"""Agent loop with a scripted provider (no network)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from agent.agent import (
    EMPTY_REPLY,
    AgentError,
    AllowAll,
    error_text,
    initial_messages,
    message_text,
    message_to_dict,
    parse_tool_args,
    render_tool_result,
    result_status,
    run_agent,
)
from agent.tools import Tool

# --- fakes ------------------------------------------------------------------


class FakeMessage:
    """Mimics the SDK's pydantic ChatCompletionMessage."""

    def __init__(
        self,
        content: object = None,
        tool_calls: list[Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls
        self.extra = extra or {}

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in self.tool_calls
            ]
        data.update(self.extra)
        if exclude_none:
            data = {k: v for k, v in data.items() if v is not None}
        return data


def tool_call(call_id: str, name: str, arguments: object) -> SimpleNamespace:
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(id=call_id, type="function", function=function)


def reply(message: FakeMessage | None) -> SimpleNamespace:
    choices = [] if message is None else [SimpleNamespace(message=message)]
    return SimpleNamespace(choices=choices)


class FakeProvider:
    def __init__(self, responses: list[Any], model: str = "fake") -> None:
        self.model = model
        self.responses = list(responses)
        self.calls: list[tuple[list[dict], list[dict]]] = []

    def respond(self, messages: list[dict], tools: list[dict]) -> Any:
        self.calls.append(([dict(m) for m in messages], list(tools)))
        return self.responses.pop(0)


class Deny:
    def approve(self, tool: Tool, args: dict) -> bool:
        return False


def make_tool(name: str, execute: Any, needs_approval: bool = False) -> Tool:
    return Tool(
        name=name,
        definition={"type": "function", "function": {"name": name}},
        needs_approval=lambda _args: needs_approval,
        trust=lambda _args: "always",
        describe=lambda _args: name,
        execute=execute,
    )


ECHO = make_tool("echo", lambda args: f"echo:{args.get('text', '')}")
BOOM = make_tool("boom", lambda _args: (_ for _ in ()).throw(OSError("disk")))
DATA = make_tool("data", lambda _args: {"n": 1, "items": ["a"]})
FAKE_REGISTRY = {ECHO.name: ECHO, BOOM.name: BOOM, DATA.name: DATA}


def run(
    provider: FakeProvider,
    registry: dict[str, Tool] | None = None,
    **kwargs: Any,
) -> tuple[Any, list[dict]]:
    events: list[dict] = []
    kwargs.setdefault("permissions", AllowAll())
    kwargs.setdefault("instructions", "SYS")
    result = run_agent(
        "do it",
        provider=provider,
        registry=registry if registry is not None else FAKE_REGISTRY,
        on_event=events.append,
        **kwargs,
    )
    return result, events


# --- helpers ----------------------------------------------------------------


class TestHelpers:
    def test_error_text(self) -> None:
        assert error_text(ValueError("bad")) == "bad"
        assert error_text(ValueError()) == "ValueError"
        assert error_text("text") == "text"
        assert error_text(None) == ""
        assert error_text(42) == "42"

    def test_render_string_and_json(self) -> None:
        assert render_tool_result("plain") == "plain"
        rendered = render_tool_result({"a": 1, "b": ["x"]})
        assert json.loads(rendered) == {"a": 1, "b": ["x"]}
        assert "\n" in rendered  # indented

    def test_render_truncates(self) -> None:
        rendered = render_tool_result("x" * 100, max_chars=10)
        assert rendered == "x" * 10 + "\n...[tool result truncated by harness]"

    def test_message_text_variants(self) -> None:
        assert message_text(FakeMessage("hi")) == "hi"
        assert message_text(FakeMessage(None)) == ""
        parts = [{"type": "text", "text": "a"}, SimpleNamespace(text="b"), "c"]
        assert message_text(FakeMessage(parts)) == "abc"
        assert message_text({"content": "dict"}) == "dict"

    def test_result_status(self) -> None:
        assert result_status("ERROR: x") == "error"
        assert result_status("DENIED: x") == "denied"
        assert result_status("fine") == "ok"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ('{"a": 1}', {"a": 1}),
            ("", {}),
            ("  ", {}),
            (None, {}),
            ({"pre": "parsed"}, {"pre": "parsed"}),
            ("not json", None),
            ("[1, 2]", None),
            ('"str"', None),
            (5, None),
        ],
    )
    def test_parse_tool_args(self, raw: object, expected: object) -> None:
        assert parse_tool_args(raw) == expected

    def test_message_to_dict_drops_none_and_keeps_extras(self) -> None:
        call = tool_call("c1", "echo", {"text": "x"})
        message = FakeMessage(None, [call], extra={"reasoning": "hmm"})
        data = message_to_dict(message)
        assert "content" not in data  # tool_calls present: content optional
        assert data["reasoning"] == "hmm"
        assert data["tool_calls"][0]["function"]["name"] == "echo"

    def test_message_to_dict_empty_reply_gets_content(self) -> None:
        assert message_to_dict(FakeMessage(None)) == {
            "role": "assistant",
            "content": "",
        }

    def test_message_to_dict_rejects_unknown(self) -> None:
        with pytest.raises(TypeError):
            message_to_dict(object())

    def test_initial_messages(self) -> None:
        fresh = initial_messages("t", "SYS", None)
        assert fresh == [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "t"},
        ]
        prior = [{"role": "system", "content": "OLD"}]
        continued = initial_messages("t", "SYS", prior)
        assert continued[0] == {"role": "system", "content": "OLD"}
        assert continued[-1] == {"role": "user", "content": "t"}
        assert initial_messages("t", "SYS", []) == fresh


# --- loop -------------------------------------------------------------------


class TestRunAgent:
    def test_final_answer_without_tools(self) -> None:
        provider = FakeProvider([reply(FakeMessage("done"))])
        result, events = run(provider)
        assert result.text == "done"
        assert [m["role"] for m in result.messages] == [
            "system",
            "user",
            "assistant",
        ]
        assert result.messages[0]["content"] == "SYS"
        assert [e["type"] for e in events] == ["step", "assistant"]
        assert events[0] == {
            "type": "step",
            "step": 1,
            "max_steps": 30,
            "model": "fake",
        }
        # tools definitions are sent to the provider
        assert {t["function"]["name"] for t in provider.calls[0][1]} == {
            "echo",
            "boom",
            "data",
        }

    def test_tool_call_round_trip(self) -> None:
        call = tool_call("c1", "echo", {"text": "hi"})
        provider = FakeProvider(
            [reply(FakeMessage(None, [call])), reply(FakeMessage("final"))]
        )
        result, events = run(provider)
        assert result.text == "final"
        roles = [m["role"] for m in result.messages]
        assert roles == ["system", "user", "assistant", "tool", "assistant"]
        assert result.messages[3] == {
            "role": "tool",
            "tool_call_id": "c1",
            "content": "echo:hi",
        }
        assert [e["type"] for e in events] == [
            "step",
            "tool",
            "result",
            "step",
            "assistant",
        ]
        assert events[1] == {
            "type": "tool",
            "name": "echo",
            "args": {"text": "hi"},
            "args_text": '{"text": "hi"}',
        }
        assert events[2]["status"] == "ok"
        assert events[2]["preview"] == "echo:hi"
        # second request carries the full history so far
        second_messages = provider.calls[1][0]
        assert [m["role"] for m in second_messages] == roles[:-1]

    def test_assistant_text_alongside_tool_calls_is_emitted(self) -> None:
        call = tool_call("c1", "echo", {})
        provider = FakeProvider(
            [
                reply(FakeMessage("thinking...", [call])),
                reply(FakeMessage("ok")),
            ]
        )
        _result, events = run(provider)
        assert events[1] == {"type": "assistant", "text": "thinking..."}

    def test_multiple_calls_in_one_step_run_in_order(self) -> None:
        calls = [
            tool_call("c1", "echo", {"text": "1"}),
            tool_call("c2", "echo", {"text": "2"}),
        ]
        provider = FakeProvider(
            [reply(FakeMessage(None, calls)), reply(FakeMessage("ok"))]
        )
        result, _events = run(provider)
        tool_messages = [m for m in result.messages if m["role"] == "tool"]
        assert [m["tool_call_id"] for m in tool_messages] == ["c1", "c2"]
        assert [m["content"] for m in tool_messages] == ["echo:1", "echo:2"]

    def test_dict_result_is_rendered_as_json(self) -> None:
        call = tool_call("c1", "data", {})
        provider = FakeProvider(
            [reply(FakeMessage(None, [call])), reply(FakeMessage("ok"))]
        )
        result, _events = run(provider)
        assert json.loads(result.messages[3]["content"]) == {
            "n": 1,
            "items": ["a"],
        }

    def test_unknown_tool(self) -> None:
        call = tool_call("c1", "nope", {})
        provider = FakeProvider(
            [reply(FakeMessage(None, [call])), reply(FakeMessage("ok"))]
        )
        result, events = run(provider)
        content = result.messages[3]["content"]
        assert content == "ERROR: Unknown tool requested: nope"
        assert events[2]["status"] == "error"

    def test_invalid_arguments(self) -> None:
        call = tool_call("c1", "echo", "{broken")
        provider = FakeProvider(
            [reply(FakeMessage(None, [call])), reply(FakeMessage("ok"))]
        )
        result, events = run(provider)
        assert result.messages[3]["content"] == "ERROR: Invalid tool arguments."
        assert events[1]["args"] is None

    def test_tool_exception_becomes_error_result(self) -> None:
        call = tool_call("c1", "boom", {})
        provider = FakeProvider(
            [reply(FakeMessage(None, [call])), reply(FakeMessage("ok"))]
        )
        result, events = run(provider)
        assert result.messages[3]["content"] == "ERROR: disk"
        assert events[2]["status"] == "error"

    def test_denied_by_permissions(self) -> None:
        executed: list[dict] = []
        spy = make_tool("spy", lambda args: executed.append(args) or "ran")
        call = tool_call("c1", "spy", {})
        provider = FakeProvider(
            [reply(FakeMessage(None, [call])), reply(FakeMessage("ok"))]
        )
        result, events = run(provider, {"spy": spy}, permissions=Deny())
        content = result.messages[3]["content"]
        assert content == "DENIED: User did not approve this tool call."
        assert events[2]["status"] == "denied"
        assert executed == []

    def test_empty_reply_placeholder(self) -> None:
        provider = FakeProvider([reply(FakeMessage(None))])
        result, events = run(provider)
        assert result.text == EMPTY_REPLY
        assert events[-1] == {"type": "assistant", "text": EMPTY_REPLY}
        assert result.messages[-1] == {"role": "assistant", "content": ""}

    def test_max_steps_exceeded_keeps_history(self) -> None:
        call = tool_call("c1", "echo", {})
        provider = FakeProvider([reply(FakeMessage(None, [call]))] * 3)
        with pytest.raises(AgentError, match="maximum of 2 steps") as info:
            run(provider, max_steps=2)
        assert len(provider.calls) == 2
        roles = [m["role"] for m in info.value.messages]
        assert roles == [
            "system",
            "user",
            "assistant",
            "tool",
            "assistant",
            "tool",
        ]

    def test_no_message_in_response(self) -> None:
        provider = FakeProvider([reply(None)])
        with pytest.raises(AgentError, match="no message"):
            run(provider)

    def test_prior_messages_continue_conversation(self) -> None:
        prior = [
            {"role": "system", "content": "OLD"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "one"},
        ]
        provider = FakeProvider([reply(FakeMessage("two"))])
        result, _events = run(provider, prior_messages=prior)
        assert result.messages[0]["content"] == "OLD"
        assert len(result.messages) == 5
        assert len(prior) == 3  # caller's list is not mutated

    def test_result_truncation(self) -> None:
        big = make_tool("big", lambda _args: "y" * 70_000)
        call = tool_call("c1", "big", {})
        provider = FakeProvider(
            [reply(FakeMessage(None, [call])), reply(FakeMessage("ok"))]
        )
        result, events = run(provider, {"big": big})
        content = result.messages[3]["content"]
        assert content.endswith("...[tool result truncated by harness]")
        assert len(content) < 60_100
        assert len(events[2]["preview"]) == 4_000

    def test_no_on_event_is_fine(self) -> None:
        provider = FakeProvider([reply(FakeMessage("done"))])
        result = run_agent(
            "t", provider=provider, registry={}, permissions=AllowAll()
        )
        assert result.text == "done"
        assert provider.calls[0][1] == []


class TestWithRealTools:
    def test_read_tool_end_to_end(self, registry: dict[str, Tool]) -> None:
        call = tool_call("c1", "read", {"path": "README.md"})
        provider = FakeProvider(
            [reply(FakeMessage(None, [call])), reply(FakeMessage("ok"))]
        )
        result, _events = run(provider, registry)
        assert result.messages[3]["content"] == "1|# Title\n2|hello world"

    def test_workspace_error_is_reported(
        self, registry: dict[str, Tool]
    ) -> None:
        call = tool_call("c1", "read", {"path": "../secret"})
        provider = FakeProvider(
            [reply(FakeMessage(None, [call])), reply(FakeMessage("ok"))]
        )
        result, _events = run(provider, registry)
        content = result.messages[3]["content"]
        assert content == "ERROR: Path escapes workspace: ../secret"
