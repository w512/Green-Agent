"""Agent loop: model response -> tool calls -> tool results -> repeat.

    user task
       |
    model response
       |
    tool calls? -- no --> final answer
       | yes
    approve + execute local tools
       |
    append tool results, ask the model again

Events emitted through `on_event` (dicts with a "type" key):

- step:      step, max_steps, model
- assistant: text
- tool:      name, args, args_text
- result:    name, args, status ("ok" | "error" | "denied"), preview
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from agent.tools import Args, Tool, definitions

INSTRUCTIONS_FILE = Path(__file__).with_name("instructions.md")
INSTRUCTIONS = INSTRUCTIONS_FILE.read_text(encoding="utf-8").strip()
DEFAULT_MAX_STEPS = 30
MAX_RESULT_CHARS = 60_000
LOG_RESULT_CHARS = 4_000
EMPTY_REPLY = "(Agent finished without a text response.)"

Message = dict[str, Any]
Event = dict[str, Any]
OnEvent = Callable[[Event], None]


class Approver(Protocol):
    def approve(self, tool: Tool, args: Args) -> bool: ...


class AgentError(Exception):
    """The loop stopped abnormally; `messages` holds the history so far."""

    def __init__(self, message: str, messages: list[Message]) -> None:
        super().__init__(message)
        self.messages = messages


@dataclass
class AgentResult:
    text: str
    messages: list[Message]


def error_text(error: object) -> str:
    if isinstance(error, BaseException):
        return str(error) or type(error).__name__
    if error is None:
        return ""
    return str(error)


def render_tool_result(value: object, max_chars: int = MAX_RESULT_CHARS) -> str:
    if isinstance(value, str):
        serialized = value
    else:
        serialized = json.dumps(
            value, indent=2, ensure_ascii=False, default=str
        )
    if len(serialized) <= max_chars:
        return serialized
    return f"{serialized[:max_chars]}\n...[tool result truncated by harness]"


def _part_text(part: object) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        return str(part.get("text") or "")
    return str(getattr(part, "text", None) or "")


def _field(obj: object, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def message_text(message: object) -> str:
    content = _field(message, "content")
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "".join(_part_text(part) for part in content)
    return ""


def result_status(rendered: str) -> str:
    if rendered.startswith("ERROR:"):
        return "error"
    if rendered.startswith("DENIED:"):
        return "denied"
    return "ok"


def parse_tool_args(args_text: object) -> Args | None:
    """Parse the model's JSON arguments; None when unusable."""
    if isinstance(args_text, dict):
        return args_text
    if args_text is None:
        return {}
    if not isinstance(args_text, str):
        return None
    if not args_text.strip():
        return {}
    try:
        parsed = json.loads(args_text)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def message_to_dict(message: object) -> Message:
    """Convert an SDK message (pydantic) to a plain dict for the history."""
    if isinstance(message, dict):
        result = dict(message)
    else:
        dump = getattr(message, "model_dump", None)
        if not callable(dump):
            raise TypeError(f"Unsupported message object: {type(message)!r}")
        result = dump(exclude_none=True)
    result.setdefault("role", "assistant")
    if not result.get("tool_calls"):
        result.setdefault("content", "")
    return result


@dataclass(frozen=True)
class ToolContext:
    registry: dict[str, Tool]
    permissions: Approver
    emit: Callable[[str, dict[str, Any]], None]


def run_tool_call(call: object, context: ToolContext) -> Message:
    function = _field(call, "function")
    name = _field(function, "name")
    args_text = _field(function, "arguments")
    tool = context.registry.get(name) if isinstance(name, str) else None
    args = parse_tool_args(args_text)

    context.emit("tool", {"name": name, "args": args, "args_text": args_text})

    result: object
    try:
        if tool is None:
            raise ValueError(f"Unknown tool requested: {name}")
        if args is None:
            raise ValueError("Invalid tool arguments.")
        if context.permissions.approve(tool, args):
            result = tool.execute(args)
        else:
            result = "DENIED: User did not approve this tool call."
    except Exception as error:  # tool failures go back to the model
        result = f"ERROR: {error_text(error)}"

    rendered = render_tool_result(result)
    context.emit(
        "result",
        {
            "name": name,
            "args": args,
            "status": result_status(rendered),
            "preview": rendered[:LOG_RESULT_CHARS],
        },
    )
    return {
        "role": "tool",
        "tool_call_id": _field(call, "id"),
        "content": rendered,
    }


def initial_messages(
    task: str,
    instructions: str,
    prior_messages: Sequence[Message] | None,
) -> list[Message]:
    user: Message = {"role": "user", "content": task}
    if prior_messages:
        return [*prior_messages, user]
    return [{"role": "system", "content": instructions}, user]


def run_agent(
    task: str,
    *,
    provider: Any,
    registry: dict[str, Tool],
    permissions: Approver,
    max_steps: int = DEFAULT_MAX_STEPS,
    instructions: str = INSTRUCTIONS,
    on_event: OnEvent | None = None,
    prior_messages: Sequence[Message] | None = None,
    stop: Callable[[], bool] | None = None,
) -> AgentResult:
    """Run the tool loop until the model answers without tool calls.

    `stop` is polled before each step; when it returns True the loop ends
    with AgentError("Stopped by user.") and a history that is consistent
    (every tool call so far has its result), so the chat can continue.
    """

    def emit(kind: str, data: dict[str, Any]) -> None:
        if on_event is not None:
            on_event({"type": kind, **data})

    messages = initial_messages(task, instructions, prior_messages)
    context = ToolContext(registry, permissions, emit)
    tools = definitions(registry)

    for step in range(1, max_steps + 1):
        if stop is not None and stop():
            raise AgentError("Stopped by user.", messages)
        model = provider.model
        emit("step", {"step": step, "max_steps": max_steps, "model": model})

        response = provider.respond(messages, tools)
        choices = _field(response, "choices") or []
        message = _field(choices[0], "message") if choices else None
        if message is None:
            raise AgentError("Model returned no message.", messages)

        messages.append(message_to_dict(message))

        text = message_text(message)
        calls = _field(message, "tool_calls") or []
        if not calls:
            final_text = text or EMPTY_REPLY
            emit("assistant", {"text": final_text})
            return AgentResult(text=final_text, messages=messages)
        if text:
            emit("assistant", {"text": text})

        for call in calls:
            messages.append(run_tool_call(call, context))

    raise AgentError(
        f"Agent exceeded the maximum of {max_steps} steps.", messages
    )
