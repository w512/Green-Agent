"""One conversation: the message history plus the outcome of each task.

Both frontends (console and Textual) drive the agent through a Session,
so history handling, step/tool counting, and failure reporting live here
instead of being repeated in every UI.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from agent.agent import (
    DEFAULT_MAX_STEPS,
    AgentError,
    Approver,
    Event,
    Message,
    ModelProvider,
    OnEvent,
    run_agent,
)
from agent.tools import Tool


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


@dataclass(frozen=True)
class Outcome:
    """What happened to one task; `error` is None on success."""

    text: str | None
    error: str | None
    steps: int
    tool_calls: int
    seconds: float

    @property
    def ok(self) -> bool:
        return self.error is None

    def summary(self) -> str:
        steps = plural(self.steps, "step")
        tools = plural(self.tool_calls, "tool call")
        return f"[{steps} · {tools} · {self.seconds:.1f}s]"


class Session:
    def __init__(
        self,
        *,
        provider: ModelProvider,
        registry: dict[str, Tool],
        permissions: Approver,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.permissions = permissions
        self.max_steps = max_steps
        self.history: list[Message] | None = None

    def reset(self) -> None:
        """Forget the conversation; the next task starts fresh."""
        self.history = None

    def run(
        self,
        task: str,
        *,
        on_event: OnEvent | None = None,
        stop: Callable[[], bool] | None = None,
    ) -> Outcome:
        """Run one task on top of the history.

        Failures become an Outcome with `error` set. An AgentError keeps
        its (consistent) history so the chat can continue; any other
        exception leaves the history untouched. KeyboardInterrupt is not
        caught: the frontend decides what Ctrl-C means.
        """
        steps = tool_calls = 0

        def observe(event: Event) -> None:
            nonlocal steps, tool_calls
            if event["type"] == "step":
                steps += 1
            elif event["type"] == "tool":
                tool_calls += 1
            if on_event is not None:
                on_event(event)

        started = time.monotonic()

        def outcome(text: str | None, error: str | None) -> Outcome:
            elapsed = time.monotonic() - started
            return Outcome(text, error, steps, tool_calls, elapsed)

        try:
            result = run_agent(
                task,
                provider=self.provider,
                registry=self.registry,
                permissions=self.permissions,
                max_steps=self.max_steps,
                on_event=observe,
                prior_messages=self.history,
                stop=stop,
            )
        except AgentError as failure:
            self.history = failure.messages
            return outcome(None, str(failure))
        except Exception as failure:  # noqa: BLE001 - frontends must survive
            return outcome(None, f"{type(failure).__name__}: {failure}")
        self.history = result.messages
        return outcome(result.text, None)
