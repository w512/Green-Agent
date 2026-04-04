"""Textual frontend, driven headlessly with App.run_test()."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("textual")

from textual.widgets import Label, Markdown, Static  # noqa: E402

from agent.permissions import Decision  # noqa: E402
from agent.tools import Tool  # noqa: E402
from agent.workspace import Workspace  # noqa: E402
from tui import (  # noqa: E402
    AgentApp,
    ApprovalScreen,
    ChatLog,
    FileViewer,
    HelpScreen,
    ProjectTree,
    TaskInput,
)

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


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
        self.calls = 0

    def respond(self, messages: list, tools: list) -> Any:
        self.calls += 1
        return self.responses.pop(0)


class BlockingProvider(FakeProvider):
    """Waits on an event before answering, to observe the busy state."""

    def __init__(self, responses: list[Any]) -> None:
        super().__init__(responses)
        self.release = threading.Event()

    def respond(self, messages: list, tools: list) -> Any:
        self.release.wait(timeout=5)
        return super().respond(messages, tools)


def make_app(
    ws: Workspace,
    registry: dict[str, Tool],
    responses: list[Any],
    *,
    auto_approve: bool = True,
    provider: FakeProvider | None = None,
) -> AgentApp:
    return AgentApp(
        provider=provider or FakeProvider(responses),
        registry=registry,
        workspace=ws,
        auto_approve=auto_approve,
        max_steps=5,
    )


async def wait_idle(app: AgentApp, timeout: float = 5.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while app.busy:
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError("agent did not finish")
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.05)  # let the last call_from_thread settle


def plain(widget: Static) -> str:
    content = widget.content
    return content.plain if hasattr(content, "plain") else str(content)


def chat_texts(app: AgentApp) -> list[str]:
    texts: list[str] = []
    for widget in app.query_one(ChatLog).children:
        if isinstance(widget, Markdown):
            texts.append(widget.source)
        elif isinstance(widget, Static):
            texts.append(plain(widget))
    return texts


def viewer_code(app: AgentApp) -> str:
    content = app.query_one("#viewer-body", Static).content
    return str(getattr(content, "code", content))


# --- tests ------------------------------------------------------------------


class TestLayout:
    async def test_startup_banner_and_focus(
        self, ws: Workspace, registry: dict[str, Tool]
    ) -> None:
        app = make_app(ws, registry, [])
        async with app.run_test() as pilot:
            await pilot.pause()
            texts = chat_texts(app)
            assert any("Workspace:" in t for t in texts)
            assert any("auto-approve is on" in t for t in texts)
            assert isinstance(app.focused, TaskInput)
            assert plain(app.query_one("#status", Label)) == "Ready"

    async def test_tree_hides_skipped_dirs(
        self, ws: Workspace, registry: dict[str, Tool]
    ) -> None:
        app = make_app(ws, registry, [])
        async with app.run_test() as pilot:
            await pilot.pause()
            tree = app.query_one(ProjectTree)
            names = {str(node.label) for node in tree.root.children}
            assert "src" in names and "README.md" in names
            assert "node_modules" not in names and ".git" not in names

    async def test_toggle_panes(
        self, ws: Workspace, registry: dict[str, Tool]
    ) -> None:
        app = make_app(ws, registry, [])
        async with app.run_test() as pilot:
            await pilot.press("ctrl+b")
            assert app.query_one(ProjectTree).display is False
            await pilot.press("ctrl+b")
            assert app.query_one(ProjectTree).display is True
            await pilot.press("ctrl+o")
            assert app.query_one(FileViewer).display is False

    async def test_help_screen(
        self, ws: Workspace, registry: dict[str, Tool]
    ) -> None:
        app = make_app(ws, registry, [])
        async with app.run_test() as pilot:
            await pilot.press("f1")
            assert isinstance(app.screen, HelpScreen)
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, HelpScreen)


class TestViewer:
    async def test_show_file_and_refresh(
        self, ws: Workspace, registry: dict[str, Tool], project: Path
    ) -> None:
        app = make_app(ws, registry, [])
        async with app.run_test() as pilot:
            viewer = app.query_one(FileViewer)
            viewer.show(project / "src" / "util.py")
            await pilot.pause()
            assert viewer.border_title == "src/util.py"
            (project / "src" / "util.py").write_text("CHANGED = 1\n")
            viewer.refresh_file()
            assert "CHANGED" in viewer_code(app)

    async def test_binary_and_missing(
        self, ws: Workspace, registry: dict[str, Tool], project: Path
    ) -> None:
        app = make_app(ws, registry, [])
        async with app.run_test() as pilot:
            viewer = app.query_one(FileViewer)
            viewer.show(project / "bin" / "data.bin")
            await pilot.pause()
            assert "binary" in viewer_code(app)
            viewer.show(project / "README.md")
            (project / "README.md").unlink()
            viewer.refresh_file()
            assert viewer.path is None
            assert viewer.border_title == "viewer"


class TestTaskInput:
    async def test_enter_submits_and_clears(
        self, ws: Workspace, registry: dict[str, Tool]
    ) -> None:
        app = make_app(ws, registry, [reply("done")])
        async with app.run_test() as pilot:
            await pilot.press(*"hello")
            await pilot.press("enter")
            await wait_idle(app)
            assert app.query_one(TaskInput).text == ""
            texts = chat_texts(app)
            assert "> hello" in texts
            assert "done" in texts

    async def test_ctrl_j_and_backslash_add_newlines(
        self, ws: Workspace, registry: dict[str, Tool]
    ) -> None:
        app = make_app(ws, registry, [])
        async with app.run_test() as pilot:
            await pilot.press(*"one", "ctrl+j", *"two")
            assert app.query_one(TaskInput).text == "one\ntwo"
            await pilot.press("backslash", "enter")
            assert app.query_one(TaskInput).text == "one\ntwo\n"
            assert app.busy is False

    async def test_empty_enter_does_nothing(
        self, ws: Workspace, registry: dict[str, Tool]
    ) -> None:
        app = make_app(ws, registry, [])
        async with app.run_test() as pilot:
            await pilot.press("enter")
            assert app.busy is False


class TestRunningTasks:
    async def test_tool_round_trip_updates_viewer_and_summary(
        self, ws: Workspace, registry: dict[str, Tool], project: Path
    ) -> None:
        edit = call(
            "edit",
            '{"path": "src/util.py", "old_text": "42", "new_text": "7"}',
        )
        app = make_app(ws, registry, [reply(None, [edit]), reply("ok")])
        async with app.run_test():
            app.query_one(FileViewer).show(project / "src" / "util.py")
            app.submit_task("change it")
            await wait_idle(app)
            texts = chat_texts(app)
            assert any(t.startswith("-> edit") for t in texts)
            assert "Edited src/util.py (-1 +1 lines)." in texts
            assert "ok" in texts
            assert any(t.startswith("[2 steps · 1 tool call") for t in texts)
            assert "VALUE = 7" in viewer_code(app)
            assert app.history is not None and len(app.history) == 5

    async def test_error_and_history_kept(
        self, ws: Workspace, registry: dict[str, Tool]
    ) -> None:
        looping = reply(None, [call("read", '{"path": "README.md"}')])
        app = make_app(ws, registry, [looping] * 6)
        app.max_steps = 2
        async with app.run_test():
            app.submit_task("loop")
            await wait_idle(app)
            texts = chat_texts(app)
            assert "Agent exceeded the maximum of 2 steps." in texts
            assert app.history is not None and len(app.history) > 2

    async def test_busy_rejects_second_task_and_stop(
        self, ws: Workspace, registry: dict[str, Tool]
    ) -> None:
        provider = BlockingProvider(
            [reply(None, [call("read", '{"path": "README.md"}')]), reply("x")]
        )
        app = make_app(ws, registry, [], provider=provider)
        async with app.run_test() as pilot:
            app.submit_task("first")
            await pilot.pause()
            assert app.busy is True
            app.submit_task("second")  # ignored with a notification
            await pilot.press("escape")  # request stop
            provider.release.set()
            await wait_idle(app)
            texts = chat_texts(app)
            assert "Stopped by user." in texts
            assert "> second" not in texts
            assert provider.calls == 1

    async def test_new_conversation(
        self, ws: Workspace, registry: dict[str, Tool]
    ) -> None:
        app = make_app(ws, registry, [reply("one")])
        async with app.run_test() as pilot:
            app.submit_task("a")
            await wait_idle(app)
            assert app.history is not None
            await pilot.press("ctrl+n")
            assert app.history is None
            assert "New conversation." in chat_texts(app)


class TestApprovals:
    async def test_dialog_allow_and_deny(
        self, ws: Workspace, registry: dict[str, Tool], project: Path
    ) -> None:
        write = call("write", '{"path": "new.txt", "content": "hi"}')
        responses = [reply(None, [write]), reply("done")]
        app = make_app(ws, registry, responses, auto_approve=False)
        async with app.run_test() as pilot:
            app.submit_task("write it")
            for _ in range(100):
                await pilot.pause()
                if isinstance(app.screen, ApprovalScreen):
                    break
            assert isinstance(app.screen, ApprovalScreen)
            assert "write new.txt" in app.screen.description
            await pilot.press("y")
            await wait_idle(app)
            assert (project / "new.txt").read_text() == "hi"
            assert "Created new.txt (2 B)." in chat_texts(app)

    async def test_dialog_deny(
        self, ws: Workspace, registry: dict[str, Tool], project: Path
    ) -> None:
        delete = call("delete", '{"path": "README.md"}')
        app = make_app(
            ws,
            registry,
            [reply(None, [delete]), reply("ok")],
            auto_approve=False,
        )
        async with app.run_test() as pilot:
            app.submit_task("delete it")
            for _ in range(100):
                await pilot.pause()
                if isinstance(app.screen, ApprovalScreen):
                    break
            await pilot.press("escape")
            await wait_idle(app)
            assert (project / "README.md").exists()
            assert any("DENIED" in t for t in chat_texts(app))

    async def test_always_remembered(
        self, ws: Workspace, registry: dict[str, Tool], project: Path
    ) -> None:
        first = call("write", '{"path": "a.txt", "content": "1"}')
        second = call("write", '{"path": "b.txt", "content": "2"}')
        responses = [reply(None, [first]), reply(None, [second]), reply("ok")]
        app = make_app(ws, registry, responses, auto_approve=False)
        async with app.run_test() as pilot:
            app.submit_task("write twice")
            for _ in range(100):
                await pilot.pause()
                if isinstance(app.screen, ApprovalScreen):
                    break
            await pilot.press("a")
            await wait_idle(app)
            assert (project / "a.txt").exists() and (project / "b.txt").exists()

    def test_screen_decisions(self) -> None:
        screen = ApprovalScreen("x")
        assert Decision("allow") is Decision.ALLOW
        assert screen.description == "x"
