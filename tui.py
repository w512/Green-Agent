"""Textual frontend: project tree, file viewer, chat, approval dialogs.

Optional; needs the `tui` extra (`uv sync --extra tui`). The agent runs in
a worker thread. Events and approval requests are marshalled onto the UI
loop with `call_from_thread`, so the core stays synchronous and unaware of
the UI, exactly as with the console frontend.
"""

from __future__ import annotations

import contextlib
import threading
from pathlib import Path
from typing import Any

from rich.syntax import Syntax
from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DirectoryTree,
    Footer,
    Header,
    Label,
    Markdown,
    Static,
    TextArea,
)
from textual.widgets.tree import TreeNode

from agent.agent import DEFAULT_MAX_STEPS, ModelProvider
from agent.permissions import Decision, Permissions
from agent.render import compact_args, preview
from agent.session import Outcome, Session
from agent.textfile import TextFileError, read_text_file
from agent.tools import Tool
from agent.walk import is_skipped_name
from agent.workspace import Workspace

FILE_TOOLS = frozenset({"write", "edit", "patch", "delete"})
MAX_VIEW_CHARS = 200_000

HELP_TEXT = """\
# Green Agent

**Chat**: type a task and press Enter. Ctrl+J inserts a newline; a trailing
backslash also continues on the next line. Pasted text keeps its lines.

**Keys**

| Key | Action |
|-----|--------|
| Enter | Send the task |
| Esc | Stop the running task after the current step |
| Ctrl+N | New conversation (forget the history) |
| Ctrl+B | Show or hide the project tree |
| Ctrl+O | Show or hide the file viewer |
| Tab / Shift+Tab | Move focus between panes |
| F1 | This help |
| Ctrl+Q | Quit |

**Tree**: Enter or click opens a file in the viewer. The viewer reloads when
the agent changes the open file.

**Approvals**: `y` allow once, `a` allow this tool for the session,
`n` or Esc deny.
"""


# --- widgets ----------------------------------------------------------------


class ProjectTree(DirectoryTree):
    """Directory tree without VCS, dependency, and cache folders."""

    def filter_paths(self, paths: Any) -> list[Path]:
        return [
            path for path in sorted(paths) if not is_skipped_name(path.name)
        ]

    def find_node(self, target: Path) -> TreeNode[Any] | None:
        """Expanded node whose directory is `target`, if any."""
        pending: list[TreeNode[Any]] = [self.root]
        while pending:
            node = pending.pop()
            data = node.data
            if data is not None and Path(data.path) == target:
                return node
            pending.extend(node.children)
        return None


class FileViewer(VerticalScroll):
    """Read-only, syntax-highlighted view of one workspace file."""

    def __init__(self, root: Path, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.root = root
        self.path: Path | None = None

    def compose(self) -> ComposeResult:
        yield Static(id="viewer-body", expand=True)

    def on_mount(self) -> None:
        self.show(None)

    def show(self, path: Path | None) -> None:
        self.path = path
        body = self.query_one("#viewer-body", Static)
        if path is None:
            self.border_title = "viewer"
            body.update(Text("Select a file in the tree.", style="dim"))
            return
        try:
            content = read_text_file(path)
        except (OSError, TextFileError) as error:
            self.border_title = path.name
            body.update(Text(str(error), style="red"))
            return
        note = ""
        if len(content) > MAX_VIEW_CHARS:
            content = content[:MAX_VIEW_CHARS]
            note = " (truncated)"
        try:
            relative = str(path.relative_to(self.root))
        except ValueError:
            relative = str(path)
        self.border_title = relative + note
        body.update(
            Syntax(
                content,
                Syntax.guess_lexer(str(path), content),
                theme="ansi_dark",
                line_numbers=True,
                word_wrap=False,
            )
        )
        self.scroll_home(animate=False)

    def refresh_file(self) -> None:
        if self.path is not None:
            if self.path.exists():
                self.show(self.path)
            else:
                self.show(None)


class ChatLog(VerticalScroll):
    """Transcript: user tasks, assistant markdown, tool calls and results."""

    def add(self, widget: Static | Markdown) -> None:
        self.mount(widget)
        self.call_after_refresh(self.scroll_end, animate=False)

    def add_user(self, text: str) -> None:
        self.add(Static(Text(f"> {text}", style="bold"), classes="user"))

    def add_assistant(self, text: str) -> None:
        self.add(Markdown(text, classes="assistant"))

    def add_note(self, text: str, classes: str = "note") -> None:
        self.add(Static(Text(text), classes=classes))

    def add_event(self, event: dict[str, Any]) -> None:
        kind = event["type"]
        if kind == "assistant":
            self.add_assistant(event["text"])
        elif kind == "tool":
            args = compact_args(event["args"], event["args_text"])
            self.add_note(f"-> {event['name']} {args}", "tool")
        elif kind == "result":
            self.add_note(
                preview(event["preview"]), f"result {event['status']}"
            )


class TaskInput(TextArea):
    """Multi-line input where Enter sends and Ctrl+J inserts a newline."""

    BINDINGS = [Binding("ctrl+j", "newline", "Newline", show=False)]

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    async def _on_key(self, event: events.Key) -> None:
        if event.key != "enter":
            await super()._on_key(event)
            return
        event.prevent_default()
        event.stop()
        text = self.text
        if text.rstrip().endswith("\\"):
            self.text = text.rstrip()[:-1] + "\n"
            self.move_cursor(self.document.end)
            return
        if text.strip():
            self.post_message(self.Submitted(text.strip()))
        self.clear()

    def action_newline(self) -> None:
        self.insert("\n")


# --- screens ----------------------------------------------------------------


class ApprovalScreen(ModalScreen[Decision]):
    BINDINGS = [
        Binding("y", "decide('allow')", "Allow"),
        Binding("a", "decide('always')", "Always"),
        Binding("n", "decide('deny')", "Deny"),
        Binding("escape", "decide('deny')", "Deny", show=False),
    ]

    def __init__(self, description: str) -> None:
        super().__init__()
        self.description = description

    def compose(self) -> ComposeResult:
        with Vertical(id="approval"):
            yield Label("Approve this action?", id="approval-title")
            yield Static(Text(self.description), id="approval-body")
            with Horizontal(id="approval-buttons"):
                yield Button("Allow once (y)", id="allow", variant="success")
                yield Button("Always (a)", id="always", variant="warning")
                yield Button("Deny (n)", id="deny", variant="error")

    @on(Button.Pressed)
    def _pressed(self, event: Button.Pressed) -> None:
        self.action_decide(event.button.id or "deny")

    def action_decide(self, choice: str) -> None:
        self.dismiss(Decision(choice))


class HelpScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("f1", "dismiss", "Close", show=False),
    ]

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="help"):
            yield Markdown(HELP_TEXT)
            yield Button("Close (Esc)", id="close")

    @on(Button.Pressed)
    def _close(self) -> None:
        self.dismiss()


# --- application ------------------------------------------------------------


class AgentApp(App[int]):
    TITLE = "Green Agent"
    CSS = """
    #main { height: 1fr; }
    ProjectTree { width: 34; border: solid $panel; }
    #right { width: 1fr; }
    FileViewer { height: 45%; border: solid $panel; border-title-color: $text; }
    ChatLog { height: 1fr; border: solid $panel; padding: 0 1; }
    ChatLog .user { margin-top: 1; }
    ChatLog .assistant { margin: 0; }
    ChatLog .note { color: $text-muted; }
    ChatLog .tool { color: $secondary; }
    ChatLog .result { color: $text-muted; }
    ChatLog .denied { color: $warning; }
    ChatLog .error { color: $error; }
    #status { height: 1; padding: 0 1; color: $text-muted; }
    TaskInput { height: 4; border: tall $accent; }
    ApprovalScreen, HelpScreen { align: center middle; }
    #approval { width: 76; height: auto; padding: 1 2; background: $surface;
                border: thick $warning; }
    #approval-title { text-style: bold; margin-bottom: 1; }
    #approval-body { margin-bottom: 1; }
    #approval-buttons { height: auto; align-horizontal: center; }
    #approval-buttons Button { margin: 0 1; }
    #help { width: 76; height: 80%; padding: 1 2; background: $surface;
            border: thick $accent; }
    #help Button { margin-top: 1; }
    """
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("escape", "stop_task", "Stop"),
        Binding("ctrl+n", "new_conversation", "New chat"),
        Binding("ctrl+b", "toggle_tree", "Tree"),
        Binding("ctrl+o", "toggle_viewer", "Viewer"),
        Binding("f1", "help", "Help"),
    ]

    def __init__(
        self,
        *,
        provider: ModelProvider,
        registry: dict[str, Tool],
        workspace: Workspace,
        auto_approve: bool = False,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        super().__init__()
        self.workspace = workspace
        self.permissions = Permissions(
            workspace, self.ask_approval, auto_approve=auto_approve
        )
        self.session = Session(
            provider=provider,
            registry=registry,
            permissions=self.permissions,
            max_steps=max_steps,
        )
        self.busy = False
        self._stop = threading.Event()
        self._pending_ask: threading.Event | None = None

    # -- layout --

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            yield ProjectTree(self.workspace.root, id="tree")
            with Vertical(id="right"):
                yield FileViewer(self.workspace.root, id="viewer")
                yield ChatLog(id="chat")
        yield Label("Ready", id="status")
        yield TaskInput(
            id="input",
            soft_wrap=True,
            show_line_numbers=False,
            tab_behavior="focus",
        )
        yield Footer()

    def on_mount(self) -> None:
        self._update_subtitle()
        chat = self.query_one(ChatLog)
        git = self.workspace.git_root
        mode = (
            "auto-approve is on"
            if self.permissions.auto_approve
            else "writes and commands ask unless git can undo them"
            if git
            else "not a git repository: every write or command asks"
        )
        chat.add_note(f"Workspace: {self.workspace.root}")
        chat.add_note(f"Tools: {', '.join(self.session.registry)}")
        chat.add_note(f"Approvals: {mode}. F1 for help.")
        self.query_one(TaskInput).focus()

    # -- helpers --

    def _update_subtitle(self) -> None:
        model = self.session.provider.model
        self.sub_title = f"{self.workspace.root.name} · {model}"

    def _set_status(self, text: str) -> None:
        self.query_one("#status", Label).update(text)

    def _reload_tree_dir(self, directory: Path) -> None:
        tree = self.query_one(ProjectTree)
        node = tree.find_node(directory)
        if node is not None:
            tree.reload_node(node)

    def _after_file_change(self, args: dict[str, Any]) -> None:
        relative = args.get("path")
        if not isinstance(relative, str):
            return
        target = (self.workspace.root / relative).resolve()
        viewer = self.query_one(FileViewer)
        if viewer.path is not None and viewer.path.resolve() == target:
            viewer.refresh_file()
        self._reload_tree_dir(target.parent)

    # -- events from the worker thread --

    def _event_from_thread(self, event: dict[str, Any]) -> None:
        with contextlib.suppress(RuntimeError):  # app is shutting down
            self.call_from_thread(self._handle_event, event)

    def _handle_event(self, event: dict[str, Any]) -> None:
        kind = event["type"]
        if kind == "step":
            step = f"step {event['step']}/{event['max_steps']}"
            self._set_status(
                f"{step} · {event['model']} · working… (Esc stops)"
            )
            return
        self.query_one(ChatLog).add_event(event)
        changed_file = (
            kind == "result"
            and event["status"] == "ok"
            and event["name"] in FILE_TOOLS
            and isinstance(event["args"], dict)
        )
        if changed_file:
            self._after_file_change(event["args"])

    def ask_approval(self, description: str) -> Decision:
        """Permissions callback; runs in the worker thread and blocks."""
        answer: list[Decision] = []
        done = threading.Event()
        self._pending_ask = done

        def finish(decision: Decision | None) -> None:
            answer.append(decision or Decision.DENY)
            done.set()

        def open_dialog() -> None:
            self.push_screen(ApprovalScreen(description), callback=finish)

        try:
            self.call_from_thread(open_dialog)
        except RuntimeError:
            return Decision.DENY
        done.wait()
        self._pending_ask = None
        return answer[0] if answer else Decision.DENY

    # -- running tasks --

    @on(TaskInput.Submitted)
    def _submitted(self, message: TaskInput.Submitted) -> None:
        self.submit_task(message.text)

    def submit_task(self, task: str) -> None:
        if self.busy:
            self.notify(
                "The agent is busy. Press Esc to stop it.", severity="warning"
            )
            return
        self.busy = True
        self._stop.clear()
        self.query_one(ChatLog).add_user(task)
        self._set_status("working…")
        self._run_task(task)

    @work(thread=True, exclusive=True, group="agent")
    def _run_task(self, task: str) -> None:
        outcome = self.session.run(
            task, on_event=self._event_from_thread, stop=self._stop.is_set
        )
        with contextlib.suppress(RuntimeError):  # app is shutting down
            self.call_from_thread(self._task_finished, outcome)

    def _task_finished(self, outcome: Outcome) -> None:
        self.busy = False
        self._stop.clear()
        chat = self.query_one(ChatLog)
        if outcome.error is not None:
            chat.add_note(outcome.error, "error")
        chat.add_note(outcome.summary())
        self._set_status("Ready")
        self._update_subtitle()
        self.query_one(TaskInput).focus()

    # -- actions --

    def action_stop_task(self) -> None:
        if not self.busy:
            return
        self._stop.set()
        self._set_status("stopping after the current step…")

    def action_new_conversation(self) -> None:
        if self.busy:
            self.notify("Stop the running task first.", severity="warning")
            return
        self.session.reset()
        self.query_one(ChatLog).add_note("New conversation.")

    def action_toggle_tree(self) -> None:
        tree = self.query_one(ProjectTree)
        tree.display = not tree.display

    def action_toggle_viewer(self) -> None:
        viewer = self.query_one(FileViewer)
        viewer.display = not viewer.display

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    async def action_quit(self) -> None:
        self._stop.set()
        if self._pending_ask is not None:
            self._pending_ask.set()
        self.exit(0)

    @on(DirectoryTree.FileSelected)
    def _file_selected(self, event: DirectoryTree.FileSelected) -> None:
        viewer = self.query_one(FileViewer)
        viewer.display = True
        viewer.show(event.path)


def run_tui(
    *,
    provider: ModelProvider,
    registry: dict[str, Tool],
    workspace: Workspace,
    auto_approve: bool = False,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> int:
    app = AgentApp(
        provider=provider,
        registry=registry,
        workspace=workspace,
        auto_approve=auto_approve,
        max_steps=max_steps,
    )
    result = app.run()
    return result if isinstance(result, int) else 0
