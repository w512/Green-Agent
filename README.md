# Green Agent

A coding agent for your terminal. Point it at a project, describe a task, and
it reads the code, searches it, makes edits, and runs your tests — asking
before anything risky. Works with any OpenAI-compatible API: OpenAI, Gemini,
Ollama (cloud or local), or a self-hosted server.

- Understands the project through `read`, `grep`, and `glob` before touching
  anything; edits with precise, unique-match replacements rather than rewriting
  files.
- Runs shell commands (tests, builds, git, package managers) with a timeout and
  no interactive hangs.
- Keeps you in control: file changes and commands need your approval unless
  git can undo them; destructive commands always ask.
- Never leaves the project directory with file operations — symlinks included.
- Multi-turn chat with history, or one-shot mode for scripts and CI.
- Optional full-screen mode with a project tree and a file viewer that
  follows the agent's edits.
- Model fallback: if the primary model is busy, it retries and switches to the
  next one you configured.
- Single dependency (`openai`), no accounts or telemetry beyond your model
  provider.

## Install

Python 3.11 or newer.

```bash
git clone https://github.com/w512/green-agent.git && cd green-agent
uv sync                 # or: pip install openai
uv sync --extra tui     # optional: full-screen mode (adds Textual)
python start.py         # first run creates config.py and asks you to fill it in
```

Open `config.py` and fill in:

| Setting           | What to put there                                              |
|-------------------|----------------------------------------------------------------|
| `API_KEY`         | Your provider's API key                                        |
| `BASE_URL`        | API root the SDK appends `/chat/completions` to                |
| `MODEL`           | Default model name as the provider spells it                   |
| `FALLBACK_MODELS` | Optional list tried when the primary model returns 429/503     |

Typical `BASE_URL` values:

```python
BASE_URL = "https://api.openai.com/v1"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
BASE_URL = "https://ollama.com/v1"
BASE_URL = "http://localhost:11434/v1"  # local Ollama
```

The model must support tool calling (function calling); most current models
do.

## Use

```bash
python start.py                          # chat about the current directory
python start.py ../my-project            # chat about another project
python start.py -t "run the tests and fix what fails"   # one task, then exit
python start.py -y -t "add type hints to utils.py"      # ...without prompts
python start.py --model gpt-oss:120b     # override MODEL for this session
python start.py --max-steps 60           # allow longer tasks (default 30)
```

Interactive session:

```text
Workspace: /home/me/my-project
Git root:  /home/me/my-project
Tools:     bash, check, delete, edit, fetch, glob, grep, patch, read, todo, write
Model:     gemini-3-flash-preview

> The date parser in utils.py breaks on ISO strings with a timezone. Fix it
  and add a test.
[step 1/30 · gemini-3-flash-preview]
-> grep {"pattern": "def parse_date", "glob": "*.py"}
utils.py:42:def parse_date(text):
-> read {"path": "utils.py", "offset": 35, "limit": 30}
...
-> edit {"path": "utils.py", "old_text": "    return datetime.strptime(...", ...}
Edited utils.py (-1 +3 lines).
-> bash {"command": "python -m pytest -q"}
exit_code: 0
...
Fixed: parse_date now uses datetime.fromisoformat ... 14 passed.
[5 steps · 4 tool calls · 21.3s]
```

Chat commands: `/new` starts a fresh conversation, `/model [name]` shows or
switches the model, `/tools` lists tools, `/help` shows everything. A trailing
backslash continues a task on the next line; pasted multi-line text stays
together. `exit` or Ctrl-D quits; Ctrl-C cancels the current line or the
running task. Input history is kept in `~/.pyagent_history`.

One-shot mode (`-t`) prints the same output and exits with code 0 on success
and 1 when the task failed or ran out of steps.

### Full-screen mode

```bash
python start.py --tui [project-dir]
```

Three panes: the project tree on the left (Enter or click opens a file), a
syntax-highlighted file viewer that reloads when the agent edits the open
file, and the chat with the agent's replies rendered as Markdown. Approvals
appear as a dialog: `y` allow once, `a` allow for the session, `n`/Esc deny.

| Key | Action |
|-----|--------|
| Enter | Send the task (Ctrl+J inserts a newline) |
| Esc | Stop the running task after the current step |
| Ctrl+N | New conversation |
| Ctrl+B / Ctrl+O | Show or hide the tree / the viewer |
| F1 | Help |
| Ctrl+Q | Quit |

Requires the `tui` extra (`uv sync --extra tui`); everything else works
without it.

## What the agent can do

| Tool     | Purpose                                                           | Asks first |
|----------|-------------------------------------------------------------------|------------|
| `read`   | Read a file, or a slice of it, with line numbers                  | no         |
| `glob`   | Find files by pattern (`*`, `**`, `?`); skips `.git`, caches      | no         |
| `grep`   | Regex search across files, optionally filtered by glob            | no         |
| `todo`   | Keep a task list during long, multi-step work                     | no         |
| `check`  | Syntax-check a file, or run your project's check command          | file: no · project: yes |
| `write`  | Create a file or replace one completely                           | yes        |
| `edit`   | Replace one unique fragment in a file                             | yes        |
| `patch`  | Several replacements in one file, all-or-nothing                  | yes        |
| `delete` | Remove a file                                                     | yes        |
| `bash`   | Run a shell command in the project directory                      | yes        |
| `fetch`  | Download a web page or API response as text (HTML is converted)   | always     |

All file paths are relative to the project directory; anything that resolves
outside it (including through symlinks) is refused.

## Approvals

"Asks first" does not mean a prompt on every edit. Inside a **git
repository**, the agent works without interruptions as long as git can undo
what it does:

- writing new files or files tracked by git — allowed;
- overwriting untracked or ignored files (`.env`, `config.py`) — asks;
- shell commands whose paths stay inside the project — allowed;
- commands that touch paths outside, use `~`, `$VARIABLES`, or `$(...)` — ask.

Some commands **always** ask, whatever you answered before: `rm -rf`, `sudo`,
`curl`/`wget`/`ssh`/`scp`, destructive git (`push`, `reset`, `clean`,
`restore`, `checkout --`, `branch -D`, ...), `find -delete`, `chmod -R`,
inline code (`sh -c`, `python -c`), and similar.

When asked, answer `y` to allow once, `a` to allow that tool for the rest of
the session, or press Enter to deny. A denied call is reported to the model,
which is instructed not to retry or work around it.

Outside a git repository every write and command asks. `--yes` turns all
prompts off — use it in disposable environments only.

## Configuration

**Project check command.** `check` without a file runs a command you define,
so the agent can verify its own work the way you would:

```toml
# pyproject.toml
[tool.pyagent]
check = "ruff check . && pytest -q"
```

`package.json` `scripts.check` and a Makefile `check` target are also
recognised.

**Colors** follow the `NO_COLOR` convention and are off when output is not a
terminal.

**System prompt.** `agent/instructions.md` is the agent's standing
instructions. Edit it to change how the agent works — for example, to require a
specific test command or coding conventions for your projects.

**Fallback models.** `FALLBACK_MODELS` are tried in order when the primary
model returns 429 or 503. The agent remembers which model answered and keeps
using it for the session.

## Tips

- Run inside a git repository with a clean working tree: fewer prompts, and
  `git diff` shows exactly what the agent did.
- Define a project check command; the agent uses it to verify changes.
- For long tasks raise `--max-steps`; each model round trip is one step.
- Keep tasks specific. "Fix the failing test in test_api.py" works better than
  "fix the tests".

## Safety

File tools stay inside the project directory. `bash` does not: it can reach
the network and any path the approval heuristics fail to recognise, and a
model can be manipulated by content it reads. The approval layer is a safety
net, not a sandbox. Do not run untrusted tasks on a machine you care about,
and treat `--yes` as "I trust everything this model might do here".

Code you work on is sent to your model provider. Check the provider's data
retention and training policies before using the agent on private code.

## Extending

```text
start.py             entry point: config, arguments, wiring
cli.py               console frontend
tui.py               full-screen frontend (Textual, optional)
agent/agent.py       model/tool loop
agent/llm.py         Chat Completions client: retries, model fallback
agent/tools.py       tool registry
agent/permissions.py approval policy
agent/workspace.py   path containment
agent/command.py     subprocess runner
tools/{name}/        one directory per tool: {name}.json + {name}.py
tests/               pytest suite, runs offline
```

To add a tool, create `tools/<name>/<name>.json` (an OpenAI function
definition) and `tools/<name>/<name>.py` exposing `create_tool(env)`. The
returned object needs `execute(args) -> str | dict` and may define
`needs_approval` (bool or callable), `trust` (`"path"`, `"command"`,
`"always"`, or a callable), and `describe(args) -> str` for the approval
prompt. Use `env.workspace` to resolve paths safely.

Development checks:

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest -q
```

## License

MIT, see [LICENSE](LICENSE).
