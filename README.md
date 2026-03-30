# Green Agent

Small, modular coding-agent harness in Python. It shows the whole mechanism
behind agentic coding tools — the model/tool loop, a tool registry, workspace
containment, and an approval policy — in under three thousand lines, without a
framework in the way.

## Run

Python 3.11+ and an API key for any OpenAI-compatible Chat Completions
provider.

```bash
uv sync                 # or: pip install openai
uv run python start.py  # or: python start.py
```

The first run copies `config.template.py` to `config.py`. Fill in `API_KEY`,
`BASE_URL`, and `MODEL`; `FALLBACK_MODELS` is optional. `BASE_URL` must be
the API root the SDK appends `/chat/completions` to, e.g.
`https://ollama.com/v1` or `https://api.openai.com/v1`.

```bash
python start.py                          # chat about the current directory
python start.py ../my-project            # chat about another project
python start.py -y -t "run the tests"    # one task, auto-approved, exit 0/1
python start.py --model gpt-oss:120b     # override MODEL for this session
python start.py --help
```

In the chat: `/new` forgets the conversation, `/model [name]` shows or
switches the model, `/tools` lists tools, `/help` explains input. A trailing
backslash continues a task on the next line; pasted blocks stay together.
`exit` or Ctrl-D quits; Ctrl-C cancels the current line or the running task.

## Loop

```text
user task
   |
model response
   |
tool calls? -- no --> final answer
   | yes
approve + execute local tools
   |
append tool results, ask the model again   (up to --max-steps, default 30)
```

Everything the model sees comes from `agent/instructions.md` (system prompt),
the tool definitions in `tools/*/*.json`, and the tool results.

## Tools

| Tool     | Purpose                                                          | Approval |
|----------|------------------------------------------------------------------|----------|
| `read`   | Numbered slice of a text file (`offset`, `limit`)                 | no       |
| `glob`   | Find files by pattern (`*`, `**`, `?`); skips `.git`, caches      | no       |
| `grep`   | Regex search across files, with optional glob filter              | no       |
| `todo`   | In-session task list                                              | no       |
| `write`  | Create or overwrite a whole file                                  | yes      |
| `edit`   | Replace one unique fragment; hints when the match is off          | yes      |
| `patch`  | Several unique replacements in one file, all-or-nothing           | yes      |
| `delete` | Remove one regular file                                           | yes      |
| `bash`   | Shell command in the workspace; timeout, pagers disabled          | yes      |
| `check`  | Syntax-check a file (free) or run the project's check command     | partly   |
| `fetch`  | HTTP GET; HTML is converted to text                               | always   |

File tools accept only paths relative to the workspace root and refuse
anything that resolves outside it, symlinks included.

## Approval

Tools that change files or run commands go through `agent/permissions.py`.
Layers, in order:

1. `--yes` allows everything (for scripts and CI).
2. Dangerous shell commands always ask: `rm -rf`, `sudo`, `curl`/`ssh`/`scp`,
   destructive git (`push`, `reset`, `clean`, `checkout --`, ...),
   `find -delete`, inline interpreters (`sh -c`, `python -c`), and more.
3. A tool you answered `a` (always) for is allowed for the rest of the
   session — except for dangerous commands.
4. Inside a git repository, an action is allowed without asking when git can
   undo it: writes to new or tracked files, and commands whose paths stay
   inside the workspace. Untracked or ignored files (`.env`, `config.py`)
   still ask.
5. Otherwise you are asked: `y` once, `a` for the session, anything else
   denies. Outside git every write and command asks.

## Layout

```text
start.py            entry point: config, arguments, wiring
cli.py              console frontend: rendering, input, approvals
agent/agent.py      the loop
agent/llm.py        Chat Completions client: retries, model fallback
agent/tools.py      registry: loads tools/{name}/{name}.json + .py
agent/permissions.py approval policy
agent/workspace.py  path containment
agent/command.py    subprocess runner used by bash and check
tools/{name}/       one directory per tool
tests/              pytest suite, no network
```

### Adding a tool

Create `tools/<name>/<name>.json` (an OpenAI function definition) and
`tools/<name>/<name>.py` exposing `create_tool(env)`. The returned object
needs `execute(args) -> str | dict` and may define `needs_approval`
(bool or callable), `trust` (`"path"`, `"command"`, `"always"`, or a
callable), and `describe(args) -> str` for the approval prompt. `env.workspace`
resolves paths safely; helpers live in `agent/textfile.py`, `agent/walk.py`,
`agent/globmatch.py`, `agent/params.py`.

## Development

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest -q
```

The same sequence is wired as this repository's `check` command
(`[tool.pyagent]` in `pyproject.toml`), so the agent can verify its own
changes with the `check` tool.

## Safety

File tools stay inside the workspace. `bash` does not: it can reach the
network and any path the approval heuristics fail to recognise, and a model
can be talked into running things. The approval layer is a safety net, not a
sandbox. Do not point this at an untrusted task on a machine you care about,
and keep `--yes` for disposable environments.
