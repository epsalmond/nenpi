# nenpi

nenpi (燃費, fuel economy) reads Claude Code and Codex CLI transcripts on disk
and reports which sessions drained how much **subscription-plan quota** — not
API dollars. Codex quota is measured directly from the `rate_limits` snapshots
the CLI writes into its own rollouts; Claude writes no quota data, so Claude
sessions are modelled at API list price and, when live utilisation has been
sampled, reported against the actual plan window.

Python 3.11 or newer is required. The base reports have no runtime dependency;
the optional Textual extras add the terminal UI and browser server.

## Install

For the complete CLI, benchmark, terminal UI, and browser server, install the
browser extra from GitHub:

```
uv tool install 'nenpi[browser] @ git+https://github.com/epsalmond/nenpi'
```

From a checkout, use:

```
uv tool install '.[browser]'
```

For smaller installations, use the base package for `nenpi` and
`nenpi-bench`, or add only the terminal UI:

```sh
uv tool install 'nenpi @ git+https://github.com/epsalmond/nenpi'
uv tool install 'nenpi[ui] @ git+https://github.com/epsalmond/nenpi'
```

## Commands

`nenpi` prints quota reports. For example:

```sh
nenpi sessions --harness all --since 7d --top 25
nenpi prompts --since 7d --sort turns --top 20
nenpi prompts --session 0123abcd
nenpi tools --since 7d --top 15
```

`nenpi prompts` without `--session` ranks every prompt in range by API turns
(or context, drain, tokens, units) across sessions, each row labelled with
the first line the person typed, cut to 120 characters and scrubbed of email
addresses and secret-shaped tokens. Injected blocks and pasted content never
become a label — a turn that is only an injected block is named, not quoted.
That short **label** is the only prompt text nenpi stores or prints;
`--no-label` hides it. With `--session` it is the per-prompt breakdown of one
session, unchanged.

`nenpi tools` ranks tool calls by the context their results add, estimated as
result characters / 4 and, where a later API call measured the growth, split
across that turn's results. Only tool **names** and result **sizes** are ever
parsed, cached or printed — never tool input or output text.

`nenpi-bench plan` projects a benchmark without spending quota. A
`nenpi-bench run` executes controlled `claude -p` calls and **spends real
subscription quota** while measuring tokens per plan percent; its fitted
weights are used by `nenpi` for Claude estimates.

```sh
nenpi-bench plan --models claude-haiku-4-5 --contexts 10k,60k --cache cold,warm
```

`nenpi-ui` opens the terminal UI and requires `nenpi[ui]`. With the browser
extra, `nenpi-ui --browser` starts the browser websocket server; `nenpi-web` is
an alias that always starts browser mode:

```sh
nenpi-ui
nenpi-ui --browser --host 127.0.0.1 --port 8000
nenpi-web --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` in your browser. `HOST` and `PORT` refer to the
machine running nenpi, which can differ from the machine displaying the page.
The browser discovers Claude and Codex transcript roots under that host's home
directory and persists source choices in the same `~/.config/nenpi/config.toml`
the CLI reads (or `$NENPI_CONFIG`; an older `config.json` is imported once and
renamed to `config.json.migrated`). Switching every root of a harness off
leaves the CLI scanning nothing for it, not the defaults. Filter the session table with free text or
`harness:claude`, `project:name`, `since:YYYY-MM-DD`, and
`until:YYYY-MM-DD`; use the sort button to change the ordering. A scan runs in
a worker and cancellation keeps the previous result visible. Date filters
select sessions active in the range; the displayed session totals remain
whole-session totals.

The browser uses the same `config.toml` root and account resolution as the
terminal reports. Account labels remain attached to the session payloads used
by the browser.

## Configuration

With no flags and no config file, nenpi looks at one root per harness:
`~/.claude` (or `$CLAUDE_CONFIG_DIR`) and `~/.codex` (or `$CODEX_HOME`). Extra
roots — a second Claude or Codex account, for instance — go in
`~/.config/nenpi/config.toml`:

```toml
[claude]
roots = ["~/.claude", "~/.claude-arcade"]

[codex]
roots = ["~/.codex", "~/.codex-arcade"]
```

`--claude-root`/`--codex-root` flags override this file for one invocation.
A `~/.claude-*` or `~/.codex-*` directory that exists but is in no resolved
set gets one stderr note per run; `[general] ignore_unconfigured = true`
turns it off.
Run `nenpi config` to see which roots are resolved and what account each one
authenticates as, or `nenpi config --init` to write a starter file seeded
from every `~/.claude*`/`~/.codex*` directory found on this host. See
[Roots and config.toml](docs/drain.md#roots-and-configtoml) for precedence
details and the breaking changes from earlier versions (the implicit
`~/.claude*`/`~/.codex*` glob is gone, and `--claude-root`/`--codex-root` now
take the harness home dir rather than its `projects`/`sessions` subdirectory).

## Docs

- [docs/drain.md](docs/drain.md) — subcommands, the Codex-measured vs.
  Claude-modelled metering models, weight configuration, and what about
  subscription quota is official, community-sourced, or unknown.
- [docs/bench.md](docs/bench.md) — scenarios, tick bracketing, the budget and
  contamination guards, and the fit it writes for `nenpi`.
