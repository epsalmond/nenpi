# Usage and configuration

## Installation options

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

## Following report suggestions


Most reports end with a short **"what to run next"** footer on **stderr**,
built from the rows it just printed: `sessions` names its top sessions and
hands you the `prompts --session` line for the biggest one, that view names
the busiest prompt and hands you the `tools --session ... --prompt N` line,
and so on. The suggestions repeat the `--since`/`--until`/`--harness`/
`--account` flags you passed, so each one is runnable as printed and stays in
the same scope.

`auto` includes next actions directly in its diagnosis cards instead of a
separate footer.

```sh
nenpi sessions --since 7d          # next: nenpi prompts --session 0123abcd --since 7d
nenpi prompts --session 0123abcd --since 7d   # next: nenpi tools --session 0123abcd --prompt 27 --since 7d
nenpi tools --session 0123abcd --prompt 27 --since 7d
```

Each suggested `--session` uses the shortest prefix that resolves to one
session, and every value is shell-quoted, so the line runs as printed.

Because the footer is on stderr, stdout stays pipeable. `--quiet`/`-q` or
`NENPI_QUIET=1` turns it off; under `--json` nothing is written to stderr and
the same suggestions ride along as the payload's `next` list of
`{"cmd", "why"}` entries. `nenpi snapshot --stdin`, the statusline
passthrough, never prints a footer.

## Benchmarks

See [Benchmark guide](bench.md) for setup and measurement details.

`nenpi-bench plan` projects a benchmark without spending quota. A
`nenpi-bench run` executes controlled `claude -p` calls and **spends real
subscription quota** while measuring tokens per plan percent; its fitted
weights are used by `nenpi` for Claude estimates.

```sh
nenpi-bench plan --models claude-haiku-4-5 --contexts 10k,60k --cache cold,warm
```

## Terminal and browser UI

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

## Transcript sources


With no flags and no config file, nenpi looks at one root per harness:
`~/.claude` (or `$CLAUDE_CONFIG_DIR`) and `~/.codex` (or `$CODEX_HOME`). Extra
roots — a second Claude or Codex account, for instance — go in
`~/.config/nenpi/config.toml`:

```toml
[claude]
roots = ["~/.claude", "~/.claude-work"]

[codex]
roots = ["~/.codex", "~/.codex-work"]
```

`--claude-root`/`--codex-root` flags override this file for one invocation.
A `~/.claude-*` or `~/.codex-*` directory that exists but is in no resolved
set gets one stderr note per run; `[general] ignore_unconfigured = true`
turns it off.
Run `nenpi config` to see which roots are resolved and what account each one
authenticates as, or `nenpi config --init` to write a starter file seeded
from every `~/.claude*`/`~/.codex*` directory found on this host. See
[Roots and config.toml](drain.md#roots-and-configtoml) for precedence
details and the breaking changes from earlier versions (the implicit
`~/.claude*`/`~/.codex*` glob is gone, and `--claude-root`/`--codex-root` now
take the harness home dir rather than its `projects`/`sessions` subdirectory).


Run Nenpi on the machine holding the transcripts; it does not scan remote hosts automatically.

[Basic usage](../README.md) · [Themes](themes.md) · [Diagnostics](diagnostics.md)
