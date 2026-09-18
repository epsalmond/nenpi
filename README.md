# nenpi

## Optional browser

The terminal reports remain stdlib-only. Install the optional UI extra on
Python 3.11 or newer and launch the local Textual browser:

```sh
python -m pip install 'nenpi[ui]'
nenpi-ui
```

For the browser websocket server, install the browser extra and run:

```sh
python -m pip install 'nenpi[browser]'
nenpi-ui --browser --host 127.0.0.1 --port 8000
```

`nenpi-web` is an alias for the same browser entry point. The server binds to
localhost by default and serves the Textual websocket application, not only a
static page.

The browser discovers Claude and Codex transcript roots under the home
directory and persists source choices in `~/.config/nenpi/config.json` (or
`$NENPI_CONFIG`). Filter the session table with free text or
`harness:claude`, `project:name`, `since:YYYY-MM-DD`, and
`until:YYYY-MM-DD`; use the sort button to change the ordering. A scan runs in
a worker and cancellation keeps the previous result visible. Date filters
select sessions active in the range; the displayed session totals remain
whole-session totals.

The browser is a local view of the same transcript data and uses the same
`config.toml` root and account resolution as the terminal reports. Account
labels remain attached to the session payloads used by the browser.

nenpi (燃費, fuel economy) reads Claude Code and Codex CLI transcripts on disk
and reports which sessions drained how much **subscription-plan quota** — not
API dollars. Codex quota is measured directly from the `rate_limits`
snapshots the CLI writes into its own rollouts; Claude writes no quota data at
all, so Claude sessions are modelled at API list price and, when live
utilisation has been sampled, reported against the actual plan window. It is
standalone: Python standard library only, Python 3.11 or newer, Linux and
macOS.

`nenpi-bench` measures what one percent of a Claude plan window actually
costs, per token kind, by firing controlled `claude -p` calls and watching
utilisation tick over. It writes the fitted weight table `nenpi` uses to
convert Claude token counts into percent.

## Install

```
uv tool install git+https://github.com/epsalmond/nenpi
```

or, from a checkout:

```
uv tool install .
```

## Example commands

```
nenpi sessions --harness all --since 7d --top 25
nenpi prompts --session 0123abcd
nenpi-bench plan --models claude-haiku-4-5 --contexts 10k,60k --cache cold,warm
```

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
