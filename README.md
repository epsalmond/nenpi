# nenpi

nenpi (燃費, fuel economy) reads Claude Code and Codex CLI transcripts on disk
and reports which sessions drained how much **subscription-plan quota** — not
API dollars. Codex quota is measured directly from the `rate_limits`
snapshots the CLI writes into its own rollouts; Claude writes no quota data at
all, so Claude sessions are modelled at API list price and, when live
utilisation has been sampled, reported against the actual plan window. It is
standalone: Python standard library only, Python 3.9 or newer, Linux and
macOS.

`nenpi-bench` measures what one percent of a Claude plan window actually
costs, per token kind, by firing controlled `claude -p` calls and watching
utilisation tick over. It writes the fitted weight table `nenpi` uses to
convert Claude token counts into percent.

## Install

```
uv tool install nenpi
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

## Docs

- [docs/drain.md](docs/drain.md) — subcommands, the Codex-measured vs.
  Claude-modelled metering models, weight configuration, and what about
  subscription quota is official, community-sourced, or unknown.
- [docs/bench.md](docs/bench.md) — scenarios, tick bracketing, the budget and
  contamination guards, and the fit it writes for `nenpi`.
