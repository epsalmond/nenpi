---
description: nenpi-bench scenarios, tick bracketing, the budget and contamination guards, the NNLS fit it writes for nenpi, and what a run costs.
status: reference
read-when: Measuring what a percent of a Claude plan window buys, or fitting nenpi's Claude weights against real drain.
---

# nenpi-bench

`nenpi-bench` measures how many tokens of each kind one percent of a
Claude plan window costs, by firing controlled `claude -p` calls and watching
live utilisation tick over. Output is a per-scenario tokens-per-percent table
and a fitted weight table (percent per million tokens, per model and kind)
written where `nenpi` keeps its calibration.

It is standalone apart from `nenpi.drain`, which it imports for the
snapshot format and the NNLS solver and runs as a subprocess to sample. Python
standard library only, Python 3.11 or newer, Linux and macOS.

**Every percent it spends is real quota.** `run` prints the projection, refuses
to start without `--yes`, and aborts at `--max-percent`.

## Subcommands

```
nenpi-bench plan   [--models ...] [--contexts 10k,60k,150k] [--cache cold,warm]
                   [--outputs short,long] [--ticks 2] [--max-percent 3]
nenpi-bench run    ... --yes [--require-idle] [--config-dir PATH] [--claude PATH]
                   [--sample-interval 60] [--filler-channel prompt|system-prompt-file]
nenpi-bench report [--run-id ID] [--ticks N] [--json]
```

`plan` spends nothing: it prints the scenario list, the calls each scenario
needs, and the projected percent, so the cost is visible before anything runs.

```
scenario projection (dry run, nothing is spent)
weights: list price at 1.40 USD per percent

scenario                                   calls  needed  tokens/call     %/call   % total
claude-haiku-4-5/10k/warm/short              400    3360        10.2K     0.0009     0.357

claude-haiku-4-5/10k/warm/short needs more calls than --max-percent allows in --max-calls; raise --max-calls, use a larger context, or lower --ticks
```

That first line is the tool's main warning: a small warm context on the
cheapest model is thousands of calls per percent, so the cheap scenarios are
cheap in quota and expensive in wall time. Large contexts and cold cache reach
a tick in tens of calls.

## Why tick bracketing

`https://api.anthropic.com/api/oauth/usage` reports whole-percent floats. A
single `claude -p` call is invisible; only cumulative load that crosses a tick
is measurable. So a scenario runs calls until the utilisation percent steps up
`--ticks` times, and the estimate is

```
tokens between the first and the last tick / percents between them
```

Everything before the first tick and after the last is a partial percent of
unknown size and is discarded. `--ticks 2` is the floor that yields one fully
bracketed percent; more ticks average over more percents. `--ticks 1` produces
an estimate that includes the leading partial and is reported `unbracketed`.

Utilisation is sampled through `nenpi snapshot --oauth`, which owns the
credential read: the OAuth token is read, used and dropped inside nenpi
and never crosses into this tool. The sampler enforces a 60 s minimum interval
per config dir and backs off on HTTP 429, so `--sample-interval` below 60 has
no effect outside tests. Each read of the snapshot log is bounded to the last
hour, so a two-month log costs nothing to poll, and `nenpi snapshot
--compact` running beside a benchmark is harmless: it drops repeats and expired
records, never the newest observation.

If the five-hour window rolls over mid-scenario the bracket spans a boundary
and measures nothing; the ticks collected so far are discarded and the record
is marked `window_rollover`.

## Scenarios

A scenario is a model, a context size, a cache mode, and an output size. The
cross product of the four list flags is run cheapest first, so the early
scenarios buy the tokens-per-percent estimate that says what the expensive ones
will cost.

| Axis | Flag | Values |
| --- | --- | --- |
| model | `--models` | any model id `claude -p --model` accepts |
| context | `--contexts` | `10k`, `60k`, `150k` tokens of deterministic filler |
| cache | `--cache` | `cold`, `warm`, `write-heavy` |
| output | `--outputs` | `short` (one word), `long` (a 2000-word essay) |

- **cold** generates fresh filler for every call, so nothing caches and the
  input side is uncached tokens.
- **warm** reuses one identical prefix back to back, inside the 5-minute cache,
  so cache reads dominate.
- **write-heavy** reuses the prefix but waits `--write-heavy-gap` (360 s by
  default) between calls, so every call pays cache creation. It is slow by
  construction.
- **long** isolates the output weight; **short** keeps output near zero so the
  input side is what moves.

Filler is deterministic pseudo-word text built in memory from a seed, not read
from anywhere. The `10k` in a scenario name is the target; the real size comes
from the first call's usage and is recorded as `measured_context_tokens`.
`--filler-channel prompt` (the default) prepends the filler to the prompt on
stdin. `--filler-channel system-prompt-file` writes it to a scratch file and
passes `--system-prompt-file`; that flag appears in the CLI's own `--bare` help
text but not in its option list, so it is opt-in rather than the default.

## Guards

Three things can make a run spend more than it should, and each one stops it.

**The budget.** `--max-percent` (3 by default) caps the five-hour window and
`--max-percent-weekly` (1) caps the seven-day window, both counted from the
baseline sample taken before the first call and both enforced whatever
`--window` selects. The check runs before every call; reaching either stops the
run and exits 3. `--max-calls` (400) caps a single scenario, and `run` refuses
to start at all when the projection already exceeds the cap, unless
`--force-projection` says otherwise.

**Observability.** A budget check reads the newest sample, so a call is
launched only while that sample is younger than `--max-sample-age` (twice
`--sample-interval`, at least 90 s). A sampler that exits non-zero, crashes, or
logs no new observation stops the run before the next call and exits 4. This
is the difference between spending one more call and spending the rest of the
window blind.

**Contamination.** Any other Claude session burning quota during the run lands
in the same window and corrupts the measurement. Before each scenario and after
it, `nenpi sessions --harness claude --since 6h --json` is read and every
Claude session that overlaps the scenario and is not one of this run's own
marks it `contaminated`. A check that could not run is `check_failed`, which
counts as contamination rather than as a clean window: under `--require-idle`
it refuses to start (exit 4), and otherwise it keeps the scenario out of the
fit. Contaminated scenarios stay in the logs.

## The fit

Each scenario that was fully bracketed, uncontaminated and free of cache drift
contributes one row: bracketed tokens per model and kind against the bracketed
percent. Non-negative least squares over those rows gives percent per million
tokens for each `(model, kind)` pair, using nenpi's solver. Every model a
call touched keeps its own row, so a subagent or internal helper call is not
priced at the requested model's weight.

A scenario with a single tick is an upper bound, not a measurement: it divides
everything spent since the scenario started, including the partial percent
burnt before that tick, by one percent. Those rows are excluded unless
`--include-unbracketed` asks for them, as are drifted ones without
`--include-drifted`; the report names every exclusion.

The report prints the measured ratios against the API list ratios — cache read
at 0.1x uncached input, output at 5x — which is the open question the fit
exists to answer: Claude Code quota is reported to charge cache reads by more
than the API's 10% (anthropics/claude-code issue #24147).

Scenarios that all stop at the same tick count carry no variance in the target,
which leaves R² undefined; the report then grades the fit by relative residual
instead and says so. A fit with fewer rows than columns is marked
`UNDERDETERMINED` and is not `usable`.

Least squares returns a number for every column, including ones the scenarios
cannot separate. Those are reported `unidentified` with a reason and stored as
null rather than as a confident zero, because a zero weight prices that kind as
free — the same convention `nenpi calibrate` uses. The rules differ from
nenpi's, because its rows are time buckets of whatever happened to run
while these rows are designed scenarios: a column is unidentified when it
correlates above 0.95 with another, when it lands at zero with under 5% of the
token mass, or when the fit is underdetermined and the column appears in fewer
than two scenarios. A single-scenario run therefore identifies nothing, which
is the honest answer.

The fit also reports `fallback_scale`, one scalar mapping API list price onto
the fitted percent unit (percent per USD, so its inverse is the measured
dollars per percent). Unidentified coefficients are priced through it, which
keeps every model on one scale instead of mixing fitted percents with raw
dollars.

## State

```
~/.local/state/nenpi/bench/<run-id>/meta.json        run arguments, baseline, budget
~/.local/state/nenpi/bench/<run-id>/calls.jsonl      one line per call
~/.local/state/nenpi/bench/<run-id>/scenarios.jsonl  one line per scenario, with ticks
~/.local/state/nenpi/bench/<run-id>/report.json      estimates and fit
~/.local/state/nenpi/claude-weights.json             the fitted weights
```

A call line holds the scenario key, the requested model, per-model usage,
timings, the reported cost, the session id, whether the call risked falling
outside the cache TTL, and the utilisation sample nearest in time. meta.json
records the flags every call carried and how many extra `--claude-arg` values
were passed, never their contents.
Prompt text, filler text and the OAuth token are never printed, logged or
stored; a failing `claude` call is reported by exit status only, because its
stderr can echo the prompt back.

`claude-weights.json` carries the shapes nenpi reads: a top-level
`claude` section shaped like the `codex` section of `codex-weights.json`, so
`merge_weights` consumes it unchanged, and a `windows` section shaped like the
`calibrate --harness claude` payload.

```json
{
  "harness": "claude",
  "unit": "percent_per_mtok",
  "usable": true,
  "identified": 3,
  "fallback_scale": 0.71,
  "claude": {
    "unit": "percent_per_mtok",
    "models": { "claude-haiku-4-5": { "input": 0.9, "cache_read": 0.12, "output": 4.7 } }
  },
  "fit": {
    "unit": "percent_per_mtok",
    "models": {
      "claude-haiku-4-5": {
        "input": 0.9, "cache_read": 0.12, "output": 4.7, "cache_write_5m": null
      }
    },
    "diagnostics": { "claude-haiku-4-5": { "cache_write_5m": { "reasons": ["collinear"] } } }
  },
  "windows": { "five_hour": { "models": { "claude-haiku-4-5": { "input": 0.9 } } } }
}
```

The `claude` section is the price table a reader merges, so it carries measured
numbers only: a kind the fit could not identify is absent and keeps its
built-in price. The nulls and the reason for each live in `fit`, which is a
record of the run rather than a price table. Only a usable fit is written to
`claude-weights.json` at all; an unusable one lands beside it as
`claude-weights.unusable.json`, which nothing loads.

### Merging a fit into nenpi by hand

Copying the `claude` section into `~/.config/nenpi/weights.json` mixes
two units. The fitted numbers are percent per million tokens; every kind the
section omits keeps its built-in price, which is USD per million tokens. A
partial fit merged that way prices some kinds in percent and the rest in
dollars, and the two differ by orders of magnitude, so one kind's tokens will
swamp the rest. `fallback_scale` is what converts the omitted kinds onto the
fitted unit, and nenpi does not yet apply it when merging a Claude fit.

Until it does, merge only a fit that identifies all four kinds for the models
you care about — `input`, `cache_read`, `cache_write_5m` and `output` — so no
kind falls back. `nenpi-bench report` names every unidentified kind and its
reason.

`NENPI_HOME_DIR`, `NENPI_CACHE_DIR`, `NENPI_STATE_DIR` and
`NENPI_CONFIG_DIR` relocate all of it, exactly as they do for
nenpi.

## What the result JSON is read from

`claude -p --output-format json` returns one result envelope. nenpi-bench reads
`modelUsage` — per-model totals for every model call the run made, which the
Agent SDK type documentation names as the field for token accounting — and
falls back to `usage`, the main-loop-only block, when a build reports no
`modelUsage`. Keys used: `type`, `is_error`, `session_id`, `total_cost_usd`,
`modelUsage.<model>.{inputTokens, cacheReadInputTokens,
cacheCreationInputTokens, outputTokens}`, and on the fallback path
`usage.{input_tokens, cache_read_input_tokens, cache_creation_input_tokens,
output_tokens}` plus the `cache_creation.ephemeral_*_input_tokens` split when
present. Verified against the installed CLI's help and the bundled
`@anthropic-ai/claude-agent-sdk` type declarations for 2.1.273, not by
spending a run.

## Exit codes

| code | meaning |
| --- | --- |
| 0 | done |
| 1 | usage error, or no scenario reached its tick target |
| 2 | missing dependency: no nenpi, no `claude`, no baseline sample |
| 3 | budget reached |
| 4 | a guard stopped the run: the sampler, or a contaminated window under `--require-idle` |
| 5 | a scenario hit `--max-calls` before its tick target |

## Limits

- Percent resolution is 1% of the window, so every estimate is an average over
  whole percents and a scenario that cannot cross two ticks inside the budget
  cannot be measured at all.
- Cache behaviour is observed, not controlled: `warm` is a label for calls made
  back to back with an identical prefix, and whether the prefix actually caches
  is a property of the harness and the account. A warm call made more than
  240 s after the previous one is recorded as `cache_ttl_risk` and its scenario
  as `warm_drift`, which keeps it out of the fit unless `--include-drifted`.
- The endpoint cannot be polled more than once a minute, so a run is blind
  between samples and the budget is enforced at sample resolution. The overshoot
  is bounded by what one sample interval of calls can spend, not by the cap, so
  size scenarios and `--sample-interval` with that in mind.
- The projection prices a fresh prefix as cache creation at 1.25x, including in
  `cold` mode. Claude Code sets its own cache breakpoints, so this is the
  conservative reading rather than a verified one; `--no-cold-as-creation`
  prices cold as uncached input instead.
- Calls run with the built-in tools off (`--tools ""`). The 2.1.273 `--help`
  lists no turn limit, and with no tools there is nothing for a further turn to
  do, so a run cannot fan out into work the scenario did not ask for.
- The fit assumes drain is linear in tokens with no per-request or per-session
  component. A constant per-call cost would show up as residual, not as a term.
- The planning projection uses a prior of 1.40 USD per percent when no fit and
  no dollar field exists, because Anthropic publishes no dollar ceiling and
  every account seen reports `limit_dollars` null. It scales projected call
  counts only; nothing measured depends on it.
- Contamination detection sees only what nenpi can parse from
  transcripts, so a session writing no transcript is invisible.

## Verification

```sh
uv run --python 3.11 python -m unittest discover -s tests -v  # 3.11 floor, no quota spent
uv run --python 3.13 python -m unittest discover -s tests -v
```

The suite puts a fake `claude` on PATH and a fake nenpi sampler that
bills the calls at known weights, so tick bracketing, the budget abort, the
contamination flag and the fit are all checked without a network call or a
percent of quota. A real smoke run costs real quota and needs an explicit
decision each time; start with one cheap scenario:

```sh
nenpi-bench run --models claude-haiku-4-5-20251001 --contexts 10k \
    --cache warm --ticks 1 --max-percent 1 --require-idle --yes
```

One tick produces an upper bound rather than a measurement, and a single
scenario identifies nothing, so that run writes no weights file: it proves the
plumbing, and the matrix is what produces a fit.

## Follow-ups

Known work this tool does not do yet:

- **Retry a transient sampler failure.** One failed poll ends the run today.
  A bounded retry window — a few attempts inside the time the freshness limit
  allows — would let a run survive a single timeout without ever letting it
  spend blind.
- **Scale omitted kinds on merge.** Applying `fallback_scale` to the kinds a
  fit leaves out would put a partial fit on one unit and remove the
  merge-by-hand caveat above.
- **Project spend inside the run.** The budget is checked against the newest
  sample, so calls made since then are unaccounted. Stopping when observed
  spend plus the projected spend of the calls made since the last sample
  exceeds the five-hour budget would close the blind window.
