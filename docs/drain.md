---
description: nenpi subcommands, the Codex measured and Claude modelled metering models, weight configuration, and what about subscription quota is official, community-sourced, or unknown.
status: reference
read-when: Attributing subscription-plan quota to agent sessions, tuning nenpi weights, or wiring its snapshot sampler into a timer.
---

# nenpi

`nenpi` reads Claude Code and Codex CLI transcripts on disk and
reports which sessions drained how much **subscription-plan quota** — not API
dollars. It is standalone: Python standard library only, Python 3.11 or newer,
Linux and macOS.

Two harnesses, two very different evidence bases:

- **Codex is measured.** The CLI writes `rate_limits` snapshots of the shared
  account pool into its own rollouts. Drain between consecutive snapshots is
  real, and the tool splits it across the sessions that burned tokens in that
  interval.
- **Claude is modelled.** Nothing about quota is written into Claude
  transcripts. Sessions are priced at API list rates and reported as a dollar
  equivalent and a share, unless live utilisation snapshots have been sampled
  (see [Snapshots](#snapshots)).

The second thing it models is **why long sessions cost so much**: each user
prompt triggers a fan-out of API calls that every re-send the whole context,
so cost grows faster than session length. See
[Fan-out and context growth](#fan-out-and-context-growth).

## Subcommands

```
nenpi sessions    [--harness claude|codex|all] [--since 7d|2026-09-10] [--until ...]
                  [--window auto|five_hour|weekly] [--top 25]
                  [--sort drain|tokens|start] [--json] [--no-color] [--width N]
nenpi prompts     [--sort turns|context|drain|tokens|units] [--top N]
                  [--label | --no-label] [--session <id-prefix> [--first] [--tools]]
nenpi tools       [--session <id-prefix>] [--first] [--prompt N] [--top N]
                  [--sort context|calls|mean] [--harness ...] [--json]
nenpi fanout      [--since ...] [--harness ...] [--sort tokens|turns|...]
nenpi reductions  [--since ...]
nenpi timeline    [--bucket 1h|5h|1d]
nenpi windows     [--harness codex]
nenpi calibrate   [--harness codex|claude] [--since ...]
nenpi verify      [--since ...]
nenpi snapshot    [--stdin | --oauth [--config-dir PATH] | --compact]
nenpi config      [--claude-root PATH] [--codex-root PATH] [--json]
nenpi config      --init [--force]

Every reporting subcommand also takes --quiet/-q (drop the stderr footer).
```

Common flags on every reporting subcommand: `--claude-root PATH` and
`--codex-root PATH` (both repeatable) *replace* the resolved roots for that
harness — see [Roots and config.toml](#roots-and-configtoml) for how they are
resolved when omitted. `--account LABEL` limits sessions, events, and quota
windows to one account's pool, by root basename (e.g. `.codex-arcade`; see
`nenpi config` for the labels in use). `--claude-cache-read-weight FLOAT`
overrides the
disputed cache-read price,
`--long-context-multiplier FLOAT` scales requests over 200K tokens (a no-op at
its default of 1.0), `--use-calibrated` prefers a stored fit,
`--whole-session` reports each selected session's whole life rather than the
part inside the range, `--ascii` draws bars without block glyphs, and
`--rebuild-cache` discards the parse cache.

Every subcommand ends with a **"what to run next"** footer of at most three
lines on **stderr**, derived from the rows it just printed: the top sessions
and the `prompts --session` line for the biggest, the busiest prompt and the
`tools --session ID --prompt N` line for it, the busiest timeline bucket and
the `sessions --since ... --until ...` line that opens it. Suggestions repeat
the `--since`, `--until`, `--harness`, and `--account` flags of the run that
produced them, so each is runnable as printed and stays in the same scope; a
flag the suggestion sets itself (the bucket's own `--since`) wins.

stdout is untouched, so `nenpi sessions | tee` stays clean. `--quiet`/`-q`
and `NENPI_QUIET=1` suppress the footer. Under `--json` nothing goes to
stderr and the same suggestions are the payload's additive `next` key, a list
of `{"cmd", "why"}` objects for scripts and the TUI. `nenpi snapshot --stdin`
is a byte-exact statusline passthrough and never writes a footer. Color is
used only when stderr is a tty.

`--profile` (or `NENPI_PROFILE=1`) prints one line per phase — scan, totals,
prompts, intervals, attribute, total — to stderr, so a run that got slower
says which phase did it without reaching for cProfile. stdout is untouched,
so `--json --profile` still pipes cleanly.

`--since` and `--until` window the **events**, not just the session list. A
session that started weeks ago and ran again this morning reports only this
morning's tokens; `--whole-session` opts back into its lifetime totals.

### sessions

One row per session, ranked by each row's share of its own harness peak, with
a detail line underneath:

```
H session  cwd      model         start            dur    in     cached  write  out     units    drain    prm resent share
X 0123abcd projects gpt-6-astra   2026-09-13 21:14 3h20m  2.1M   90.4M   0      210.0K  1180.55  4.220%   9   62%   ████████
          api turns 140, turns/prompt p90 31, peak context 168.4K; subagents 22 requests / 4.10 units
```

`H` is `C` for Claude and `X` for Codex. `units` is weighted tokens: credit
units for Codex, dollars for Claude. `drain` is the measured percent of the
Codex window, or the Claude dollar equivalent. `resent` is the share of the
session's cost spent re-sending context it had already sent
([below](#fan-out-and-context-growth)).

Codex drain and Claude drain are different units and cannot share a scale, so
the ranking and the bar both use each row's fraction of its own harness's
largest drainer. The header says so.

### prompts

Without `--session`, every prompt in range is ranked together, one row per
prompt: rank, harness, short session id, cwd, prompt index, API turns,
subagent turns, peak context, measured drain, and the prompt's label.
`--sort` picks `turns` (default), `context`, `drain`, `tokens` (input tokens
sent) or `units`, `--top N` cuts the list (default 10), and `--no-label`
drops the label column. `--json` returns the same rows with the full
`session_id`, `prompt_index`, `cwd` and `label`.

```
top 10 prompts by turns across 63 sessions

rank harness session    cwd              #  turns    sub   ctx peak    drain prompt
   1 codex   01a0ab96   nenpi            7    929    713     116.8K    2.26% rerun the tests
```

The **label** is the one line the person typed that opened the prompt, cut to
120 characters — see [What is stored](#performance-and-state) for what is
scrubbed out of it first.

With `--session`, the per-prompt breakdown for that session, plus two charts (input tokens sent per
prompt, split cached and uncached; peak context per prompt) and a fitted
growth summary:

```
growth: quadratic fits better | linear slope 41.2 u/prompt (R^2 0.71) |
quadratic x^2 term 9.98 (R^2 0.998) | last 20% of prompts = 41% of session cost
```

The fit is ordinary least squares of per-prompt weighted units against prompt
index, once linear and once quadratic; `better` names the higher R², requiring
at least a 0.01 margin so near-ties report as linear.

### tools

Ranks tool names by the context their results add. `--session` takes the same
ID prefix as `prompts` (busiest wins when several match) and `--prompt N`
narrows to one prompt of that session; without either, the report is
corpus-wide. Columns:

- **calls** — completed calls of that tool in range.
- **est tokens** — *estimate*: total result characters / 4. Labelled an
  estimate everywhere; no tokenizer is run.
- **measured** — the input-token growth actually billed on the next API call
  of the same thread, split across that turn's tool results in proportion to
  their sizes. A turn with no following call measures nothing, so measured is
  0 there; where other things also grew the context (reasoning, assistant
  output, pasted text) the split charges them to the tools of that turn, so
  measured is an UPPER BOUND - the table says so under its header, and
  `--json` carries `measured_is_upper_bound`. On a real corpus it runs
  several times the estimate for that reason.
- **mean** / **max** — mean and largest single result, in estimated tokens.
- **share** — measured tokens over the total positive input growth of the
  sessions in scope.

A closing section lists the five largest single results by tool, session and
prompt index. `--sort` picks `context` (default, estimated tokens), `calls`,
or `mean`.

`prompts --session --tools` adds the same numbers per prompt: call count,
estimated tokens, and the tool behind the largest single result of that
prompt. `--label` adds the prompt label there too - it is on by default in
the ranking and off with `--session`, and `--label`/`--no-label` work in
both. Without either flag the per-session `prompts` table is unchanged.

Only tool **names** and result **sizes** are read. Tool inputs and outputs are
never stored in the cache, printed, or hashed.

### fanout

Across every session in range: the turns-per-prompt distribution (histogram
plus p50/p90/max), the peak-context-per-prompt distribution, and the top 15
single prompts by input tokens sent; `--sort` reranks that table by `turns`,
`context`, `drain` or `units` instead.

### reductions

Points where a session's context shrank sharply — see
[Context reductions](#context-reductions).

### windows

Each observed Codex quota window: account, start, `resets_at`, peak
`used_percent`, and the sessions that drained it. Two roots on different
accounts always get separate windows, even when they share the same
`limit_id`/`plan_type`/`window_minutes` - see
[Roots and config.toml](#roots-and-configtoml) for how a root's account is
detected. Pass `--account LABEL` (a root's basename, e.g. `.codex-arcade`) to
any report command to see one pool at a time.

### verify

Cross-checks the parse against the harnesses' own summaries: Claude deduped
sums against the `cost-state` line's `modelUsage`, Codex summed usage deltas
against the final `thread_token_usage`.

### calibrate

Fits weights against measured drain instead of trusting the rate card. Both
harnesses use the same non-negative least squares over snapshot intervals:
each interval contributes its summed tokens per (model, kind) as features and
its measured percent drain as the target.

Least squares returns a number for every coefficient, including ones the data
cannot separate: a model that only ever ran beside another, or one with too
little token mass to move the target. Each coefficient reports the number of
buckets it appears in, its share of token mass, its highest correlation with
any other coefficient, and whether it sits at the non-negativity boundary. A
coefficient is `unidentified` when it appears in fewer than 3 buckets, or
correlates above 0.95 with another, or sits at zero with under 5% of the token
mass. Unidentified coefficients are stored as null, never as a confident zero,
because a zero weight prices that model as free.

`--use-calibrated` converts the **whole** Codex table to percent per Mtok:
fitted values where they are identified, and the rate card multiplied by one
fitted global scale everywhere else. Half a table in fitted percents and half
in rate-card credit units is not a scale — the two differ by orders of
magnitude, so one model's event would absorb a whole interval while the rest
rounded to nothing. If that global scale cannot be fit, no calibration is
applied at all.

Both vendors report utilisation in **whole percent**, so a single interval's
target is almost always exactly 1.0 while its token features vary by orders of
magnitude. Intervals are therefore summed into fixed time buckets
(`--calibrate-bucket-hours`, default 2) before fitting. Bucketing by *time*
and not by drain matters: equal-drain buckets would make every target the same
by construction and leave the fit no variance to explain. On 14 days of real
Codex rollouts this is the difference between R² of -1.9 and R² of 0.93.

A fit is `usable` only with R² of at least 0.5 and at least one identified
coefficient; a Codex fit additionally needs the global fallback scale, which
Claude has no rate card for and does not require. Otherwise it is stored with
`"usable": false`, reported as `NOT USABLE`, and refused by
`--use-calibrated`.

`--harness codex` writes the fit to
`~/.local/state/nenpi/codex-weights.json`; `--use-calibrated` then
prefers it. `--harness claude` needs sampled snapshots and reports the
measured cache-read rate beside the uncached-input rate, so the disputed 0.1x
list ratio can be tested against observation.

## Metering model

### Codex: measured

Every Codex `event_msg` of type `token_count` may carry a `rate_limits` block
describing the shared account pool: `limit_id`, `plan_type`, and a `primary`
and optional `secondary` window with `used_percent`, `window_minutes`
(300 = five-hour, 10080 = weekly) and `resets_at` (Unix seconds).

Observations from every Codex root are merged into one timeline per
`(account, limit_id, plan_type, window_minutes)`. Within a timeline the tool
walks forward, keeping a running maximum:

- **(a) A bucket change with drain evidence** (`used_percent` dropped, or rose
  by more than nothing) **is a rollover.** Its own `used_percent` is that
  window's drain so far, attributed as a `rollover` interval that never
  reaches back past the moment the window opened (`resets_at` minus its
  length).
- **(b) A bucket change with no drain evidence is not a rollover.** Both
  vendors re-stamp `resets_at` on every poll even for an idle pool - Codex
  slides an untouched window's stamp toward `now + 7d`, drifting by roughly a
  minute per reading - so a bucket change where `used_percent` is unchanged,
  or an always-zero pool, carries no evidence anything reset. No interval is
  emitted, and the running maximum and its anchor timestamp carry forward
  unchanged, so the next real rise still spans back to the last reading that
  actually moved.
- **(c) Within one bucket, a decrease is bounded jitter, not a reset.** A
  same-bucket decrease of `JITTER_TOLERANCE` (2 points) or less is vendor
  jitter and is ignored, keeping the running maximum and its anchor; a larger
  decrease is a real drop and becomes the new baseline (but still emits no
  interval - a same-bucket reset cannot happen), so the next rise is not
  re-charged against percent already attributed to the old high-water mark.
- A reading from a window that has already rolled over is a stale poll from a
  concurrent session and is dropped.
- Otherwise (same bucket, a rise past the running maximum) drain is the
  increase over it.

`windows` and `sessions --json`'s `pools` group readings by `resets_at`,
clustered within a five-minute tolerance per `(account, window_minutes)`
rather than rounded to the minute: the same continuous drift that rule (b)
keeps from creating phantom intervals would otherwise still fragment one
idle window's `windows` output into dozens of near-identical, zero-peak rows.

Exactly one `window_minutes` value is ever used — whichever the snapshots
report most often, or the one `--window` names — because a five-hour percent
and a weekly percent have different denominators and adding them doubles every
session's drain. Ties go to the shorter window. The header names the window in
use. A window this tool has no name for is still selected and reported by its
minute count.

Each interval's drain is split across the sessions that recorded token deltas
inside it, in proportion to their weighted tokens. A session's measured drain
is the sum of its shares; `share_of_window` is its share of everything
attributed to the same window instance - the denominator is the interval's
*measured* drain, not the sum of what got attributed, so a capped window's
shares do not have to sum to 100%. A rolled-over window's interval starts
no earlier than the moment that window opened (`resets_at` minus its length),
so its drain is never charged to sessions that had already finished.

**The attribution cap.** Sparse readings can still make a proportional split
implausible: the whole jump between two readings is split only across the
sessions with a token delta inside that interval, so a handful of turns that
happened to land inside a big jump can outrank a session that did a thousand
times the work in a smaller one. After the proportional split, each session's
share is capped at `CAP_FACTOR` (3) times its *plausible cost* - a per-account
rate, in measured percent per weighted unit, times its own weighted units in
that interval. A session over its cap is clamped to it; the freed drain is
redistributed proportionally among the *other capped* sessions still under
their own cap, which can repeat until none are over. Whatever the cap will
not let any session absorb becomes `interval.unattributed` - usage from a
client of *this* account that was never scanned (another machine, another
login copy of the same credentials). It is never another account's drain: an
interval only ever holds events from its own account's sessions
(`account_for_root`), so this is not the cross-account bleed a per-account
pool key already rules out. An interval with drain but no local session at
all (every byte of it came from a client nas never saw) reports its whole
drain as `unattributed` rather than silently dropping it - the invariant
`sum(session shares) + unattributed == interval drain` holds for every
interval, not only the ones with local activity. `nenpi prompts --session`
splits a session's own turns out of its *capped* share, so a session's
prompts always sum to the same figure `sessions` reports for it, never the
interval's uncapped drain.

The rate is the median of each qualifying interval's own drain/units ratio
(not a pooled sum/sum, which one foreign-contaminated interval could drag
up), taken over that account's own non-rollover intervals whose units are all
on the weighted-unit scale. A pool needs at least 3 such intervals and at
least 1 weighted unit of coverage before its own rate is trusted; short of
that, `--use-calibrated`'s implied rate (1.0 - a calibrated table is already
percent per weighted unit) or a usable stored fit's `fallback_scale` is used
instead, and with neither, capping is skipped for that pool with a single
warning. A session whose weighted units came from the raw-token fallback (a
model with no weight, or a zeroed fitted coefficient) is exempt from the cap
in either direction - that fallback's scale is not the weighted-unit scale, so
capping against it would silently zero out a session that really did the
work. That exemption also keeps such a session out of the water-fill
redistribution pool: it always keeps its own raw proportional share (never
more, even when another session's overflow is freed alongside it), and a
capped session's freed drain goes only to other sessions with a finite,
unmet cap, or to `unattributed` when none remain - an exempt session must
never become an uncapped sink for everyone else's overflow. The one
exception is a fallback session with no other session to share the interval
with: there is nothing to cap it against, so it keeps its raw (i.e. full)
share, which is what falls out of the loop naturally rather than a special
case.

Surfaced wherever drain is: the `sessions` header adds a line per account with
measurable unattributed drain (`unattributed: .codex 14.2% (usage from
clients not in the scanned roots)`); `windows` prints an `unattributed N%`
line under any window that has one; `sessions --json` adds a top-level
`pools` array, one entry per (account, window instance), with
`account_label`, `window_resets_at`, `peak`, `attributed`, and `unattributed`.
`calibrate --harness codex` fits weights against measured drain unchanged -
the cap only reshuffles one interval's drain across its own sessions and
never reaches the fit, and subtracting `unattributed` from the fit target
would be circular - but reports the corpus's overall unattributed share as a
diagnostic, since a large one means a contaminated corpus.

Two details matter and are easy to get wrong:

- **`resets_at` jitters between readings, in both harnesses.** Codex moves its
  Unix seconds by a second or two; Claude stamps fresh microseconds onto its
  ISO timestamp on every poll. Comparing either exactly makes every reading
  look like a window rollover and inflates drain by more than an order of
  magnitude. The tool parses both forms and buckets them to the nearest
  minute.
- **`used_percent` is reported in whole percent.** A 1% step covers everything
  since the previous change, so the tool spans each interval back to the last
  reading at which the value moved, rather than charging the step to whichever
  session happened to be running at the final poll.

Token accounting follows the Codex protocol definitions
(`codex-rs/protocol/src/protocol.rs`): `cached_input_tokens` is a **subset** of
`input_tokens` (`TokenUsage::non_cached_input` subtracts it) and
`reasoning_output_tokens` is a **subset** of `output_tokens`. Neither is added
again. `cache_write_input_tokens` is reported for display only: it is part of
the input count and OpenAI publishes no separate cache-write rate. Codex's own
rollout budget weighs `output_tokens * sampling_weight + non_cached_input *
prefill_weight`, which is the same shape this tool uses.

Deltas come from `token_usage_record.payload.usage`. Rollouts old enough to
lack that record fall back to diffing consecutive
`event_msg token_count info.total_token_usage`; when both exist the records
win.

### Claude: modelled

No rate-limit or quota data is written into Claude transcripts. Usage lives on
`assistant` lines at `message.usage`, and the same usage object is repeated
once per streamed content block — typically three times, with identical
`message.id` and `requestId`. **Deduplicating on `message.id` is mandatory**;
without it totals are roughly 3x too high.

Dedup is corpus-wide, not per file. A forked or resumed session writes a new
transcript under a **new `sessionId`** that replays the original's assistant
lines verbatim — same `message.id`, same timestamps, same usage. On this host
2.1% of tokens in the 600 newest transcripts were cross-file replays. The
oldest file to record an API call keeps it; a session that lost calls to
another reports `fork_of` and `duplicate_turns`. Codex is deduped the same way
on `response_id`.

Quota draw is modelled as API list-price dollars per model, because the
officially stated factors are model, length and effort, and because the
`limit_dollars` / `used_dollars` field names in `.claude.json` indicate
dollar-denominated metering. Built-in prices (USD per million tokens):

| model | input | cache read | cache write 5m | cache write 1h | output |
| --- | --- | --- | --- | --- | --- |
| claude-opus-5 | 5.00 | 0.50 | 6.25 | 10.00 | 25.00 |
| claude-opus-4-8 | 5.00 | 0.50 | 6.25 | 10.00 | 25.00 |
| claude-fable-5-1 | 10.00 | 0.25 | 12.50 | 20.00 | 50.00 |
| claude-fable-5 | 10.00 | 1.00 | 12.50 | 20.00 | 50.00 |
| claude-sonnet-5 | 2.00 | 0.20 | 2.50 | 4.00 | 10.00 |
| claude-haiku-4-5 | 1.00 | 0.10 | 1.25 | 2.00 | 5.00 |

Model ids are normalised by stripping a trailing date suffix, so
`claude-haiku-4-5-20251001` prices as `claude-haiku-4-5`. `<synthetic>` and
unrecognised model strings are counted in an `unweighted` bucket and priced at
zero; the header reports the total.

Working directories are stored and printed as a **basename only**, with a
stable hash of the full path for grouping. The full path is customer-
identifying and never reaches the cache or the report.

Subagent transcripts live at `<session-dir>/subagents/agent-<id>.jsonl`
(sometimes nested deeper) and carry the parent's `sessionId` with
`isSidechain: true`. Their usage rolls up into the parent session and is also
reported separately as "of which subagents".

**The Claude denominator is UNKNOWN.** Without sampled snapshots the tool
reports a dollar equivalent and each session's share of the Claude sessions
listed. With snapshots it can fit percent (see below), and labels those
figures `est`.

### Codex sessions span several rollout files

A Codex *session* is not a rollout. `session_meta.payload` carries `id` (this
thread), `session_id` (the umbrella session) and, for a spawned subagent,
`parent_thread_id` plus a `source` naming the spawn. Every subagent runs in
its own `rollout-*.jsonl` under the parent's `session_id`.

The tool keys sessions by `session_id` and rolls subagent threads up into the
parent, matching the Claude behaviour, so a Codex session reports an
"of which subagents" figure too.

## Fan-out and context growth

A prompt is one thing a person typed. Everything the harness does until the
next prompt belongs to it, including subagent work started under it.

- **Claude:** a `type: user` line on the main session (not `isSidechain`, not
  `isMeta`) whose `message.content` is a string or contains a `text` block,
  with no `toolUseResult` key and no `tool_result` block. Lines carrying tool
  results are fan-out steps, not prompts.
- **Codex:** `token_usage_record.payload.turn_id` groups API calls into turns
  exactly. Where it is absent, a prompt starts at an `event_msg` of type
  `task_started` or at a `turn_context` line; Codex writes both within a
  second or two of each other for the same turn, so they are collapsed.

Per prompt the tool records wall time, API turns (distinct Claude
`message.id` / Codex `response_id`), context size at the start and at the
peak, input tokens summed across the fan-out split into uncached, cache read
and cache write, output tokens, weighted units, measured or estimated drain,
and the tool-call count, total result size, estimated and measured tool
context, and largest single result (see [tools](#tools)).

**Resent share** is the fraction of a session's weighted cost that went on
context it had already sent: for each prompt, the input-side weighted cost of
every API turn after the first, divided by the session's total weighted cost.
On real sessions this runs 75-90%.

## Context reductions

`nenpi reductions` finds points where one thread's context shrank
sharply, and prices what that saved.

- **`compact`** — a Codex `compacted` record was written between the two
  calls. Its fields sit directly on the record (`window_number`, `window_id`,
  `previous_window_id`, `compaction_response_id`,
  `latest_token_usage_record`). The history fields alongside them (`message`,
  `replacement_history`, `guardian_history`, `retained_context`) hold prompt
  text; the tool never reads or stores them.
- **`unmarked`** — a drop with no marker. The ported `/shake` writes no event
  into Codex rollouts, so every shake lands here alongside manual context
  edits.

A reduction requires a drop of more than 30% in context between consecutive
API calls of the **same thread**, from a base of at least 20K tokens, that
still holds three calls later. Each guard exists for a reason:

- Comparisons stay inside one thread because a Codex session runs several
  concurrent subagent threads whose small contexts would otherwise read as
  drops against the root thread's.
- The drop must persist because Codex interleaves a second, smaller-context
  call stream into the root thread; those dips bounce straight back and are
  not reductions.

Savings are the removed tokens times the number of later turns in the thread,
priced at the cache-read rate (removed context would mostly have been cache
hits) with the uncached rate printed as an upper bound.

`unmarked` detection is a heuristic. **An explicit shake marker in the Codex
fork would make this exact.** It should carry `kind`, `before`, `after`,
`timestamp`, and `turn_id`.

## Roots and config.toml

A **root** is a harness home dir: `~/.claude`, `~/.codex`, or a differently
named one such as `~/.claude-arcade`. It holds the transcripts (`projects/`
for Claude, `sessions/` for Codex) and the credentials (`.claude.json` /
`.credentials.json` for Claude, `auth.json` for Codex).

Roots are resolved per harness, in this order, and each source *replaces*
rather than adds to the ones after it:

1. `--claude-root PATH` / `--codex-root PATH` (repeatable). A path ending in
   `projects` or `sessions` is still accepted, with the leaf stripped and a
   deprecation warning — before this change the flags took that leaf path,
   not the home dir.
2. `[claude].roots` / `[codex].roots` in `config.toml` (below).
3. Defaults: `[$CLAUDE_CONFIG_DIR, ~/.claude]` for Claude,
   `[$CODEX_HOME, ~/.codex]` for Codex, with the environment variable first.

At every step, missing directories are dropped silently and the list is
deduplicated; a harness that resolves to zero roots gets one warning. There is
no implicit `~/.claude*` / `~/.codex*` glob any more — a root that is not the
default location has to be named, in a flag or in `config.toml`.

`config.toml` lives at `~/.config/nenpi/config.toml` (`NENPI_CONFIG_FILE`
overrides the path):

```toml
[claude]
roots = ["~/.claude", "~/.claude-arcade"]
disabled = ["~/.claude-old"]   # written by the UI; not scanned
ignored = ["~/.claude-tmp"]    # written by the UI; not offered again

[codex]
roots = ["~/.codex", "~/.codex-arcade"]

[plan]            # optional, display only
claude = "max_20x"
codex = "pro"

[general]
ignore_unconfigured = true     # silence the unconfigured-sibling note
```

This one file is the whole store. The Textual Sources screen reads and
writes it too: adding a source appends to `roots`, switching one off moves
it to `disabled`, and removing a discovered one records it under `ignored`.
An older `~/.config/nenpi/config.json` (the UI's previous store) is imported
on first use — enabled state preserved, each source's harness re-derived
from its layout, since the Sources form used to save `~/.codex-*` roots as
Claude — and the old file is renamed to `config.json.migrated`. Nothing
reads it afterwards. `$NENPI_CONFIG`, the UI's own override, still works and
is now an alias of `$NENPI_CONFIG_FILE`; it names the TOML file, and a
`.json` value is read as the `.toml` beside it, with a warning. The import
also runs for a CLI-only user, on the first command after the upgrade. A
`config.json` that listed sources for one harness only imports as an empty
`roots = []` table for the other, which now means "scan nothing" rather than
the defaults — run `nenpi config --init` afterwards, or add that harness's
roots to the file (or delete its table) to get the defaults back.

An empty `roots` in a `[claude]`/`[codex]` table the file actually has means
"scan nothing for this harness": disabling every root in the UI keeps the
CLI away from `~/.claude` too. The defaults below apply only when the
harness has no table at all.

Writing the file (the Sources screen, or `nenpi config --init --force`)
preserves every table and key, including ones this version does not
recognise, but not comments: `--init` re-seeds `roots` and rewrites the rest
from what it parsed, so hand-written comments are lost. `--init --force`
keeps `disabled`, `ignored`, `[plan]`, `[general]` and unknown tables, and
never re-seeds a root that is listed as disabled or ignored.

A missing file falls back to the defaults above. A malformed file is a hard
error naming the file and the parse problem; an unknown key is a warning, not
an error. `[plan]` is cosmetic — it never overrides a measured tier or the
`plan_type` snapshots are grouped by; use it to label the header when nothing
has been sampled yet.

`nenpi config` prints the config file in use (or that none was found) and,
for every resolved root, its harness, path, whether it exists, and the
account it authenticates as: a label (the root's basename) and a key (Codex:
`auth.json`'s `tokens.account_id`; Claude: `.claude.json`'s
`oauthAccount.organizationUuid`, falling back to `accountUuid`, then to the
label). Tokens are never read or printed. `nenpi config --init` writes a
starter `config.toml`, seeded with every root the old glob would have found
on this host (refuses to overwrite an existing file without `--force`).

When a `~/.claude-*` or `~/.codex-*` directory that looks like a harness
home (it has `projects/` or `sessions/`) is present but in nothing's
resolved set, one line goes to stderr per run:

```
nenpi: found unconfigured harness dirs: ~/.claude-arcade, ~/.codex-arcade; run `nenpi config --init` to include them (or set [general] ignore_unconfigured = true)
```

It costs one glob per harness, is suppressed for `--json` output and for
`snapshot --stdin`, never fires for roots given with `--claude-root` /
`--codex-root`, and skips the default locations and anything listed under
`disabled` or `ignored`. `nenpi config` prints the same set as an
`unconfigured:` line (`unconfigured_roots` in `--json`).

**Breaking change:** with defaults narrowed to one location per harness, a
host that relied on the old glob picking up e.g. `~/.codex-arcade` or
`~/.claude-work` needs those roots added to `config.toml` (or run
`nenpi config --init` once, before upgrading further) — otherwise they drop
out of every report silently.

Per-account Claude statusline snapshots (`nenpi snapshot --stdin`, wired into
the statusline command) need `$CLAUDE_CONFIG_DIR` set in the statusline's own
environment, not just the shell that launches Claude — otherwise every
account's statusline reads and dedups as the default root.

## Snapshots

`nenpi snapshot` logs Claude quota observations to
`~/.local/state/nenpi/snapshots.jsonl`. Three sources:

### `--oauth` (recommended)

Samples live utilisation from the Claude usage endpoint. This is the only
source that is both current and available non-interactively: the cached copy
in `.claude.json` is stale, and the statusline only runs while someone is
using the CLI.

```
nenpi snapshot --oauth [--config-dir ~/.claude]
```

`--config-dir` is repeatable; with none given, every resolved Claude root
(see [Roots and config.toml](#roots-and-configtoml)) holding a
`.credentials.json` with a `claudeAiOauth` block is sampled. **On
macOS the CLI keeps these credentials in the login Keychain instead of on
disk**, so `--oauth` finds nothing there and says so; reading the Keychain is
deliberately not implemented. The
access token is read from that file, held in memory for the request, and never
printed, logged, or written anywhere. `expiresAt` is checked first; an expired
token is skipped with a note on stderr and no refresh is attempted.

Guards: one attempt per invocation, a 15 s timeout, a minimum of 60 s between
calls per config dir, a 10-minute backoff after an HTTP 429, and a private
opener that refuses redirects — urllib would otherwise forward the bearer
token to whatever host answered. Poll state
lives in `~/.local/state/nenpi/oauth-poll.json` and holds timestamps
only.

Stored per observation: `utilization` and `resets_at` for `five_hour`,
`seven_day`, `seven_day_opus` and `seven_day_sonnet`; the `limits[]` entries
(`kind`, `group`, `percent`, `severity`, `is_active`, `resets_at`, and the
scoped model id or display name); and `spend.percent`. Organization and
account identifiers are never stored.

To sample every five minutes, without installing anything here:

```ini
# ~/.config/systemd/user/nenpi-snapshot.service
[Unit]
Description=Sample Claude quota utilization

[Service]
Type=oneshot
ExecStart=%h/.local/bin/nenpi snapshot --oauth
```

```ini
# ~/.config/systemd/user/nenpi-snapshot.timer
[Unit]
Description=Sample Claude quota utilization every 5 minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=5min
Persistent=true

[Install]
WantedBy=timers.target
```

`systemctl --user enable --now nenpi-snapshot.timer`. The macOS
equivalent is a launchd agent in `~/Library/LaunchAgents` with
`StartInterval` set to `300` and `ProgramArguments` of the script path plus
`snapshot` and `--oauth`.

### `--stdin`

Reads the Claude statusline JSON on stdin, extracts
`rate_limits.*.used_percentage` and `resets_at`, and passes stdin through to
stdout byte for byte so it can sit inside the statusline pipeline:

```sh
... | nenpi snapshot --stdin | bash ~/.claude/statusline-command.sh
```

It never fails the pipeline: the passthrough is wrapped end to end, every
error is swallowed, and it always exits 0. It appends only when a window's
utilisation or reset time actually moved, so a statusline that runs on every
prompt does not grow the log.

`nenpi snapshot --compact` drops repeated entries and anything older
than 60 days. Reporting subcommands read only the entries inside the requested
range.

### no flag

Reads `cachedUsageUtilization` from every resolved Claude root's
`.claude.json` and appends when `fetchedAtMs` is newer than the last logged
value for that root.
`accountUuid` is never stored. This source is stale by design — it is whatever
the CLI last cached.

## Weights

`~/.config/nenpi/weights.json` overrides any built-in weight, per model,
merged over the defaults:

```json
{
  "version": 1,
  "codex": {
    "unit": "credit_units_per_mtok",
    "models": {
      "gpt-6-astra": { "input": 250, "cached_input": 25, "output": 1250 }
    }
  },
  "claude": {
    "unit": "usd_per_mtok",
    "models": {
      "claude-opus-5": {
        "input": 5.0, "cache_read": 0.5,
        "cache_write_5m": 6.25, "cache_write_1h": 10.0, "output": 25.0
      }
    }
  }
}
```

Built-in Codex weights come from the OpenAI credit rate card
(learn.chatgpt.com/docs/pricing), per million tokens:

| model | input | cached | output |
| --- | --- | --- | --- |
| gpt-6-astra | 250 | 25 | 1250 |
| gpt-5.6-sol | 100 | 10 | 500 |
| gpt-5.6-terra | 50 | 5 | 300 |
| gpt-5.6-luna | 5 | 0.5 | 30 |

`gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.3-codex` and
`gpt-5.3-codex-spark` have no published rate. They are guessed at the sol rate
and carry `"guessed": true`, which the report header names. Model strings are
lowercased, and `luna` / `gpt-luna` map to `gpt-5.6-luna`; anything else
unrecognised is left unweighted.

## Performance and state

Roughly 10 GB of rollouts. Files stream line by line with a byte-offset cache
of one JSON shard per transcript under `~/.cache/nenpi/`, keyed by path
with `size`, `mtime` and `offset`. Each shard also stores a hash of the file's
first 4 KiB and of the 256 bytes before the resume offset: size and mtime
alone miss an in-place rewrite that happens to grow the file, which would
resume mid-record and silently skip the rest. A grown file resumes from its
stored offset; a file that shrank, whose mtime moved backwards, or whose
fingerprints changed is reparsed whole; a partial trailing line is left for
the next run. A full sweep prunes shards for transcripts that no longer exist,
and superseded schema directories are removed on every run. `--since` prunes by Codex
directory date and by file mtime before anything is opened, so a narrow query
never touches the shards of files outside the window.

Sharding rather than a single index is deliberate: one index of this corpus
reaches nine figures of JSON and has to be parsed in full on every
invocation, which dominates the runtime of a narrow query.

Measured on a 2600-transcript Claude corpus (1.2 GB) and 3400 Codex
rollouts (9.1 GB):

| run | wall | peak rss |
| --- | --- | --- |
| full history, cold cache | ~80 s | 577 MB |
| full history, warm cache | ~7 s | 629 MB |
| `--since 3d`, cold cache | ~10 s | 88 MB |
| `--since 3d`, warm cache | ~1.2 s | 62 MB |

A cold full sweep is the price of corpus-wide dedup: every transcript has to
be read once before a replayed API call can be told from a new one.

Only the structural fields are read: `type`, `message.usage`, `message.id`,
`message.model`, timestamps, ids, `cwd`, the Codex `rate_limits` and
`turn_id`, and — for tool accounting — `tool_use.name` / `function_call.name`
(MCP names kept whole, Codex namespaces qualified as `namespace.name`) with
the character SIZE of each `tool_result` / `function_call_output`. Tool result
text and message content are never parsed into anything stored or printed: a
result is measured and dropped.

One deliberate exception: each prompt's **label** is stored in its shard and
printed by `prompts`. The label is the first line the person typed, and
nothing else — the rest of the prompt is discarded before anything is kept:

- Injected blocks are stripped, whether they close on the same line or span
  several. The list is fixed: `system-reminder`, `user_instructions`,
  `environment_context`, `recommended_plugins`, `pasted_content`,
  `command-name`, `command-message`, `command-args`,
  `local-command-stdout`, `local-command-stderr`, `local-command-caveat`,
  `ide_selection`, `ide_opened_file`, `task-notification`,
  `cross-session-message`. Anything else in angle brackets is something the
  person typed and is kept, and a self-closing `<tag …/>` is dropped while
  the typed text beside it stays.
- A **pasted-content placeholder** (`[Pasted text …]`, `[Image #1]`) ends the
  scan. The lines after it are the paste, and the paste is never a label.
- A turn that is *only* an injected block — a task notification, a bare
  slash-command expansion — is labelled with the block's name in
  parentheses, e.g. `(task-notification)`. The name comes from the list
  above, so such a label is a fixed vocabulary and carries nothing from
  inside the block. Only a closing tag at the start of a line ends a block,
  so a body that quotes its own closing tag cannot hand back a line; a block
  that never closes yields its name and nothing else.
- Whitespace is collapsed and the line is cut to 120 characters.
- These become `[redacted]` first: email addresses, `sk-…`,
  `ghp_`/`gho_`/`github_pat_…`, `xox…` and `AKIA…` keys, JWTs, `Bearer …`
  values, hex runs of 32 characters or more, and base64-looking runs of 40 or
  more. The runs are matched with lookarounds rather than word boundaries, so
  `api_key_<hex>` is caught too.

`--no-label` hides the column, and nothing longer than the label ever reaches
the cache; delete `~/.cache/nenpi/` to drop the labels already stored.

Claude user lines carrying tool results still bypass the prompt path on the
raw-bytes screen; they now go through the size-only tool parser instead of
being skipped.

Reading those lines costs something: on a 14-day sweep of a real corpus the
cold parse went from ~24 s to ~32 s and the cache from 34 MB to 43 MB (a
later pass trimmed that back to ~24 s: only `"tool_result"` admits a line,
and a structured result is sized by walking its strings rather than by
re-serializing it, which under-counts JSON punctuation by a fraction of a
percent). Warm runs are unchanged. An issued call is forgotten once the next
user turn starts in that file, so a result can only be named by a call of
its own turn.

Labels add two more Codex line kinds to the screen (`"role":"user"` and
`"task_complete"`, the latter bounding how long a user message can wait for
the prompt it opens), worth about 12% on a cold Codex-only scan.

State lives in `~/.local/state/nenpi/` (`snapshots.jsonl`,
`codex-weights.json`, `oauth-poll.json`), config in
`~/.config/nenpi/` (`weights.json`, `config.toml`), cache in
`~/.cache/nenpi/`. The `NENPI_HOME_DIR`, `NENPI_CACHE_DIR`,
`NENPI_STATE_DIR` and `NENPI_CONFIG_DIR` environment variables relocate all
four for tests; `NENPI_CONFIG_FILE` relocates `config.toml` on its own. The
old `~/.cache/quota-drain` (and the matching state/config dirs) and
`QUOTA_DRAIN_*` names still work for one release: on first run, an old
default dir is moved to its new name if the new one does not already exist,
and each `QUOTA_DRAIN_*` variable still honoured prints one deprecation
warning.

## What is known

**OFFICIAL.** Claude has a 5-hour rolling window and a weekly window; Max 5x
and 20x are multipliers on Pro; usage depends on model, length, effort and
features; extra-usage overflow bills at API rates. Codex has 5-hour primary
and weekly windows; usage depends on model, context, reasoning, tool use and
caching; the per-model credit rate card is above; Fast mode costs 2.5x on
Astra.

**COMMUNITY.** Claude quota approximates API cost. Cache reads are reported to
count against Claude Code quota by more than the API's 10%
(anthropics/claude-code issue #24147); the default here is the list price and
`--claude-cache-read-weight` overrides it, with `calibrate --harness claude`
reporting the measured ratio once snapshots exist. Opus and Sonnet weekly caps
are separate.

**UNKNOWN.** The exact Claude formula, and the dollar ceiling per tier — every
account seen so far reports `limit_dollars` and `used_dollars` as null. Any
long-context multiplier for either vendor on subscription plans: Anthropic
removed the API 1M premium on 2026-03-13 and OpenAI publishes none, so none is
implemented; `--long-context-multiplier` exists as a no-op-by-default knob
applied above 200K tokens so it can be tested later.

## Known limitations

Four known-wrong behaviours, none of them blocking, each with the fix it
wants:

- **The attribution cap's rate is per account, not per model or session.** A
  pool that mixes a cheap, chatty model with an expensive, quiet one measures
  one blended rate; a session on the expensive model in a sparse interval
  could still be capped tighter than its real cost. The fix is a per-model
  rate, which needs enough non-rollover coverage per model to be worth
  fitting - most pools do not have it yet.
- **An interval that straddles `--since` keeps its whole drain, but only the
  in-range events share it.** The interval's start is clamped to the range
  while its measured percent is not, so at most one interval per window group
  is over-attributed to the range. The fix is to attribute against unwindowed
  events and then report only the in-range sessions' shares.
- **Fork dedup picks the winner by file mtime.** A transcript restored from
  backup, or otherwise touched after its fork was written, loses its own API
  calls to the fork. The fix is to order by earliest line timestamp and fall
  back to mtime only on a tie.
- **The Codex cumulative fallback path has no dedup key.** Rollouts too old to
  carry `token_usage_record` are read by diffing
  `event_msg token_count info.total_token_usage`, which yields no
  `response_id`, so those events cannot be deduped corpus-wide. A resumed old
  rollout double-counts. The fix is a synthetic key from thread id and
  cumulative totals.

## Limits

- Fast mode is not detectable from transcripts, so Astra's 2.5x is not applied.
- Bars use Unicode block glyphs and fall back to ASCII when stdout is not
  UTF-8 or under `--ascii`.
- Codex `used_percent` arrives in whole percent, which caps how finely drain
  can be attributed; calibration needs multi-day ranges and time bucketing
  before it means anything.
- Quota timelines are keyed by account first (root's detected `auth.json`
  `tokens.account_id` / `.claude.json` `oauthAccount.organizationUuid`,
  falling back to the root's basename), then `limit_id`/`plan_type`/
  `window_minutes`; two roots on the same account still merge into one pool,
  which is correct.
- `unmarked` reductions are heuristic until the Codex fork writes a shake
  marker.
- A green parse is not proof of a correct model: `nenpi verify` compares
  the parse against the harnesses' own totals, and `calibrate` compares the
  weights against measured drain. Use both before trusting a number.

## Verification

```sh
uv run --python 3.11 python -m unittest discover -s tests -v  # 3.11 floor
uv run --python 3.13 python -m unittest discover -s tests -v
```

Fixtures are synthetic and must stay that way. Real transcripts hold prompts
and customer data and are never copied into this repository.
