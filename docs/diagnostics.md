# Usage diagnostics

Run `nenpi` for activity usage in the last **72 hours**, across
configured Claude and Codex roots. It ranks by **input tokens processed** and
shows usage and repeated status checks or identical reads in two compact tables. Model
responses count once, including responses containing several tool calls.
Different harness/account weights are kept separate in JSON.

```sh
nenpi
nenpi --since 7d
nenpi --project ~/project-dir
nenpi --exclude-project ~/project-dir
nenpi activities --activity 'Waiting for CI'
nenpi activities --json
```

The report includes runnable commands to inspect an activity. Use `nenpi sessions` for a session overview.
Activities combine across projects; operation identities remain scoped to a
thread and prompt. A loop inside one tool call is one model response. Tiny
results alone do not establish repetition. Literal shell commands and native
tools are recognized; dynamic code and mixed work remain visible as such.
Read repetition resets after operations that may change files. Repetition counts
input on repeated operations; it does not assume every repeated call is removable.
`Other / unknown` means unrecognized tools or commands. `No linked tool call`
means the response has no matched tool record, including text-only responses.

## Project filters

Project filters work on all report commands and carry through drilldown links.
A directory selects itself and its descendants; `--project` can be repeated for
alternatives, and `--exclude-project` always wins. A bare name matches a project
basename; quoted globs such as `--project '*/project-dir*'` are also supported.
Full paths are read from transcript metadata without adding them to the cache.
Quota attribution is calculated before project selection, so filtering does not
assign the rest of an account's usage to the selected project.

For example:

```sh
nenpi sessions --harness all --since 7d --top 25
nenpi prompts --since 7d --sort turns --top 20
nenpi prompts --session 0123abcd
nenpi prompts --harness claude --session 0123abcd --prompt 1 --drilldown
nenpi tools --since 7d --top 15
```

## Session diagnostics

Use `nenpi auto --session ID` for **session diagnostics and Shake payback**.
It examines the last seven days of Codex activity and shows the five prompts
with the most input processed, including cached reads:

```sh
nenpi auto
nenpi auto --session 0123abcd --prompt 2
nenpi auto --color always         # force colors, including in captured output
nenpi auto --json                 # structured findings, without ANSI escapes
```

The diagnosis connects response count, context size, cached/uncached input,
model output and tool-result size. For example, 400 responses reading 100K
tokens each process 40M input tokens even if every tool result says only `ok`.
Small-result findings show the cumulative input on those responses, the amount
of preceding tool output, and the longest uninterrupted run within one thread.
Root and subagent usage stay separate and reconcile to the combined total.

See [Themes](themes.md) for colors and terminal output options.

`auto` ranks observed input burden, not a claimed percentage of avoidable
quota. Unknown model weights do not hide a large workload. Small-result
associations require strictly ordered result/usage timestamps in the same
thread and prompt; ambiguous timestamps are excluded from that diagnostic.
Use the suggested `tools --explain` command to inspect the activities behind
a finding, or `prompts --drilldown` to inspect its agents.

## Hypothetical Shake savings

Both the activity report and `auto --session` also show **Estimated Shake
savings** where no Shake ran in that thread/prompt, including Claude sessions.
The `old-tool-outputs-v1` policy replaces older tool outputs of at least 512
estimated tokens with 120-token stubs and protects the latest 16K tokens.
It selects the best single intervention per thread/prompt from the observed
calls. A context decrease or prompt boundary ends the projection. Only outputs
seen entering context within the selected range are considered.

The estimate charges for rewriting the remaining cached suffix after the first
changed output, preserving the prefix. Claude uses its cache creation rates;
unknown duration uses the more expensive one-hour rate. A request with no cache
reads uses a cold baseline: immediate avoided input/cache creation, no rewrite
penalty. Savings are cached-read-equivalent tokens. These estimates and savings
from eliminating repeated calls **are not additive**. This simulates a defined
policy, not Claude support for running `/shake`.

Activity metadata caches contain only classifications and opaque operation
identities, never commands or result bodies. Source changes invalidate them.
Harness parsing lives in `activity_model.py`; repetition and activity grouping
consume its normalized responses in `activity_report.py`, while
`context_savings.py` owns the shared hypothetical policy. The existing applied
Shake evidence and payback table remain available in `auto --session`.

## Applied Shake

`auto` also recognizes Shake in the `epsalmond/codex` fork. Applied markers
are compared with surrounding same-thread requests; optional read-only
`logs_2.sqlite` stats add Shake's own estimated tokens freed and transform
counts. The report keeps that estimate separate from observed context and
cached/uncached input. A smaller post-Shake context can initially cost more
uncached input while the cache rebuilds. Missing telemetry, prompt changes,
model changes, and several Shakes between requests limit the comparison.

`Tiny output → large read` counts model calls following a small tool-result
batch while processing a large context. It is a size-based signal, not proof
of polling or repeated commands; `tools --explain` provides the activity breakdown.
The Shake table contains runs from the selected prompt's main agent and
subagents. The column labels identify which; repeated thread IDs mean several
runs on the same agent. There is one row per Shake run.
`Post-Shake Turns` counts requests after that Shake, including the first, until the next
Shake or prompt end. `Context` shows before → after with the reduction in parentheses; `Cache write*` shows the extra uncached input
on the first later request. `Break-even turn*` uses the selected model's input weights:

- Per-call benefit = context reduction × cached-input weight.
- Initial rebuild premium = extra uncached input × (input weight − cached-input weight).
- Break-even call = ceiling(rebuild premium / per-call benefit), minimum 1.

`Saved / cost*` is net cached-read-equivalent tokens: reduction × turns minus
the rebuild premium converted at the cached-input weight. Positive values are
green; negative costs are red with a minus sign. The agent column includes its
short thread hash followed by its recorded nickname, with a role label when no name exists.
Rows sort by thread ID, then chronologically within each thread. Post-Shake
turns are red when they fall short of break-even. An asterisk marks modeled columns; cache write here means inferred extra
uncached input, not a separately reported cache-write counter.

This estimate holds the context reduction constant, treats avoided reads as
cached, and charges one initial rebuild. Generated output is held unchanged.
It does not project a fixed context size: later context can grow equally in
both scenarios. Further cache misses, compaction, or restoring removed content
can change the result. Missing weights, model/prompt changes, or ambiguous
markers leave break-even unavailable. Cold-resume Shakes use a presumed-expired baseline: the first avoided read is
credited at the uncached-input weight and subsequent reads at the cached-input
weight. The survivor rebuild would happen either way, so no incremental rewrite
penalty is charged and break-even is turn 1 (cold). The observed extra uncached
input remains visible in the cache-write column. Expiry is inferred from the
trigger, not proven by the provider; the score remains an estimate. Full measurements, assumptions and the
estimated net input-weighted benefit remain in `--json` under `shake.events[].payback`.

For a thread that still carries a large context, the report can suggest
reviewing `/shake` when continuing it. That command previews the change in
the affected Codex thread; Nenpi never executes it. No applied marker does
not prove Shake was never run: a preview or no-op may leave no such marker.
The structured `shake` report under `--json` includes all observed events
and coverage details.

## Prompt and agent drilldowns

`nenpi prompts` without `--session` ranks every prompt in range by API turns
(or context, drain, tokens, units) across sessions, each row labelled with
the first line the person typed, cut to 120 characters and scrubbed of email
addresses and secret-shaped tokens. Injected blocks and pasted content never
become a label — a turn that is only an injected block is named, not quoted.
That short **label** is the only prompt text nenpi stores or prints;
`--no-label` hides it. With `--session` it is the per-prompt breakdown of one
session, unchanged.

Add `--prompt N --drilldown` to inspect one prompt's root, child-agent, and
unknown-thread totals for either harness. Claude child lineage uses exact
`toolUseId` matches from agent sidecars; missing or unmatched lineage stays
unknown. The report includes disjoint cache-read and cache-write token counts,
tool-use sizes and result sizes, and only explicit Claude `TaskOutput` waits.

## Tool explanations

`nenpi tools` ranks tool calls by the context their results add, estimated as
result characters / 4 and, where a later API call measured the growth, split
across that turn's results. By default, it reports only tool **names** and
result **sizes**. To investigate repeated model work, use:

```sh
nenpi tools --session 0123abcd --prompt 2 --since 7d --explain
```

`--explain` rereads live transcripts to unpack Codex `exec` code and shell
commands, with bounded command examples. The Codex activity ranking shows
tool calls, generating model responses, root/subagent call counts, and
uncached input, cached input, and output tokens. Outer tool-family totals
contain their nested activities, with the same usage denominator and tool
result sizes alongside each. It ranks by weighted usage
of the matched responses, counting each response once even when it generated
multiple tool calls. This exposes work that adds little tool output but
repeatedly sends a large context to the model, such as polling or agent messages.

These are associations, not guaranteed savings or measured quota percentages.
The entire response includes reasoning and other actions; its cost cannot be
assigned exactly to an individual tool. Static code inspection cannot resolve
dynamic commands or tell how many times a loop ran. Missing response IDs remain
unpriced. Use `--top 0` for every activity group, or `--json` for the full
structured `activity` report. Tool input and output text are never cached;
command examples appear only with `--explain`.

Activity output keeps total input and repeated input separate. `--activity Mixed`
and `--activity Other` show operation groups; each response belongs to one group,
so their input totals do not overlap. Activity drilldowns put repeated input
first, then total input. The default shows one positive hypothetical Shake
opportunity; full applied-Shake evidence remains in `auto --session`.

[Basic usage](../README.md)
