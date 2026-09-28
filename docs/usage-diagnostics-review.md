# Usage diagnostics review

Three Luna/max reviewers examined accounting, CLI/product behavior, and
efficiency/privacy/testing. The review centered on reducing subscription usage:
small tool results must not hide hundreds of model responses processing a large
retained context.

## Findings and changes

- **Result size is only one input to usage.** The former ranking emphasized
  added context and its positive growth. `auto` now reports response count,
  average context, cumulative input, cached/uncached input, and model output.
  It separately identifies responses following small tool-result batches.
- **Generating and consuming responses differ.** `tools --explain` describes
  the response that generated a call. `auto` associates a preceding result
  with the next same-thread response. The reports label these relationships
  and do not assign an exact causal cost to a tool.
- **Two unrelated rankings caused confusion.** Explained tool reports now
  show outer tool-family usage followed by the activities within each family.
  They use the same denominator and preserve result size alongside usage.
- **Incomplete telemetry must stay incomplete.** Missing response IDs and
  interrupted source batches remain unpriced; timestamp ties and prompt
  boundaries are excluded from small-result associations. Unknown model
  weights remain visible and do not remove a large input workload from `auto`.
- **The default journey required too many commands.** `nenpi auto` starts with
  recent Codex activity and provides scoped follow-up commands. Semantic color
  distinguishes input, repeated work, model output, and next actions. Commands
  remain copyable; plain, ASCII, narrow-terminal, and JSON output are supported.
- **Shake estimates and observed effects differ.** Persisted applied markers
  are compared with surrounding same-thread usage. Optional read-only local
  SQLite stats supply the transform's own estimates. Cache rebuilding, model
  changes, multiple Shakes, missing telemetry, and prompt boundaries limit
  conclusions. A suggested `/shake` previews the affected thread's context;
  Nenpi does not run it or claim it can recover historical usage.

## Acceptance evidence

Synthetic tests cover a 400-response loop processing 40M input tokens while
its tool results contain only 800 characters; a single large result does not
hide it. Additional cases cover shared generating responses, root/subagent
reconciliation, tool-free responses, prompt boundaries, timestamp ties,
missing weights, warm caches, bounded examples, and scope preservation.

Shake fixtures cover applied markers and structured stats separately from
before/after usage, including adverse cache effects and unknown outcomes.
All fixtures are synthetic; real transcript contents are not stored in this
repository. Live checks use Eric's selected session to verify that the report
finds the repeated-input pattern and the fork's actual emitted Shake records.

## Limits

Cumulative input measures processing volume, including cached reads; it is
not a quota percentage or a count of identical bytes. Weighted usage depends
on available model weights. Small-result associations use strict timestamp
ordering, not provider-side causal tracing. Static JavaScript inspection
cannot prove how often nested code ran. Before/after Shake observations do
not establish net quota savings or whether removed history was useful.

Shake inspection makes a second read of selected transcripts. Header parsing
is limited to 64 KiB, but Python still materializes each JSONL line before
slicing it; unusually large compacted-history records can increase memory and
read time. Collecting marker metadata during the main scan is a future
performance improvement. A proposed ambiguity for distinct same-time Shake
markers was checked with a regression fixture: both are counted and their
joint context change remains unattributed.

## Activity-first default

Bare `nenpi` now runs the 72-hour activity overview. `activity_model.py` adapts
Claude and Codex tool records into a common response/operation model;
`activity_report.py` groups and ranks these records by input processed. Weighted
usage remains partitioned by harness/account. The repetition estimate includes
only responses consisting entirely of checks of previously seen literal targets
or identical reads within the same thread/prompt. Potential file changes reset
read repetition. Agent coordination resets the agent-wait history.
Mixed responses have one accounting bucket, so their costs are never duplicated.

A source-signature cache stores classifications and opaque target fingerprints,
not commands or output bodies. Literal shell wrappers and simple command chains
are supported; dynamic arguments remain unknown. The largest repetition finding
is highlighted even when its activity falls below the five largest rows.

`context_savings.py` supplies the shared hypothetical reduction policy for both
harnesses. Applied Shake retains its existing observed-context payback report.
Hypothetical reductions preserve a recent tail, account for the remaining cached
suffix and harness-specific write rates, and stop at a context decrease or prompt
boundary. Estimates select one intervention per thread/prompt and do not add to
repetition savings. Synthetic tests cover both adapters, cache invalidation,
thread/prompt separation, mixed responses, cold/warm costs, and Claude sessions
without Shake markers.

Read-only validation against all four NAS transcript roots found approximately
305M input tokens associated with repeated checks over 72 hours. The tested
snapshot included 74 Codex and 182 Claude hypothetical-reduction candidates.
A fresh isolated cache took about 57 seconds; its subsequent run took 7 seconds.
These are snapshot measurements, not fixed expected counts. No remote installation
or harness behavior changes were made.

## Compact report review

A Sol/high review identified repeated-input drilldowns sorted by total input,
opaque activity buckets, and redundant default output. Drilldowns now rank
repeated input first. Mixed and unknown activity expose exclusive operation
groups on demand, with totals reconciled against the parent bucket. The default
retains usage and repetition tables plus one positive hypothetical Shake result;
a representative snapshot fell from 38 to 20 lines. Include/exclude project
filters preserve attribution and propagate to follow-up commands.

The resulting wheel was installed locally and on NAS. Live NAS checks verified
project-filter partitions, operation-group totals, and the installed renderer.
