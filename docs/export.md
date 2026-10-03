# Native analytics export

Run `nenpi export --identity-map identities.json`. The source mapping belongs
to the consumer; Nenpi never derives exported identity from credentials:

```json
{"sources":[{"harness":"codex","source_root":"/home/example/.codex","provider":"openai","account_alias":"personal"}]}
```

Labels allow letters, digits, dots, underscores and dashes, at most 48 characters.
The longest matching source root wins. Equal-length conflicting mappings yield
`identity_status=ambiguous`; unmapped identity resolution yields `unknown`.
Only mapped roots are scanned. Map every account separately; different accounts
with the same alias are explicitly treated as one pool. No account key, prompt
label, working directory, source path, command, tool argument, result body,
target hash or classifier signature enters the output.

The JSONL schema is version 1. `analytics_prompt`, `analytics_partition` and
`analytics_session` carry stable `record_id`, monotonic `revision`, source and
observation timestamps, configured provider/account identity and native session
IDs. Prompt IDs use a native turn ID or human boundary timestamp rather than
the display index. Calls lacking a boundary/turn occupy an unknown prompt
bucket and are excluded from the human-prompt denominator. Claude descendant
membership remains temporal; it is not proof that a late child was requested
by the current human prompt. Full native lineage IDs and allowlisted native
Claude agent types are exported. An implementation role is known only when
the native type explicitly says `implementation`.

Native token kinds stay separate: Claude input, cache read, write at five-minute
TTL, write at one-hour TTL, write with unknown TTL, and output; Codex uncached
input, cached input, cache write and output. `token_kinds_known` distinguishes
an absent measurement from measured zero. Reasoning is an output subset with
its own known-turn count. Tool-result estimates are not generated output.
Each partition counts a whole generating response once: a single linked tool
or one combined family batch. Unlinked responses retain `tool_association=unknown`.

Classification uses the existing closed polling/recipe contract, threshold 3
and window 30. Categories `pure_poll`, `watch`, `recipe`, `pure_poll_recipe`
and `watch_recipe` make overlaps explicit. `unflagged` only means this classifier
did not flag linked operations. Missing operation joins are `unknown`.

Delivery is at least once. The final `analytics_export_batch` names an immutable
pending page. Accept and flush **all** records, then run:

```sh
nenpi export --identity-map identities.json --ack BATCH_ID
```

Until acknowledged, every invocation returns the same page without scanning.
Replayed records must be deduplicated by `(record_id, revision)`. Consumers
select the latest snapshot, including tombstones, and never sum revisions.
An idle prompt is never permanently finalized: resumes, sidecar edits and late
children revise existing logical records. Updated prompts reemit all current
partitions and zero-valued tombstones for disappeared partitions. Journal
flushing is best effort; it is not a Loki acknowledgement.

Defaults are a 24-hour initial source-mtime backfill, 10,000 discovered files,
64 MiB of incremental native parsing, 64 MiB of operation/result classification
reads, 90 seconds before beginning another scan unit, 4,000 records and 8 MiB
per delivery page. Native parse offsets persist when a scan budget expires;
the next scheduled invocation continues. A single JSONL record larger than
the parse budget requires increasing `--max-scan-bytes`. Large classification
sources retain unknown association rather than exceeding the read budget.
Consumers should impose an outer process timeout; it also bounds the current
parse/analysis unit. Projection and pending-page state lives in a private
SQLite store under `~/.local/state/nenpi/analytics.sqlite`.

`analytics_export_health` reports complete/incomplete scan coverage when no
batch is produced; pending pages also carry coverage on their final marker.
Source signatures skip unchanged projection work. Native shard schema 10 and
operation cache schema 7 preserve token presence and bounded tool families;
old shards rebuild on the next read. The cache and export do not change quota
weights, calibration, statusline sampling or the burn governor.
