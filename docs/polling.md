# Polling and recipe classification

nenpi owns one classification of repeated model checks and hand-driven
command recipes. Two consumers share it:

- `nenpi polling` and the `polling` section of `nenpi activities --json`
  replay it over local transcripts and attach cost.
- management-plane `scripts/burn-governor` vendors
  `tests/fixtures/command-classification.json` and reimplements these rules
  without importing nenpi.

The fixture is the contract. It asserts classification only. Signature
strings, hashes, and the offline mechanical-run score are internal to nenpi
and can change without breaking burn-governor.

## Fixture schema

```json
{
  "defaults": {"threshold": 3, "window_steps": 30},
  "cases": [{
    "name": "...",
    "options": {"threshold": 2},
    "steps": [
      {"tool": "Bash", "command": "gh pr checks 12", "cwd": "/repo", "result": "pending"},
      {"tool": "write_stdin", "input": {"session_id": 7, "chars": ""}},
      {"tool": "Edit", "input": {"file_path": "/repo/a.py"}}
    ],
    "expect": [{"polling": false, "kind": "none", "recipe_id": null}, "..."]
  }]
}
```

Each case is one thread within one prompt, in order. A step is a shell step
(`command`, `cwd`) or a tool step (`tool`, `input`). `result` is optional
tool-result text; absent means unknown. `expect` has one entry per step.
`polling` is true exactly when `kind` is not `none`. `recipe_id` is one of
`wait-for-status`, `merge-pr-when-green`, `query-logs`, or null.

## Rules

1. **Judged before the step runs.** A step's verdict uses only earlier steps.
   Its own `result` matters only to later steps, as a PreToolUse hook sees it.
2. **Window.** Up to the `window_steps` most recent earlier steps since the
   last write. Writes are the tools `Write`, `Edit`, `MultiEdit`,
   `NotebookEdit`, `apply_patch` (case-insensitive; also `multi_edit`,
   `patch`). A write is never flagged and empties the window. The offline
   report also starts a new window at each prompt and in each thread.
3. **Exempt steps are never flagged and never counted toward a recipe.** Exempt
   steps are:
   - a shell statement whose executable basename is `wait-for-status`,
     `merge-pr-when-green`, or `query-logs`;
   - any statement with a `--watch` word, `gh run watch`, or the `watch`
     command;
   - a script that owns its own sleeps: a statement starting with `while`,
     `until`, `for`, or `select`, or any command containing a heredoc (`<<`).
4. **Shell signature.** Split the command on `;`, `&`, `|`, and newlines outside
   quotes. `>&` and `&>` are redirections, not separators. In each statement:
   - Strip leading `rtk`, `rtk proxy`, `command`, `env`, `NAME=value`, and
     `sudo` with its leading `-flags`.
   - Unwrap `sh|bash|zsh -c|-lc STRING`.
   - Reduce the executable to its basename.
   - Drop `cd DIR` statements and join DIR onto the working directory.
   - Drop `sleep N` statements. A command of only sleeps has the signature
     `sleep`.
   - For `gh pr checks|view` and `gh run view`, drop `--json`, `--jq`, `-q`,
     `--template`, and `-t` with their values, plus `--compact` and
     `--exit-status`.

   The signature is the effective working directory plus the remaining word
   lists.
5. **Tool signature.** The lowercased tool leaf name (after the last `.` or
   `__`) plus the input object without `description`, `timeout`, `timeout_ms`,
   `yield_time_ms`, or `max_output_tokens`. `write_stdin` with non-empty
   `chars` sends input and has no signature.
6. **Polling.** A step is polling when its signature appears in the window at
   least `threshold - 1` times. Compare the results of the most recent
   `max(threshold - 1, 2)` same-signature window steps. The kind is
   `pure_poll` when there are at least two results, all are known, and all are
   equal. Otherwise it is `watch`. Before comparing, mask each run of six or
   more hex characters that contains a digit, then each run of digits, as `0`.
   Collapse whitespace. Elapsed times, chunk IDs, and wall times therefore do
   not count as change.
7. **Recipes** look at the first statement left after rule 4:
   - `wait-for-status`: `gh pr checks`, `gh pr view`, or `gh run view`. Its
     source is the `-R/--repo` value, `pr` or `run`, and the first positional
     target. The recipe applies when the window holds at least
     `threshold - 1` steps with the same source.
   - `query-logs`: `journalctl`, `docker ... logs`, `docker-compose logs`, or
     `tail` of a path containing `log`. The journal source is
     `-u/--unit/--user-unit/-t/--identifier` values plus `--user`. The Docker
     source is the last positional after `logs`. The tail source is the path.
     The recipe uses the same-source count as `wait-for-status`.
   - `merge-pr-when-green`: `gh pr merge`, `gh api` on
     `.../pulls|issues/N/comments|reviews`, or `gh api graphql` with
     `resolveReviewThread`. The recipe applies when the window holds an
     earlier `wait-for-status` or `merge-pr-when-green` family step.

   A recipe can apply without polling, for example when flags change between
   checks of the same PR.

## Calibration report

```sh
nenpi polling --since 7d            # small human summary
nenpi polling --since 7d --json     # adds sessions -> prompts, threshold sweep, mechanical runs
```

The report answers: what would the advisory have flagged, and what did it cost?
The cost of a flag is the input tokens and weighted units of the model
response that issued the flagged call. A response with several flagged calls
counts once.

The report includes totals by kind, recipe, and harness; top patterns by
input; and a threshold sweep (2–5). Patterns use the command family and
subcommand, or the activity for non-shell tools. It also simulates one
advisory per `(session, pattern)`: flags per pattern, the time between
repeated flags, and the cost of flags after the first.

`mechanical_runs` is offline discovery only. A run is a sequence of
small-result responses without writes, scored by length and distinct-command
ratio. It finds candidate new recipes and is not part of the fixture or hook
contract.

Reports and caches hold only command-family labels, opaque hashes, and
counts. They never hold command arguments or result text.
