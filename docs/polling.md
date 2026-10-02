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
(`command`, `cwd`) or a tool step (`tool`, `input`). A Codex code-mode call
is the tool step `exec` with its code in `input.code`. `result` is optional
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
3. **Exempt steps are never flagged and never counted toward a recipe.** A
   step is exempt when any of its statements (after rule 4's stripping and
   unwrapping) is:
   - a recommended script: executable basename `wait-for-status`,
     `merge-pr-when-green`, or `query-logs`, however it is launched
     (`bash scripts/wait-for-status pr 45`, `python3 scripts/query-logs ...`,
     `nohup scripts/wait-for-status ... &`,
     `if ...; then scripts/merge-pr-when-green 45; fi`);
   - a watcher: any statement with a `--watch` word, `gh run watch`, or the
     `watch` command;
   - a script that owns its own sleeps: a statement starting with `while`,
     `until`, `for`, or `select`, or any command containing a heredoc (`<<`).
     This includes loops behind wrappers, subshells, and `sh -c`, such as
     `timeout 600 bash -c 'until gh pr checks 45; do sleep 10; done'` and
     `(while ! curl ...; do sleep 2; done)`. Sleeping inside a script or loop
     is the recommended shape.
4. **Shell signature.** Split the command on `;`, `&`, `|`, and newlines outside
   quotes. `>&` and `&>` are redirections, not separators. Split each
   statement into words with POSIX shell quoting (Python `shlex.split`); a
   command that does not parse has no signature. In each statement:
   - Repeatedly strip from the front:
     - the shell words `(`, `{`, `!`, `if`, `elif`, `then`, `else`, `do`, and a
       `(` or `{` glued to the first word (`(while` becomes `while`);
     - `rtk` and `rtk proxy`;
     - `NAME=value` words;
     - a wrapper command and its leading `-flags` (stopping after `--`). A
       flag in the wrapper's value-flag list also drops the next word.
       `timeout` also drops its DURATION word after the flags.

       | wrapper | flags that take a value |
       |---|---|
       | `command`, `nohup` | none |
       | `env` | `-u --unset -C --chdir` |
       | `sudo` | `-u --user -g --group -p --prompt -C --close-from -D --chdir -r --role -t --type -U --other-user` |
       | `timeout` | `-s --signal -k --kill-after` |
       | `nice` | `-n --adjustment` |
       | `time` | `-f --format -o --output` |
   - Drop a statement made only of `)`, `}`, `fi`, `done`, or `esac`.
   - Unwrap script runners. For `sh`, `bash`, or `zsh`, skip leading flags; a
     flag matching `-[a-z]*c` (`-c`, `-lc`, `-ec`) makes the next word a
     command string, which is split and processed as statements in place of
     this one. Otherwise, for `sh|bash|zsh|python|python3`, skip leading flags
     other than `-`, `-m`, and `-c`; if the next word is not a flag, the
     statement becomes that FILE and its arguments, stripped again as above.
     `python3 -c`, `python3 -m`, and `python3 -` are not unwrapped.
   - Reduce the executable to its basename.
   - Drop `cd DIR` statements and join DIR onto the working directory.
   - Drop `sleep N` statements. A command of only sleeps has the signature
     `sleep`.
   - For a `wait-for-status` family statement (rule 7), drop `--json`,
     `--jq`, `-q`, `--template`, and `-t` with their values, plus `--compact`
     and `--exit-status`. A PR read keeps them.

   The signature is the effective working directory plus the remaining word
   lists.
5. **Tool signature.** The lowercased tool leaf name (after the last `.` or
   `__`) plus the input object without `description`, `timeout`, `timeout_ms`,
   `yield_time_ms`, or `max_output_tokens`. `write_stdin` with non-empty
   `chars` sends input and has no signature. A call is always one step: a Codex
   code-mode `exec` call is one tool step even when its code calls several
   tools, so repeats inside one call never flag that call. A hook may sign an
   `exec` call by its input like any other tool. nenpi's offline report reads
   the tools the code calls and joins their signatures in order; the call is
   a write or exempt if any part is, and takes the recipe family of the first
   part that has one.
6. **Polling.** A step is polling when its signature appears in the window at
   least `threshold - 1` times. Compare the results of the most recent
   `max(threshold - 1, 2)` same-signature window steps. The kind is
   `pure_poll` when there are at least two results, all are known, and all are
   equal. Otherwise it is `watch`. Before comparing, mask each run of six or
   more hex characters that contains a digit, then each run of digits, as `0`.
   Collapse whitespace. Elapsed times, chunk IDs, and wall times therefore do
   not count as change.
7. **Recipes** look at the first statement left after rule 4:
   - `wait-for-status`: a status check, which is `gh pr checks`, `gh run
     view`, or `gh pr view` that is not a PR read. A PR read is `gh pr view`
     with `--comments`, `-c`, or a `--json` field (comma-separated, also
     `--json=...`) outside the status fields: `state`, `statusCheckRollup`,
     `mergeable`, `mergeStateStatus`, `reviewDecision`, `isDraft`,
     `mergedAt`, `mergedBy`, `closed`, `closedAt`, `autoMergeRequest`,
     `headRefOid`, `headRefName`, `baseRefName`, `number`, `url`, `id`.
     Reading a PR's body, files, or comments during review is not a status
     check. The source is the `-R/--repo` value (also `--repo=`), `pr` or
     `run`, and the first positional after the group and subcommand. A word
     in the gh value-flag list drops the next word before positionals are
     taken: `--json --jq -q --template -t -R --repo -X --method -H --header
     -f -F --field --raw-field --input -L --limit -b --body -B --base -s
     --state -A --author --workflow --branch -j --job -a --attempt -i
     --interval`. `-w` is `--web` on `gh pr view|checks` and `gh run view`
     and takes no value. The recipe applies when the window holds at least
     `threshold - 1` steps with the same source.
   - `query-logs`: `journalctl`, `docker ... logs`, `docker-compose logs`, or
     `tail` of a log path. The journal source is
     `-u/--unit/--user-unit/-t/--identifier` values (also `--unit=`) plus
     `--user`. The Docker source is the last positional after `logs`, skipping
     `--since --until --tail -n --index` and their values, so
     `docker logs web --tail 50` targets `web`. The tail source is the last
     positional, skipping `-n -c --lines --bytes -s --sleep-interval --pid`
     and their values; it is a log path when it matches the regular
     expression `(^|/)logs?/|\.log(\.[0-9]+)?$` ignoring case (`app.log`,
     `/var/log/syslog`, `logs/worker.out`, not `CHANGELOG.md`). The recipe
     uses the same-source count as `wait-for-status`.
   - `merge-pr-when-green`: `gh pr merge`, `gh api` on
     `.../pulls|issues/N/comments|reviews`, or `gh api graphql` with
     `resolveReviewThread`. The recipe applies when the window holds an
     earlier `wait-for-status` or `merge-pr-when-green` family step.

   A recipe can apply without polling, for example when flags change between
   checks of the same PR.

## Labels

Labels name a flagged pattern in reports. They are not part of the fixture
contract, but a reimplementation that shows a label must keep it
content-free: build it only from fixed names, never from argument values,
paths, or script names.

- A recommended script is labeled by its name, for example `wait-for-status`.
- An executable outside nenpi's allowlist (`LABEL_EXECUTABLES` in
  `src/nenpi/command_classification.py`) is `script` when invoked by path and
  `shell` otherwise.
- `gh` is `gh GROUP SUB` when GROUP and SUB are known names, `gh api`, or
  `gh api graphql`. `docker logs` and `docker-compose logs` keep `logs`.
- `git`, `docker`, `docker-compose`, `kubectl`, `npm`, `pnpm`, `yarn`,
  `cargo`, `go`, `uv`, `systemctl`, and `terraform` add the first positional
  after their global value flags (`git -C DIR`, `kubectl -n NS`) when it is a
  known subcommand. Otherwise the label is the executable alone.

Signatures still hash full arguments internally; only labels are emitted.

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
