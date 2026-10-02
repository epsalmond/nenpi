"""Polling and scripted-recipe classification of ordered tool calls.

`tests/fixtures/command-classification.json` is the contract; burn-governor in
management-plane reimplements these rules without importing nenpi, so keep
them small and documented in docs/polling.md. Steps carry only opaque hashes
and command-family labels, never command or result text.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import posixpath
import re
import shlex

THRESHOLD = 3
WINDOW_STEPS = 30
RECIPES = {
    "wait-for-status": "scripts/wait-for-status",
    "merge-pr-when-green": "scripts/merge-pr-when-green",
    "query-logs": "scripts/query-logs",
}
KINDS = ("pure_poll", "watch", "none")
WRITE_TOOLS = frozenset({"write", "edit", "multiedit", "multi_edit", "notebookedit", "apply_patch", "patch"})
SHELL_TOOLS = frozenset({"bash", "exec_command", "shell", "shell_command", "local_shell", "local_shell_call"})
VOLATILE_KEYS = frozenset({"description", "timeout", "timeout_ms", "yield_time_ms", "max_output_tokens"})
SHELLS = frozenset({"sh", "bash", "zsh"})
SCRIPT_RUNNERS = SHELLS | {"python", "python3"}
LOOP_KEYWORDS = frozenset({"while", "until", "for", "select"})
# Shell grammar words skipped at the start of a statement, and closers dropped.
STATEMENT_PREFIXES = frozenset({"(", "{", "!", "if", "elif", "then", "else", "do"})
STATEMENT_CLOSERS = frozenset({")", "}", "fi", "done", "esac"})
# Wrapper commands stripped from a statement, with the flags that take a value.
WRAPPER_VALUE_FLAGS = {
    "command": frozenset(),
    "nohup": frozenset(),
    "env": frozenset({"-u", "--unset", "-C", "--chdir"}),
    "sudo": frozenset({"-u", "--user", "-g", "--group", "-p", "--prompt", "-C", "--close-from",
                       "-D", "--chdir", "-r", "--role", "-t", "--type", "-U", "--other-user"}),
    "timeout": frozenset({"-s", "--signal", "-k", "--kill-after"}),
    "nice": frozenset({"-n", "--adjustment"}),
    "time": frozenset({"-f", "--format", "-o", "--output"}),
}
GH_FORMAT_VALUE_FLAGS = frozenset({"--json", "--jq", "-q", "--template", "-t"})
GH_FORMAT_FLAGS = frozenset({"--compact", "--exit-status"})
GH_VALUE_FLAGS = GH_FORMAT_VALUE_FLAGS | {
    "-R", "--repo", "-X", "--method", "-H", "--header", "-f", "-F", "--field",
    "--raw-field", "--input", "-L", "--limit", "-b", "--body", "-B", "--base",
    "-s", "--state", "-A", "--author", "--workflow", "--branch", "-j", "--job",
    "-a", "--attempt", "-i", "--interval",
}
# `gh pr view --json` fields that report status; any other field is a PR read.
GH_STATUS_FIELDS = frozenset({
    "state", "statusCheckRollup", "mergeable", "mergeStateStatus", "reviewDecision", "isDraft",
    "mergedAt", "mergedBy", "closed", "closedAt", "autoMergeRequest", "headRefOid", "headRefName",
    "baseRefName", "number", "url", "id",
})
JOURNAL_SOURCE_FLAGS = frozenset({"-u", "--unit", "--user-unit", "-t", "--identifier"})
DOCKER_LOG_VALUE_FLAGS = frozenset({"--since", "--until", "--tail", "-n", "--index"})
TAIL_VALUE_FLAGS = frozenset({"-n", "-c", "--lines", "--bytes", "-s", "--sleep-interval", "--pid"})
LOG_PATH = re.compile(r"(?:^|/)logs?/|\.log(?:\.\d+)?$", re.IGNORECASE)
REVIEW_PATH = re.compile(r"(?:^|/)(?:pulls|issues)/[^/]+/(?:comments|reviews)(?:[/?]|$)|(?:^|/)pulls/comments(?:[/?]|$)")

# Labels reach caches and reports, so they are built only from these names.
LABEL_EXECUTABLES = frozenset({
    "adb", "ansible", "ansible-playbook", "awk", "bun", "cargo", "cat", "cd", "chmod", "claude", "cmake", "codex",
    "cp", "curl", "cut", "date", "deno", "df", "diff", "dig", "docker", "docker-compose", "du", "echo", "env",
    "false", "fd", "find", "flux", "free", "gh", "git", "go", "gradle", "gradlew", "grep", "gzip", "head", "helm",
    "java", "journalctl", "jq", "just", "kill", "kubectl", "less", "ln", "ls", "make", "mise", "mkdir", "mv",
    "nc", "nenpi", "ninja", "node", "npm", "npx", "openssl", "patch", "perl", "pgrep", "ping", "pip", "pip3",
    "pkill", "pnpm", "printf", "ps", "psql", "pytest", "python", "python3", "rg", "rm", "rsync", "ruby",
    "scp", "sed", "sh", "bash", "zsh", "sleep", "sort", "sqlite3", "ssh", "stat", "swift", "systemctl", "tail",
    "tailscale", "tar", "tee", "terraform", "test", "tmux", "top", "touch", "tr", "true", "uniq", "unzip",
    "uv", "virsh", "watch", "wc", "wget", "which", "xargs", "xcodebuild", "xcrun", "yarn", "zfs", "zpool",
})
# Executables labeled with a subcommand, and their global flags that take a value.
SUBCOMMAND_VALUE_FLAGS = {
    "git": frozenset({"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--config-env"}),
    "docker": frozenset({"-H", "--host", "-c", "--context", "--config", "-l", "--log-level"}),
    "docker-compose": frozenset({"-f", "--file", "-p", "--project-name", "--project-directory", "--profile", "--env-file"}),
    "kubectl": frozenset({"-n", "--namespace", "--context", "--cluster", "--user", "--kubeconfig", "-s", "--server",
                          "-l", "--selector", "-o", "--output", "-f", "--filename", "-c", "--container"}),
    "npm": frozenset({"--prefix", "-w", "--workspace"}),
    "pnpm": frozenset({"-C", "--dir", "-F", "--filter"}),
    "yarn": frozenset({"--cwd"}),
    "cargo": frozenset({"-C", "--manifest-path", "-Z", "--config", "--color", "-p", "--package"}),
    "go": frozenset({"-C"}),
    "uv": frozenset({"--directory", "--project", "-p", "--python", "--with"}),
    "systemctl": frozenset({"-H", "--host", "-M", "--machine", "-t", "--type", "-p", "--property"}),
    "terraform": frozenset(),
}
GH_GROUPS = frozenset({"pr", "run", "issue", "repo", "workflow", "release", "auth", "search", "label", "gist",
                       "cache", "secret", "variable", "project", "browse", "status", "extension", "ruleset"})
SAFE_SUBCOMMANDS = frozenset({
    "add", "am", "api", "apply", "auth", "bench", "bisect", "blame", "branch", "build", "cancel", "cat-file",
    "check", "checkout", "checks", "cherry-pick", "ci", "clean", "clippy", "clone", "close", "comment", "commit",
    "compose", "config", "cp", "create", "delete", "describe", "diff", "disable", "doc", "down", "download",
    "edit", "enable", "exec", "fetch", "fmt", "fork", "format-patch", "gc", "get", "grep", "images", "init",
    "inspect", "install", "is-active", "is-failed", "list", "lock", "log", "logs", "ls", "ls-files", "ls-remote",
    "merge", "merge-base", "metadata", "mod", "mv", "outdated", "plan", "port-forward", "ps", "pull", "push",
    "ready", "rebase", "reflog", "reload", "remote", "remove", "reopen", "rerun", "reset", "restart", "restore",
    "rev-list", "rev-parse", "review", "rm", "rollout", "run", "show", "start", "stash", "status", "stop",
    "submodule", "switch", "sync", "tag", "test", "tool", "top", "tree", "up", "update", "validate", "venv",
    "vet", "view", "watch", "worktree",
})


@dataclass(frozen=True)
class Step:
    """Content-free classification features of one tool call."""
    signature: str = ""    # opaque; "" never repeats
    family: str = ""       # "ci" | "logs" | "merge" | ""
    source: str = ""       # opaque target within the family
    label: str = ""        # command family/subcommand only
    write: bool = False
    exempt: bool = False
    result: str | None = None  # opaque masked result; None is unknown


@dataclass(frozen=True)
class Verdict:
    polling: bool = False
    kind: str = "none"
    recipe_id: str | None = None

    def to_json(self):
        return dict(polling=self.polling, kind=self.kind, recipe_id=self.recipe_id)


NOT_FLAGGED = Verdict()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, default=str).encode()).hexdigest()[:24]


HEX_ID = re.compile(r"\b(?=[0-9a-f]*[0-9])[0-9a-f]{6,}\b")


def result_key(text):
    """Results compare equal after masking hex IDs and digit runs, collapsing whitespace."""
    if text is None:
        return None
    if not isinstance(text, str):
        text = json.dumps(text, sort_keys=True, default=str)
    return digest(" ".join(re.sub(r"\d+", "0", HEX_ID.sub("0", text)).split()))


def split_statements(command):
    """Split on ; & | and newlines outside quotes; `2>&1` and `&>` are redirections."""
    quote, escaped, start, parts = "", False, 0, []
    for i, char in enumerate(command):
        if escaped:
            escaped = False
        elif char == "\\" and quote != "'":
            escaped = True
        elif quote:
            if char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
        elif char in ";|\n" or (char == "&" and command[i - 1:i] != ">" and command[i + 1:i + 2] != ">"):
            if command[start:i].strip():
                parts.append(command[start:i].strip())
            start = i + 1
    if command[start:].strip():
        parts.append(command[start:].strip())
    return parts


def _strip_wrappers(words):
    """Drop grammar prefixes, wrapper commands with their flags, and NAME=value words."""
    words = list(words)
    while words:
        head = words[0]
        if head in STATEMENT_PREFIXES:
            words.pop(0)
        elif head[:1] in "({" and len(head) > 1:
            words[0] = head[1:]
        elif head == "rtk":
            words.pop(0)
            if words and words[0] == "proxy":
                words.pop(0)
        elif re.match(r"^[A-Za-z_][A-Za-z_0-9]*=", head):
            words.pop(0)
        elif head in WRAPPER_VALUE_FLAGS:
            words.pop(0)
            while words and words[0].startswith("-") and words[0] != "-":
                flag = words.pop(0)
                if flag == "--":
                    break
                if flag in WRAPPER_VALUE_FLAGS[head] and words:
                    words.pop(0)
            if head == "timeout" and words:
                words.pop(0)  # DURATION
        else:
            break
    return words


def _unwrap_runner(words):
    """`sh -c STRING` -> its statements; `bash|sh|python3 FILE ARGS` -> FILE ARGS; else None."""
    if posixpath.basename(words[0]) not in SCRIPT_RUNNERS:
        return None
    shell = posixpath.basename(words[0]) in SHELLS
    index = 1
    while index < len(words) and words[index].startswith("-") and words[index] not in {"-", "-m"}:
        if shell and re.fullmatch(r"-[a-z]*c", words[index]):
            return ("code", words[index + 1]) if index + 1 < len(words) else None
        if words[index] == "-c":
            return None
        index += 1
    if index < len(words) and not words[index].startswith("-"):
        return ("file", words[index:])
    return None


def _statements(command):
    """Shell statements as (words, path) pairs, wrappers stripped and runners unwrapped; None if unparsable.

    words[0] is the executable basename; path is whether it was invoked by path.
    """
    found = []
    for part in split_statements(command):
        try:
            words = _strip_wrappers(shlex.split(part))
        except ValueError:
            return None
        if not words or all(word in STATEMENT_CLOSERS for word in words):
            continue
        unwrapped = _unwrap_runner(words)
        if unwrapped and unwrapped[0] == "code":
            inner = _statements(unwrapped[1])
            if inner is None:
                return None
            found.extend(inner)
            continue
        if unwrapped:
            words = _strip_wrappers(unwrapped[1])
            if not words:
                continue
        found.append(([posixpath.basename(words[0])] + words[1:], "/" in words[0]))
    return found


def _gh(words):
    """(positionals, repo) for a gh command, skipping flag values."""
    positionals, repo, skip = [], "", None
    for word in words[1:]:
        if skip:
            if skip in {"-R", "--repo"}:
                repo = word
            skip = None
        elif word in GH_VALUE_FLAGS:
            skip = word
        elif word.startswith("-"):
            if word.startswith("--repo="):
                repo = word.split("=", 1)[1]
        else:
            positionals.append(word)
    return positionals, repo


def _drop_gh_format(words):
    kept, skip = [], False
    for word in words:
        if skip:
            skip = False
        elif word in GH_FORMAT_VALUE_FLAGS:
            skip = True
        elif word in GH_FORMAT_FLAGS or word.split("=", 1)[0] in GH_FORMAT_VALUE_FLAGS:
            continue
        else:
            kept.append(word)
    return kept


def _positionals(words, value_flags):
    found, skip = [], False
    for word in words:
        if skip:
            skip = False
        elif word in value_flags:
            skip = True
        elif not word.startswith("-"):
            found.append(word)
    return found


def _subcommand(words, value_flags):
    """First positional after skipping global flags and their values, if it is a known subcommand."""
    skip = False
    for word in words[1:]:
        if skip:
            skip = False
        elif word in value_flags:
            skip = True
        elif not word.startswith(("-", "+")):  # +toolchain selects a cargo toolchain
            return word if word in SAFE_SUBCOMMANDS else ""
    return ""


def _label(words, path):
    """Content-free label: an allowlisted executable and subcommand, else `script` or `shell`."""
    exe = words[0]
    if exe in RECIPES:
        return exe
    if exe not in LABEL_EXECUTABLES:
        return "script" if path else "shell"
    if exe == "gh":
        positionals, _repo = _gh(words)
        group, sub = (positionals + ["", ""])[:2]
        if group == "api":
            return "gh api graphql" if sub == "graphql" else "gh api"
        if group not in GH_GROUPS:
            return "gh"
        return "gh " + group + (" " + sub if sub in SAFE_SUBCOMMANDS else "")
    if exe in {"docker", "docker-compose"} and "logs" in words:
        return exe + " logs"
    if exe in SUBCOMMAND_VALUE_FLAGS:
        sub = _subcommand(words, SUBCOMMAND_VALUE_FLAGS[exe])
        return exe + " " + sub if sub else exe
    return exe


def _gh_pr_read(words):
    """`gh pr view` asking for comments or any non-status `--json` field reads the PR, not its status."""
    if "--comments" in words or "-c" in words:
        return True
    fields = []
    for index, word in enumerate(words):
        if word == "--json" and index + 1 < len(words):
            fields += words[index + 1].split(",")
        elif word.startswith("--json="):
            fields += word.split("=", 1)[1].split(",")
    return any(field.strip() not in GH_STATUS_FIELDS for field in fields if field.strip())


def _family(words, path=False):
    """(family, source key, label) of a primary statement."""
    exe = words[0]
    label = _label(words, path)
    if exe == "gh":
        positionals, repo = _gh(words)
        group, sub = (positionals + ["", ""])[:2]
        if (group == "pr" and sub == "checks") or (group == "run" and sub == "view") or (
                group == "pr" and sub == "view" and not _gh_pr_read(words)):
            target = positionals[2] if len(positionals) > 2 else ""
            return "ci", ["ci", repo, group, target], label
        if group == "pr" and sub == "merge":
            return "merge", ["merge"], label
        if group == "api":
            if sub == "graphql" and any("resolveReviewThread" in w for w in words):
                return "merge", ["merge"], label
            if sub and REVIEW_PATH.search(sub):
                return "merge", ["merge"], label
        return "", None, label
    if exe == "journalctl":
        units, skip = [], None
        for word in words[1:]:
            if skip:
                units.append(word)
                skip = None
            elif word in JOURNAL_SOURCE_FLAGS:
                skip = word
            elif word.split("=", 1)[0] in JOURNAL_SOURCE_FLAGS and "=" in word:
                units.append(word.split("=", 1)[1])
        return "logs", ["journal", sorted(units), "--user" in words], label
    if exe in {"docker", "docker-compose"} and "logs" in words:
        tail = words[words.index("logs") + 1:]
        containers = _positionals(tail, DOCKER_LOG_VALUE_FLAGS)
        return "logs", ["docker", containers[-1] if containers else ""], label
    if exe == "tail":
        paths = _positionals(words[1:], TAIL_VALUE_FLAGS)
        if paths and LOG_PATH.search(paths[-1]):
            return "logs", ["tail", paths[-1]], label
    return "", None, label


def _exempt_statement(words):
    return (words[0] in LOOP_KEYWORDS or words[0] == "watch" or words[0] in RECIPES
            or "--watch" in words or words[:3] == ["gh", "run", "watch"])


def shell_step(command, cwd="", result=None):
    """Features of one shell command; see docs/polling.md for each rule."""
    if not isinstance(command, str) or not command.strip():
        return Step(result=result)
    statements = _statements(command)
    if statements is None:
        return Step(label="shell", result=result)
    if "<<" in command or any(_exempt_statement(words) for words, _path in statements):
        primary = statements[0] if statements else (["shell"], False)
        return Step(label=_family(*primary)[2], exempt=True, result=result)
    directory, kept, slept = cwd or "", [], False
    for words, path in statements:
        if words[0] == "cd":
            directory = posixpath.join(directory, words[1] if len(words) > 1 else "~")
        elif words[0] == "sleep":
            slept = True
        else:
            kept.append((words, path))
    if not kept:
        if not slept:
            return Step(label="cd", result=result)
        return Step(digest(["shell", directory, [["sleep"]]]), label="sleep", result=result)
    normalized = []
    for words, path in kept:
        if words[0] == "gh" and _family(words, path)[0] == "ci":
            words = _drop_gh_format(words)
        normalized.append(words)
    family, source, label = _family(*kept[0])
    return Step(digest(["shell", directory, normalized]), family,
                digest(source) if source is not None else "", label, result=result)


def merge_steps(steps, result=None):
    """One step for a call that ran several tools (Codex code-mode `exec`).

    Repeats inside one call never count against each other: the call's
    signature is its tools' signatures in order, its recipe family the first
    one found, and it is a write or exempt if any part is.
    """
    steps = [s for s in steps if s is not None]
    if len(steps) == 1:
        return replace(steps[0], result=result)
    signatures = [s.signature for s in steps]
    primary = next((s for s in steps if s.family), steps[0] if steps else Step())
    return Step(digest(["exec", signatures]) if any(signatures) else "", primary.family, primary.source,
                primary.label, write=any(s.write for s in steps), exempt=any(s.exempt for s in steps),
                result=result)


def tool_leaf(name):
    return re.split(r"\.|__", name or "")[-1].lower()


def tool_step(name, args, cwd="", result=None):
    """Features of one tool call from its harness name and input object."""
    leaf = tool_leaf(name)
    args = args if isinstance(args, dict) else {}
    if leaf in WRITE_TOOLS:
        return Step(label=leaf, write=True, result=result)
    if leaf in SHELL_TOOLS:
        command = args.get("cmd", args.get("command"))
        if isinstance(command, list):
            command = shlex.join(str(w) for w in command)
        return shell_step(command, args.get("workdir", args.get("cwd", cwd)) or cwd, result)
    if leaf == "write_stdin" and args.get("chars", "") != "":
        return Step(label=leaf, result=result)
    stable = {k: v for k, v in args.items() if k not in VOLATILE_KEYS}
    return Step(digest(["tool", leaf, stable]), label=leaf, result=result)


def fixture_step(step):
    """A fixture step: shell `command`/`cwd`, or `tool` with `input`."""
    result = result_key(step.get("result"))
    if "command" in step:
        return shell_step(step["command"], step.get("cwd", ""), result)
    code = (step.get("input") or {}).get("code") if isinstance(step.get("input"), dict) else None
    if tool_leaf(step.get("tool", "")) == "exec" and isinstance(code, str):
        from .activity_model import exec_operations
        return merge_steps([op.step for op in exec_operations(code)], result)
    return tool_step(step.get("tool", ""), step.get("input", {}), result=result)


def _judge(step, recent, threshold):
    if step.exempt:
        return NOT_FLAGGED
    polling, kind = False, "none"
    if step.signature:
        same = [s for s in recent if s.signature == step.signature]
        if len(same) + 1 >= threshold:
            polling = True
            compared = [s.result for s in same[-max(threshold - 1, 2):]]
            identical = len(compared) >= 2 and None not in compared and len(set(compared)) == 1
            kind = "pure_poll" if identical else "watch"
    recipe = None
    if step.family in {"ci", "logs"}:
        repeats = sum(1 for s in recent if s.family == step.family and s.source == step.source)
        if repeats + 1 >= threshold:
            recipe = "wait-for-status" if step.family == "ci" else "query-logs"
    elif step.family == "merge" and any(s.family in {"ci", "merge"} for s in recent):
        recipe = "merge-pr-when-green"
    return Verdict(polling, kind, recipe)


def classify(steps, threshold=THRESHOLD, window_steps=WINDOW_STEPS):
    """One verdict per step, judged before it runs from earlier steps only."""
    verdicts, window = [], []
    for step in steps:
        if step.write:
            window = []
            verdicts.append(NOT_FLAGGED)
            continue
        recent = window[-window_steps:] if window_steps else window
        verdicts.append(_judge(step, recent, threshold))
        window.append(step)
    return verdicts


def classify_steps(steps, threshold=THRESHOLD, window_steps=WINDOW_STEPS):
    """Fixture entry point: raw step dicts in, `{polling, kind, recipe_id}` dicts out."""
    return [v.to_json() for v in classify([fixture_step(s) for s in steps], threshold, window_steps)]
