"""Polling and scripted-recipe classification of ordered tool calls.

`tests/fixtures/command-classification.json` is the contract; burn-governor in
management-plane reimplements these rules without importing nenpi, so keep
them small and documented in docs/polling.md. Steps carry only opaque hashes
and command-family labels, never command or result text.
"""
from __future__ import annotations

from dataclasses import dataclass
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
LOOP_KEYWORDS = frozenset({"while", "until", "for", "select"})
GH_FORMAT_VALUE_FLAGS = frozenset({"--json", "--jq", "-q", "--template", "-t"})
GH_FORMAT_FLAGS = frozenset({"--compact", "--exit-status"})
GH_VALUE_FLAGS = GH_FORMAT_VALUE_FLAGS | {
    "-R", "--repo", "-X", "--method", "-H", "--header", "-f", "-F", "--field",
    "--raw-field", "--input", "-L", "--limit", "-b", "--body", "-B", "--base",
    "-s", "--state", "-A", "--author", "-w", "--workflow", "--branch", "--job",
}
JOURNAL_SOURCE_FLAGS = frozenset({"-u", "--unit", "--user-unit", "-t", "--identifier"})
DOCKER_LOG_VALUE_FLAGS = frozenset({"--since", "--until", "--tail", "-n"})
TAIL_VALUE_FLAGS = frozenset({"-n", "-c", "--lines", "--bytes", "-s", "--sleep-interval", "--pid"})
REVIEW_PATH = re.compile(r"(?:^|/)(?:pulls|issues)/[^/]+/(?:comments|reviews)(?:[/?]|$)|(?:^|/)pulls/comments(?:[/?]|$)")
SAFE_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,39}")


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
    words = list(words)
    while words:
        head = words[0]
        if head == "rtk":
            words.pop(0)
            if words and words[0] == "proxy":
                words.pop(0)
        elif head in {"command", "env"} or re.match(r"^[A-Za-z_][A-Za-z_0-9]*=", head):
            words.pop(0)
        elif head == "sudo":
            words.pop(0)
            while words and words[0].startswith("-"):
                words.pop(0)
        else:
            break
    return words


def _statements(command):
    """Shell statements as word lists, wrappers stripped and `sh -c` unwrapped; None if unparsable."""
    found = []
    for part in split_statements(command):
        try:
            words = _strip_wrappers(shlex.split(part))
        except ValueError:
            return None
        if not words:
            continue
        if len(words) == 3 and posixpath.basename(words[0]) in SHELLS and re.fullmatch(r"-[a-z]*c", words[1]):
            inner = _statements(words[2])
            if inner is None:
                return None
            found.extend(inner)
            continue
        found.append([posixpath.basename(words[0])] + words[1:])
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


def _family(words):
    """(family, source key, label) of a primary statement."""
    exe = words[0]
    label = exe if SAFE_WORD.fullmatch(exe) else "shell"
    if exe == "gh":
        positionals, repo = _gh(words)
        group, sub = (positionals + ["", ""])[:2]
        label = " ".join(["gh"] + [w for w in (group, sub) if SAFE_WORD.fullmatch(w)])
        if (group == "pr" and sub in {"checks", "view"}) or (group == "run" and sub == "view"):
            target = positionals[2] if len(positionals) > 2 else ""
            return "ci", ["ci", repo, group, target], label
        if group == "pr" and sub == "merge":
            return "merge", ["merge"], label
        if group == "api":
            label = "gh api graphql" if sub == "graphql" else "gh api"
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
        label = exe + " logs"
        containers = _positionals(tail, DOCKER_LOG_VALUE_FLAGS)
        return "logs", ["docker", containers[-1] if containers else ""], label
    if exe == "tail":
        paths = _positionals(words[1:], TAIL_VALUE_FLAGS)
        if paths and "log" in paths[-1].lower():
            return "logs", ["tail", paths[-1]], label
        return "", None, label
    if exe in {"git", "docker", "kubectl", "npm", "pnpm", "cargo", "go", "uv", "make"}:
        sub = next((w for w in words[1:] if not w.startswith("-")), "")
        if SAFE_WORD.fullmatch(sub) and exe != "make":
            label = exe + " " + sub
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
    if "<<" in command or any(_exempt_statement(words) for words in statements):
        primary = statements[0] if statements else ["shell"]
        return Step(label=_family(primary)[2], exempt=True, result=result)
    directory, kept, slept = cwd or "", [], False
    for words in statements:
        if words[0] == "cd":
            directory = posixpath.join(directory, words[1] if len(words) > 1 else "~")
        elif words[0] == "sleep":
            slept = True
        else:
            kept.append(words)
    if not kept:
        if not slept:
            return Step(label="cd", result=result)
        return Step(digest(["shell", directory, [["sleep"]]]), label="sleep", result=result)
    normalized = []
    for words in kept:
        if words[0] == "gh" and _family(words)[0] == "ci":
            words = _drop_gh_format(words)
        normalized.append(words)
    family, source, label = _family(kept[0])
    return Step(digest(["shell", directory, normalized]), family,
                digest(source) if source is not None else "", label, result=result)


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
