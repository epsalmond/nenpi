"""Harness adapters for normalized usage and operation records.

No command or result bodies are persisted. Operation identities are hashes of
literal targets, scoped by source/thread by the shared repetition detector.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import os
import tempfile
from pathlib import Path
import re
import shlex

from .command_classification import (LABEL_EXECUTABLES, SHELL_TOOLS, WRITE_TOOLS, Step, merge_steps, result_key,
                                     tool_leaf, tool_step)
from .tool_activity import _JS_TOKEN, _object_properties, _source_activity


@dataclass(frozen=True)
class Operation:
    activity: str
    target: str = ""
    polling: bool = False
    detail: str = ""
    # Polling/recipe features (opaque hashes and a command-family label only).
    step: Step | None = field(default=None, compare=False)


@dataclass
class Response:
    harness: str
    account: str
    session: str
    thread: str
    prompt: int
    project: str
    timestamp: float
    response_id: str
    model: str
    context: int
    cached: int
    output: int
    usage: float
    prices: dict
    operations: list[Operation] = field(default_factory=list)
    results: list[tuple[float, int]] = field(default_factory=list)
    reset: bool = False
    cache_write_kind: str = ""
    native_tokens: dict = field(default_factory=dict)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()[:24]


def shell_parts(command):
    """Split literal shell statements outside quotes; never interpret code."""
    quote, escaped, start = "", False, 0
    parts = []
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
        elif char in ";&|\n":
            value = command[start:i].strip()
            if value:
                parts.append(value)
            start = i + 1
    if command[start:].strip():
        parts.append(command[start:].strip())
    return parts


def shell_operation(command):
    if not isinstance(command, str) or not command.strip():
        return Operation("Other / unknown")
    if re.search(r"(?:^|\s)(?:python[0-9.]*|node|ruby|perl)(?:\s|$)", command) and "<<" in command:
        return Operation("Running scripts")
    if re.match(r"\s*(?:for|while|if)\s", command):
        return Operation("Shell scripts")
    parts = shell_parts(command)
    if len(parts) > 1:
        operations, directories = [], []
        for part in parts:
            try:
                args = shlex.split(part)
            except ValueError:
                return Operation("Other / unknown")
            if args and args[0] == "cd" and len(args) == 2:
                directories.append(args[1])
            elif args and args[0] in {"echo", "printf", "jq", "head"}:
                continue
            else:
                operations.append(shell_operation(part))
        categories = {op.activity for op in operations}
        if len(categories) == 1:
            polling = all(op.polling and op.target for op in operations)
            target = fingerprint([directories, [op.target for op in operations]]) if all(op.target for op in operations) else ""
            return Operation(next(iter(categories)), target, polling)
        return Operation("Mixed activity", detail=" + ".join(sorted(categories)) or "Shell formatting")
    try:
        words = shlex.split(command)
    except ValueError:
        return Operation("Other / unknown")
    # Strip only transparent wrappers. Preserve flags/targets for identity.
    while words and words[0] in {"rtk", "proxy", "command", "env"}:
        words.pop(0)
    while words and re.match(r"^[A-Za-z_][A-Za-z_0-9]*=", words[0]):
        words.pop(0)
    if len(words) >= 3 and Path(words[0]).name in {"sh", "bash", "zsh"} and words[1] in {"-c", "-lc"}:
        return shell_operation(words[2])
    if not words:
        return Operation("Other / unknown")
    # Complex shell programs are not treated as one repeated status operation.
    words[0] = Path(words[0]).name
    executable = words[0]
    prefix = words[:3]
    activity, polling = "Other / unknown", False
    if prefix[:2] in (["gh", "run"], ["gh", "pr"]) and len(words) > 2 and words[2] in {"view", "checks", "watch", "list"}:
        activity, polling = "Waiting for CI", True
    elif executable in {"kubectl", "helm", "argocd", "flux", "fly", "vercel"} and any(w in {"status", "get", "wait", "logs"} for w in words[1:]):
        activity, polling = "Checking deployments", True
    elif executable in {"sleep", "wait"}:
        activity, polling = "Waiting for time", True
    elif executable in {"ps", "pgrep"} or (executable == "tail" and any(w in {"-f", "-F"} for w in words[1:])):
        activity, polling = "Waiting for processes", True
    elif executable in {"pytest", "cargo", "make", "cmake", "ninja", "xcodebuild", "swift", "just", "npm", "pnpm", "yarn", "uv"} and (executable in {"pytest", "make", "ninja", "xcodebuild"} or any(re.search(r"test|build|check|compile", w) for w in words[1:])):
        activity = "Running builds/tests"
    elif executable in {"rg", "grep", "cat", "sed", "head", "find", "ls", "wc", "tail", "nl", "jq", "cut", "sort", "awk", "head", "stat"} or prefix[:2] in (["git", "diff"], ["git", "show"], ["git", "status"], ["git", "log"]):
        activity = "Reading/searching code"
    elif executable == "git":
        activity = "Version control"
    elif executable == "gh":
        activity = "Working with GitHub"
    elif executable in {"python", "python3", "node", "ruby", "perl"}:
        activity = "Running scripts"
    elif executable in {"ssh", "scp", "rsync"}:
        activity = "Remote commands/transfers"
    elif executable in {"curl", "wget"}:
        activity = "HTTP requests"
    # CI output-format flags do not change the operation being checked.
    identity = words
    if activity == "Waiting for CI":
        identity = []
        skip = False
        for word in words:
            if skip:
                skip = False
                continue
            if word in {"--json", "--jq", "--template", "--interval"}:
                skip = True
            elif word not in {"--watch", "--exit-status", "--compact"}:
                identity.append(word)
        if len(identity) > 2 and identity[1] == "run" and identity[2] in {"view", "watch"}:
            identity[2] = "status"
    return Operation(activity, fingerprint(identity) if polling or (activity == "Reading/searching code" and executable not in {"sed", "awk"}) else "", polling, "shell: " + executable if executable in LABEL_EXECUTABLES else "Shell command")


def operation(name, args, *, target_known=True, cwd=""):
    return [replace(op, step=_step(name, args, target_known, cwd)) for op in _operation(name, args, target_known=target_known)]


def _step(name, args, target_known, cwd):
    leaf = tool_leaf(name)
    if not target_known:
        return Step(write=leaf in WRITE_TOOLS)
    step = tool_step(name, args, cwd)
    # Tool names stay out of the cache; reports label non-shell steps by activity.
    return step if leaf in SHELL_TOOLS else replace(step, label="")


def _operation(name, args, *, target_known=True):
    leaf = re.split(r"\.|__", name)[-1].lower()
    args = args if isinstance(args, dict) else {}
    if leaf in {"bash", "exec_command", "shell", "shell_command", "local_shell", "local_shell_call"}:
        command = args.get("cmd", args.get("command"))
        if isinstance(command, list):
            command = shlex.join(str(w) for w in command)
        op = shell_operation(command)
        if op.target and not target_known:
            op = Operation(op.activity, "", op.polling, op.detail)
        if op.target:
            op = Operation(op.activity, fingerprint([op.target, args.get("workdir", args.get("cwd", ""))]), op.polling, op.detail)
        return [op]
    if leaf in {"write_stdin", "taskoutput", "bashoutput", "wait", "wait_agent", "sleep", "curr_time"}:
        if leaf == "write_stdin" and args.get("chars", "") != "":
            return [Operation("Other / unknown")]
        target = args.get("session_id", args.get("cell_id", args.get("task_id", args.get("bash_id", args.get("target", args.get("ids"))))))
        if target is None and target_known and leaf in {"wait_agent", "curr_time"}:
            target = "all-agents" if leaf == "wait_agent" else "clock"
        activity = "Waiting for agents" if leaf == "wait_agent" else "Waiting for time" if leaf in {"sleep", "curr_time"} else "Waiting for processes"
        # No literal target: classify the activity but do not invent repetition.
        return [Operation(activity, fingerprint(target) if target is not None else "", True)]
    if leaf in {"sendmessage", "taskstop"}:
        return [Operation("Coordinating agents")]
    if leaf in {"request_user_input", "request_user_input_async", "askuserquestion"}:
        return [Operation("Asking the user")]
    if leaf in {"webfetch", "websearch"} or name == "web__run":
        return [Operation("Searching the web")]
    if leaf in {"toolsearch", "skill"}:
        return [Operation("Loading tools/skills")]
    if "hindsight" in name:
        return [Operation("Reading/writing memory")]
    if leaf == "view_image":
        return [Operation("Inspecting images")]
    if leaf in {"read", "read_file", "glob", "grep", "list_directory"}:
        return [Operation("Reading/searching code", fingerprint([leaf, args]) if args and target_known else "")]
    if leaf in {"edit", "write", "multiedit", "apply_patch"}:
        return [Operation("Editing code")]
    if leaf in {"spawn_agent", "send_message", "followup_task", "agent", "task"}:
        return [Operation("Coordinating agents")]
    if leaf == "list_agents":
        return [Operation("Waiting for agents", fingerprint("agent-list"), True)]
    label = name if isinstance(name, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,95}", name) else "Unrecognized tool"
    return [Operation("Other / unknown", detail=label)]


def exec_operations(code):
    if not isinstance(code, str):
        return [Operation("Other / unknown")]
    tokens = [m.group() for m in _JS_TOKEN.finditer(code) if m.lastgroup != "comment"]
    found = []
    for i in range(len(tokens) - 3):
        if tokens[i:i + 2] != ["tools", "."] or tokens[i + 3] != "(":
            continue
        args, names, uncertain = _object_properties(tokens, i + 4, numbers=True)
        name = tokens[i + 2]
        if name.split("__")[-1] == "write_stdin" and "chars" in names and "chars" not in args:
            found.append(Operation("Other / unknown"))
        else:
            target_known = not uncertain and all(k in args for k in names)
            found.extend(operation(name, {} if uncertain else args, target_known=target_known))
    return found or [Operation("Other / unknown")]


def codex_operations(name, payload, item):
    from . import drain as d
    if name.split(".")[-1] == "exec" and item == "custom_tool_call":
        return exec_operations(payload.get("input"))
    if item == "local_shell_call":
        command = d.command_from_payload(payload, item)
        return [replace(shell_operation(command), step=tool_step(item, {"command": command}))]
    args = payload.get("arguments", payload.get("input", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {}
    return operation(name, args)


def claude_batches(path):
    """Claude streaming blocks share a message ID; merge them, never price twice."""
    try:
        with Path(path).open(encoding="utf-8") as stream:
            for raw in stream:
                if '"tool_use"' not in raw:
                    continue
                try:
                    record = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(record, dict) or record.get("type") != "assistant":
                    continue
                message = record.get("message") or {}
                rid = message.get("id") or record.get("requestId")
                batch = {}
                content = message.get("content") or []
                for block in content if isinstance(content, list) else []:
                    if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id"):
                        batch[block["id"]] = (operation(block.get("name", ""), block.get("input"), cwd=record.get("cwd") or ""), [], block.get("name", ""))
                if rid:
                    yield rid, batch
    except OSError:
        return


def _result_text(content):
    if isinstance(content, list):
        return "\n".join(str(b.get("text", "")) if isinstance(b, dict) else str(b) for b in content)
    return content


def result_keys(harness, path):
    """Masked-result hashes by call ID; result bodies are never kept."""
    from . import drain as d
    keys = {}
    markers = ('"tool_result"',) if harness == "claude" else ('_call_output"',)
    try:
        with Path(path).open(encoding="utf-8") as stream:
            for raw in stream:
                if not any(marker in raw for marker in markers):
                    continue
                try:
                    record = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(record, dict):
                    continue
                if harness == "claude":
                    content = (record.get("message") or {}).get("content") if record.get("type") == "user" else None
                    for block in content if isinstance(content, list) else []:
                        if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id"):
                            keys[block["tool_use_id"]] = result_key(_result_text(block.get("content", "")))
                else:
                    payload = record.get("payload")
                    if (record.get("type") == "response_item" and isinstance(payload, dict)
                            and payload.get("type") in d.CODEX_TOOL_OUTPUT_ITEMS and payload.get("call_id")):
                        keys[payload["call_id"]] = result_key(_result_text(payload.get("output", "")))
    except OSError:
        pass
    return keys


def _cached_operation(op):
    step = op.get("step")
    return Operation(**dict(op, step=Step(**step) if isinstance(step, dict) else None))


def _one_step_per_call(ops, result):
    """The call's step rides on its first operation; the rest carry none (Codex exec runs several tools)."""
    if not ops:
        return ops
    step = merge_steps([op.step or Step() for op in ops], result)
    return [replace(ops[0], step=step)] + [replace(op, step=None) for op in ops[1:]]


def operation_batches(harness, path, read_budget=None):
    """Cache only classifications and opaque identities; invalidate on source change."""
    from . import drain as d
    source = Path(path)
    try:
        before = source.stat()
    except OSError:
        return []
    signature = [before.st_size, before.st_mtime_ns, before.st_ctime_ns]
    destination = d.cache_dir() / "activities-v7" / (fingerprint([harness, str(source.resolve())]) + ".json")
    try:
        if destination.stat().st_size <= 64 * 1024 * 1024:
            cached = json.loads(destination.read_text())
            if cached["signature"] == signature:
                return [(rid, {cid: ([_cached_operation(op) for op in value[0]], [], value[1])
                    for cid, value in batch.items()}) for rid, batch in cached["batches"]]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if read_budget is not None:
        # Classification rereads each changed source for operations and masked
        # results. Native usage can still be exported when this budget runs out.
        needed = 2 * before.st_size
        if needed > read_budget["remaining"]:
            read_budget["incomplete"] = True
            return []
        read_budget["remaining"] -= needed
    batches = list(claude_batches(path) if harness == "claude" else _source_activity(path, codex_operations))
    results = result_keys(harness, path)
    batches = [(rid, {cid: (_one_step_per_call(ops, results.get(cid)), examples, name)
                      for cid, (ops, examples, name) in batch.items()}) for rid, batch in batches]
    from .export import tool_family
    encoded = [[rid, {cid: [[asdict(op) for op in ops], tool_family(name)] for cid, (ops, _, name) in batch.items()}]
               for rid, batch in batches if rid and batch]
    temporary = None
    try:
        after = source.stat()
        if signature == [after.st_size, after.st_mtime_ns, after.st_ctime_ns]:
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(dir=destination.parent)
            with os.fdopen(fd, "w") as output:
                json.dump(dict(signature=signature, batches=encoded), output)
            os.replace(temporary, destination)
    except OSError:
        pass
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return batches


def normalize(analysis, session_key=None, prompt_index=None):
    """The sole bridge from harness-specific records to shared analytics."""
    from . import drain as d
    responses = []
    by_id = {}
    for harness, events in analysis.scan.events.items():
        kinds = d.CLAUDE_KINDS if harness == "claude" else d.CODEX_KINDS
        for event in events:
            session = event[d.EVENT_SESSION]
            key = (harness, session)
            if session_key and key != session_key:
                continue
            if prompt_index is not None and event[d.EVENT_PROMPT] != prompt_index:
                continue
            model = event[d.EVENT_MODEL]
            tokens = d.event_tokens(event, kinds)
            price = analysis.weights.price_row(harness, model, analysis.args.claude_cache_read_weight)
            price_kinds = d.CLAUDE_KINDS if harness == "claude" else d.CODEX_FIT_KINDS
            multiplier = analysis.args.long_context_multiplier if event[d.EVENT_LONG] else 1
            prices = {k: p * multiplier for k, p in zip(price_kinds, price)} if price else {}
            summary = analysis.scan.sessions.get(key)
            write_kind = next((k for k in ("cache_write_1h", "cache_write_5m", "cache_write_unknown") if tokens.get(k)), "")
            response = Response(
                harness, analysis.scan.session_accounts.get(key, ("", harness))[1],
                session, str(event[d.EVENT_THREAD] or (session if not event[d.EVENT_SUB] else "")),
                event[d.EVENT_PROMPT], str(summary.cwd or "") if summary else "", event[d.EVENT_TS],
                event[d.EVENT_ID], model, int(d.event_context(event, harness)),
                int(tokens.get("cache_read", tokens.get("cached_input", 0))), tokens["output"],
                d.event_units(harness, event, kinds, analysis.weights, analysis.args), prices,
                cache_write_kind=write_kind,
                native_tokens=tokens,
            )
            responses.append(response)
            if response.response_id:
                by_id[(harness, session, response.response_id)] = response
    # Existing provenance points at canonical deduplicated transcript files.
    sources = {}
    for entry, harness, source in analysis.scan._tool_sources:
        if source and any((harness, e[d.EVENT_SESSION], e[d.EVENT_ID]) in by_id for e in entry.events):
            sources[(harness, str(source))] = {e[d.EVENT_ID]: e[d.EVENT_SESSION] for e in entry.events}
    tools = {(c.harness, c.source, c.call_id): c for c in analysis.tool_calls}
    linked = set()
    for (harness, path), ids in sources.items():
        d.check_cancelled(analysis.cancellation)
        batches = operation_batches(harness, path, getattr(analysis.args, "activity_read_budget", None))
        for rid, batch in batches:
            response = by_id.get((harness, ids.get(rid), rid))
            if response is None:
                continue
            for cid, (ops, _, _) in batch.items():
                unique = (harness, response.session, rid, cid)
                if unique in linked:
                    continue
                linked.add(unique)
                response.operations.extend(ops)
                call = tools.get((harness, path, cid))
                if call:
                    response.results.append((call.ts, int(call.est_tokens)))
    responses.sort(key=lambda r: (r.harness, r.session, r.thread, r.timestamp, r.response_id))
    previous = {}
    for response in responses:
        key = (response.harness, response.session, response.thread)
        before = previous.get(key)
        markers = analysis.scan.compactions.get((response.harness, response.session), [])
        response.reset = bool(before and (response.context < before.context or any(before.timestamp < t <= response.timestamp for t in markers)))
        previous[key] = response
    return responses
