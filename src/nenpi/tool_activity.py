"""Opt-in, transient explanations of tool activity and its generating responses.

Static code inspection only: never execute a captured command or JavaScript.
Command examples and tool bodies are never written to Nenpi's transcript cache.
"""

import ast
import json
import re
from collections import defaultdict
from pathlib import Path


_JS_TOKEN = re.compile(
    r"(?P<comment>//[^\n]*|/\*[\s\S]*?\*/)"
    r"|(?P<string>\"(?:\\[\s\S]|[^\"\\])*\"|'(?:\\[\s\S]|[^'\\])*'|`(?:\\[\s\S]|[^`\\])*`)"
    r"|(?P<regex>/(?:\\[^\n]|[^/\\\n])+/[a-z]*)"
    r"|(?P<word>[A-Za-z_$][\w$]*)|(?P<number>[0-9]+)|(?P<other>[^\s])"
)


def _literal(token):
    if not token or token[0] not in "\"'`":
        return None
    if token[0] == "`":
        # A template with substitutions is not a literal command.
        return token[1:-1] if "${" not in token else None
    try:
        value = json.loads(token) if token[0] == '"' else ast.literal_eval(token)
        return value if isinstance(value, str) else None
    except (ValueError, SyntaxError):
        return None


def _object_properties(tokens, opening, *, numbers=False):
    """Read simple properties from a literal JS object without evaluating it."""
    fields = {}
    field_names = set()
    uncertain = False
    if opening >= len(tokens) or tokens[opening] != "{":
        return fields, field_names, uncertain

    def consume(start, end):
        nonlocal uncertain
        prop = tokens[start:end]
        if not prop:
            return
        if prop[:3] == [".", ".", "."]:
            # A spread can override any property, including a literal cmd.
            uncertain = True
            return
        if prop[0] == "[":
            # A computed top-level key can override any literal property.
            uncertain = True
            return
        if len(prop) == 1 and prop[0] == "chars":
            field_names.add("chars")
            return

        # Property values may contain arrays, member access, calls, or nested
        # objects. Only a colon immediately after a simple key is structural.
        if len(prop) < 2 or prop[1] != ":":
            return
        raw_key = prop[0]
        key = _literal(raw_key) if raw_key[:1] in ('"', "'") else raw_key
        if not isinstance(key, str):
            return
        field_names.add(key)
        fields.pop(key, None)
        if numbers and len(prop) > 3 and prop[2] == "[":
            try:
                value = ast.literal_eval("".join(prop[2:]))
            except (SyntaxError, ValueError):
                value = None
            if isinstance(value, list) and all(isinstance(v, (str, int)) for v in value):
                fields[key] = value
        if len(prop) == 3:
            value = _literal(prop[2])
            if value is None and numbers and re.fullmatch(r"[0-9]+", prop[2]):
                value = int(prop[2])
            if value is not None:
                fields[key] = value

    start = opening + 1
    depth = 1
    stack = ["}"]
    pairs = {"{": "}", "[": "]", "(": ")"}
    for i in range(opening + 1, len(tokens)):
        token = tokens[i]
        if token in pairs:
            stack.append(pairs[token])
            depth += 1
        elif token in ("}", "]", ")"):
            if depth == 1 and token == "}":
                consume(start, i)
                return fields, field_names, uncertain
            if stack and token == stack[-1]:
                stack.pop()
                depth -= 1
        elif depth == 1 and token == ",":
            consume(start, i)
            start = i + 1
    return fields, field_names, True


def exec_activity(code):
    """Describe literal tools.foo(...) references, not runtime invocation counts.

    Skip strings/comments; computed tool names, aliases and templates with
    substitutions remain unknown. Loops and branches are not evaluated.
    """
    from .drain import command_shape

    tokens = [m.group() for m in _JS_TOKEN.finditer(code) if m.lastgroup != "comment"]
    activities = set()
    examples = []
    for i in range(len(tokens) - 3):
        if tokens[i:i + 2] != ["tools", "."] or tokens[i + 3] != "(":
            continue
        name = tokens[i + 2]
        # Only literal properties of the first object argument are inspected.
        fields, field_names, uncertain_object = (
            _object_properties(tokens, i + 4)
            if i + 4 < len(tokens) and tokens[i + 4] == "{"
            else ({}, set(), False)
        )
        if name == "exec_command":
            command = fields.get("cmd") if not uncertain_object else None
            activities.add("shell: " + command_shape(command) if command else "shell: dynamic command")
            if command:
                examples.append(command)
        elif name == "write_stdin":
            # Missing chars is a poll, but a nonliteral chars expression is unknown.
            if (not uncertain_object and i + 4 < len(tokens) and tokens[i + 4] == "{"
                    and ("chars" not in field_names or fields.get("chars") == "")):
                activities.add("process poll")
            else:
                activities.add("process input/dynamic")
        else:
            activities.add(name)
    return sorted(activities) or ["exec: unclassified code"], examples


_SECRET_LABEL = (
    r"[A-Za-z0-9_-]*(?:api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"auth[_-]?token|client[_-]?(?:secret|key)|private[_-]?key|"
    r"password|passwd|secret|token|credential)[A-Za-z0-9_-]*"
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?P<prefix>(?<![\w-])(?:--)?" + _SECRET_LABEL + r"\s*(?:=|:)\s*)"
    r"(?P<value>\"[^\"]*\"|'[^']*'|[^\s,;&]+)", re.IGNORECASE
)
_SECRET_OPTION = re.compile(
    r"(?P<prefix>(?<![\w-])--?(?:api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"auth[_-]?token|client[_-]?(?:secret|key)|private[_-]?key|"
    r"password|passwd|secret|token|credential)\s+)"
    r"(?P<value>\"[^\"]*\"|'[^']*'|[^\s,;&]+)", re.IGNORECASE
)
_AUTH_HEADER = re.compile(
    r"(?P<prefix>\b(?:proxy-)?authorization\s*[:=]\s*)"
    r"(?P<scheme>bearer|basic|token)(?P<space>\s+)(?P<value>[^\s\"']+)", re.IGNORECASE
)
_OUTPUT_RECORD_PREFIX = re.compile(
    r'^\s*\{\s*"type"\s*:\s*"response_item"\s*,.*?'
    r'"payload"\s*:\s*\{\s*"type"\s*:\s*"'
    r'(?:custom_tool_call_output|function_call_output|local_shell_call_output)"'
)


def _safe_example(command, limit):
    """Bound and redact common credential assignments in an opt-in example."""
    from .drain import escaped_command

    value = _AUTH_HEADER.sub(
        lambda m: m.group("prefix") + m.group("scheme") + m.group("space") + "<redacted>",
        command,
    )

    def replace_value(match):
        raw = match.group("value")
        if raw[:1] in ('"', "'") and raw[-1:] == raw[:1]:
            return match.group("prefix") + raw[:1] + "<redacted>" + raw[-1:]
        return match.group("prefix") + "<redacted>"

    value = _SECRET_ASSIGNMENT.sub(replace_value, value)
    value = _SECRET_OPTION.sub(replace_value, value)
    return escaped_command(value, limit)


def _json_string_end(raw, start):
    if start >= len(raw) or raw[start] != '"':
        return None
    index = start + 1
    while index < len(raw):
        char = raw[index]
        if char == "\\":
            index += 2
            continue
        if char == '"':
            return index + 1
        index += 1
    return None


def _skip_json_value(raw, start):
    while start < len(raw) and raw[start].isspace():
        start += 1
    if start >= len(raw):
        return start
    if raw[start] == '"':
        return _json_string_end(raw, start) or len(raw)
    if raw[start] in "{[":
        pairs = {"{": "}", "[": "]"}
        stack = [pairs[raw[start]]]
        index = start + 1
        while index < len(raw) and stack:
            char = raw[index]
            if char == '"':
                index = _json_string_end(raw, index) or len(raw)
                continue
            if char in pairs:
                stack.append(pairs[char])
            elif char in "}]" and char == stack[-1]:
                stack.pop()
            index += 1
        return index
    index = start
    while index < len(raw) and raw[index] not in ",}]":
        index += 1
    return index


def _json_member(raw, member, start=0, *, start_only=False):
    """Return the value slice for one shallow JSON object member."""
    index = start
    while index < len(raw) and raw[index].isspace():
        index += 1
    if index >= len(raw) or raw[index] != "{":
        return None
    index += 1
    while index < len(raw):
        while index < len(raw) and (raw[index].isspace() or raw[index] == ","):
            index += 1
        if index >= len(raw) or raw[index] == "}":
            return None
        key_end = _json_string_end(raw, index)
        if key_end is None:
            return None
        try:
            key = json.loads(raw[index:key_end])
        except (ValueError, TypeError):
            return None
        index = key_end
        while index < len(raw) and raw[index].isspace():
            index += 1
        if index >= len(raw) or raw[index] != ":":
            return None
        value_start = index + 1
        while value_start < len(raw) and raw[value_start].isspace():
            value_start += 1
        if key == member and start_only:
            return value_start, value_start
        value_end = _skip_json_value(raw, value_start)
        if key == member:
            return value_start, value_end
        index = value_end
    return None


def _record_types(raw):
    """Read record and payload type strings without decoding other values."""
    raw = raw[:65536]
    record_type = _json_member(raw, "type")
    payload = _json_member(raw, "payload", start_only=True)
    if not record_type or not payload:
        return None, None
    try:
        root = json.loads(raw[slice(*record_type)])
        payload_type = _json_member(raw, "type", payload[0])
        child = json.loads(raw[slice(*payload_type)]) if payload_type else None
    except (ValueError, TypeError):
        return None, None
    return root, child


def _source_activity(path, describe=None):
    """Join calls to usage records in transcript order, flushing uncertain batches."""
    from . import drain as d

    pending = {}
    saw_output = False
    try:
        with Path(path).open(encoding="utf-8") as source:
            for raw in source:
                if not any(marker in raw for marker in (
                    '"token_usage_record"', '"custom_tool_call"',
                    '"function_call"', '"local_shell_call"',
                    '"task_started"', '"task_complete"', '"turn_context"',
                    '"custom_tool_call_output"', '"function_call_output"',
                    '"local_shell_call_output"',
                )):
                    continue
                if _OUTPUT_RECORD_PREFIX.search(raw):
                    if pending:
                        saw_output = True
                    continue
                if any(item in raw for item in (
                    '"custom_tool_call_output"', '"function_call_output"',
                    '"local_shell_call_output"',
                )):
                    record_type, payload_type = _record_types(raw)
                    if record_type == "response_item" and payload_type in d.CODEX_TOOL_OUTPUT_ITEMS:
                        if pending:
                            saw_output = True
                        continue
                try:
                    record = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                if not isinstance(record, dict):
                    continue
                payload = record.get("payload")
                if not isinstance(payload, dict):
                    continue
                item = payload.get("type")
                if record.get("type") == "response_item" and item in d.CODEX_TOOL_OUTPUT_ITEMS:
                    if pending:
                        saw_output = True
                    continue
                if record.get("type") == "event_msg" and item in ("task_started", "task_complete"):
                    if pending:
                        yield "", pending
                        pending = {}
                    saw_output = False
                    continue
                if record.get("type") == "turn_context" and pending:
                    yield "", pending
                    pending = {}
                    saw_output = False
                    continue
                if record.get("type") == "response_item" and item in d.CODEX_TOOL_CALL_ITEMS:
                    if pending and saw_output:
                        # A later call after outputs may be from another model
                        # response. Without its usage record, do not price it
                        # against whatever response appears next.
                        yield "", pending
                        pending = {}
                        saw_output = False
                    elif not pending:
                        # Outputs can follow a usage record in some transcripts;
                        # they must not mark the next response's batch as stale.
                        saw_output = False
                    call_id = payload.get("call_id") or payload.get("id")
                    if not isinstance(call_id, str):
                        continue
                    name = d.codex_tool_name(payload, item)
                    if describe is not None:
                        activity, examples = describe(name, payload, item), []
                    elif item == "custom_tool_call" and name.split(".")[-1] == "exec":
                        code = payload.get("input")
                        activity, examples = exec_activity(code) if isinstance(code, str) else (["exec: unclassified code"], [])
                    elif item == "local_shell_call" or name.lower().split(".")[-1] in d.SHELL_TOOL_NAMES:
                        command = d.command_from_payload(payload, item)
                        activity = ["shell: " + d.command_shape(command)] if command else [name]
                        examples = [command] if command else []
                    else:
                        activity, examples = [name], []
                    pending[call_id] = (activity, examples, name)
                elif record.get("type") == "token_usage_record":
                    response = payload.get("response_id")
                    yield response if isinstance(response, str) else "", pending
                    pending = {}
                    saw_output = False
        if pending:
            yield "", pending
    except OSError:
        return


def explain_activity(analysis, calls):
    """Count whole generating responses once; do not divide cost among tools."""
    from . import drain as d

    selected = {(c.source, c.call_id): c for c in calls if c.harness == "codex" and c.source and c.call_id}
    scopes = {(c.session_id, c.prompt) for c in calls if c.harness == "codex"}
    events = defaultdict(list)
    scope_events = defaultdict(list)
    for event in analysis.scan.events.get("codex", []):
        if (event[d.EVENT_SESSION], event[d.EVENT_PROMPT]) not in scopes:
            continue
        session_id = event[d.EVENT_SESSION]
        response_id = event[d.EVENT_ID]
        if response_id:
            key = (session_id, response_id)
            events[key].append(event)
            scope_events[key].append(event)
        else:
            # The scan already deduplicates transcript files. Keep unidentified
            # response records in the scope denominator, but never join them to
            # a tool call by timestamp or token similarity.
            key = (session_id, "", event[d.EVENT_TS], len(scope_events))
            scope_events[key].append(event)
    groups = {}
    families = {}
    matched = set()
    responses = {}
    unpriced_response_calls = 0
    matched_unweighted_responses = 0
    for source in sorted({key[0] for key in selected}):
        for response_id, batch in _source_activity(source):
            chosen = [(key, selected[key]) for cid in batch if (key := (source, cid)) in selected and key not in matched]
            if not chosen:
                continue
            keys = {(call.session_id, response_id) for _, call in chosen}
            key = next(iter(keys)) if len(keys) == 1 and response_id else (source, chosen[0][0])
            entry = responses.setdefault(key, {
                "calls": [], "names": set(), "families": set(), "examples": [], "other": False,
            })
            entry["other"] |= any((source, cid) not in selected for cid in batch)
            for call_key, call in chosen:
                matched.add(call_key)
                entry["calls"].append(call)
                names, examples, family = batch[call_key[1]]
                entry["names"].update(names)
                entry["families"].add(family)
                entry["examples"].extend(examples)

    scope_model_responses = len(scope_events)
    scope_weighted_units = 0.0
    ambiguous_scope_responses = 0
    scope_unweighted_responses = 0
    scope_uncached_input_tokens = 0
    scope_cached_input_tokens = 0
    scope_output_tokens = 0
    for key, candidates in scope_events.items():
        if len(candidates) != 1:
            ambiguous_scope_responses += 1
            continue
        event = candidates[0]
        if analysis.weights.price_row("codex", event[d.EVENT_MODEL]) is None:
            scope_unweighted_responses += 1
        scope_weighted_units += d.event_units(
            "codex", event, d.CODEX_KINDS, analysis.weights, analysis.args
        )
        scope_uncached_input_tokens += event[d.EVENT_KINDS]
        scope_cached_input_tokens += event[d.EVENT_KINDS + 1]
        scope_output_tokens += event[d.EVENT_KINDS + 3]
    unpriced_response_batches = sum(
        1 for key in responses if len(events.get(key, [])) != 1
    )

    def empty_row(activity, tool_family):
        return {
            "activity": activity,
            "tool_family": tool_family,
            "tool_calls": 0,
            "model_responses": 0,
            "uncached_input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "weighted_units": 0.0,
            "unweighted_responses": 0,
            "result_chars": 0,
            "unmatched_calls": 0,
            "examples": [],
            "root_calls": 0,
            "subagent_calls": 0,
        }

    for key, entry in responses.items():
        names = sorted(entry["names"])
        family_names = set(entry["families"])
        if entry["other"]:
            names = sorted(set(names) | {"other calls in response"})
            family_names.add("other calls in response")
        label = " + ".join(names)
        family_label = " + ".join(sorted(family_names))
        group_key = (label, family_label)
        group = groups.setdefault(group_key, empty_row(label, family_label))
        family = families.setdefault(
            family_label, empty_row(family_label, family_label)
        )
        for call in entry["calls"]:
            for row in (group, family):
                row["tool_calls"] += 1
                row["result_chars"] += call.chars
                row["subagent_calls" if call.subagent else "root_calls"] += 1
        for example in entry["examples"]:
            example = _safe_example(example, d.EXPLAIN_EXAMPLE_CHARS)
            if example not in group["examples"] and len(group["examples"]) < d.EXPLAIN_MAX_EXAMPLES:
                group["examples"].append(example)
        # Ambiguous or absent response IDs are explicitly unpriced.
        candidates = events.get(key, [])
        if len(candidates) != 1:
            unpriced_response_calls += len(entry["calls"])
            for row in (group, family):
                row["unmatched_calls"] += len(entry["calls"])
            continue
        event = candidates[0]
        unweighted = analysis.weights.price_row("codex", event[d.EVENT_MODEL]) is None
        for row in (group, family):
            row["model_responses"] += 1
            row["unweighted_responses"] += 1 if unweighted else 0
            row["uncached_input_tokens"] += event[d.EVENT_KINDS]
            row["cached_input_tokens"] += event[d.EVENT_KINDS + 1]
            row["output_tokens"] += event[d.EVENT_KINDS + 3]
            row["weighted_units"] += d.event_units(
                "codex", event, d.CODEX_KINDS, analysis.weights, analysis.args
            )
        if unweighted:
            matched_unweighted_responses += 1
    codex_calls = [call for call in calls if call.harness == "codex"]
    orphan_calls = [
        call for call in codex_calls if (call.source, call.call_id) not in matched
    ]
    unexplained_result_chars = sum(call.chars for call in orphan_calls)
    result_chars = sum(call.chars for call in codex_calls)
    explained_result_chars = result_chars - unexplained_result_chars
    if orphan_calls:
        unlinked = empty_row("tool activity unlinked", "unlinked tool calls")
        unlinked["tool_calls"] = len(orphan_calls)
        unlinked["result_chars"] = unexplained_result_chars
        unlinked["unmatched_calls"] = len(orphan_calls)
        unlinked["root_calls"] = sum(not call.subagent for call in orphan_calls)
        unlinked["subagent_calls"] = sum(bool(call.subagent) for call in orphan_calls)
        groups[(unlinked["activity"], unlinked["tool_family"])] = unlinked
        families[unlinked["tool_family"]] = unlinked.copy()
        families[unlinked["tool_family"]]["examples"] = []

    rows = sorted(
        groups.values(),
        key=lambda r: (r["weighted_units"], r["tool_calls"], r["activity"]),
        reverse=True,
    )
    for row in rows:
        row["est_result_tokens"] = row["result_chars"] / d.CHARS_PER_TOKEN
    family_rows = sorted(
        families.values(),
        key=lambda r: (r["weighted_units"], r["tool_calls"], r["tool_family"]),
        reverse=True,
    )
    for row in family_rows:
        row["est_result_tokens"] = row["result_chars"] / d.CHARS_PER_TOKEN
    matched_weighted_units = sum(r["weighted_units"] for r in rows)
    matched_model_responses = sum(r["model_responses"] for r in rows)
    return {
        "groups": rows,
        "tool_families": family_rows,
        "tool_calls": len(codex_calls),
        "result_chars": result_chars,
        "est_result_tokens": result_chars / d.CHARS_PER_TOKEN,
        "explained_result_chars": explained_result_chars,
        "unexplained_result_chars": unexplained_result_chars,
        "explained_calls": len(matched),
        "model_responses": matched_model_responses,
        "unmatched_calls": len(codex_calls) - len(matched) + unpriced_response_calls,
        "unpriced_tool_call_responses": unpriced_response_batches,
        "weighted_units": matched_weighted_units,
        "scope_model_responses": scope_model_responses,
        "scope_weighted_units": scope_weighted_units,
        "scope_unweighted_responses": scope_unweighted_responses,
        "matched_unweighted_responses": matched_unweighted_responses,
        "scope_uncached_input_tokens": scope_uncached_input_tokens,
        "scope_cached_input_tokens": scope_cached_input_tokens,
        "scope_output_tokens": scope_output_tokens,
        "scope_ambiguous_responses": ambiguous_scope_responses,
        "uncovered_model_responses": max(0, scope_model_responses - matched_model_responses),
        "coverage_model_responses": (
            matched_model_responses / scope_model_responses if scope_model_responses else None
        ),
        "coverage_weighted_usage": (
            matched_weighted_units / scope_weighted_units if scope_weighted_units else None
        ),
        "poll_only_calls": sum(r["tool_calls"] for r in rows if set(r["activity"].split(" + ")) <= {"process poll", "clock__curr_time"}),
        "note": "Each uniquely matched generating response is counted once, including reasoning and other actions. Activity rows associate a response with static tool references, not exact per-tool cost or avoidable usage. Missing, ambiguous or out-of-scope response IDs remain unpriced. Scope totals include responses without selected tool calls; ambiguous duplicate IDs are excluded from scope weighted usage. Zero weighted units can mean that model weights are unavailable.",
    }
