"""Read-only, evidence-bounded reports for Codex Shake activity.

Successful Shake operations leave a small ``compacted`` marker in the rollout.
The replacement history on the same record can be very large and sensitive, so
this module scans only its scalar marker prefix and never retains that payload.
Optional ``logs_2.sqlite`` records add the transform's estimated counts; actual
before/after request context is measured separately from token-usage records.
"""

import json
import math
import os
import re
import sqlite3
import tomllib
from collections import defaultdict
from pathlib import Path


_SHAKE_PREFIX = "[shake] context reduced surgically"
_MARKER_HEADER_SCAN_BYTES = 64 * 1024
_SHAKE_KINDS = (
    (" (automatic, cold resume)", "automatic_cold_resume"),
    (" (automatic, escalated)", "automatic_escalated"),
    (" (automatic)", "automatic"),
    ("", "manual"),
)
_SHAKE_TARGET = "codex_core::shake"
_LOG_FIELDS = (
    "trigger", "mode", "tokens_freed", "tool_outputs_elided",
    "blocks_elided", "images_dropped", "thinking_dropped", "model",
)
_LOG_MATCH_SECONDS = 30.0


def _event_thread(row, d):
    thread = row[d.EVENT_THREAD]
    if thread:
        return str(thread)
    return "" if row[d.EVENT_SUB] else str(row[d.EVENT_SESSION])


def _event_inputs(event, analysis, d):
    context = int(d.event_context(event, "codex"))
    cached = int(event[d.EVENT_KINDS + 1])
    uncached = max(0, context - cached)
    model = str(event[d.EVENT_MODEL] or "")
    weighted = float(d.event_units("codex", event, d.CODEX_KINDS,
                                   analysis.weights, analysis.args))
    return {
        "context": context,
        "cached": cached,
        "uncached": uncached,
        "model": model or None,
        "weighted": weighted,
    }


def estimate_payback(reduction, extra_uncached, calls, input_weight, cached_weight, *, cold_resume=False):
    """Constant retained-context estimate, with one initial cache rebuild.

    Avoided reads would have been cached. The rebuild changes cached tokens to
    uncached, so its premium is the DIFFERENCE between their weights. The first
    post-Shake call both pays that premium and benefits from reduced context.
    On a presumed-expired cache, the first avoided read is uncached and the
    survivor rebuild would happen either way, so no rebuild premium is charged.
    Output is unchanged in this model. Prices are units per million tokens.
    """
    if reduction <= 0:
        return {"status": "no_reduction"}
    if cached_weight <= 0 or input_weight < cached_weight:
        return {"status": "unsupported_weights"}
    read_savings = reduction * calls
    premium = (0 if cold_resume else max(0, extra_uncached) * (input_weight - cached_weight) / 1_000_000)
    per_call = reduction * cached_weight / 1_000_000
    first_call_bonus = reduction * (input_weight - cached_weight) / 1_000_000 if cold_resume and calls else 0
    net_units = per_call * calls + first_call_bonus - premium
    break_even = max(1, math.ceil(premium / per_call - 1e-12))
    return {
        "status": "estimated", "model": "cold_first_read_uncached" if cold_resume else "constant_reduction_one_rebuild",
        "cache_baseline": "presumed_expired" if cold_resume else "warm",
        "first_call_uncached_tokens_avoided": reduction if cold_resume and calls else 0,
        "reduction_tokens_per_call": reduction,
        "extra_uncached_tokens": max(0, extra_uncached),
        "input_weight": input_weight, "cached_input_weight": cached_weight,
        "cache_rebuild_premium_units": premium,
        "saved_units_per_call": per_call,
        "break_even_calls": break_even,
        "post_shake_calls": calls,
        "break_even_reached": calls >= break_even,
        "read_tokens_saved": read_savings,
        "net_input_units_saved": net_units,
        "cache_write_read_equivalent_tokens": premium * 1_000_000 / cached_weight,
        "net_read_equivalent_tokens_saved": net_units * 1_000_000 / cached_weight,
    }


def _payback(before, after, post_events, reduction, same_prompt, marker_count, analysis, d, kind):
    if before is None or after is None or reduction is None:
        return {"status": "missing_or_ambiguous_usage"}
    if marker_count != 1:
        return {"status": "multiple_shakes"}
    if not same_prompt:
        return {"status": "prompt_changed"}
    model = before[d.EVENT_MODEL]
    if any(event[d.EVENT_MODEL] != model for event in [after, *post_events]):
        return {"status": "model_changed"}
    prices = analysis.weights.price_row("codex", model)
    if prices is None:
        return {"status": "unknown_weights"}
    multiplier = analysis.args.long_context_multiplier
    if multiplier != 1 and any(event[d.EVENT_LONG] != before[d.EVENT_LONG]
                               for event in [after, *post_events]):
        return {"status": "weight_regime_changed"}
    scale = multiplier if before[d.EVENT_LONG] else 1
    extra_uncached = int(after[d.EVENT_KINDS]) - int(before[d.EVENT_KINDS])
    return estimate_payback(reduction, extra_uncached, len(post_events),
                            prices[0] * scale, prices[1] * scale,
                            cold_resume=kind == "automatic_cold_resume")


def _json_string_end(raw, start):
    if start >= len(raw) or raw[start] != 34:
        return None
    index = start + 1
    while index < len(raw):
        byte = raw[index]
        if byte == 92:
            index += 2
            continue
        if byte == 34:
            return index + 1
        index += 1
    return None


def _json_value_end(raw, start):
    if start >= len(raw):
        return None
    if raw[start] == 34:
        return _json_string_end(raw, start)
    if raw[start] not in (91, 123):
        index = start
        while index < len(raw) and raw[index] not in b",}] \t\r\n":
            index += 1
        return index
    stack = [93 if raw[start] == 91 else 125]
    index = start + 1
    while index < len(raw) and stack:
        byte = raw[index]
        if byte == 34:
            end = _json_string_end(raw, index)
            if end is None:
                return None
            index = end
            continue
        if byte == 91:
            stack.append(93)
        elif byte == 123:
            stack.append(125)
        elif byte == stack[-1]:
            stack.pop()
        index += 1
    return index if not stack else None


def _object_prefix_fields(raw, start, wanted, *, nested_payload=False):
    heavy = {"replacement_history", "retained_context", "guardian_history", "mcp_resources"}
    decoder = json.JSONDecoder()
    fields = {}
    index = start
    while index < len(raw) and raw[index] in b" \t\r\n":
        index += 1
    if index >= len(raw) or raw[index] != 123:
        return {}
    index += 1
    while index < len(raw):
        while index < len(raw) and raw[index] in b" \t\r\n,":
            index += 1
        if index >= len(raw) or raw[index] == 125:
            break
        key_end = _json_string_end(raw, index)
        if key_end is None:
            return {}
        try:
            key = decoder.decode(raw[index:key_end].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
        index = key_end
        while index < len(raw) and raw[index] in b" \t\r\n":
            index += 1
        if index >= len(raw) or raw[index] != 58:
            return {}
        index += 1
        while index < len(raw) and raw[index] in b" \t\r\n":
            index += 1
        if key in heavy:
            break
        if key == "payload" and not nested_payload and "message" not in fields:
            payload_fields = _object_prefix_fields(raw, index, {"message"}, nested_payload=True)
            if "message" in payload_fields:
                fields["message"] = payload_fields["message"]
            break
        if key in wanted:
            end = _json_string_end(raw, index)
            if end is None:
                return {}
            try:
                fields[key] = decoder.decode(raw[index:end].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return {}
            index = end
            if key == "message" and wanted.issubset(fields):
                break
        else:
            end = _json_value_end(raw, index)
            if end is None:
                return {}
            index = end
    return fields


def _prefix_fields(raw):
    """Extract only marker scalars from top-level or rollout payload headers."""
    return _object_prefix_fields(raw, 0, {"type", "timestamp", "message"})


def _shake_kind(message):
    if not isinstance(message, str) or not message.startswith(_SHAKE_PREFIX):
        return None
    suffix = message[len(_SHAKE_PREFIX):]
    for expected, kind in _SHAKE_KINDS:
        if suffix == expected:
            return kind
    return None


def _iter_markers(source, entry, harness):
    if harness != "codex" or not source:
        return
    path = Path(source)
    root = None
    for parent in path.parents:
        if parent.name == "sessions":
            root = parent.parent.resolve()
            break
    session_id = getattr(entry, "last_session", None)
    if not session_id:
        sessions = getattr(entry, "sessions", {})
        if sessions:
            session_id = next(iter(sessions))
    thread_id = getattr(entry, "thread_id", None)
    if not session_id or not thread_id:
        return
    try:
        stream = path.open("rb")
    except OSError:
        return
    agent_name = None
    with stream:
        for line_number, raw in enumerate(stream, 1):
            prefix = raw[:_MARKER_HEADER_SCAN_BYTES]
            if b'"session_meta"' in prefix[:512]:
                try:
                    record = json.loads(prefix)
                    payload = record.get("payload") or {}
                    if record.get("type") == "session_meta" and payload.get("id") == thread_id:
                        source = payload.get("source")
                        spawn = (((source.get("subagent") or {}).get("thread_spawn") or {})
                                 if isinstance(source, dict) and isinstance(source.get("subagent"), dict) else {})
                        name = payload.get("agent_nickname") or spawn.get("agent_nickname")
                        if isinstance(name, str) and name.strip():
                            agent_name = " ".join(name.split())[:64]
                except (ValueError, TypeError, AttributeError):
                    pass
            if _SHAKE_PREFIX.encode("ascii") not in prefix:
                continue
            fields = _prefix_fields(prefix)
            if fields.get("type") != "compacted":
                continue
            message = fields.get("message")
            kind = _shake_kind(message)
            timestamp = fields.get("timestamp")
            if not kind or not timestamp:
                continue
            try:
                from . import drain as d

                epoch = d.parse_timestamp(timestamp)
            except (TypeError, ValueError, OverflowError):
                continue
            if epoch is None:
                continue
            yield {
                "session_id": str(session_id),
                "thread_id": str(thread_id),
                "timestamp": float(epoch),
                "kind": kind,
                "agent_name": agent_name,
                "source_line": line_number,
                "_root": root,
            }


def _logs_paths(root):
    paths = [root / "logs_2.sqlite"]
    configured = None
    config_path = root / "config.toml"
    try:
        with config_path.open("rb") as stream:
            config = tomllib.load(stream)
        configured = config.get("sqlite_home")
    except (OSError, tomllib.TOMLDecodeError):
        pass
    if isinstance(configured, str) and configured.strip():
        home = Path(configured).expanduser()
        if not home.is_absolute():
            home = root / home
        paths.insert(0, home / "logs_2.sqlite")

    # CODEX_SQLITE_HOME is honored only when this is the active Codex home;
    # this avoids accidentally reading an unrelated host database while Nenpi
    # analyzes an explicit transcript fixture or archive root.
    sqlite_home = os.environ.get("CODEX_SQLITE_HOME")
    codex_home = os.environ.get("CODEX_HOME")
    if not codex_home:
        codex_home = str(Path.home() / ".codex")
    try:
        is_active_home = root.resolve() == Path(codex_home).expanduser().resolve()
    except OSError:
        is_active_home = False
    if sqlite_home and is_active_home:
        home = Path(sqlite_home).expanduser()
        if not home.is_absolute():
            home = root / home
        paths.insert(0, home / "logs_2.sqlite")

    unique = []
    seen = set()
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _epoch(ts, nanos):
    try:
        return float(ts) + float(nanos or 0) / 1_000_000_000
    except (TypeError, ValueError, OverflowError):
        return None


def _log_field(message, key):
    match = re.search(
        r"(?:^|[\s{,])" + re.escape(key) + r"=(\"(?:\\.|[^\"])*\"|[^\s,}]+)",
        message,
    )
    if not match:
        return None
    value = match.group(1)
    if value.startswith('"'):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value[1:-1]
    return value


def _read_stats(path, thread_ids):
    """Read only Shake metric rows. URI mode=ro prevents database creation."""
    if not thread_ids or not path.is_file():
        return None
    try:
        uri = path.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=0.15)
        try:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(logs)")}
            required = {"target", "thread_id", "ts", "ts_nanos", "level", "feedback_log_body"}
            if not required.issubset(columns):
                return None
            placeholders = ",".join("?" for _ in thread_ids)
            query = (
                "SELECT ts, ts_nanos, level, feedback_log_body, thread_id FROM logs "
                "WHERE target = ? AND level = 'INFO' AND thread_id IN (" + placeholders + ")"
            )
            rows = connection.execute(query, (_SHAKE_TARGET, *sorted(thread_ids))).fetchall()
        finally:
            connection.close()
    except (OSError, sqlite3.Error, ValueError):
        return None

    parsed = []
    for ts, nanos, level, message, thread_id in rows:
        if str(level).upper() != "INFO" or not isinstance(message, str):
            continue
        fields = {key: _log_field(message, key) for key in _LOG_FIELDS}
        if fields["trigger"] not in dict((kind, suffix) for suffix, kind in _SHAKE_KINDS).keys():
            continue
        record = {
            "timestamp": _epoch(ts, nanos),
            "thread_id": str(thread_id),
            "trigger": fields["trigger"],
            "mode": fields["mode"],
            "model": fields["model"],
        }
        for source_key, output_key in (
            ("tokens_freed", "tokens_freed_estimate"),
            ("tool_outputs_elided", "tool_outputs_elided"),
            ("blocks_elided", "blocks_elided"),
            ("images_dropped", "images_dropped"),
            ("thinking_dropped", "thinking_dropped"),
        ):
            try:
                value = int(fields[source_key])
            except (TypeError, ValueError):
                continue
            record[output_key] = value
        parsed.append(record)
    return parsed


def _attach_stats(markers, database_records):
    candidates_by_marker = {}
    candidates_by_record = defaultdict(list)
    for marker_index, marker in enumerate(markers):
        candidates = []
        for record_index, record in enumerate(database_records):
            if (record["thread_id"] != marker["thread_id"]
                    or record["trigger"] != marker["kind"]
                    or record["timestamp"] is None):
                continue
            distance = abs(record["timestamp"] - marker["timestamp"])
            if distance <= _LOG_MATCH_SECONDS:
                candidates.append(record_index)
                candidates_by_record[record_index].append(marker_index)
        candidates_by_marker[marker_index] = candidates

    matches = {}
    for marker_index, candidates in candidates_by_marker.items():
        if len(candidates) != 1 or len(candidates_by_record[candidates[0]]) != 1:
            continue
        record_index = candidates[0]
        record = dict(database_records[record_index])
        record.pop("timestamp", None)
        record.pop("thread_id", None)
        record["evidence"] = _SHAKE_TARGET
        matches[marker_index] = record
    return matches, len(matches)


def _marker_in_window(marker, analysis):
    if getattr(analysis.args, "whole_session", False):
        return True
    return ((analysis.since is None or marker["timestamp"] >= analysis.since)
            and (analysis.until is None or marker["timestamp"] <= analysis.until))


def analyze_shakes(analysis, session_key=None, prompt_index=None):
    """Describe persisted Shake markers and observed same-thread usage around them.

    ``session_key`` is the usual ``(harness, session_id)`` selector. A marker
    means Shake changed history and persisted a new checkpoint. The adjacent
    context difference is observational; it includes any intervening prompt
    input and is not a quota-savings counterfactual.
    """
    from . import drain as d

    events_by_thread = defaultdict(list)
    allowed_sessions = set()
    for event in analysis.scan.events.get("codex", []):
        key = ("codex", str(event[d.EVENT_SESSION]))
        if session_key and key != session_key:
            continue
        allowed_sessions.add(key[1])
        thread_id = _event_thread(event, d)
        events_by_thread[(key[1], thread_id)].append(event)
    for rows in events_by_thread.values():
        rows.sort(key=lambda event: event[d.EVENT_TS])

    all_markers = []
    seen_markers = set()
    source_roots = set()
    for entry, harness, source in getattr(analysis.scan, "_tool_sources", ()):
        if harness != "codex" or not source:
            continue
        if session_key and session_key[0] == "codex":
            indexed_session = str(getattr(entry, "last_session", "") or "")
            indexed_sessions = getattr(entry, "sessions", {})
            if (indexed_session and indexed_session != session_key[1]
                    and session_key[1] not in indexed_sessions):
                continue
        source_path = Path(source)
        for parent in source_path.parents:
            if parent.name == "sessions":
                source_roots.add(parent.parent.resolve())
                break
        for marker in _iter_markers(source, entry, harness):
            if marker["session_id"] not in allowed_sessions:
                continue
            if not _marker_in_window(marker, analysis):
                continue
            dedupe = (marker["session_id"], marker["thread_id"],
                      marker["timestamp"], marker["kind"])
            if dedupe in seen_markers:
                continue
            seen_markers.add(dedupe)
            all_markers.append(marker)
    all_markers.sort(key=lambda marker: (marker["timestamp"], marker["session_id"],
                                         marker["thread_id"], marker["source_line"]))

    records = []
    roots_read = 0
    seen_databases = set()
    for root in sorted(source_roots):
        thread_ids = {
            marker["thread_id"] for marker in all_markers if marker["_root"] == root
        }
        if not thread_ids:
            continue
        db_paths = _logs_paths(root)
        found_for_root = False
        for path in db_paths:
            if path in seen_databases:
                continue
            seen_databases.add(path)
            db_records = _read_stats(path, thread_ids)
            if db_records is None:
                continue
            roots_read += 1
            found_for_root = True
            records.extend(db_records)
            break
        if found_for_root:
            continue

    # Match records by thread, trigger, and nearby timestamp. The lookup is
    # intentionally unique-only so adjacent runs never inherit each other's
    # estimated transform counts.
    matches, matched_count = _attach_stats(all_markers, records)

    output = []
    for marker_index, marker in enumerate(all_markers):
        key = (marker["session_id"], marker["thread_id"])
        stream = events_by_thread.get(key, [])
        tied = [event for event in stream
                if event[d.EVENT_TS] == marker["timestamp"]]
        before = None
        after = None
        if not tied:
            before = next((event for event in reversed(stream)
                           if event[d.EVENT_TS] < marker["timestamp"]), None)
            after = next((event for event in stream
                          if event[d.EVENT_TS] > marker["timestamp"]), None)
        owner = after or (tied[0] if tied else before)
        owner_prompt = owner[d.EVENT_PROMPT] if owner is not None else None
        if prompt_index is not None and owner_prompt != prompt_index:
            continue

        before_data = _event_inputs(before, analysis, d) if before is not None else None
        after_data = _event_inputs(after, analysis, d) if after is not None else None
        intervening_shake_count = 0
        if before is not None and after is not None:
            intervening_shake_count = sum(
                other["session_id"] == marker["session_id"]
                and other["thread_id"] == marker["thread_id"]
                and before[d.EVENT_TS] < other["timestamp"] < after[d.EVENT_TS]
                for other in all_markers
            )
        same_prompt = bool(before is not None and after is not None
                           and before[d.EVENT_PROMPT] == after[d.EVENT_PROMPT])
        same_model = bool(before_data and after_data
                          and before_data["model"] == after_data["model"])
        weighted_comparable = bool(
            same_prompt and same_model and before_data["weighted"] > 0
            and after_data["weighted"] > 0
            and intervening_shake_count == 1
        )

        next_marker = next((other["timestamp"] for other in all_markers[marker_index + 1:]
                            if other["session_id"] == marker["session_id"]
                            and other["thread_id"] == marker["thread_id"]
                            and other["timestamp"] > marker["timestamp"]), None)
        post_events = [event for event in stream
                       if event[d.EVENT_TS] > marker["timestamp"]
                       and event[d.EVENT_PROMPT] == owner_prompt
                       and (next_marker is None or event[d.EVENT_TS] < next_marker)]
        post_contexts = [int(d.event_context(event, "codex")) for event in post_events]
        before_context = before_data["context"] if before_data else None
        after_context = after_data["context"] if after_data else None
        reduction = (before_context - after_context
                     if before_context is not None and after_context is not None
                     and intervening_shake_count <= 1 else None)
        if tied:
            effect = "ambiguous_timestamp"
        elif intervening_shake_count > 1:
            effect = "multiple_shakes_between_requests"
        elif reduction is None:
            effect = "insufficient_context_data"
        elif reduction > 0:
            effect = "context_lower_after_marker"
        elif reduction < 0:
            effect = "context_higher_after_marker"
        else:
            effect = "no_context_change_observed"

        stats = matches.get(marker_index)
        role = d._drill_thread_classification(
            analysis.scan.thread_metadata, marker["session_id"],
            marker["thread_id"], {}, harness="codex",
        )
        if role == "descendant":
            role = "subagent"
        output.append({
            "session_id": marker["session_id"],
            "thread_id": marker["thread_id"],
            "role": role,
            "agent_name": marker.get("agent_name"),
            "prompt": owner_prompt,
            "timestamp": marker["timestamp"],
            "kind": marker["kind"],
            "status": "applied",
            "effect": effect,
            "cross_prompt": (not same_prompt if before is not None and after is not None else None),
            "intervening_shake_count": intervening_shake_count,
            "before_request_timestamp": before[d.EVENT_TS] if before is not None else None,
            "after_request_timestamp": after[d.EVENT_TS] if after is not None else None,
            "idle_before_shake_seconds": marker["timestamp"] - before[d.EVENT_TS] if before is not None else None,
            "before_context_tokens": before_context,
            "after_context_tokens": after_context,
            "reduction_tokens": reduction,
            "before_cached_input_tokens": before_data["cached"] if before_data else None,
            "after_cached_input_tokens": after_data["cached"] if after_data else None,
            "before_uncached_input_tokens": before_data["uncached"] if before_data else None,
            "after_uncached_input_tokens": after_data["uncached"] if after_data else None,
            "before_model": before_data["model"] if before_data else None,
            "after_model": after_data["model"] if after_data else None,
            "weighted_units_before": before_data["weighted"] if before_data else None,
            "weighted_units_after": after_data["weighted"] if after_data else None,
            "weighted_delta_units": (after_data["weighted"] - before_data["weighted"]
                                     if weighted_comparable else None),
            "weighted_comparable": weighted_comparable,
            "subsequent_response_count": len(post_events),
            "subsequent_input_tokens": sum(post_contexts),
            "payback": _payback(before, after, post_events, reduction, same_prompt,
                                 intervening_shake_count, analysis, d, marker["kind"]),
            "stats": stats,
            "evidence": "persisted_compacted_marker",
        })

    roots_with_logs = len({
        path for root in source_roots for path in _logs_paths(root) if path.is_file()
    })
    selected_marker_count = len(output)
    selected_matched_count = sum(item["stats"] is not None for item in output)
    if not selected_marker_count:
        stats_status = "not_applicable"
    elif roots_read and selected_matched_count == selected_marker_count:
        stats_status = "available"
    elif selected_matched_count:
        stats_status = "partial"
    else:
        stats_status = "unavailable"
    notes = [
        "A persisted [shake] marker proves an applied history change; its absence cannot distinguish no-op, disabled, or unrecorded Shake activity.",
        "The adjacent context difference includes intervening prompt input and is an observed comparison, not guaranteed quota savings.",
        "Cached and uncached input are separate; a lower total context can still have higher uncached input after the cache rebuild.",
        "Shake's tokens_freed_estimate is a transform estimate, separate from observed request context and later usage.",
    ]
    if not selected_marker_count:
        notes[0] = "No applied Shake marker was found in the selected transcript scope; no-op, disabled, and never-run states are not distinguishable from that absence."
    return {
        "events": output,
        "coverage": {
            "observed_run_status": "applied_markers_found" if selected_marker_count else "no_applied_marker_observed",
            "applied_markers_found": selected_marker_count,
            "events_with_context_comparison": sum(
                item["before_context_tokens"] is not None and item["after_context_tokens"] is not None
                for item in output
            ),
            "stats_status": stats_status,
            "stats_databases_found": roots_with_logs,
            "stats_databases_read": roots_read,
            "stats_records_matched": selected_matched_count,
            "markers_without_stats": selected_marker_count - selected_matched_count,
            "no_op_status": "unknown_without_a_persisted_skip/no-op record",
        },
        "notes": notes,
    }
