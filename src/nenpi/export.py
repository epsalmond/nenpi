"""Content-free native accounting snapshots with durable, acknowledged delivery.

SQLite stores the latest projection and an immutable pending batch. A consumer
acknowledges only after accepting its records; stdout alone is not an ack.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import time

from . import drain as d
from .activity_model import normalize
from .polling_report import flag_responses

SCHEMA = 1
CLASSIFIER = "closed-recipes-v1"
LABEL = re.compile(r"[a-zA-Z][a-zA-Z0-9_.-]{0,47}\Z")
NATIVE_ID = re.compile(r"[A-Za-z0-9_.:-]{1,160}\Z")
AGENT_TYPES = {"general-purpose", "Explore", "Plan", "implementation", "review", "implementer", "reviewer"}
TOOL_FAMILIES = {"shell", "read", "edit", "wait", "agents", "web", "memory", "other"}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def native_id(value):
    return value if isinstance(value, str) and NATIVE_ID.fullmatch(value) else "unknown"


def tool_family(name):
    if name in TOOL_FAMILIES:
        return name
    leaf = re.split(r"\.|__", name)[-1].lower()
    if leaf in {"exec", "bash", "exec_command", "shell", "shell_command", "local_shell_call"}:
        return "shell"
    if leaf in {"read", "read_file", "glob", "grep", "list_directory"}:
        return "read"
    if leaf in {"edit", "write", "multiedit", "apply_patch"}:
        return "edit"
    if leaf in {"write_stdin", "wait", "wait_agent", "sleep", "curr_time", "taskoutput", "bashoutput"}:
        return "wait"
    if leaf in {"agent", "task", "spawn_agent", "send_message", "followup_task", "list_agents"}:
        return "agents"
    if leaf in {"webfetch", "websearch", "run"}:
        return "web"
    return "memory" if "hindsight" in name.lower() else "other"


def identity_sources(path):
    if path.stat().st_size > 65536:
        raise ValueError("identity map exceeds 64 KiB")
    payload = json.loads(path.read_text())
    rows = payload.get("sources")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 16:
        raise ValueError("identity map needs 1..16 sources")
    output = []
    for row in rows:
        if not isinstance(row, dict) or row.get("harness") not in {"claude", "codex"}:
            raise ValueError("invalid identity harness")
        for key in ("provider", "account_alias"):
            if not isinstance(row.get(key), str) or not LABEL.fullmatch(row[key]):
                raise ValueError("invalid bounded identity label")
        root = Path(row["source_root"]).expanduser().resolve()
        output.append(dict(row, root=root))
    return output


def resolve_identity(path, harness, sources):
    matches = [s for s in sources if s["harness"] == harness and path.is_relative_to(s["root"])]
    if not matches:
        return dict(harness=harness, provider="unknown", account_alias="unknown", identity_status="unknown")
    longest = max(len(s["root"].parts) for s in matches)
    candidates = {(s["provider"], s["account_alias"]) for s in matches if len(s["root"].parts) == longest}
    if len(candidates) != 1:
        return dict(harness=harness, provider="ambiguous", account_alias="ambiguous", identity_status="ambiguous")
    provider, alias = candidates.pop()
    return dict(harness=harness, provider=provider, account_alias=alias, identity_status="known")


def discover(sources, args, lower):
    """Bound discovery and deduplicate overlapping roots before attribution."""
    found = {}
    visited = 0
    for source in sources:
        leaf = source["root"] / ("projects" if source["harness"] == "claude" else "sessions")
        if not leaf.is_dir():
            leaf = source["root"]
        iterator = (Path(directory) / name for directory, _, filenames in os.walk(leaf)
            for name in filenames if name.endswith(".jsonl") and (source["harness"] == "claude" or name.startswith("rollout-")))
        for path in iterator:
            visited += 1
            if visited > args.max_files:
                raise ValueError("source file budget exceeded; narrow source roots")
            key = (source["harness"], str(path.resolve()))
            if key in found:
                continue
            if len(found) >= args.max_files:
                raise ValueError("source file budget exceeded; narrow source roots")
            stat = path.stat()
            if stat.st_mtime < lower:
                continue
            identity = resolve_identity(path.resolve(), source["harness"], sources)
            sidecar_signature, _ = d.claude_sidecar(path) if source["harness"] == "claude" else ("", {})
            found[key] = (path, identity, [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, sidecar_signature])
    return list(found.values())


def totals(events, harness):
    kinds = d.CLAUDE_KINDS if harness == "claude" else d.CODEX_KINDS
    result = dict.fromkeys(kinds, 0)
    for event in events:
        for kind, value in d.event_tokens(event, kinds).items():
            result[kind] += value
    return result


def accounting(events, analysis, identity):
    harness = identity["harness"]
    counts = dict(root=0, descendant=0, unknown=0)
    reasoning = 0
    known_reasoning = 0
    for event in events:
        thread = str(event[d.EVENT_THREAD] or (event[d.EVENT_SESSION] if not event[d.EVENT_SUB] else ""))
        role = d._drill_thread_classification(analysis.scan.thread_metadata, event[d.EVENT_SESSION], thread, {}, harness=harness)
        role = role if role in counts else "unknown"
        counts[role] += 1
        if len(event) > d.EVENT_REASONING_KNOWN and event[d.EVENT_REASONING_KNOWN] is True:
            reasoning += int(event[d.EVENT_REASONING])
            known_reasoning += 1
    tokens = totals(events, harness)
    # Claude records carry native TTL buckets. Codex cache writes have no TTL.
    return dict(tokens=tokens, turns=len(events), root_turns=counts["root"], descendant_turns=counts["descendant"],
        cache_write_tokens=sum(value for kind, value in tokens.items() if kind.startswith("cache_write")), output_tokens=tokens.get("output", 0),
        unknown_turns=counts["unknown"], reasoning_tokens=reasoning,
        reasoning_known_turns=known_reasoning, context_start=d.event_context(events[0], harness) if events else 0,
        context_peak=max((d.event_context(e, harness) for e in events), default=0),
        token_kinds_known={kind: bool(events) and all(len(e) > d.EVENT_TOKEN_KINDS_KNOWN and kind in e[d.EVENT_TOKEN_KINDS_KNOWN] for e in events) for kind in tokens},
        attribution_complete=counts["unknown"] == 0)


def project(analysis, identity, args):
    """Project native deduplicated usage; partitions never split response tokens."""
    harness = identity["harness"]
    responses = normalize(analysis)
    flags = {id(f["response"]): f for f in flag_responses(responses, args.threshold, args.window_steps)}
    response_by_id = {(r.session, r.response_id): r for r in responses if r.response_id}
    # Native generating-response joins retain call counts even for cached batches.
    calls = defaultdict(list)
    from .activity_model import operation_batches
    for entry, _, path in analysis.scan._tool_sources:
        if not path:
            continue
        sessions = {e[d.EVENT_ID]: e[d.EVENT_SESSION] for e in entry.events}
        for rid, batch in operation_batches(harness, path, getattr(analysis.args, "activity_read_budget", None)):
            for cid, (_, _, name) in batch.items():
                calls[(sessions.get(rid), rid)].append((cid, tool_family(name)))
    by_prompt = defaultdict(list)
    for event in analysis.scan.events[harness]:
        by_prompt[(event[d.EVENT_SESSION], event[d.EVENT_PROMPT])].append(event)
    records = []
    session_prompts = defaultdict(list)
    for (session, index), events in sorted(by_prompt.items()):
        events.sort(key=lambda e: (e[d.EVENT_TS], str(e[d.EVENT_ID])))
        prompt = next((p for p in analysis.prompts.get((harness, session), []) if p.index == index), None)
        # The native boundary timestamp survives display-index changes and resumes.
        bounds = analysis.scan.boundaries.get((harness, session), [])
        boundary = prompt.start if prompt and prompt.start in bounds else None
        turns = {e[d.EVENT_TURN] for e in events if not e[d.EVENT_SUB] and e[d.EVENT_TURN]}
        turn_id = next(iter(turns)) if len(turns) == 1 else None
        logical = digest([identity, native_id(session), "prompt", turn_id or boundary])
        known_prompt = bool(boundary is not None or turn_id)
        base = dict(identity, session_id=native_id(session), prompt_id=logical, prompt_index=index,
            prompt_membership="native_turn" if turn_id else "temporal_boundary" if known_prompt else "unknown", source_timestamp=max(e[d.EVENT_TS] for e in events),
            classifier_version=CLASSIFIER, threshold=args.threshold, window_steps=args.window_steps)
        record = dict(base, event="analytics_prompt", record_id=logical, **accounting(events, analysis, identity))
        record["human_prompt"] = known_prompt
        records.append(record)
        session_prompts[session].append(record)
        partitions = defaultdict(list)
        for event in events:
            thread = str(event[d.EVENT_THREAD] or (session if not event[d.EVENT_SUB] else ""))
            metadata = analysis.scan.thread_metadata.get((harness, session, thread), {})
            role = d._drill_thread_classification(analysis.scan.thread_metadata, session, thread, {}, harness=harness)
            role = role if role in {"root", "descendant"} else "unknown"
            agent_type = metadata.get("agent_type")
            agent_type = agent_type if agent_type in AGENT_TYPES else "unknown"
            model = str(event[d.EVENT_MODEL])
            model = model if LABEL.fullmatch(model) else "unknown"
            response = response_by_id.get((session, event[d.EVENT_ID]))
            flag = flags.get(id(response)) if response else None
            classification = (flag["kind"] if flag and flag["kind"] != "none" else "unflagged" if response and response.operations else "unknown")
            if flag and flag.get("recipe_id"):
                classification = "recipe" if classification in {"unflagged", "unknown"} else classification + "_recipe"
            linked_calls = dict(calls.get((session, event[d.EVENT_ID]), [])) if response else {}
            batch = "+".join(sorted(set(linked_calls.values()))) if linked_calls else "unknown"
            association = "single_tool" if len(linked_calls) == 1 else "combined_batch" if linked_calls else "unknown"
            key = (model, role, agent_type, classification, batch, association, native_id(thread), native_id(metadata.get("parent_thread_id")))
            partitions[key].append(event)
        for key, partition in sorted(partitions.items()):
            model, role, agent_type, classification, batch, association, thread, parent = key
            records.append(dict(base, event="analytics_partition", record_id=digest([logical, key]),
                model=model, lineage=role, agent_type=agent_type, implementation_role="implementation" if agent_type == "implementation" else "unknown",
                classification=classification, tool_batch=batch, tool_association=association,
                thread_id=thread, parent_thread_id=parent, **accounting(partition, analysis, identity)))
    for (source_harness, session), bounds in analysis.scan.boundaries.items():
        if source_harness != harness:
            continue
        present = {r["prompt_id"] for r in session_prompts[session]}
        for index, boundary in enumerate(sorted(bounds), start=1):
            logical = digest([identity, native_id(session), "prompt", boundary])
            # A native turn ID already owns the boundary of a nonempty prompt.
            if logical in present or any(p.start == boundary for p in analysis.prompts.get((harness, session), [])):
                continue
            record = dict(identity,event="analytics_prompt",record_id=logical,prompt_id=logical,session_id=native_id(session),
                prompt_index=index,prompt_membership="temporal_boundary",source_timestamp=boundary,human_prompt=True,
                classifier_version=CLASSIFIER,threshold=args.threshold,window_steps=args.window_steps,**accounting([],analysis,identity))
            records.append(record)
            session_prompts[session].append(record)
    for session, prompts in sorted(session_prompts.items()):
        events = [e for e in analysis.scan.events[harness] if e[d.EVENT_SESSION] == session]
        records.append(dict(identity, event="analytics_session", record_id=digest([identity, native_id(session), "session"]),
            session_id=native_id(session), source_timestamp=max([e[d.EVENT_TS] for e in events] + [p["source_timestamp"] for p in prompts]),
            human_prompts=sum(p["human_prompt"] for p in prompts), **accounting(events, analysis, identity)))
    return records


def open_store(path):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise ValueError("export store must not be a symlink")
    connection = sqlite3.connect(path, timeout=2)
    os.chmod(path, 0o600)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS projections (id TEXT PRIMARY KEY, revision INTEGER NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS pending (batch TEXT NOT NULL, position INTEGER PRIMARY KEY, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS sources (identity TEXT PRIMARY KEY, signature TEXT NOT NULL);
    """)
    return connection


def make_batch(store, records, now, args, coverage="complete", updated_scopes=None):
    """Update all disappeared partitions with tombstones in the same transaction."""
    old = {rid: (revision, json.loads(payload)) for rid, revision, payload in store.execute("SELECT id, revision, payload FROM projections")}
    changed = []
    current_ids = {r["record_id"] for r in records}
    updated_prompts = {r["prompt_id"] for r in records if r["event"] == "analytics_prompt"}
    for rid, (revision, previous) in old.items():
        identity = {key:previous[key] for key in ("harness","provider","account_alias","identity_status")}
        disappeared = updated_scopes is not None and encoded([identity, previous["session_id"]]) in updated_scopes and rid not in current_ids
        if ((previous["event"] == "analytics_partition" and previous.get("prompt_id") in updated_prompts and rid not in current_ids) or disappeared) and not previous.get("deleted"):
            tombstone = dict(previous, deleted=True)
            tombstone["tokens"] = dict.fromkeys(previous["tokens"], 0)
            for key in ("turns", "root_turns", "descendant_turns", "unknown_turns", "reasoning_tokens", "reasoning_known_turns", "context_start", "context_peak", "cache_write_tokens", "output_tokens"):
                tombstone[key] = 0
            if "human_prompt" in tombstone:
                tombstone["human_prompt"] = False
            if "human_prompts" in tombstone:
                tombstone["human_prompts"] = 0
            records.append(tombstone)
    dirty_prompts = {r.get("prompt_id") for r in records if r["event"] in {"analytics_prompt", "analytics_partition"}
        and (r["record_id"] not in old or old[r["record_id"]][1] != r)}
    for record in records:
        rid = record["record_id"]
        previous = old.get(rid)
        if previous and previous[1] == record and record.get("prompt_id") not in dirty_prompts:
            continue
        revision = previous[0] + 1 if previous else 1
        changed.append(dict(record, schema_version=SCHEMA, revision=revision, observed_at=now))
        store.execute("INSERT OR REPLACE INTO projections VALUES (?, ?, ?)", (rid, revision, encoded(record)))
    if not changed:
        return []
    # Pages are durable before delivery. The following invocation returns the
    # oldest unacknowledged page rather than scanning or advancing it again.
    pages = []
    page = []
    size = 512
    for record in changed:
        record_size = len(encoded(record).encode()) + 128
        if record_size + 512 > args.max_output_bytes:
            raise ValueError("single projection exceeds delivery budget")
        if page and (len(page) >= args.max_records - 1 or size + record_size > args.max_output_bytes):
            pages.append(page)
            page, size = [], 512
        page.append(record)
        size += record_size
    if page:
        pages.append(page)
    position = 0
    outputs = []
    for page in pages:
        batch_id = digest(page)
        output = [dict(r, batch_id=batch_id) for r in page]
        output.append(dict(event="analytics_export_batch", schema_version=SCHEMA, batch_id=batch_id, records=len(page), observed_at=now, coverage=coverage))
        outputs.append(output)
        for record in output:
            store.execute("INSERT INTO pending VALUES (?, ?, ?)", (batch_id, position, encoded(record)))
            position += 1
    return outputs[0]


def command_export(args):
    started = time.monotonic()
    try:
        with open_store(Path(args.export_state)) as store:
            store.execute("BEGIN IMMEDIATE")
            if args.ack:
                pending = store.execute("SELECT DISTINCT batch FROM pending").fetchall()
                first = store.execute("SELECT batch FROM pending ORDER BY position LIMIT 1").fetchone()
                if first and first[0] != args.ack:
                    raise ValueError("acknowledgement does not name pending batch")
                store.execute("DELETE FROM pending WHERE batch = ?", (args.ack,))
                return 0
            pending = [json.loads(row[0]) for row in store.execute("SELECT payload FROM pending WHERE batch=(SELECT batch FROM pending ORDER BY position LIMIT 1) ORDER BY position")]
            if pending:
                for record in pending:
                    print(encoded(record))
                return 0
            sources = identity_sources(Path(args.identity_map))
            now = time.time()
            row = store.execute("SELECT value FROM meta WHERE key='backfill_start'").fetchone()
            lower = float(row[0]) if row else (d.parse_since(args.since, now) or now - 86400)
            store.execute("INSERT OR IGNORE INTO meta VALUES ('backfill_start', ?)", (str(lower),))
            files = discover(sources, args, lower)
            grouped = defaultdict(list)
            for source in sources:
                if source["root"].is_dir():
                    identity = resolve_identity(source["root"],source["harness"],sources)
                    grouped[encoded(identity)]
            for path, identity, signature in files:
                grouped[encoded(identity)].append((path, signature))
            records = []
            updated_scopes = set()
            incomplete = any(not source["root"].is_dir() for source in sources)
            total_bytes = 0
            activity_bytes_remaining = args.max_scan_bytes
            for identity_key, paths in sorted(grouped.items()):
                identity = json.loads(identity_key)
                signature = digest([[str(p), s] for p, s in paths])
                previous = store.execute("SELECT signature FROM sources WHERE identity=?", (identity_key,)).fetchone()
                if previous and previous[0] == signature:
                    continue
                if time.monotonic() - started > args.scan_seconds:
                    incomplete = True
                    break
                # Prime incremental native shards within a finite byte budget.
                cache = d.Cache(d.cache_dir(), False)
                for path, _ in paths:
                    stat = path.stat()
                    sidecar_signature, metadata = d.claude_sidecar(path) if identity["harness"] == "claude" else ("", {})
                    with d.serialized_cache(cache.root):
                        entry, stale = cache.entry_for(path, identity["harness"], stat, sidecar_signature)
                        needed = max(0, stat.st_size - entry.offset)
                        if stale:
                            if time.monotonic() - started > args.scan_seconds:
                                incomplete = True
                                break
                            entry.claude_sidecar_signature = sidecar_signature
                            entry.claude_agent_metadata = metadata
                            budget = {"remaining": args.max_scan_bytes - total_bytes, "exhausted": False}
                            token = d.PARSE_BYTE_BUDGET.set(budget)
                            try:
                                (d.parse_claude_file if identity["harness"] == "claude" else d.parse_codex_file)(path, entry)
                            finally:
                                d.PARSE_BYTE_BUDGET.reset(token)
                            entry.stamp(path, stat)
                            if budget["exhausted"]:
                                # A partial native shard must be revisited even if the source stops changing.
                                entry.size = entry.offset
                            cache.mark(path)
                            cache._flush_unlocked()
                            total_bytes = args.max_scan_bytes - budget["remaining"]
                            if budget["exhausted"]:
                                incomplete = True
                                break
                        cache.forget(path)
                if incomplete:
                    break
                scan = d.Scan()
                for path, _ in sorted(paths, key=lambda pair: (pair[0].stat().st_mtime, str(pair[0]))):
                    entry = cache.stored(str(path), identity["harness"])
                    if entry is None:
                        raise ValueError("native cache missing after scan")
                    d.absorb(scan, entry, identity["harness"], identity_key, identity["account_alias"], path)
                    cache.forget(path)
                d.resolve_claude_thread_metadata(scan)
                d.rebuild_totals(scan)
                closures = defaultdict(list)
                path_signatures = {str(path):stamp for path,stamp in paths}
                for entry, _, source_path in scan._tool_sources:
                    for session in entry.sessions:
                        closures[session].append([str(source_path),path_signatures[str(source_path)]])
                complete_group = True
                prefix = "scope:" + identity_key + ":"
                existing_scopes = {key for (key,) in store.execute("SELECT identity FROM sources WHERE substr(identity,1,?)=?", (len(prefix),prefix))}
                for session in sorted(closures):
                    scope_key = prefix + encoded(native_id(session))
                    existing_scopes.discard(scope_key)
                    native_events = [e for e in scan.events[identity["harness"]] if e[d.EVENT_SESSION] == session]
                    closure_signature = digest([closures[session], [[e[d.EVENT_ID],e[d.EVENT_SUB],e[d.EVENT_THREAD]] for e in native_events]])
                    previous_closure = store.execute("SELECT signature FROM sources WHERE identity=?", (scope_key,)).fetchone()
                    if previous_closure and previous_closure[0] == closure_signature:
                        continue
                    if time.monotonic() - started > args.scan_seconds:
                        incomplete = True
                        complete_group = False
                        break
                    selected = d.Scan()
                    for field in ("sessions","session_accounts","boundaries","compactions"):
                        setattr(selected,field,{key:value for key,value in getattr(scan,field).items() if key == (identity["harness"],session)})
                    selected.thread_metadata = {key:value for key,value in scan.thread_metadata.items() if key[0:2] == (identity["harness"],session)}
                    selected.events[identity["harness"]] = native_events
                    for entry, source_harness, source_path in scan._tool_sources:
                        if session in entry.sessions:
                            selected.defer_tools(entry,source_harness,source_path)
                    selected.filter_tools(lambda _h,rows: [r for r in rows if r[d.TOOL_SESSION] == session])
                    analysis_args = copy.copy(args)
                    analysis_args.activity_read_budget = {"remaining":activity_bytes_remaining,"incomplete":False}
                    analysis = d.Analysis(selected,d.load_weights(False),None,None,analysis_args)
                    analysis.prompts = d.assemble_prompts(selected,analysis.weights,analysis_args)
                    records.extend(project(analysis,identity,args))
                    updated_scopes.add(encoded([identity,native_id(session)]))
                    activity_bytes_remaining = analysis_args.activity_read_budget["remaining"]
                    if not analysis_args.activity_read_budget["incomplete"]:
                        store.execute("INSERT OR REPLACE INTO sources VALUES (?,?)", (scope_key,closure_signature))
                    else:
                        complete_group = False
                if complete_group:
                    for key in existing_scopes:
                        removed_session = json.loads(key[len(prefix):])
                        updated_scopes.add(encoded([identity,removed_session]))
                        store.execute("DELETE FROM sources WHERE identity=?", (key,))
                    store.execute("INSERT OR REPLACE INTO sources VALUES (?, ?)", (identity_key, signature))
            output = make_batch(store, records, now, args, "incomplete" if incomplete else "complete", updated_scopes)
            health = dict(event="analytics_export_health", schema_version=SCHEMA, observed_at=now,
                coverage="incomplete" if incomplete else "complete", scan_bytes=total_bytes, files=len(files),
                duration_seconds=time.monotonic() - started, classifier_version=CLASSIFIER)
            for record in output:
                print(encoded(record))
            if not output:
                print(encoded(health))
            else:
                print(encoded(health), file=sys.stderr)
        return 0
    except (OSError, ValueError, sqlite3.Error) as error:
        # Exceptions may contain a private path; keep diagnostics content-free.
        print("nenpi export: " + (str(error) if isinstance(error, ValueError) else type(error).__name__), file=sys.stderr)
        return 1


def add_parser(sub):
    parser = sub.add_parser("export", help="content-free revisioned native analytics JSONL")
    d.add_common(parser)
    parser.add_argument("--identity-map", required=True, help="JSON source-root to bounded provider/account labels")
    parser.add_argument("--export-state", default=str(d.state_dir() / "analytics.sqlite"))
    parser.add_argument("--ack", help="acknowledge the accepted immutable pending batch")
    parser.add_argument("--max-files", type=int, default=10000)
    parser.add_argument("--max-scan-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-records", type=int, default=4000)
    parser.add_argument("--max-output-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--scan-seconds", type=int, default=90)
    parser.add_argument("--threshold", type=int, default=3)
    parser.add_argument("--window-steps", type=int, default=30)
    parser.set_defaults(handler=command_export, since="24h", quiet=True, json=False)
