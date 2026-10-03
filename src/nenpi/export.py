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
# Projection semantics can change without changing the consumer's wire schema.
PROJECTION_VERSION = 4
LABEL = re.compile(r"[a-zA-Z][a-zA-Z0-9_.-]{0,47}\Z")
NATIVE_ID = re.compile(r"[A-Za-z0-9_.:-]{1,160}\Z")
AGENT_TYPES = {"general-purpose", "Explore", "Plan", "implementation", "review", "implementer", "reviewer"}
TOOL_FAMILIES = {"shell", "read", "edit", "wait", "agents", "web", "memory", "other"}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def native_id(value):
    if isinstance(value,str) and NATIVE_ID.fullmatch(value):
        return value
    return "opaque-" + digest(value)[:32] if value else "unknown"


def native_lineage(analysis, harness, session, thread):
    metadata = analysis.scan.thread_metadata.get((harness,session,thread))
    if not metadata or (metadata.get("classification") != "root" and not metadata.get("parent_thread_id")):
        return "unknown"
    return d._drill_thread_classification(analysis.scan.thread_metadata,session,thread,{},harness=harness)


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
    started = time.monotonic()
    for source in sources:
        leaf = source["root"] / ("projects" if source["harness"] == "claude" else "sessions")
        if not leaf.is_dir():
            leaf = source["root"]
        iterator = (Path(directory) / name for directory, _, filenames in os.walk(leaf)
            for name in filenames if name.endswith(".jsonl") and (source["harness"] == "claude" or name.startswith("rollout-")))
        for path in iterator:
            if time.monotonic() - started > args.scan_seconds:
                raise ValueError("source discovery time budget exceeded")
            key = (source["harness"], str(path.resolve()))
            if key in found:
                continue
            stat = path.stat()
            if stat.st_mtime < lower:
                continue
            if len(found) >= args.max_files:
                raise ValueError("source file budget exceeded; narrow source roots")
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
        role = native_lineage(analysis,harness,event[d.EVENT_SESSION],thread)
        role = role if role in counts else "unknown"
        counts[role] += 1
        if len(event) > d.EVENT_REASONING_KNOWN and event[d.EVENT_REASONING_KNOWN] is True:
            reasoning += int(event[d.EVENT_REASONING])
            known_reasoning += 1
    tokens = totals(events, harness)
    # Claude records carry native TTL buckets. Codex cache writes have no TTL.
    return dict(tokens=tokens, turns=len(events), root_turns=counts["root"], descendant_turns=counts["descendant"],
        response_identity_known_turns=sum(isinstance(e[d.EVENT_ID],str) and bool(NATIVE_ID.fullmatch(e[d.EVENT_ID])) for e in events),
        cache_write_tokens=sum(value for kind, value in tokens.items() if kind.startswith("cache_write")), output_tokens=tokens.get("output", 0),
        unknown_turns=counts["unknown"], reasoning_tokens=reasoning,
        reasoning_known_turns=known_reasoning, context_start=d.event_context(events[0], harness) if events else 0,
        context_peak=max((d.event_context(e, harness) for e in events), default=0),
        token_kinds_known={kind: bool(events) and all(len(e) > d.EVENT_TOKEN_KINDS_KNOWN and kind in e[d.EVENT_TOKEN_KINDS_KNOWN]
            and (harness != "codex" or kind != "input" or "cached_input" in e[d.EVENT_TOKEN_KINDS_KNOWN]) for e in events) for kind in tokens},
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
            role = native_lineage(analysis,harness,session,thread)
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
        CREATE TABLE IF NOT EXISTS response_usage (id TEXT PRIMARY KEY, identity TEXT NOT NULL, tokens TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS account_counters (identity TEXT NOT NULL, kind TEXT NOT NULL, value INTEGER NOT NULL, PRIMARY KEY(identity,kind));
        CREATE TABLE IF NOT EXISTS account_measurements (identity TEXT NOT NULL, kind TEXT NOT NULL, value INTEGER NOT NULL, PRIMARY KEY(identity,kind));
        CREATE TABLE IF NOT EXISTS projection_scopes (id TEXT PRIMARY KEY, scope TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS projection_scope_lookup ON projection_scopes(scope);
        CREATE TABLE IF NOT EXISTS scope_files (scope TEXT PRIMARY KEY, paths TEXT NOT NULL);
    """)
    if not connection.execute("SELECT value FROM meta WHERE key='projection_scope_schema'").fetchone():
        connection.execute("""INSERT OR IGNORE INTO projection_scopes
        SELECT id,json_array(json_object('account_alias',json_extract(payload,'$.account_alias'),
            'harness',json_extract(payload,'$.harness'),'identity_status',json_extract(payload,'$.identity_status'),
            'provider',json_extract(payload,'$.provider')),json_extract(payload,'$.session_id'))
        FROM projections WHERE json_extract(payload,'$.event')!='analytics_account'
        AND id NOT IN (SELECT id FROM projection_scopes)""")
        connection.execute("INSERT INTO meta VALUES ('projection_scope_schema','1')")
        connection.commit()
    return connection


def account_counter_record(store,scan,identity,args,now):
    """Native response IDs keep forks and prompt reassignment out of counters."""
    identity_key = encoded(identity)
    count = store.execute("SELECT COUNT(*) FROM response_usage").fetchone()[0]
    kinds = ("input_processed","cache_write","output")
    changed = False
    complete_key = "counter_complete:" + identity_key
    limit_key = "counter_limit:" + identity_key
    old_limit = store.execute("SELECT value FROM meta WHERE key=?",(limit_key,)).fetchone()
    old_complete = store.execute("SELECT value FROM meta WHERE key=?",(complete_key,)).fetchone()
    budget_complete = not old_complete or old_complete[0] == "1" or not old_limit or int(old_limit[0]) != args.max_response_ids
    for event in scan.events[identity["harness"]]:
        response_id = event[d.EVENT_ID]
        if not isinstance(response_id,str) or not NATIVE_ID.fullmatch(response_id):
            continue
        key = digest([identity,response_id])
        previous = store.execute("SELECT tokens FROM response_usage WHERE id=?",(key,)).fetchone()
        if previous is None and count >= args.max_response_ids:
            budget_complete = False
            continue
        before_record = json.loads(previous[0]) if previous else {}
        before = before_record.get("tokens",{})
        before_known = set(before_record.get("known",[]))
        native_kinds = d.CLAUDE_KINDS if identity["harness"] == "claude" else d.CODEX_KINDS
        native_tokens = d.event_tokens(event,native_kinds)
        tokens = dict(input_processed=d.event_context(event,identity["harness"]),cache_write=sum(v for k,v in native_tokens.items() if k.startswith("cache_write")),output=native_tokens.get("output",0))
        presence = set(event[d.EVENT_TOKEN_KINDS_KNOWN]) if len(event) > d.EVENT_TOKEN_KINDS_KNOWN else set()
        if identity["harness"] == "codex":
            context_known = "input" in presence
            write_known = "cache_write" in presence
        else:
            write_known = "cache_write_unknown" in presence or {"cache_write_5m","cache_write_1h"} <= presence
            context_known = {"input","cache_read"} <= presence and write_known
        known = ({"input_processed"} if context_known else set()) | ({"cache_write"} if write_known else set()) | ({"output"} if "output" in presence else set())
        if any(type(value) is not int or value < 0 for value in tokens.values()):
            raise ValueError("invalid native usage")
        observed = {kind:max(value if kind in known else 0,before.get(kind,0)) for kind,value in tokens.items()}
        if previous is None:
            count += 1
            changed = True
        for kind,value in observed.items():
            delta = value - before.get(kind,0)
            if delta:
                changed = True
                store.execute("INSERT INTO account_counters VALUES (?,?,?) ON CONFLICT(identity,kind) DO UPDATE SET value=value+excluded.value",(identity_key,kind,delta))
        for kind in known - before_known:
            changed = True
            store.execute("INSERT INTO account_measurements VALUES (?,?,1) ON CONFLICT(identity,kind) DO UPDATE SET value=value+1",(identity_key,kind))
        payload = encoded(dict(tokens=observed,known=sorted(known | before_known)))
        if previous is None or previous[0] != payload:
            store.execute("INSERT OR REPLACE INTO response_usage VALUES (?,?,?)",(key,identity_key,payload))
    started_key = "counter_started:" + identity_key
    source_key = "counter_updated:" + identity_key
    store.execute("INSERT OR IGNORE INTO meta VALUES (?,?)",(started_key,str(now)))
    if changed:
        store.execute("INSERT OR REPLACE INTO meta VALUES (?,?)",(source_key,str(now)))
    source_row = store.execute("SELECT value FROM meta WHERE key=?",(source_key,)).fetchone()
    store.execute("INSERT OR REPLACE INTO meta VALUES (?,?)",(complete_key,"1" if budget_complete else "0"))
    store.execute("INSERT OR REPLACE INTO meta VALUES (?,?)",(limit_key,str(args.max_response_ids)))
    tokens = dict.fromkeys(kinds,0)
    tokens.update(dict(store.execute("SELECT kind,value FROM account_counters WHERE identity=?",(identity_key,))))
    return dict(identity,event="analytics_account",record_id=digest([identity,"account"]),source_timestamp=float(source_row[0]) if source_row else 0,
        counter_started_at=float(store.execute("SELECT value FROM meta WHERE key=?",(started_key,)).fetchone()[0]),tokens=tokens,
        native_responses=store.execute("SELECT COUNT(*) FROM response_usage WHERE identity=?",(identity_key,)).fetchone()[0],
        measurements={kind:dict(store.execute("SELECT kind,value FROM account_measurements WHERE identity=?",(identity_key,))).get(kind,0) for kind in kinds},
        missing_response_ids_excluded=True,counter_identity_budget_complete=budget_complete)


def make_batch(store, records, now, args, coverage="complete", updated_scopes=None):
    """Update all disappeared partitions with tombstones in the same transaction."""
    if not records and not updated_scopes:
        return []
    scopes = sorted(updated_scopes or [])
    rows = []
    for offset in range(0,len(scopes),200):
        selected = scopes[offset:offset + 200]
        rows.extend(store.execute("SELECT p.id,p.revision,p.payload FROM projections p JOIN projection_scopes s ON s.id=p.id WHERE s.scope IN (" + ",".join("?" for _ in selected) + ")",selected))
    for record in records:
        if record["event"] == "analytics_account":
            rows.extend(store.execute("SELECT id,revision,payload FROM projections WHERE id=?",(record["record_id"],)))
    old = {rid:(revision,json.loads(payload)) for rid,revision,payload in rows}
    changed = []
    current_ids = {r["record_id"] for r in records}
    updated_prompts = {r["prompt_id"] for r in records if r["event"] == "analytics_prompt"}
    for rid, (revision, previous) in old.items():
        identity = {key:previous[key] for key in ("harness","provider","account_alias","identity_status")}
        disappeared = previous["event"] != "analytics_account" and updated_scopes is not None and encoded([identity,previous["session_id"]]) in updated_scopes and rid not in current_ids
        if ((previous["event"] == "analytics_partition" and previous.get("prompt_id") in updated_prompts and rid not in current_ids) or disappeared) and not previous.get("deleted"):
            tombstone = dict(previous, deleted=True)
            tombstone["tokens"] = dict.fromkeys(previous["tokens"], 0)
            for key in ("turns", "root_turns", "descendant_turns", "unknown_turns", "reasoning_tokens", "reasoning_known_turns", "context_start", "context_peak", "cache_write_tokens", "output_tokens", "response_identity_known_turns"):
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
        if record["event"] != "analytics_account":
            identity = {key:record[key] for key in ("harness","provider","account_alias","identity_status")}
            store.execute("INSERT OR REPLACE INTO projection_scopes VALUES (?,?)",(rid,encoded([identity,record["session_id"]])))
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
            requested = d.parse_since(args.since,now)
            lower = max(float(row[0]),now - 86400) if row else (requested if requested is not None else now - 86400)
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
            source_incomplete = any(not source["root"].is_dir() for source in sources)
            incomplete = False
            classification_incomplete = False
            total_bytes = 0
            closure_file_count = len(files)
            activity_bytes_remaining = args.max_scan_bytes
            for identity_key, paths in sorted(grouped.items()):
                identity = json.loads(identity_key)
                signature = digest([PROJECTION_VERSION,d.CACHE_SCHEMA,CLASSIFIER,args.threshold,args.window_steps,args.max_response_ids,[[str(p),s] for p,s in paths]])
                previous = store.execute("SELECT signature FROM sources WHERE identity=?", (identity_key,)).fetchone()
                if previous and previous[0] == signature:
                    continue
                if time.monotonic() - started > args.scan_seconds:
                    incomplete = True
                    break
                # Prime incremental native shards within a finite byte budget.
                cache = d.Cache(d.cache_dir(), False)
                known_paths = {str(path) for path, _ in paths}
                # The rolling window discovers changed work. A touched session
                # must still include its retained historical parent/descendants.
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
                        for session in entry.sessions:
                            scope_key = "scope:" + identity_key + ":" + encoded(native_id(session))
                            historical = store.execute("SELECT paths FROM scope_files WHERE scope=?", (scope_key,)).fetchone()
                            for member in json.loads(historical[0]) if historical else []:
                                if member in known_paths:
                                    continue
                                historical_path = Path(member)
                                if not historical_path.is_file():
                                    continue
                                if encoded(resolve_identity(historical_path.resolve(), identity["harness"], sources)) != identity_key:
                                    continue
                                if closure_file_count >= args.max_files or time.monotonic() - started > args.scan_seconds:
                                    incomplete = True
                                    break
                                member_stat = historical_path.stat()
                                sidecar, _ = d.claude_sidecar(historical_path) if identity["harness"] == "claude" else ("", {})
                                paths.append((historical_path, [member_stat.st_size, member_stat.st_mtime_ns, member_stat.st_ctime_ns, sidecar]))
                                known_paths.add(member)
                                closure_file_count += 1
                            if incomplete:
                                break
                        cache.forget(path)
                    if incomplete:
                        break
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
                counter_events = []
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
                    closure_signature = digest([PROJECTION_VERSION,d.CACHE_SCHEMA,CLASSIFIER,args.threshold,args.window_steps,args.max_response_ids,closures[session],[[e[d.EVENT_ID],e[d.EVENT_SUB],e[d.EVENT_THREAD]] for e in native_events]])
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
                    counter_events.extend(native_events)
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
                    store.execute("INSERT OR REPLACE INTO scope_files VALUES (?,?)",(scope_key,encoded([p for p,_ in closures[session]])))
                    activity_bytes_remaining = analysis_args.activity_read_budget["remaining"]
                    if not analysis_args.activity_read_budget["incomplete"]:
                        store.execute("INSERT OR REPLACE INTO sources VALUES (?,?)", (scope_key,closure_signature))
                    else:
                        classification_incomplete = True
                        complete_group = False
                if complete_group:
                    for key in existing_scopes:
                        previous_files = store.execute("SELECT paths FROM scope_files WHERE scope=?",(key,)).fetchone()
                        if previous_files and any(Path(p).exists() for p in json.loads(previous_files[0])):
                            # An inactive source retains its historical snapshot;
                            # expiration of the scan window is not deletion.
                            continue
                        removed_session = json.loads(key[len(prefix):])
                        updated_scopes.add(encoded([identity,removed_session]))
                        store.execute("DELETE FROM sources WHERE identity=?", (key,))
                        store.execute("DELETE FROM scope_files WHERE scope=?",(key,))
                    store.execute("INSERT OR REPLACE INTO sources VALUES (?, ?)", (identity_key, signature))
                counter_scan = d.Scan()
                counter_scan.events[identity["harness"]] = counter_events
                records.append(account_counter_record(store,counter_scan,identity,args,now))
            coverage = "incomplete" if incomplete or classification_incomplete or source_incomplete else "complete"
            output = make_batch(store, records, now, args, coverage, updated_scopes)
            health = dict(event="analytics_export_health", schema_version=SCHEMA, observed_at=now,
                coverage=coverage, scan_bytes=total_bytes, files=len(files),
                duration_seconds=time.monotonic() - started, classifier_version=CLASSIFIER)
            for record in output:
                print(encoded(record))
            if not output:
                print(encoded(health))
            else:
                print(encoded(health), file=sys.stderr)
        return 0
    except (OSError, ValueError, TypeError, sqlite3.Error) as error:
        # Parser conversion errors can contain native record values as well as
        # paths. Diagnostics never repeat exception messages from transcripts.
        print("nenpi export: rejected configuration, accounting or scan budget (" + type(error).__name__ + ")",file=sys.stderr)
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
    parser.add_argument("--max-response-ids",type=int,default=1_000_000,help="native response identities retained for monotonic counter deduplication")
    parser.add_argument("--threshold", type=int, default=3)
    parser.add_argument("--window-steps", type=int, default=30)
    parser.set_defaults(handler=command_export, since="24h", quiet=True, json=False)
