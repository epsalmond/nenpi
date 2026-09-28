"""Find repeated model work from deduplicated usage and result-size metadata."""

import bisect
import json
import time
from collections import defaultdict


SMALL_RESULT_CHARS = 1024
LARGE_CONTEXT_TOKENS = 32768
REPEATED_RESPONSES = 10


def _totals():
    return dict(responses=0, root_responses=0, subagent_responses=0, unknown_responses=0,
                input_tokens=0, cached_input_tokens=0, uncached_input_tokens=0,
                output_tokens=0, weighted_units=0.0, unweighted_responses=0,
                tool_calls=0, result_chars=0, small_result_responses=0,
                small_result_input_tokens=0, small_result_result_chars=0, small_result_max_streak=0,
                small_result_cached_input_tokens=0, small_result_uncached_input_tokens=0,
                small_result_weighted_units=0.0, small_result_unweighted_responses=0,
                linked_result_responses=0, unlinked_tool_results=0)


def _thread(row, *, tool=False):
    from . import drain as d

    session = row[d.TOOL_SESSION if tool else d.EVENT_SESSION]
    thread = row[d.TOOL_THREAD if tool else d.EVENT_THREAD]
    sub = row[d.TOOL_SUB if tool else d.EVENT_SUB]
    # Never combine legacy sidechains with no stable thread identity.
    return str(thread) if thread else ("" if sub else str(session))


def _role(analysis, harness, session, thread):
    from . import drain as d

    classification = d._drill_thread_classification(
        analysis.scan.thread_metadata, session, thread, {}, harness=harness,
    )
    return "subagent" if classification == "descendant" else classification


def _add_event(total, event, harness, role, analysis):
    from . import drain as d

    context = d.event_context(event, harness)
    cached = int(event[d.EVENT_KINDS + 1])
    kinds = d.CODEX_KINDS if harness == "codex" else d.CLAUDE_KINDS
    tokens = d.event_tokens(event, kinds)
    total["responses"] += 1
    total[role + "_responses"] += 1
    total["input_tokens"] += context
    total["cached_input_tokens"] += cached
    total["uncached_input_tokens"] += context - cached
    total["output_tokens"] += tokens["output"]
    units = d.event_units(harness, event, kinds, analysis.weights, analysis.args)
    total["weighted_units"] += units
    known_weight = analysis.weights.event_vector(harness, event[d.EVENT_MODEL], analysis.args.claude_cache_read_weight) is not None
    if not known_weight:
        total["unweighted_responses"] += 1
    return units, known_weight


def _merge(target, source):
    for key in _totals():
        if key == "small_result_max_streak":
            target[key] = max(target[key], source[key])
        else:
            target[key] += source[key]


def build_report(analysis, session_key=None, prompt_index=None):
    """Report input burden without inferring exact tool cost or quota savings.

    A result strictly between two same-thread usage records is associated with
    the later record. Timestamp ties, unknown threads and prompt boundaries are
    excluded from the small-result diagnostic; totals still include their usage.
    """
    from . import drain as d

    args = analysis.args
    source_flags = []
    if args.whole_session:
        source_flags.append("--whole-session")
    for root_flag, attr in (("--codex-root", "codex_root"), ("--claude-root", "claude_root")):
        for path in getattr(args, attr, []) or []:
            source_flags.extend([root_flag, path])
    if args.use_calibrated:
        source_flags.append("--use-calibrated")
    if args.long_context_multiplier != 1:
        source_flags.extend(["--long-context-multiplier", args.long_context_multiplier])
    if args.claude_cache_read_weight is not None:
        source_flags.extend(["--claude-cache-read-weight", args.claude_cache_read_weight])
    prompt_events = defaultdict(list)
    streams = defaultdict(list)
    for harness, events in analysis.scan.events.items():
        for event in events:
            key = (harness, event[d.EVENT_SESSION])
            if session_key and key != session_key:
                continue
            index = event[d.EVENT_PROMPT]
            if prompt_index is not None and index != prompt_index:
                continue
            prompt_events[(*key, index)].append(event)
            streams[(*key, _thread(event))].append(event)
    # Include neighbouring events outside the selected prompt to avoid assigning
    # a preceding prompt's results to the first response in a narrowed report.
    stream_history = defaultdict(list)
    for harness, events in analysis.scan.events.items():
        for event in events:
            key = (harness, event[d.EVENT_SESSION], _thread(event))
            if key in streams:
                stream_history[key].append(event)
    result_buckets = defaultdict(list)
    stream_tools = defaultdict(list)
    for harness, rows in analysis.scan.tools.items():
        for row in rows:
            key = (harness, row[d.TOOL_SESSION], _thread(row, tool=True))
            if key in streams:
                stream_tools[key].append(row)
    unlinked = defaultdict(int)
    tool_counts = defaultdict(lambda: [0, 0])
    for key, history in stream_history.items():
        history.sort(key=lambda e: e[d.EVENT_TS])
        times = [e[d.EVENT_TS] for e in history]
        for result in stream_tools[key]:
            ts = result[d.TOOL_TS]
            position = bisect.bisect_right(times, ts)
            owner = history[min(bisect.bisect_left(times, ts), len(times) - 1)][d.EVENT_PROMPT]
            if prompt_index is not None and owner != prompt_index:
                continue
            counts = tool_counts[(*key, owner)]
            counts[0] += 1
            counts[1] += result[d.TOOL_CHARS]
            if not key[2] or position == 0 or position == len(times) or ts == times[position - 1]:
                unlinked[key] += 1
                continue
            before, after = history[position - 1], history[position]
            if (before[d.EVENT_PROMPT] != after[d.EVENT_PROMPT]
                    or (prompt_index is not None and after[d.EVENT_PROMPT] != prompt_index)):
                unlinked[key] += 1
                continue
            result_buckets[id(after)].append(result)
    findings = []
    summary = _totals()
    harness_units = defaultdict(float)
    known = d.analysis_session_ids(analysis)
    for (harness, session, index), events in prompt_events.items():
        threads = {}
        streaks = defaultdict(int)
        for event in sorted(events, key=lambda e: e[d.EVENT_TS]):
            thread_id = _thread(event)
            role = _role(analysis, harness, session, thread_id)
            thread = threads.setdefault(thread_id, dict(
                _totals(), thread_id=thread_id or None, role=role,
            ))
            units, known_weight = _add_event(thread, event, harness, role, analysis)
            thread["last_context_tokens"] = d.event_context(event, harness)
            results = result_buckets[id(event)]
            if results:
                thread["linked_result_responses"] += 1
            small = (bool(results) and sum(r[d.TOOL_CHARS] for r in results) <= SMALL_RESULT_CHARS
                     and d.event_context(event, harness) >= LARGE_CONTEXT_TOKENS)
            if small:
                thread["small_result_responses"] += 1
                thread["small_result_input_tokens"] += d.event_context(event, harness)
                thread["small_result_result_chars"] += sum(r[d.TOOL_CHARS] for r in results)
                cached = int(event[d.EVENT_KINDS + 1])
                thread["small_result_cached_input_tokens"] += cached
                thread["small_result_uncached_input_tokens"] += d.event_context(event, harness) - cached
                thread["small_result_weighted_units"] += units
                thread["small_result_unweighted_responses"] += not known_weight
                streaks[thread_id] += 1
                thread["small_result_max_streak"] = max(thread["small_result_max_streak"], streaks[thread_id])
            else:
                streaks[thread_id] = 0
        totals = _totals()
        for thread in threads.values():
            counts = tool_counts[(harness, session, thread["thread_id"] or "", index)]
            thread["tool_calls"], thread["result_chars"] = counts
            thread["mean_context_tokens"] = thread["input_tokens"] / thread["responses"] if thread["responses"] else 0
            _merge(totals, thread)
        totals["mean_context_tokens"] = totals["input_tokens"] / totals["responses"] if totals["responses"] else 0
        totals["small_result_weighted_share"] = totals["small_result_weighted_units"] / totals["weighted_units"] if totals["weighted_units"] else None
        totals["context_peak_tokens"] = max(d.event_context(e, harness) for e in events)
        _merge(summary, totals)
        harness_units[harness] += totals["weighted_units"]
        prompt = next((p for p in analysis.prompts.get((harness, session), []) if p.index == index), None)
        short = d.pick_id(session, known)
        flags = ["--session", short, "--prompt", index, *source_flags]
        next_steps = [
            d.hint(d.suggest(args, "tools", *flags, "--explain"), "inspect the repeated tool activity"),
            d.hint(d.suggest(args, "prompts", *flags, "--drilldown"), "compare root and subagent usage"),
        ]
        repeat = totals["small_result_responses"] >= REPEATED_RESPONSES
        many = totals["responses"] >= REPEATED_RESPONSES
        evidence = [
            "%d responses × %s average context = %s input tokens processed"
            % (totals["responses"], d.format_tokens(totals["mean_context_tokens"]), d.format_tokens(totals["input_tokens"])),
            "%s cached reads; %s uncached input; %s generated output"
            % tuple(d.format_tokens(totals[k]) for k in ("cached_input_tokens", "uncached_input_tokens", "output_tokens")),
        ]
        if totals["small_result_responses"]:
            evidence.append(
                "%d responses followed ≤%d estimated tokens of tool output at ≥%s context; longest same-thread run %d"
                % (totals["small_result_responses"], SMALL_RESULT_CHARS // int(d.CHARS_PER_TOKEN),
                   d.format_tokens(LARGE_CONTEXT_TOKENS), totals["small_result_max_streak"])
            )
        findings.append({
            "kind": "small_results" if repeat else "repeated_input" if many else "usage",
            "severity": "high" if repeat else "info", "harness": harness,
            "session_id": session, "short_id": short, "prompt": index,
            "cwd": prompt.cwd if prompt else "-",
            "title": "Small results, repeated large context" if repeat else "Repeated large input" if many else "Input workload",
            "evidence": evidence, "values": totals,
            "threads": sorted(threads.values(), key=lambda t: t["input_tokens"], reverse=True),
            "next": next_steps,
            "detail": (
                "Inspect polling, retries and repeated reads. Batch independent work and wait for completion before returning to the model; reduce retained context when it is no longer needed."
                if repeat else "Review the response count and retained context together; the input total counts cached reads too."
            ),
        })
    for finding in findings:
        denominator = harness_units[finding["harness"]]
        finding["share"] = finding["values"]["weighted_units"] / denominator if denominator else None
    # Repeated input stays visible even when a model has no known quota weights.
    findings.sort(key=lambda f: (f["values"]["input_tokens"], f["values"]["responses"], f["session_id"], f["prompt"]), reverse=True)
    summary["unlinked_tool_results"] = sum(unlinked.values())
    summary["mean_context_tokens"] = summary["input_tokens"] / summary["responses"] if summary["responses"] else 0
    summary["weighted_units_by_harness"] = dict(harness_units)
    small_units_by_harness = defaultdict(float)
    for finding in findings:
        small_units_by_harness[finding["harness"]] += finding["values"]["small_result_weighted_units"]
    summary["small_result_weighted_units_by_harness"] = dict(small_units_by_harness)
    if len(harness_units) > 1:
        summary["weighted_units"] = None  # different harness weight scales are not additive
        summary["small_result_weighted_units"] = None
    scope = {"harness": args.harness, "since": args.since, "until": args.until,
             "session_id": session_key[1] if session_key else None, "prompt": prompt_index}
    shown = findings[:args.top] if args.top else findings
    widen_command = d.suggest(args, "prompts", "--session", session_key[1], "--since", "30d", *source_flags) if session_key else d.suggest(args, "sessions", "--since", "30d", *source_flags)
    return {
        "schema": d.JSON_SCHEMA, "command": "auto", "generated_at": time.time(),
        "scope": scope, "summary": summary, "findings": shown,
        "findings_total": len(findings), "weight_source": analysis.weights.source_label,
        "next": shown[0]["next"] if shown else [d.hint(widen_command, "widen the range"), d.hint("nenpi config", "inspect transcript sources")],
        "thresholds": {"small_result_chars": SMALL_RESULT_CHARS, "minimum_context_tokens": LARGE_CONTEXT_TOKENS, "repeated_responses": REPEATED_RESPONSES},
        "notes": [
            "Ranked by cumulative input processed, including cached reads; this is not a quota percentage or guaranteed savings.",
            "Small-result links use strictly ordered timestamps within the same thread and prompt. Ties, missing threads and boundary results are excluded.",
            "Tool-result tokens are estimated as characters / 4. Generated output and retained context also contribute to usage.",
            "Root, subagent and unknown-thread responses are disjoint; cumulative thread counters are not added to response usage.",
        ],
    }


def attach_shake_findings(report, shakes):
    """Connect persisted Shake evidence to the workload; suggest previews only."""
    from . import drain as d

    report["shake"] = shakes
    for finding in report["findings"]:
        if finding.get("harness") != "codex":
            continue
        events = [e for e in shakes.get("events", [])
                  if e.get("session_id") == finding["session_id"] and e.get("prompt") == finding["prompt"]]
        if events:
            lower = sum(e.get("effect") == "context_lower_after_marker" for e in events)
            with_stats = sum(bool(e.get("stats")) for e in events)
            finding["evidence"].append("Shake: %d applied; %d had a lower-context request afterward; %d have matching transform stats." % (len(events), lower, with_stats))
        examples = sorted(events, key=lambda e: e.get("subsequent_response_count", 0), reverse=True)[:2]
        with_stats = next((e for e in sorted(events, key=lambda e: e.get("subsequent_response_count", 0), reverse=True) if e.get("stats")), None)
        if len(examples) == 2 and with_stats and not any(e.get("stats") for e in examples):
            examples[1] = with_stats
        for event in examples:
            before, after = event.get("before_context_tokens"), event.get("after_context_tokens")
            thread = d.short_id(event.get("thread_id") or "")
            if before is not None and after is not None:
                text = "Shake in thread %s: context %s → %s; uncached input %s → %s." % (
                    thread, d.format_tokens(before), d.format_tokens(after),
                    d.format_tokens(event.get("before_uncached_input_tokens") or 0),
                    d.format_tokens(event.get("after_uncached_input_tokens") or 0),
                )
                text += " Observed before/after, not measured quota savings."
            else:
                text = "Shake applied in thread %s; no unique before/after usage pair to assess its effect." % thread
            freed = (event.get("stats") or {}).get("tokens_freed_estimate")
            if freed is not None:
                text += " Shake's transform estimated %s tokens freed." % d.format_tokens(freed)
            finding["evidence"].append(text)
            later = event.get("subsequent_response_count", 0)
            finding["evidence"].append("%d later responses in that prompt processed %s input before the next Shake; this is observed usage, not tokens saved." % (later, d.format_tokens(event.get("subsequent_input_tokens", 0))))
            if event.get("cross_prompt"):
                finding["evidence"].append("Shake comparison crosses a prompt boundary; new instructions can change context too.")
            if event.get("before_model") and event.get("after_model") and event["before_model"] != event["after_model"]:
                finding["evidence"].append("The model also changed across Shake; weighted usage is not directly comparable.")
            if event.get("intervening_shake_count", 0) > 1:
                finding["evidence"].append("Multiple Shakes occurred between these requests; the combined context change cannot be assigned to one run.")
        candidates = [thread for thread in finding["threads"]
                      if thread["small_result_responses"] >= REPEATED_RESPONSES
                      and thread.get("last_context_tokens", 0) >= 100_000 and thread.get("thread_id")]
        if candidates:
            thread = max(candidates, key=lambda item: item["small_result_input_tokens"])
            finding["shake_candidate"] = {
                "thread_id": thread["thread_id"], "role": thread["role"],
                "last_context_tokens": thread["last_context_tokens"],
                "command": "/shake", "removable_tokens": None,
            }
            finding["evidence"].append(
                "Shake candidate on continuation: %s thread %s last read %s context tokens. Token counts alone do not establish what can be removed."
                % (thread["role"], d.short_id(thread["thread_id"]), d.format_tokens(thread["last_context_tokens"]))
            )
            finding["next"].append(d.hint("/shake", "when continuing Codex %s thread %s, review the preview before confirming a context reduction" % (thread["role"], thread["thread_id"])))
    for note in shakes.get("notes", []):
        if note not in report["notes"]:
            report["notes"].append(note)
    return report


def command_auto(args):
    from . import drain as d

    if args.prompt is not None and not args.session:
        d.warn("--prompt requires --session")
        return 2
    analysis = d.prepare(args)
    key = None
    if args.session:
        key, code = d.resolve_session_prefix(d.session_unit_totals(analysis), args.session, args.first)
        if key is None:
            return code
    report = build_report(analysis, key, args.prompt)
    from .shake_report import analyze_shakes

    attach_shake_findings(report, analyze_shakes(analysis, key, args.prompt))
    from .activity_model import normalize
    from .context_savings import POLICY, estimate_savings
    report["estimated_shake_savings"] = estimate_savings(
        normalize(analysis, key, args.prompt), report["shake"].get("events", []))
    report["savings_policy"] = POLICY
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        from .report_render import render_auto

        print(render_auto(report, args))
    return 0
