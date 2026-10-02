"""Activity-first reporting; all analytics consume normalized harness records."""
from collections import defaultdict
import json
import socket

from .activity_model import normalize
from .context_savings import POLICY, estimate_savings


def bucket_responses(responses):
    buckets = {}
    seen = defaultdict(set)
    for r in responses:
        activities = {op.activity for op in r.operations}
        activity = next(iter(activities)) if len(activities) == 1 else "Mixed activity" if activities else "No linked tool call"
        scope = (r.harness, r.account, r.session, r.thread, r.prompt)
        if "Coordinating agents" in activities:
            seen[scope] = {target for target in seen[scope] if target[0] != "Waiting for agents"}
        if any(not op.polling and op.activity != "Reading/searching code" for op in r.operations):
            seen[scope] = {target for target in seen[scope] if target[0] != "Reading/searching code"}
        def repeatable(op):
            return op.target and (op.polling or op.activity == "Reading/searching code")
        targets = {(op.activity, op.target) for op in r.operations if repeatable(op)}
        repeated = bool(r.thread and len(activities) == 1 and targets and all(repeatable(op) for op in r.operations) and targets <= seen[scope])
        seen[scope].update(targets)
        bucket = buckets.setdefault(activity, dict(
            activity=activity, responses=0, input_tokens=0, cached_input_tokens=0,
            output_tokens=0, repeated_responses=0, repeated_input_tokens=0,
            projects=set(), locations={}, usage_by_account={}, unweighted_responses=0, operation_groups={},
        ))
        if activity in {"Mixed activity", "Other / unknown"}:
            labels = sorted({op.detail if op.activity in {"Mixed activity", "Other / unknown"} and op.detail else op.activity for op in r.operations})
            label = " + ".join(labels)
            group = bucket["operation_groups"].setdefault(label, dict(operation=label, responses=0, input_tokens=0))
            group["responses"] += 1
            group["input_tokens"] += r.context
        bucket["responses"] += 1
        bucket["input_tokens"] += r.context
        bucket["cached_input_tokens"] += r.cached
        bucket["output_tokens"] += r.output
        bucket["repeated_responses"] += int(repeated)
        bucket["repeated_input_tokens"] += r.context if repeated else 0
        bucket["projects"].add(r.project)
        bucket["unweighted_responses"] += int(not r.prices)
        pool = (r.harness, r.account)
        units = bucket["usage_by_account"].setdefault(pool, dict(harness=r.harness, account=r.account, units=0, repeated_units=0))
        units["units"] += r.usage
        units["repeated_units"] += r.usage if repeated else 0
        location = bucket["locations"].setdefault(scope, dict(
            harness=r.harness, account=r.account, session_id=r.session, thread_id=r.thread,
            prompt=r.prompt, project=r.project, responses=0, input_tokens=0,
            repeated_responses=0, repeated_input_tokens=0,
        ))
        location["responses"] += 1
        location["input_tokens"] += r.context
        location["repeated_responses"] += int(repeated)
        location["repeated_input_tokens"] += r.context if repeated else 0
    for bucket in buckets.values():
        bucket["projects"] = sorted(bucket["projects"])
        bucket["locations"] = sorted(bucket["locations"].values(), key=lambda row: (-row["repeated_input_tokens"], -row["input_tokens"], row["session_id"]))
        bucket["operation_groups"] = sorted(bucket["operation_groups"].values(), key=lambda row: (-row["input_tokens"], row["operation"]))
        bucket["usage_by_account"] = list(bucket["usage_by_account"].values())
    return sorted(buckets.values(), key=lambda row: (-row["input_tokens"], row["activity"]))


def _polling_section(responses, top):
    """Polling and recipe flags keyed by session and prompt (docs/polling.md)."""
    from .polling_report import build
    report = build(responses, top=top, sweep=False)
    return {key: report[key] for key in ("threshold", "window_steps", "summary", "by_kind", "by_recipe", "sessions")}


def scoped_command(analysis, command, *flags):
    from . import drain as d
    flags = list(flags)
    for flag, attr in (("--codex-root", "codex_root"), ("--claude-root", "claude_root")):
        for root in getattr(analysis.args, attr, []) or []:
            flags.extend((flag, str(root)))
    if analysis.args.whole_session:
        flags.append("--whole-session")
    if analysis.args.use_calibrated:
        flags.append("--use-calibrated")
    if analysis.args.claude_cache_read_weight is not None:
        flags.extend(("--claude-cache-read-weight", str(analysis.args.claude_cache_read_weight)))
    if analysis.args.long_context_multiplier != 1:
        flags.extend(("--long-context-multiplier", str(analysis.args.long_context_multiplier)))
    return d.suggest(analysis.args, command, *flags)


def build_report(analysis, session_key=None, prompt_index=None):
    from . import drain as d
    from .shake_report import analyze_shakes
    responses = normalize(analysis, session_key, prompt_index)
    all_buckets = bucket_responses(responses)
    requested = getattr(analysis.args, "activity", None)
    matching = [row for row in all_buckets if not requested or requested.casefold() in row["activity"].casefold()]
    shown = matching[:analysis.args.top] if analysis.args.top else matching
    shakes = analyze_shakes(analysis, session_key, prompt_index)
    estimates = estimate_savings(responses, shakes.get("events", []))
    if requested:
        included = {(loc["harness"], loc["session_id"], loc["thread_id"], loc["prompt"])
                    for bucket in matching for loc in bucket["locations"]}
        estimates = [e for e in estimates if (e["harness"], e["session_id"], e["thread_id"], e["prompt"]) in included]
        shakes = dict(shakes, events=[e for e in shakes.get("events", [])
            if ("codex", e["session_id"], e["thread_id"], e.get("prompt")) in included])
    known = d.analysis_session_ids(analysis)
    for bucket in matching:
        for location in bucket["locations"]:
            location["inspect"] = scoped_command(analysis, "auto", "--harness", location["harness"],
                "--session", d.pick_id(location["session_id"], known), "--prompt", location["prompt"])
        bucket["inspect"] = scoped_command(analysis, "activities", "--activity", bucket["activity"])
    for estimate in estimates:
        estimate["inspect"] = scoped_command(analysis, "auto", "--harness", estimate["harness"],
            "--session", d.pick_id(estimate["session_id"], known), "--prompt", estimate["prompt"])
    roots = []
    for harness in ("codex", "claude"):
        if analysis.args.harness not in (harness, "all"):
            continue
        for root in d.resolve_roots(harness, getattr(analysis.args, harness + "_root", None) or [], d.load_config()):
            roots.append(dict(host=socket.gethostname(), harness=harness, path=str(root), account=root.name))
    return dict(
        schema_version=1, scope=dict(since=analysis.args.since, until=analysis.args.until,
            harness=analysis.args.harness, session_id=session_key[1] if session_key else None,
            prompt=prompt_index, activity=requested, project=getattr(analysis.args, "project", []),
            exclude_project=getattr(analysis.args, "exclude_project", [])), sources=roots,
        ranking="input_tokens", summary=dict(responses=sum(b["responses"] for b in matching),
            input_tokens=sum(b["input_tokens"] for b in matching),
            repeated_responses=sum(b["repeated_responses"] for b in matching),
            repeated_input_tokens=sum(b["repeated_input_tokens"] for b in matching),
            unclassified_responses=sum(b["responses"] for b in matching if b["activity"] == "Other / unknown")),
        activities=shown, repetition=sorted([b for b in matching if b["repeated_responses"]],
            key=lambda b: -b["repeated_input_tokens"])[:3],
        total_activities=len(matching), shake=shakes,
        estimated_shake_savings=estimates, savings_policy=POLICY,
        polling=_polling_section(responses, analysis.args.top),
        next=[dict(cmd=scoped_command(analysis, "sessions"), why="See sessions"),
              dict(cmd=scoped_command(analysis, "activities", "--top", "0"), why="See all activities")],
    )


def command_activities(args):
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
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        from .report_render import render_activities
        print(render_activities(report, args))
    return 0
